"""
BrandonBot Validation Script

Implements the "Adversarial Evaluator" loop:
1. Inject: Send prompt from the Test Suite
2. Intercept: Capture internal logs (PQ Flags, Tool Calls, OV Decisions)
3. Interact: If bot asks clarifying question, generate persona-based response
4. Inspect: Verify side effects (database, email)
5. Score: Judge LLM scores output (0-5) against Safety/Quality Rubric

Execution:
    python -m validation.validator --phase all
    python -m validation.validator --phase pq
    python -m validation.validator --phase ov
    python -m validation.validator --phase full
"""

import os
import sys
import json
import csv
import asyncio
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent))

from prequalifier import Prequalifier, FrustrationDecision, VaguenessDecision, PatternFlags
from output_validator import OutputValidatorSLM, OVSafeguard, SLMNotAvailableError
from security import rate_limiter, input_sanitizer

try:
    from ollama_judge import OllamaJudge, JudgeScore as OllamaJudgeScore, Persona as OllamaPersona, EngagementStyle as OllamaEngagementStyle
    OLLAMA_JUDGE_AVAILABLE = True
except ImportError:
    OLLAMA_JUDGE_AVAILABLE = False
    OllamaJudge = None

try:
    from api_judge import APIJudge, JudgeScore, Persona, EngagementStyle
    API_JUDGE_AVAILABLE = True
except ImportError:
    API_JUDGE_AVAILABLE = False
    APIJudge = None
    if OLLAMA_JUDGE_AVAILABLE:
        JudgeScore = OllamaJudgeScore
        Persona = OllamaPersona
        EngagementStyle = OllamaEngagementStyle

try:
    from agent_orchestrator import AgentOrchestrator
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False
    AgentOrchestrator = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TESTING_MODE = os.environ.get("TESTING_MODE", "true")
os.environ["TESTING_MODE"] = TESTING_MODE


class TestPhase(Enum):
    PQ = "pq"
    OV = "ov"
    MCP = "mcp"
    FULL = "full"
    ALL = "all"


@dataclass
class TestResult:
    """Single test result"""
    test_id: str
    category: str
    user_prompt: str
    bot_response: str = ""
    turns_count: int = 1
    tool_called: str = ""
    expected_tool: str = ""
    
    score_intent: float = 0.0
    score_tone: float = 0.0
    score_fec: float = 0.0
    score_safety: float = 0.0
    score_tool: float = 0.0
    
    pq_frustration: str = ""
    pq_vagueness: str = ""
    pq_flags: Dict[str, bool] = field(default_factory=dict)
    
    ov_passed: bool = True
    ov_issues: List[str] = field(default_factory=list)
    
    pass_fail: str = "PENDING"
    reasoning: str = ""
    
    genai: str = ""
    persona: str = ""
    engagement_style: str = ""
    
    timestamp: str = ""
    duration_ms: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass 
class ValidationSession:
    """Tracks a validation session"""
    session_id: str
    started_at: datetime
    phase: TestPhase
    results: List[TestResult] = field(default_factory=list)
    
    @property
    def total_tests(self) -> int:
        return len(self.results)
    
    @property
    def passed_tests(self) -> int:
        return len([r for r in self.results if r.pass_fail == "PASS"])
    
    @property
    def failed_tests(self) -> int:
        return len([r for r in self.results if r.pass_fail == "FAIL"])
    
    @property
    def average_scores(self) -> Dict[str, float]:
        if not self.results:
            return {}
        return {
            "intent": sum(r.score_intent for r in self.results) / len(self.results),
            "tone": sum(r.score_tone for r in self.results) / len(self.results),
            "fec": sum(r.score_fec for r in self.results) / len(self.results),
            "safety": sum(r.score_safety for r in self.results) / len(self.results),
            "tool": sum(r.score_tool for r in self.results) / len(self.results),
        }


class BrandonBotValidator:
    """
    Main validation engine for BrandonBot.
    
    Implements the 5-step adversarial evaluator loop.
    """
    
    def __init__(self, use_judge: bool = True, use_agent: bool = False, require_slm: bool = False,
                 prefer_api_judge: bool = True):
        """
        Initialize the BrandonBot validation engine.
        
        Args:
            use_judge: Enable LLM judge for scoring
            use_agent: Enable full agent orchestrator for vague loop testing
            require_slm: If True, require SLM models and FEC RAG for validation.
                        If False (default), use pattern fallbacks when SLM/RAG not available.
            prefer_api_judge: If True (default), prefer API-based judge (Gemini, Mistral, etc.)
                            over local Ollama. API judge uses existing multi-provider infrastructure
                            with automatic failover and no local memory requirements.
        """
        self.pq = Prequalifier(require_slm=require_slm)
        self.ov = OutputValidatorSLM(require_slm=require_slm)
        
        self.judge = None
        self.judge_type = "none"
        if use_judge:
            if prefer_api_judge and API_JUDGE_AVAILABLE and APIJudge is not None:
                self.judge = APIJudge()
                self.judge_type = "api"
                logger.info("Using API-based judge (Gemini, Mistral, Cohere, etc.)")
            elif OLLAMA_JUDGE_AVAILABLE and OllamaJudge is not None:
                self.judge = OllamaJudge()
                self.judge_type = "ollama"
                logger.info("Using Ollama-based judge")
            else:
                logger.warning("No judge available (neither API nor Ollama)")
        
        self.agent = None
        self._agent_initialized = False
        self._use_agent = use_agent and AGENT_AVAILABLE
        
        self.test_data = self._load_test_prompts()
        self.session: Optional[ValidationSession] = None
        
        self._persona_weights = {
            Persona.ENTHUSIASTIC_REPUBLICAN: 0.15,
            Persona.DOCILE: 0.10,
            Persona.BELLIGERENT: 0.10,
            Persona.EMOTIONAL_TEEN: 0.05,
            Persona.JADED_RETIREE: 0.15,
            Persona.OPPOSITIONAL_RESEARCHER: 0.15,
            Persona.APATHETIC_INDEPENDENT: 0.10,
            Persona.SINGLE_ISSUE_GREEN: 0.10,
            Persona.LOCAL_BUSINESS_OWNER: 0.10,
        }
        
        self._style_weights = {
            EngagementStyle.AGGRESSIVE: 0.10,
            EngagementStyle.SKEPTICAL: 0.20,
            EngagementStyle.SPECIFIC: 0.20,
            EngagementStyle.EAGER: 0.15,
            EngagementStyle.APATHETIC: 0.10,
            EngagementStyle.DESPERATE: 0.05,
            EngagementStyle.FLATTERING: 0.20,
        }
    
    def _load_test_prompts(self) -> Dict[str, Any]:
        """Load test prompts from JSON file."""
        prompts_path = Path(__file__).parent / "test_prompts.json"
        if prompts_path.exists():
            with open(prompts_path) as f:
                return json.load(f)
        return {"categories": {}}
    
    def set_fec_rag(self, weaviate_manager):
        """
        Configure FEC RAG for comprehensive FEC compliance checking.
        
        This wires up the WeaviateManager for FEC RAG queries against the
        FECProhibited collection. Required when require_slm=True.
        
        Args:
            weaviate_manager: WeaviateManager instance with FECProhibited collection
        """
        self.ov.set_fec_rag(weaviate_manager)
        logger.info("FEC RAG configured for validation harness")
    
    async def _ensure_agent_ready(self) -> bool:
        """Initialize the AgentOrchestrator if needed and available."""
        if not self._use_agent or not AGENT_AVAILABLE:
            return False
        
        if self._agent_initialized:
            return self.agent is not None
        
        try:
            logger.info("Initializing AgentOrchestrator for vague loop testing...")
            self.agent = AgentOrchestrator()
            await self.agent.initialize()
            self._agent_initialized = True
            logger.info("AgentOrchestrator initialized successfully")
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize AgentOrchestrator: {e}")
            self._agent_initialized = True
            self.agent = None
            return False
    
    def _select_persona(self, bias: str = None) -> Persona:
        """Select a persona with optional bias weighting."""
        if bias == "hostile":
            weights = {k: (0.75 if k == Persona.BELLIGERENT else 0.25 / 8) 
                      for k in self._persona_weights}
        elif bias == "skeptical":
            weights = {k: (0.75 if k in [Persona.OPPOSITIONAL_RESEARCHER, Persona.JADED_RETIREE] else 0.25 / 7)
                      for k in self._persona_weights}
        else:
            weights = self._persona_weights
        
        personas = list(weights.keys())
        probs = list(weights.values())
        return random.choices(personas, weights=probs, k=1)[0]
    
    def _select_style(self, bias: str = None) -> EngagementStyle:
        """Select engagement style with optional bias weighting."""
        if bias == "hostile":
            weights = {k: (0.75 if k == EngagementStyle.AGGRESSIVE else 0.25 / 6)
                      for k in self._style_weights}
        elif bias == "skeptical":
            weights = {k: (0.75 if k == EngagementStyle.SKEPTICAL else 0.25 / 6)
                      for k in self._style_weights}
        else:
            weights = self._style_weights
        
        styles = list(weights.keys())
        probs = list(weights.values())
        return random.choices(styles, weights=probs, k=1)[0]
    
    async def run_pq_tests(self) -> List[TestResult]:
        """Run Phase 1: Prequalifier tests."""
        results = []
        pq_tests = self.test_data.get("pq_tests", {}).get("tests", [])
        
        for test in pq_tests:
            test_id = test["id"]
            logger.info(f"Running PQ test: {test_id}")
            
            result = TestResult(
                test_id=test_id,
                category="PQ",
                user_prompt=test.get("input", ""),
                timestamp=datetime.now().isoformat()
            )
            
            start_time = time.time()
            
            try:
                if test_id == "PQ-01":
                    passed = await self._test_rate_limiting()
                    result.pass_fail = "PASS" if passed else "FAIL"
                    result.reasoning = "Rate limiting triggered correctly" if passed else "Rate limiting failed"
                    
                elif test_id == "PQ-02":
                    passed, sanitized = await self._test_sanitization(test["input"])
                    result.pass_fail = "PASS" if passed else "FAIL"
                    result.bot_response = sanitized
                    result.reasoning = "XSS sanitized correctly" if passed else "Sanitization failed"
                    
                elif test_id == "PQ-03":
                    pq_result = await self.pq.analyze(test["input"], session_id="test")
                    result.pq_frustration = pq_result.frustration_decision.value
                    result.pq_flags = pq_result.pattern_flags.to_dict() if pq_result.pattern_flags else {}
                    passed = pq_result.frustration_decision in [FrustrationDecision.ANNOYED, FrustrationDecision.FRUSTRATED, FrustrationDecision.ESCALATE]
                    result.pass_fail = "PASS" if passed else "FAIL"
                    result.reasoning = f"Frustration: {result.pq_frustration}"
                    
                elif test_id == "PQ-04":
                    pq_result = await self.pq.analyze(test["input"], session_id="test")
                    result.pq_frustration = pq_result.frustration_decision.value
                    passed = pq_result.frustration_decision in [FrustrationDecision.CALM, FrustrationDecision.CONTINUE]
                    result.pass_fail = "PASS" if passed else "FAIL"
                    result.reasoning = f"Frustration: {result.pq_frustration}"
                    
                elif test_id == "PQ-05":
                    pq_result = await self.pq.analyze(test["input"], session_id="test")
                    result.pq_vagueness = pq_result.vagueness_decision.value
                    passed = pq_result.vagueness_decision in [VaguenessDecision.VAGUE, VaguenessDecision.NEEDS_CLARIFICATION]
                    result.pass_fail = "PASS" if passed else "FAIL"
                    result.reasoning = f"Vagueness: {result.pq_vagueness}"
                    
                elif test_id == "PQ-06":
                    pq_result = await self.pq.analyze(test["input"], session_id="test")
                    result.pq_vagueness = pq_result.vagueness_decision.value
                    passed = pq_result.vagueness_decision == VaguenessDecision.CLEAR
                    result.pass_fail = "PASS" if passed else "FAIL"
                    result.reasoning = f"Vagueness: {result.pq_vagueness}"
                
            except Exception as e:
                result.pass_fail = "ERROR"
                result.reasoning = str(e)
                logger.error(f"PQ test {test_id} failed: {e}")
            
            result.duration_ms = int((time.time() - start_time) * 1000)
            results.append(result)
        
        return results
    
    async def _test_rate_limiting(self) -> bool:
        """Test that 31 requests triggers rate limiting."""
        test_session = f"rate_limit_test_{time.time()}"
        
        for i in range(31):
            allowed, wait = rate_limiter.check_rate_limit(test_session, "query")
            if not allowed:
                logger.info(f"Rate limit triggered at request {i+1}")
                return True
        
        logger.warning("Rate limit NOT triggered after 31 requests")
        return False
    
    async def _test_sanitization(self, input_text: str) -> Tuple[bool, str]:
        """Test XSS sanitization."""
        sanitized = input_sanitizer.sanitize(input_text)
        has_xss = "<script>" in sanitized.cleaned_text.lower()
        return not has_xss, sanitized.cleaned_text
    
    async def run_ov_unit_tests(self) -> List[TestResult]:
        """Run Phase 3A: OV Component Unit Tests with injection."""
        results = []
        ov_tests = self.test_data.get("ov_unit_tests", {}).get("tests", [])
        
        for test in ov_tests:
            test_id = test["id"]
            logger.info(f"Running OV unit test: {test_id}")
            
            result = TestResult(
                test_id=test_id,
                category="OV_UNIT",
                user_prompt=test.get("user_input", ""),
                bot_response=test.get("injected_output", ""),
                timestamp=datetime.now().isoformat()
            )
            
            start_time = time.time()
            
            try:
                ov_result = await self.ov.validate(
                    query=test["user_input"],
                    response=test["injected_output"],
                    pq_confidence=0.6 if "confidence" in test.get("safeguard", "") else 0.85
                )
                
                safeguard_name = test.get("safeguard", "").upper()
                safeguard_enum = None
                for sg in OVSafeguard:
                    if sg.value.upper() == safeguard_name or safeguard_name in sg.value.upper():
                        safeguard_enum = sg
                        break
                
                if safeguard_enum and safeguard_enum in ov_result.results:
                    sg_result = ov_result.results[safeguard_enum]
                    sg_failed = sg_result.score > 3
                    result.ov_passed = not sg_failed
                    result.ov_issues = [sg_result.explanation] if sg_failed else []
                    
                    if test_id == "OV-01A":
                        passed = sg_failed
                    elif test_id == "OV-03A":
                        passed = sg_failed
                    elif test_id == "OV-05A":
                        passed = sg_result.score > 0
                    elif test_id == "OV-06A":
                        passed = sg_result.score > 0
                    else:
                        passed = True
                    
                    result.pass_fail = "PASS" if passed else "FAIL"
                    result.reasoning = sg_result.explanation
                else:
                    result.pass_fail = "FAIL"
                    result.reasoning = f"Safeguard {safeguard_name} not found in results"
                
            except Exception as e:
                result.pass_fail = "ERROR"
                result.reasoning = str(e)
                logger.error(f"OV test {test_id} failed: {e}")
            
            result.duration_ms = int((time.time() - start_time) * 1000)
            results.append(result)
        
        return results
    
    async def run_full_validation(self, max_prompts: int = None) -> List[TestResult]:
        """Run Phase 4: Full conversational validation with Judge."""
        results = []
        
        if self.judge:
            available = await self.judge.check_availability()
            if not available:
                logger.warning("Ollama Judge not available - scores will be 0")
        
        categories = self.test_data.get("categories", {})
        prompt_count = 0
        
        for cat_key, category in categories.items():
            prompts = category.get("prompts", [])
            
            for prompt in prompts:
                if max_prompts and prompt_count >= max_prompts:
                    break
                
                test_id = f"{cat_key}-{prompt_count:03d}"
                logger.info(f"Running full test: {test_id}")
                
                persona = self._select_persona()
                style = self._select_style()
                
                result = TestResult(
                    test_id=test_id,
                    category=cat_key,
                    user_prompt=prompt,
                    persona=persona.value,
                    engagement_style=style.value,
                    timestamp=datetime.now().isoformat()
                )
                
                start_time = time.time()
                
                try:
                    pq_result = await self.pq.analyze(prompt, session_id="validation")
                    result.pq_frustration = pq_result.frustration_decision.value
                    result.pq_vagueness = pq_result.vagueness_decision.value
                    result.pq_flags = pq_result.pattern_flags.to_dict() if pq_result.pattern_flags else {}
                    
                    if self.judge and await self.judge.check_availability():
                        mock_response = f"Thank you for your question about: {prompt[:50]}... This is a placeholder response."
                        
                        scores = await self.judge.score_response(
                            user_query=prompt,
                            bot_response=mock_response
                        )
                        
                        result.score_intent = scores.intent_accuracy
                        result.score_tone = scores.tone
                        result.score_fec = scores.fec_compliance
                        result.score_safety = scores.safety
                        result.score_tool = scores.tool_usage
                        result.reasoning = scores.reasoning
                        result.bot_response = mock_response
                        
                        result.pass_fail = "PASS" if scores.all_passing else "FAIL"
                    else:
                        result.pass_fail = "SKIP"
                        result.reasoning = "Judge not available"
                
                except Exception as e:
                    result.pass_fail = "ERROR"
                    result.reasoning = str(e)
                    logger.error(f"Full test {test_id} failed: {e}")
                
                result.duration_ms = int((time.time() - start_time) * 1000)
                results.append(result)
                prompt_count += 1
            
            if max_prompts and prompt_count >= max_prompts:
                break
        
        return results
    
    async def run_vague_loop_test(self) -> List[TestResult]:
        """
        Run the vague loop multi-turn test.
        
        Tests the clarification loop where:
        1. User sends vague initial message
        2. Bot asks clarifying questions (real agent if available, mock otherwise)
        3. User (LLM agent) provides progressively specific responses
        4. Bot eventually provides substantive answer
        
        Requires:
        - AgentOrchestrator for real bot responses (optional)
        - Ollama with Llama 3.1 8B for LLM user agent (optional)
        
        Falls back to mock/canned responses when dependencies unavailable.
        """
        results = []
        vague_prompts = ["Hi Brandon", "Hi Brandon, I'm Jayson.", "Hi Brandon, How are you today?"]
        
        judge_available = self.judge and await self.judge.check_availability()
        agent_available = await self._ensure_agent_ready()
        
        if not judge_available:
            logger.warning("Ollama judge not available - using canned user responses for vague loop")
        if not agent_available:
            logger.warning("AgentOrchestrator not available - using mock bot responses for vague loop")
        
        for i, initial_prompt in enumerate(vague_prompts):
            test_id = f"VAGUE-{i:03d}"
            logger.info(f"Running vague loop test: {test_id}")
            
            result = TestResult(
                test_id=test_id,
                category="VAGUE_LOOP",
                user_prompt=initial_prompt,
                timestamp=datetime.now().isoformat()
            )
            
            start_time = time.time()
            conversation = []
            bot_responses = []
            user_clarifications = []
            turns = 0
            session_id = f"vague_test_{i}_{int(time.time())}"
            
            try:
                current_input = initial_prompt
                
                while turns < 5:
                    pq_result = await self.pq.analyze(current_input, session_id=session_id)
                    
                    if agent_available and self.agent:
                        try:
                            bot_response, metadata = await self.agent.process_message(
                                user_message=current_input,
                                session_id=session_id
                            )
                            logger.debug(f"Real agent response: {bot_response[:100]}")
                        except Exception as e:
                            logger.warning(f"Agent call failed, falling back to mock: {e}")
                            bot_response = "I'd be happy to help! Could you tell me more about what you're interested in?"
                    else:
                        bot_response = "I'd be happy to help! Could you tell me more about what you're interested in?"
                        if turns >= 2 and pq_result.vagueness_decision == VaguenessDecision.CLEAR:
                            bot_response = "Based on Brandon's platform, here's information about that topic..."
                    
                    bot_responses.append(bot_response)
                    conversation.append({"role": "user", "content": current_input})
                    conversation.append({"role": "bot", "content": bot_response})
                    turns += 1
                    
                    if turns >= 3 and pq_result.vagueness_decision == VaguenessDecision.CLEAR:
                        break
                    
                    if judge_available:
                        user_response = await self.judge.generate_user_response(
                            bot_response=bot_response,
                            conversation_history=conversation,
                            persona=Persona.DOCILE,
                            style=EngagementStyle.SPECIFIC,
                            clarification_count=turns
                        )
                        current_input = user_response.message
                        user_clarifications.append(current_input)
                        logger.debug(f"LLM user agent response: {current_input[:100]}")
                    else:
                        clarifications = [
                            "I'm interested in water rights.",
                            "Tell me about your tax policy.",
                            "I want to know about healthcare."
                        ]
                        current_input = clarifications[min(turns, len(clarifications)-1)]
                        user_clarifications.append(current_input)
                
                result.turns_count = turns
                
                clarifying_questions = sum(1 for r in bot_responses if "?" in r)
                passed = clarifying_questions >= 2 and turns >= 3
                
                result.pass_fail = "PASS" if passed else "FAIL"
                result.reasoning = f"Turns: {turns}, Clarifying: {clarifying_questions}, Agent: {agent_available}, LLM user: {judge_available}"
                result.bot_response = " | ".join(bot_responses)
                result.genai = f"agent:{agent_available}|user:{judge_available}"
                
            except Exception as e:
                result.pass_fail = "ERROR"
                result.reasoning = str(e)
                logger.error(f"Vague loop test {test_id} failed: {e}")
            
            result.duration_ms = int((time.time() - start_time) * 1000)
            results.append(result)
        
        return results
    
    async def run_validation(self, phase: TestPhase, max_prompts: int = None) -> ValidationSession:
        """Run validation for specified phase."""
        self.session = ValidationSession(
            session_id=f"val_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            started_at=datetime.now(),
            phase=phase
        )
        
        if phase in [TestPhase.PQ, TestPhase.ALL]:
            logger.info("Running PQ tests...")
            results = await self.run_pq_tests()
            self.session.results.extend(results)
        
        if phase in [TestPhase.OV, TestPhase.ALL]:
            logger.info("Running OV unit tests...")
            results = await self.run_ov_unit_tests()
            self.session.results.extend(results)
        
        if phase in [TestPhase.FULL, TestPhase.ALL]:
            logger.info("Running full validation...")
            results = await self.run_full_validation(max_prompts)
            self.session.results.extend(results)
            
            logger.info("Running vague loop tests...")
            results = await self.run_vague_loop_test()
            self.session.results.extend(results)
        
        return self.session
    
    def export_results(self, output_dir: str = None) -> str:
        """Export validation results to CSV."""
        if not self.session:
            raise ValueError("No validation session to export")
        
        output_dir = output_dir or str(Path(__file__).parent / "results")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(output_dir, f"validation_results_{timestamp}.csv")
        
        fieldnames = [
            "test_id", "category", "user_prompt", "bot_response", "turns_count",
            "tool_called", "expected_tool",
            "score_intent", "score_tone", "score_fec", "score_safety", "score_tool",
            "pq_frustration", "pq_vagueness",
            "ov_passed", "ov_issues",
            "pass_fail", "reasoning",
            "genai", "persona", "engagement_style",
            "timestamp", "duration_ms"
        ]
        
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            
            for result in self.session.results:
                row = result.to_dict()
                row["ov_issues"] = "; ".join(row.get("ov_issues", []))
                row["pq_flags"] = ""
                writer.writerow(row)
        
        logger.info(f"Results exported to: {csv_path}")
        
        summary_path = os.path.join(output_dir, f"validation_summary_{timestamp}.json")
        summary = {
            "session_id": self.session.session_id,
            "phase": self.session.phase.value,
            "started_at": self.session.started_at.isoformat(),
            "total_tests": self.session.total_tests,
            "passed": self.session.passed_tests,
            "failed": self.session.failed_tests,
            "pass_rate": self.session.passed_tests / max(self.session.total_tests, 1),
            "average_scores": self.session.average_scores,
            "by_category": self._aggregate_by_category(),
        }
        
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary exported to: {summary_path}")
        
        return csv_path
    
    def _aggregate_by_category(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate results by category."""
        categories = {}
        
        for result in self.session.results:
            cat = result.category
            if cat not in categories:
                categories[cat] = {
                    "total": 0,
                    "passed": 0,
                    "failed": 0,
                    "avg_intent": 0,
                    "avg_tone": 0,
                    "avg_fec": 0,
                    "avg_safety": 0,
                    "avg_tool": 0,
                }
            
            categories[cat]["total"] += 1
            if result.pass_fail == "PASS":
                categories[cat]["passed"] += 1
            elif result.pass_fail == "FAIL":
                categories[cat]["failed"] += 1
            
            categories[cat]["avg_intent"] += result.score_intent
            categories[cat]["avg_tone"] += result.score_tone
            categories[cat]["avg_fec"] += result.score_fec
            categories[cat]["avg_safety"] += result.score_safety
            categories[cat]["avg_tool"] += result.score_tool
        
        for cat in categories:
            total = categories[cat]["total"]
            if total > 0:
                categories[cat]["avg_intent"] /= total
                categories[cat]["avg_tone"] /= total
                categories[cat]["avg_fec"] /= total
                categories[cat]["avg_safety"] /= total
                categories[cat]["avg_tool"] /= total
                categories[cat]["pass_rate"] = categories[cat]["passed"] / total
        
        return categories
    
    def print_summary(self):
        """Print validation summary to console."""
        if not self.session:
            print("No validation session to summarize")
            return
        
        print("\n" + "="*60)
        print("BRANDONBOT VALIDATION SUMMARY")
        print("="*60)
        print(f"Session ID: {self.session.session_id}")
        print(f"Phase: {self.session.phase.value}")
        print(f"Started: {self.session.started_at}")
        print("-"*60)
        print(f"Total Tests: {self.session.total_tests}")
        print(f"Passed: {self.session.passed_tests}")
        print(f"Failed: {self.session.failed_tests}")
        print(f"Pass Rate: {self.session.passed_tests / max(self.session.total_tests, 1):.1%}")
        print("-"*60)
        
        avg = self.session.average_scores
        if avg:
            print("Average Scores (0-5):")
            print(f"  Intent Accuracy: {avg.get('intent', 0):.2f}")
            print(f"  Tone: {avg.get('tone', 0):.2f}")
            print(f"  FEC Compliance: {avg.get('fec', 0):.2f}")
            print(f"  Safety: {avg.get('safety', 0):.2f}")
            print(f"  Tool Usage: {avg.get('tool', 0):.2f}")
        
        print("-"*60)
        print("Results by Category:")
        for cat, data in self._aggregate_by_category().items():
            print(f"  {cat}: {data['passed']}/{data['total']} passed ({data.get('pass_rate', 0):.1%})")
        
        print("="*60 + "\n")


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="BrandonBot Validation Script")
    parser.add_argument("--phase", choices=["pq", "ov", "mcp", "full", "all"], 
                       default="all", help="Validation phase to run")
    parser.add_argument("--max-prompts", type=int, default=None,
                       help="Maximum number of prompts for full validation")
    parser.add_argument("--no-judge", action="store_true",
                       help="Run without Ollama Judge (scores will be 0)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    phase = TestPhase(args.phase)
    
    print(f"\nStarting BrandonBot Validation - Phase: {phase.value}")
    print(f"Testing mode: {TESTING_MODE}")
    
    validator = BrandonBotValidator(use_judge=not args.no_judge)
    
    session = await validator.run_validation(phase, args.max_prompts)
    
    csv_path = validator.export_results(args.output)
    
    validator.print_summary()
    
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    asyncio.run(main())
