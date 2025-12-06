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
from ollama_judge import OllamaJudge, JudgeScore, Persona, EngagementStyle
from validation_debug import get_debug_db, sanitize_bot_response
from structured_response import parse_structured_response

try:
    from agent_orchestrator import AgentOrchestrator
    from weaviate_manager import WeaviateManager
    from slm_manager import SLMManager
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False
    AgentOrchestrator = None
    WeaviateManager = None
    SLMManager = None

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
class ConversationTurn:
    """Single turn in a conversation."""
    turn_number: int
    user_prompt: str
    bot_response: str
    tool_called: str = ""
    pq_frustration: str = ""
    pq_vagueness: str = ""
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TestResult:
    """Test result tracking a full conversation
    
    Scoring dimensions (0-5 scale, 0=worst, 5=best):
    - Clarity: Is the response easy to understand?
    - Empathy: Does it acknowledge the user's perspective?
    - Accuracy: Are the facts and policies correct?
    - Engagement: Does it encourage further interaction or action?
    - Tone: Is it professional yet approachable?
    - Alignment: Does it align with AZ-01 district interests?
    
    Scores apply to the ENTIRE conversation, not per-turn.
    Pass/Fail: PASS if all scores > 3 AND Tool_Called == Expected_Tool
    """
    test_id: str
    category: str
    
    turns: List[ConversationTurn] = field(default_factory=list)
    
    user_prompt: str = ""
    bot_response: str = ""
    turns_count: int = 1
    tool_called: str = ""
    expected_tool: str = ""
    
    score_clarity: float = 0.0
    score_empathy: float = 0.0
    score_accuracy: float = 0.0
    score_engagement: float = 0.0
    score_tone: float = 0.0
    score_alignment: float = 0.0
    
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
    
    def add_turn(self, user_prompt: str, bot_response: str, tool_called: str = "",
                 pq_frustration: str = "", pq_vagueness: str = "") -> None:
        """Add a conversation turn.
        
        The bot_response is processed to extract only user-facing content:
        1. First tries structured JSON parsing (reasoning + final_response)
        2. Falls back to delimiter parsing (<final_response>)
        3. Finally uses regex sanitization for chatter removal
        
        Internal reasoning is logged to debug.db for investigation.
        """
        # Primary: Try structured response parsing (JSON or delimiters)
        parsed = parse_structured_response(bot_response)
        debug_db = get_debug_db()
        
        if parsed.parse_method in ("json", "delimiter"):
            # Successfully extracted structured response
            clean_response = parsed.final_response
            
            # Log reasoning to debug DB
            if parsed.reasoning:
                debug_db.log_reasoning(
                    session_id="",
                    request_id=self.test_id,
                    reasoning=parsed.reasoning,
                    parse_method=parsed.parse_method,
                    raw_response=parsed.raw_response[:2000]
                )
        else:
            # Fallback: Use regex sanitization for remaining chatter
            clean_response = sanitize_bot_response(parsed.final_response)
            
            # Always log fallback parsing for forensic visibility
            # The "reasoning" in fallback is the chatter that was stripped
            stripped_content = bot_response[:len(bot_response) - len(clean_response)] if len(bot_response) > len(clean_response) else ""
            if bot_response != clean_response or stripped_content:
                debug_db.log_reasoning(
                    session_id="",
                    request_id=self.test_id,
                    reasoning=f"[FALLBACK STRIPPED] {stripped_content[:500] if stripped_content else 'Minor sanitization applied'}",
                    parse_method=f"fallback_{parsed.parse_method}",
                    raw_response=bot_response[:2000]
                )
                debug_db.log_raw_llm_response(
                    query=user_prompt,
                    raw_response=bot_response,
                    sanitized_response=clean_response,
                    test_id=self.test_id,
                    session_id=""
                )
        
        turn = ConversationTurn(
            turn_number=len(self.turns) + 1,
            user_prompt=user_prompt,
            bot_response=clean_response,
            tool_called=tool_called,
            pq_frustration=pq_frustration,
            pq_vagueness=pq_vagueness,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        self.turns.append(turn)
        self.turns_count = len(self.turns)
        self.user_prompt = user_prompt
        self.bot_response = clean_response
        self.tool_called = tool_called
    
    def get_full_conversation(self) -> str:
        """Get the full conversation as a formatted string for scoring."""
        parts = []
        for turn in self.turns:
            parts.append(f"User: {turn.user_prompt}")
            parts.append(f"Bot: {turn.bot_response}")
        return "\n".join(parts)
    
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
            "clarity": sum(r.score_clarity for r in self.results) / len(self.results),
            "empathy": sum(r.score_empathy for r in self.results) / len(self.results),
            "accuracy": sum(r.score_accuracy for r in self.results) / len(self.results),
            "engagement": sum(r.score_engagement for r in self.results) / len(self.results),
            "tone": sum(r.score_tone for r in self.results) / len(self.results),
            "alignment": sum(r.score_alignment for r in self.results) / len(self.results),
        }


class BrandonBotValidator:
    """
    Main validation engine for BrandonBot.
    
    Implements the 5-step adversarial evaluator loop.
    """
    
    def __init__(self, use_judge: bool = True, use_agent: bool = False, require_slm: bool = True):
        """
        Initialize the BrandonBot validation engine.
        
        Args:
            use_judge: Enable Ollama LLM judge for scoring
            use_agent: Enable full agent orchestrator for vague loop testing
            require_slm: If True (default), require SLM models for validation.
                        This ensures proper intent checking via MS-MARCO cross-encoder.
                        If False, fall back to pattern-only checking (NOT recommended).
        """
        self.pq = Prequalifier(require_slm=require_slm)
        self.ov = OutputValidatorSLM(require_slm=require_slm)
        self.judge = OllamaJudge() if use_judge else None
        self.agent = None
        self._weaviate = None
        self._slm_manager = None
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
        """Initialize the AgentOrchestrator with all required dependencies."""
        if not self._use_agent or not AGENT_AVAILABLE:
            return False
        
        if self._agent_initialized:
            return self.agent is not None
        
        try:
            logger.info("Initializing SLMManager for validation...")
            self._slm_manager = SLMManager()
            logger.info("SLMManager created (lazy loading)")
            
            logger.info("Wiring SLM provider to Prequalifier...")
            self.pq.set_slm_provider(self._slm_manager)
            logger.info("Prequalifier SLM provider configured")
            
            logger.info("Initializing WeaviateManager for AgentOrchestrator...")
            self._weaviate = WeaviateManager()
            await self._weaviate.initialize()
            logger.info("WeaviateManager initialized successfully")
            
            logger.info("Wiring Weaviate to Prequalifier...")
            self.pq.set_weaviate_manager(self._weaviate)
            
            logger.info("Initializing AgentOrchestrator...")
            self.agent = AgentOrchestrator(
                weaviate_manager=self._weaviate, 
                slm_manager=self._slm_manager
            )
            self._agent_initialized = True
            logger.info("AgentOrchestrator initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize AgentOrchestrator: {e}")
            import traceback
            traceback.print_exc()
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
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
                    issue_detected = sg_result.score > 0
                    
                    if test_id == "OV-01A":
                        passed = issue_detected
                    elif test_id == "OV-03A":
                        passed = issue_detected
                    elif test_id == "OV-05A":
                        passed = issue_detected
                    elif test_id == "OV-06A":
                        passed = issue_detected
                    else:
                        passed = not issue_detected
                    
                    result.ov_passed = passed
                    result.ov_issues = [sg_result.explanation] if not passed else []
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
        """Run Phase 4: Full conversational validation with Judge.
        
        REQUIRES: 
        - AgentOrchestrator must be available and initialized
        - LLM Judge must be available
        
        Tracks all conversation turns. Scores apply to the ENTIRE conversation.
        No mock responses - errors out if dependencies aren't available.
        """
        results = []
        
        if not AGENT_AVAILABLE:
            raise RuntimeError("AgentOrchestrator not available - cannot run full validation without real agent")
        
        agent_ready = await self._ensure_agent_ready()
        if not agent_ready or self.agent is None:
            raise RuntimeError("AgentOrchestrator failed to initialize - cannot run full validation without real agent")
        
        if not self.judge:
            raise RuntimeError("LLM Judge not configured - cannot run full validation without judge")
        
        judge_available = await self.judge.check_availability()
        if not judge_available:
            raise RuntimeError("LLM Judge not available - cannot run full validation without judge")
        
        logger.info("Full validation: Agent and Judge both available - using real responses")
        
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
                session_id = f"full_val_{test_id}_{int(time.time())}"
                
                result = TestResult(
                    test_id=test_id,
                    category=cat_key,
                    persona=persona.value,
                    engagement_style=style.value,
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
                
                start_time = time.time()
                conversation = []
                bot_responses = []
                last_metadata = None
                max_turns = 5
                
                try:
                    current_input = prompt
                    turn_count = 0
                    
                    while turn_count < max_turns:
                        pq_result = await self.pq.analyze(current_input, session_id=session_id)
                        pq_frustration = pq_result.frustration_decision.value
                        pq_vagueness = pq_result.vagueness_decision.value
                        
                        if turn_count == 0:
                            result.pq_flags = pq_result.pattern_flags.to_dict() if pq_result.pattern_flags else {}
                        
                        bot_response, metadata = await self.agent.process_message(
                            user_message=current_input,
                            session_id=session_id
                        )
                        last_metadata = metadata
                        
                        tool_called = metadata.get("tool_called", "") if metadata else ""
                        result.genai = metadata.get("model", "") if metadata else ""
                        
                        result.add_turn(
                            user_prompt=current_input,
                            bot_response=bot_response,
                            tool_called=tool_called,
                            pq_frustration=pq_frustration,
                            pq_vagueness=pq_vagueness
                        )
                        
                        bot_responses.append(bot_response)
                        conversation.append({"role": "user", "content": current_input})
                        conversation.append({"role": "bot", "content": bot_response})
                        turn_count += 1
                        
                        is_clarifying = bot_response.strip().endswith("?")
                        
                        if not is_clarifying:
                            logger.debug(f"Bot provided substantive answer at turn {turn_count}")
                            break
                        
                        if turn_count >= max_turns:
                            logger.debug(f"Max turns ({max_turns}) reached")
                            break
                        
                        user_response = await self.judge.generate_user_response(
                            bot_response=bot_response,
                            conversation_history=conversation,
                            persona=persona,
                            style=style,
                            clarification_count=turn_count
                        )
                        current_input = user_response.message
                        logger.debug(f"LLM user actor follow-up: {current_input[:100]}")
                    
                    full_conversation = result.get_full_conversation()
                    final_response = bot_responses[-1] if bot_responses else ""
                    
                    scores = await self.judge.score_response(
                        user_query=prompt,
                        bot_response=final_response,
                        context=full_conversation
                    )
                    
                    result.score_clarity = scores.clarity
                    result.score_empathy = scores.empathy
                    result.score_accuracy = scores.accuracy
                    result.score_engagement = scores.engagement
                    result.score_tone = scores.tone
                    result.score_alignment = scores.alignment
                    result.reasoning = f"Turns: {turn_count}. {scores.reasoning}"
                    
                    tool_match = (result.tool_called == result.expected_tool) if result.expected_tool else True
                    result.pass_fail = "PASS" if (scores.all_passing and tool_match) else "FAIL"
                
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
        2. Bot asks clarifying questions (real agent required)
        3. User (LLM agent) provides progressively specific responses
        4. Bot eventually provides substantive answer
        
        Tracks all turns. Scores apply to the ENTIRE conversation.
        
        REQUIRES:
        - AgentOrchestrator for real bot responses
        - LLM Judge for user simulation
        
        No mock responses - errors out if dependencies aren't available.
        """
        if not AGENT_AVAILABLE:
            raise RuntimeError("AgentOrchestrator not available - cannot run vague loop test without real agent")
        
        agent_ready = await self._ensure_agent_ready()
        if not agent_ready or self.agent is None:
            raise RuntimeError("AgentOrchestrator failed to initialize - cannot run vague loop test without real agent")
        
        if not self.judge:
            raise RuntimeError("LLM Judge not configured - cannot run vague loop test without judge")
        
        judge_available = await self.judge.check_availability()
        if not judge_available:
            raise RuntimeError("LLM Judge not available - cannot run vague loop test without judge")
        
        logger.info("Vague loop test: Agent and Judge both available - using real responses")
        
        results = []
        vague_prompts = ["Hi Brandon", "Hi Brandon, I'm Jayson.", "Hi Brandon, How are you today?"]
        
        for i, initial_prompt in enumerate(vague_prompts):
            test_id = f"VAGUE-{i:03d}"
            logger.info(f"Running vague loop test: {test_id}")
            
            result = TestResult(
                test_id=test_id,
                category="VAGUE_LOOP",
                persona=Persona.DOCILE.value,
                engagement_style=EngagementStyle.SPECIFIC.value,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            
            start_time = time.time()
            conversation = []
            bot_responses = []
            session_id = f"vague_test_{i}_{int(time.time())}"
            last_metadata = None
            
            try:
                current_input = initial_prompt
                turn_count = 0
                
                while turn_count < 5:
                    pq_result = await self.pq.analyze(current_input, session_id=session_id)
                    pq_frustration = pq_result.frustration_decision.value
                    pq_vagueness = pq_result.vagueness_decision.value
                    
                    bot_response, metadata = await self.agent.process_message(
                        user_message=current_input,
                        session_id=session_id
                    )
                    last_metadata = metadata
                    logger.debug(f"Real agent response: {bot_response[:100]}")
                    
                    tool_called = metadata.get("tool_called", "") if metadata else ""
                    result.add_turn(
                        user_prompt=current_input,
                        bot_response=bot_response,
                        tool_called=tool_called,
                        pq_frustration=pq_frustration,
                        pq_vagueness=pq_vagueness
                    )
                    
                    bot_responses.append(bot_response)
                    conversation.append({"role": "user", "content": current_input})
                    conversation.append({"role": "bot", "content": bot_response})
                    turn_count += 1
                    
                    if turn_count >= 3 and pq_result.vagueness_decision == VaguenessDecision.CLEAR:
                        break
                    
                    user_response = await self.judge.generate_user_response(
                        bot_response=bot_response,
                        conversation_history=conversation,
                        persona=Persona.DOCILE,
                        style=EngagementStyle.SPECIFIC,
                        clarification_count=turn_count
                    )
                    current_input = user_response.message
                    logger.debug(f"LLM user agent response: {current_input[:100]}")
                
                clarifying_questions = sum(1 for r in bot_responses if "?" in r)
                structure_passed = clarifying_questions >= 2 and turn_count >= 3
                
                full_conversation = result.get_full_conversation()
                scores = await self.judge.score_response(
                    user_query=initial_prompt,
                    bot_response=bot_responses[-1] if bot_responses else "",
                    context=full_conversation
                )
                
                result.score_clarity = scores.clarity
                result.score_empathy = scores.empathy
                result.score_accuracy = scores.accuracy
                result.score_engagement = scores.engagement
                result.score_tone = scores.tone
                result.score_alignment = scores.alignment
                
                passed = structure_passed and scores.all_passing
                result.pass_fail = "PASS" if passed else "FAIL"
                result.reasoning = f"Turns: {turn_count}, Clarifying: {clarifying_questions}, Scores: {scores.average:.2f}"
                result.genai = last_metadata.get("model", "") if last_metadata else ""
                
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
        """Export validation results to CSV.
        
        Format: One row per conversation turn.
        Scores appear only on the final turn (conversation-level scoring).
        Uses local time for timestamps.
        """
        if not self.session:
            raise ValueError("No validation session to export")
        
        output_dir = output_dir or str(Path(__file__).parent / "results")
        os.makedirs(output_dir, exist_ok=True)
        
        local_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(output_dir, f"validation_results_{local_timestamp}.csv")
        
        fieldnames = [
            "Test_ID", "Turn", "Category", "User_Prompt", "Bot_Response", 
            "Tool_Called", "PQ_Frustration", "PQ_Vagueness",
            "Score_Clarity", "Score_Empathy", "Score_Accuracy", "Score_Engagement", "Score_Tone", "Score_Alignment",
            "Pass_Fail", "Reasoning",
            "GenAI", "Persona", "Engagement_Style",
            "Timestamp", "Duration_ms"
        ]
        
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            
            for result in self.session.results:
                if result.turns:
                    total_turns = len(result.turns)
                    for i, turn in enumerate(result.turns):
                        is_final_turn = (i == total_turns - 1)
                        
                        row = {
                            "Test_ID": result.test_id,
                            "Turn": turn.turn_number,
                            "Category": result.category,
                            "User_Prompt": turn.user_prompt,
                            "Bot_Response": turn.bot_response,
                            "Tool_Called": turn.tool_called,
                            "PQ_Frustration": turn.pq_frustration,
                            "PQ_Vagueness": turn.pq_vagueness,
                            "Score_Clarity": result.score_clarity if is_final_turn else "",
                            "Score_Empathy": result.score_empathy if is_final_turn else "",
                            "Score_Accuracy": result.score_accuracy if is_final_turn else "",
                            "Score_Engagement": result.score_engagement if is_final_turn else "",
                            "Score_Tone": result.score_tone if is_final_turn else "",
                            "Score_Alignment": result.score_alignment if is_final_turn else "",
                            "Pass_Fail": result.pass_fail if is_final_turn else "",
                            "Reasoning": result.reasoning if is_final_turn else "",
                            "GenAI": result.genai if is_final_turn else "",
                            "Persona": result.persona,
                            "Engagement_Style": result.engagement_style,
                            "Timestamp": turn.timestamp,
                            "Duration_ms": result.duration_ms if is_final_turn else ""
                        }
                        writer.writerow(row)
                else:
                    row = {
                        "Test_ID": result.test_id,
                        "Turn": 1,
                        "Category": result.category,
                        "User_Prompt": result.user_prompt,
                        "Bot_Response": result.bot_response,
                        "Tool_Called": result.tool_called,
                        "PQ_Frustration": result.pq_frustration,
                        "PQ_Vagueness": result.pq_vagueness,
                        "Score_Clarity": result.score_clarity,
                        "Score_Empathy": result.score_empathy,
                        "Score_Accuracy": result.score_accuracy,
                        "Score_Engagement": result.score_engagement,
                        "Score_Tone": result.score_tone,
                        "Score_Alignment": result.score_alignment,
                        "Pass_Fail": result.pass_fail,
                        "Reasoning": result.reasoning,
                        "GenAI": result.genai,
                        "Persona": result.persona,
                        "Engagement_Style": result.engagement_style,
                        "Timestamp": result.timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Duration_ms": result.duration_ms
                    }
                    writer.writerow(row)
        
        logger.info(f"Results exported to: {csv_path}")
        
        summary_path = os.path.join(output_dir, f"validation_summary_{local_timestamp}.json")
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
            "by_engagement_style": self._aggregate_by_engagement_style(),
            "by_model": self._aggregate_by_model(),
            "by_persona": self._aggregate_by_persona(),
        }
        
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary exported to: {summary_path}")
        
        self._update_latest_symlinks(output_dir, csv_path, summary_path)
        
        return csv_path
    
    def _update_latest_symlinks(self, output_dir: str, csv_path: str, summary_path: str):
        """Create/update symlinks to the latest results and summary files."""
        csv_latest = os.path.join(output_dir, "validation_results_latest.csv")
        summary_latest = os.path.join(output_dir, "validation_summary_latest.json")
        
        csv_basename = os.path.basename(csv_path)
        summary_basename = os.path.basename(summary_path)
        
        try:
            if os.path.islink(csv_latest):
                os.unlink(csv_latest)
            elif os.path.exists(csv_latest):
                os.remove(csv_latest)
            os.symlink(csv_basename, csv_latest)
            logger.info(f"Updated symlink: validation_results_latest.csv -> {csv_basename}")
        except OSError as e:
            logger.warning(f"Failed to create CSV symlink: {e}")
        
        try:
            if os.path.islink(summary_latest):
                os.unlink(summary_latest)
            elif os.path.exists(summary_latest):
                os.remove(summary_latest)
            os.symlink(summary_basename, summary_latest)
            logger.info(f"Updated symlink: validation_summary_latest.json -> {summary_basename}")
        except OSError as e:
            logger.warning(f"Failed to create summary symlink: {e}")
    
    def _aggregate_by_field(self, field_name: str) -> Dict[str, Dict[str, Any]]:
        """Generic aggregation by any field on TestResult."""
        groups = {}
        
        for result in self.session.results:
            key = getattr(result, field_name, "") or "unknown"
            if not key:
                key = "unknown"
            
            if key not in groups:
                groups[key] = {
                    "total": 0,
                    "passed": 0,
                    "failed": 0,
                    "avg_clarity": 0,
                    "avg_empathy": 0,
                    "avg_accuracy": 0,
                    "avg_engagement": 0,
                    "avg_tone": 0,
                    "avg_alignment": 0,
                }
            
            groups[key]["total"] += 1
            if result.pass_fail == "PASS":
                groups[key]["passed"] += 1
            elif result.pass_fail == "FAIL":
                groups[key]["failed"] += 1
            
            groups[key]["avg_clarity"] += result.score_clarity
            groups[key]["avg_empathy"] += result.score_empathy
            groups[key]["avg_accuracy"] += result.score_accuracy
            groups[key]["avg_engagement"] += result.score_engagement
            groups[key]["avg_tone"] += result.score_tone
            groups[key]["avg_alignment"] += result.score_alignment
        
        for key in groups:
            total = groups[key]["total"]
            if total > 0:
                groups[key]["avg_clarity"] /= total
                groups[key]["avg_empathy"] /= total
                groups[key]["avg_accuracy"] /= total
                groups[key]["avg_engagement"] /= total
                groups[key]["avg_tone"] /= total
                groups[key]["avg_alignment"] /= total
                groups[key]["pass_rate"] = groups[key]["passed"] / total
        
        return groups
    
    def _aggregate_by_category(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate results by category."""
        return self._aggregate_by_field("category")
    
    def _aggregate_by_engagement_style(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate results by engagement style."""
        return self._aggregate_by_field("engagement_style")
    
    def _aggregate_by_model(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate results by LLM model (genai field)."""
        return self._aggregate_by_field("genai")
    
    def _aggregate_by_persona(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate results by user persona."""
        return self._aggregate_by_field("persona")
    
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
            print(f"  Clarity: {avg.get('clarity', 0):.2f}")
            print(f"  Empathy: {avg.get('empathy', 0):.2f}")
            print(f"  Accuracy: {avg.get('accuracy', 0):.2f}")
            print(f"  Engagement: {avg.get('engagement', 0):.2f}")
            print(f"  Tone: {avg.get('tone', 0):.2f}")
            print(f"  Alignment: {avg.get('alignment', 0):.2f}")
        
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
    
    use_agent = phase in [TestPhase.FULL, TestPhase.ALL]
    validator = BrandonBotValidator(use_judge=not args.no_judge, use_agent=use_agent)
    
    session = await validator.run_validation(phase, args.max_prompts)
    
    csv_path = validator.export_results(args.output)
    
    validator.print_summary()
    
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    asyncio.run(main())
