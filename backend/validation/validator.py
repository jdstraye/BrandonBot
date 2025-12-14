"""
BrandonBot Validation Script
Implements the full "Adversarial Evaluator" loop with 5 phases:

1. Inject: Send prompt from the Test Suite
2. Intercept: Capture internal logs (PQ Flags, Tool Calls, OV Decisions)
3. Interact: If bot asks clarifying question, generate persona-based response
4. Inspect: Verify side effects (database, email)
5. Score: Judge LLM scores output (0-5) against Safety/Quality Rubric

Execution:
    python -m validation.validator                    # Run all phases (default)
    python -m validation.validator --phase all        # Same as above
    python -m validation.validator --phase pq         # Prequalifier only (rate limit, sanitization, frustration/vagueness)
    python -m validation.validator --phase ov         # Output Validator: unit tests, E2E drift, repetition safeguard
    python -m validation.validator --phase mcp        # Tool (MCP) verification, multi-turn, callback edge cases
    python -m validation.validator --phase full       # Full adversarial conversations with LLM judge scoring

Optional flags:
    --max-prompts N     Limit number of full validation prompts
    --no-judge          Run without Ollama judge (scores = 0)
    --output DIR        Custom output directory for results

Results: CSV + JSON summary with aggregations by category, persona, model, style
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

# Per-agent-turn timeout in seconds (fail-fast when an agent turn is too slow)
AGENT_TURN_TIMEOUT = 5.0
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

# Config-driven role pinning and provider manager
try:
    from config_loader import load_config
    from backend.llm_providers import llm_manager
except Exception:
    # Best-effort imports; failures will be handled at runtime
    load_config = None
    llm_manager = None

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
    duration_ms: int = 0
    
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
    ov_retries: int = 0
    
    pass_fail: str = "PENDING"
    reasoning: str = ""
    
    genai: str = ""
    persona: str = ""
    engagement_style: str = ""
    
    timestamp: str = ""
    duration_ms: int = 0
    judge_latency_ms: int = 0
    
    def add_turn(self, user_prompt: str, bot_response: str, tool_called: str = "",
                 pq_frustration: str = "", pq_vagueness: str = "", model: str = "", duration_ms: int = 0) -> None:
        """Add a conversation turn.
        
        The bot_response is processed to extract only user-facing content:
        1. First tries structured JSON parsing (reasoning + final_response)
        2. Falls back to delimiter parsing (<final_response>)
        3. Finally uses regex sanitization for chatter removal
        
        Internal reasoning is logged to debug.db for investigation.
        
        Args:
            model: The LLM model that generated this response (e.g. "nvidia/llama-4-maverick")
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
            
            # Always log raw LLM response with model info
            debug_db.log_raw_llm_response(
                query=user_prompt,
                raw_response=bot_response,
                sanitized_response=clean_response,
                model=model,
                test_id=self.test_id,
                session_id=""
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
            
            # Always log raw LLM response with model info
            debug_db.log_raw_llm_response(
                query=user_prompt,
                raw_response=bot_response,
                sanitized_response=clean_response,
                model=model,
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
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            duration_ms=duration_ms,
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
    
    def __init__(self, use_judge: bool = True, use_agent: bool = False, require_slm: bool = True, weaviate_manager=None):
        """
        Initialize the BrandonBot validation engine.
        
        Args:
            use_judge: Enable Ollama LLM judge for scoring
            use_agent: Enable full agent orchestrator for vague loop testing
            require_slm: If True (default), require SLM models for validation.
                        This ensures proper intent checking via MS-MARCO cross-encoder.
                        If False, fall back to pattern-only checking (NOT recommended).
        """
        self._slm_manager = None
        self._weaviate = None

        # Enforce SLM requirement: fail fast if require_slm=True but SLMManager
        # isn't importable/available. Attempt a lazy import here in case the
        # module was unavailable at top-level import but is present now.
        if require_slm:
            if SLMManager is None:
                try:
                    from slm_manager import SLMManager as _SLMManager
                    globals()['SLMManager'] = _SLMManager
                except Exception:
                    raise RuntimeError(
                        "SLMManager not available but require_slm=True.\n"
                        "Set up the SLM dependencies (see developer_guide.md) or ensure the"
                        " environment can import backend.slm_manager."
                    )

        if require_slm and SLMManager is not None:
            logger.info("Initializing SLMManager for validation...")
            self._slm_manager = SLMManager()
            logger.info("SLMManager ready (lazy loading)")
        
        self.pq = Prequalifier(require_slm=require_slm, slm_provider=self._slm_manager)
        self.ov = OutputValidatorSLM(require_slm=require_slm)
        # If a WeaviateManager instance is provided at construction time,
        # wire it immediately so FEC RAG is available for validation runs.
        if weaviate_manager is not None:
            try:
                self._weaviate = weaviate_manager
                try:
                    self.pq.set_weaviate_manager(self._weaviate)
                except Exception:
                    # Not all Prequalifier implementations require explicit wiring
                    pass
                try:
                    self.set_fec_rag(self._weaviate)
                except Exception:
                    # set_fec_rag may raise if collection isn't present; let caller handle
                    pass
                logger.info("Wired provided WeaviateManager to validator at construction time")
            except Exception as e:
                logger.warning(f"Failed to wire provided WeaviateManager: {e}")
        # Note: Do not auto-initialize Weaviate at validator startup. Keep
        # fail-closed production semantics — callers/agent orchestrator should
        # explicitly wire a WeaviateManager via `set_fec_rag()` when available.
        self.judge = OllamaJudge() if use_judge else None
        # preserve the user's intent to use or skip the LLM judge
        self._use_judge = bool(use_judge)
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
        """Initialize the AgentOrchestrator with all required dependencies."""
        if not self._use_agent or not AGENT_AVAILABLE:
            return False
        
        if self._agent_initialized:
            return self.agent is not None
        
        try:
            if self._slm_manager is None and SLMManager is not None:
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
            
            logger.info("Wiring Weaviate to OutputValidator for FEC RAG...")
            self.set_fec_rag(self._weaviate)
            
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
    
    async def _ensure_pq_ready(self) -> bool:
        """Ensure Prequalifier has Weaviate and SLM wired up for tests."""
        if self._weaviate is not None:
            return True
        
        try:
            if self._slm_manager is None and SLMManager is not None:
                logger.info("Initializing SLMManager for PQ tests...")
                self._slm_manager = SLMManager()
                logger.info("SLMManager created (lazy loading)")
                
                logger.info("Wiring SLM provider to Prequalifier...")
                self.pq.set_slm_provider(self._slm_manager)
                logger.info("Prequalifier SLM provider configured")
            
            logger.info("Initializing WeaviateManager for PQ tests...")
            self._weaviate = WeaviateManager()
            await self._weaviate.initialize()
            logger.info("WeaviateManager initialized successfully")
            
            logger.info("Wiring Weaviate to Prequalifier...")
            self.pq.set_weaviate_manager(self._weaviate)
            logger.info("Prequalifier Weaviate configured")
            return True
        except Exception as e:
            logger.warning(f"Could not initialize Prequalifier dependencies: {e}")
            return False

    async def _create_session_judge(self, session_id: str) -> Optional[OllamaJudge]:
        """
        Create a per-session OllamaJudge instance pinned from `BrandonBot.ini` via `llm_manager`.

        Returns:
            - `OllamaJudge` instance if a pinned judge was created and is available
            - `None` to indicate caller should fall back to `self.judge` (when allowed)

        Raises RuntimeError when `require_llama_for_judge` is True and no pinned judge is available.
        """
        # If config loader or llm_manager not present, fall back to global judge
        if load_config is None or llm_manager is None:
            logger.debug("Config loader or llm_manager not imported; using default judge")
            return None

        cfg = load_config()

        try:
            selection = llm_manager.select_for_role(session_id, "Judge")
        except Exception as e:
            logger.warning(f"llm_manager.select_for_role failed: {e}")
            selection = None

        require_llama = bool(cfg.scoring.require_llama_for_judge) if getattr(cfg, 'scoring', None) else True

        if not selection:
            if require_llama:
                raise RuntimeError("No configured Llama judge slots available and 'require_llama_for_judge' is true")
            logger.info("No pinned judge selection found; falling back to default judge if available")
            return None

        provider_name, slot_id, model_name = selection
        provider_name = provider_name.lower() if provider_name else provider_name

        # Map provider name to judge backend
        if provider_name == "ollama":
            backend = "ollama"
        elif provider_name == "nvidia":
            backend = "nvidia"
        else:
            logger.warning(f"Configured judge provider '{provider_name}' is not recognized as a judge-capable backend")
            if require_llama:
                raise RuntimeError(f"Configured judge provider '{provider_name}' not available as judge and 'require_llama_for_judge' is true")
            return None

        # Instantiate per-session judge
        try:
            session_judge = OllamaJudge(model=model_name or None, force_backend=backend)
        except Exception as e:
            logger.error(f"Failed to instantiate session judge for {provider_name}:{model_name} - {e}")
            if require_llama:
                raise
            return None

        available = await session_judge.check_availability()
        if not available:
            if require_llama:
                raise RuntimeError(f"Pinned judge {provider_name}:{model_name} not available")
            logger.warning(f"Pinned judge {provider_name}:{model_name} not available; falling back to default judge")
            return None

        logger.info(f"Session judge pinned: {provider_name}/{slot_id}/{model_name} for session {session_id[:8]}...")
        return session_judge
    
    async def run_pq_tests(self) -> List[TestResult]:
        """Run Phase 1: Prequalifier tests."""
        await self._ensure_pq_ready()
        
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
    
    async def _ensure_fec_rag_ready(self) -> bool:
        """Ensure FEC RAG is configured for OV tests."""
        if self._weaviate is not None:
            return True
        
        try:
            logger.info("Initializing WeaviateManager for FEC RAG...")
            self._weaviate = WeaviateManager()
            await self._weaviate.initialize()
            logger.info("WeaviateManager initialized successfully")
            
            logger.info("Wiring Weaviate to OutputValidator for FEC RAG...")
            self.set_fec_rag(self._weaviate)
            return True
        except Exception as e:
            logger.warning(f"Could not initialize FEC RAG: {e}")
            return False
    
    async def run_ov_unit_tests(self) -> List[TestResult]:
        """Run Phase 3A: OV Component Unit Tests with injection."""
        await self._ensure_fec_rag_ready()
        
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
    
    async def run_mcp_tests(self, limit: int = None) -> List[TestResult]:
        """Run Phase 2: MCP Tool Verification Tests.
        
        Tests that specific prompts trigger the correct MCP tools.
        Validates tool execution and proper handling of blocked requests.
        """
        results = []
        
        if not AGENT_AVAILABLE:
            raise RuntimeError("AgentOrchestrator not available - cannot run MCP tests without real agent")
        
        agent_ready = await self._ensure_agent_ready()
        if not agent_ready or self.agent is None:
            raise RuntimeError("AgentOrchestrator failed to initialize - cannot run MCP tests without real agent")
        
        mcp_tests = self.test_data.get("mcp_tests", {}).get("tests", [])
        test_identity = self.test_data.get("mcp_tests", {}).get("test_identity", {})
        
        # Optionally limit number of MCP tests for quick runs
        if limit is not None and isinstance(limit, int) and limit > 0:
            mcp_tests = mcp_tests[:limit]
        for test in mcp_tests:
            test_id = test["id"]
            logger.info(f"Running MCP test: {test_id}")
            
            result = TestResult(
                test_id=test_id,
                category="MCP",
                user_prompt=test.get("input", ""),
                expected_tool=test.get("expected_tool", ""),
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            
            start_time = time.time()
            session_id = f"mcp_test_{test_id}_{int(time.time())}"
            
            try:
                response, metadata = await self.agent.process_query(
                    message=test["input"],
                    session_id=session_id
                )
                
                result.bot_response = sanitize_bot_response(response)
                
                tool_calls = metadata.get("tool_calls", [])
                tools_called = [tc.get("name", "") for tc in tool_calls]
                result.tool_called = ", ".join(tools_called) if tools_called else "NONE"
                
                expected = test.get("expected_tool", "")
                
                if expected == "NONE (Blocked)":
                    passed = len(tools_called) == 0
                    result.reasoning = "Correctly blocked - no tools called" if passed else f"Expected no tools, got: {tools_called}"
                elif expected:
                    passed = any(expected.lower() in tc.lower() for tc in tools_called)
                    result.reasoning = f"Expected {expected}, got {tools_called}" if not passed else f"Correct tool called: {expected}"
                else:
                    passed = True
                    result.reasoning = "No expected tool specified"
                
                result.pass_fail = "PASS" if passed else "FAIL"
                result.genai = metadata.get("model_used", metadata.get("model", ""))
                
            except Exception as e:
                result.pass_fail = "ERROR"
                result.reasoning = str(e)
                logger.error(f"MCP test {test_id} failed: {e}")
            
            result.duration_ms = int((time.time() - start_time) * 1000)
            results.append(result)
        
        return results
    
    async def run_ov_e2e_tests(self) -> List[TestResult]:
        """Run Phase 3B: End-to-End OV Drift Detection Tests.
        
        Tests full pipeline with personas to detect LLM drift and verify
        OV catches issues like intent drift, excessive rhetoric, missing citations.
        """
        results = []
        
        if not AGENT_AVAILABLE:
            raise RuntimeError("AgentOrchestrator not available - cannot run OV E2E tests without real agent")
        
        agent_ready = await self._ensure_agent_ready()
        if not agent_ready or self.agent is None:
            raise RuntimeError("AgentOrchestrator failed to initialize - cannot run OV E2E tests without real agent")
        
        if not self.judge:
            raise RuntimeError("LLM Judge not configured - cannot run OV E2E tests without judge")
        
        ov_e2e_tests = self.test_data.get("ov_e2e_tests", {}).get("tests", [])
        
        for test in ov_e2e_tests:
            test_id = test["id"]
            logger.info(f"Running OV E2E test: {test_id}")
            
            result = TestResult(
                test_id=test_id,
                category="OV_E2E",
                user_prompt=test.get("prompt", ""),
                persona=test.get("persona", ""),
                engagement_style=test.get("style", ""),
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            
            start_time = time.time()
            session_id = f"ov_e2e_test_{test_id}_{int(time.time())}"
            
            try:
                response, metadata = await self.agent.process_query(
                    message=test["prompt"],
                    session_id=session_id
                )
                
                result.bot_response = sanitize_bot_response(response)
                result.genai = metadata.get("model_used", metadata.get("model", ""))
                
                ov_data = metadata.get("ov_validation", {})
                ov_retries = metadata.get("ov_retries", 0)
                ov_final_status = ov_data.get("status", "unknown")
                
                result.ov_passed = ov_final_status in ["passed", "approved"]
                result.ov_retries = ov_retries
                
                expected_success = test.get("success", "")
                
                # If the test expects the OV to detect or trigger a safeguard,
                # pass when the OV retried or explicitly rejected the response.
                if any(term in expected_success.lower() for term in ("catch", "trigger", "detect")):
                    passed = ov_retries > 0 or not result.ov_passed
                    result.reasoning = f"OV retries: {ov_retries}, Status: {ov_final_status}"
                else:
                    passed = result.ov_passed
                    result.reasoning = f"OV passed: {result.ov_passed}"
                
                result.pass_fail = "PASS" if passed else "FAIL"
                
            except Exception as e:
                result.pass_fail = "ERROR"
                result.reasoning = str(e)
                logger.error(f"OV E2E test {test_id} failed: {e}")
            
            result.duration_ms = int((time.time() - start_time) * 1000)
            results.append(result)
        
        return results
    
    async def run_full_validation(self, max_prompts: int = None, target_prompt_index: Optional[int] = None, target_prompt_id: Optional[str] = None) -> List[TestResult]:
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
        
        if self._use_judge:
            if not self.judge:
                raise RuntimeError("LLM Judge not configured - cannot run full validation without judge")
            judge_available = await self.judge.check_availability()
            if not judge_available:
                raise RuntimeError("LLM Judge not available - cannot run full validation without judge")
            logger.info("Full validation: Agent and Judge both available - using real responses")
        else:
            logger.info("Full validation: running without LLM judge (--no-judge); scoring will be skipped or zeroed")        
        categories = self.test_data.get("categories", {})
        prompt_count = 0
        
        for cat_key, category in categories.items():
            prompts = category.get("prompts", [])
            
            for prompt in prompts:
                # If caller specified a single prompt index, skip until we reach it
                if target_prompt_index is not None and prompt_count != int(target_prompt_index):
                    prompt_count += 1
                    continue

                # If caller specified a single prompt id (e.g. 'A_VAGUE-002'), skip until match
                generated_test_id = f"{cat_key}-{prompt_count:03d}"
                if target_prompt_id is not None and target_prompt_id != generated_test_id:
                    prompt_count += 1
                    continue

                if max_prompts and prompt_count >= max_prompts:
                    break
                
                # Use generated id (may have been computed above when filtering by id)
                test_id = generated_test_id
                logger.info(f"Running full test: {test_id}")
                
                persona = self._select_persona()
                style = self._select_style()
                session_id = f"full_val_{test_id}_{int(time.time())}"
                # start timer for this test early so errors during judge setup can record a duration
                start_time = time.time()
                # Create a per-session judge pinned from INI (if configured)
                session_judge = None
                # Only attempt per-session judge creation when running with judge enabled
                if self._use_judge:
                    try:
                        session_judge = await self._create_session_judge(session_id)
                    except Exception as e:
                        # Build a minimal TestResult to record the setup error and continue
                        err_result = TestResult(
                            test_id=test_id,
                            category=cat_key,
                            persona=persona.value,
                            engagement_style=style.value,
                            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        )
                        err_result.pass_fail = "ERROR"
                        err_result.reasoning = str(e)
                        logger.error(f"Full test {test_id} failed during judge setup: {e}")
                        err_result.duration_ms = int((time.time() - start_time) * 1000)
                        results.append(err_result)
                        prompt_count += 1
                        continue
                
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

                        # Remember the initial vagueness decision for OV leniency
                        if turn_count == 0:
                            initial_pq_vagueness = pq_result.vagueness_decision
                            # Numeric confidence (0.0-1.0) for vagueness to inform OV leniency
                            initial_pq_vagueness_confidence = float(getattr(pq_result, 'vagueness_confidence', 0.0) or 0.0)
                        
                        if turn_count == 0:
                            result.pq_flags = pq_result.pattern_flags.to_dict() if pq_result.pattern_flags else {}
                        
                        # Log the user message sent to the (possibly local) agent
                        try:
                            debug_db = get_debug_db()
                            extra_meta = {
                                "query_vague": (pq_result.vagueness_decision == VaguenessDecision.VAGUE),
                                "phase": "initial",
                                "attempt": turn_count
                            }
                            # Include numeric vagueness confidence for forensic analysis
                            try:
                                extra_meta['vagueness_confidence'] = float(getattr(pq_result, 'vagueness_confidence', 0.0) or 0.0)
                            except Exception:
                                extra_meta['vagueness_confidence'] = 0.0
                            debug_db.log_llm_request(
                                system_prompt="",
                                messages=[{"role": "user", "content": current_input}],
                                tools=None,
                                provider="local_agent",
                                model="",
                                test_id=test_id,
                                session_id=session_id,
                                request_id=None,
                                extra=extra_meta,
                            )
                        except Exception:
                            logger.debug(f"[{test_id}] Failed to log agent request; continuing")

                        # Measure bot response latency
                        resp_start = time.time()
                        try:
                            # Enforce a per-agent-turn timeout to avoid long
                            # blocking LLM calls stalling the entire validation.
                            # If an agent turn exceeds AGENT_TURN_TIMEOUT, treat
                            # it as a failure (fail-fast) so the watchdog can
                            # collect diagnostics and move to deterministic
                            # fallback.
                            from backend import validation as _valmod
                            timeout_s = getattr(_valmod, 'AGENT_TURN_TIMEOUT', 5.0)
                            bot_response, metadata = await asyncio.wait_for(
                                self.agent.process_message(user_message=current_input, session_id=session_id),
                                timeout=timeout_s
                            )
                        except asyncio.TimeoutError:
                            logger.error(f"Agent turn timed out (> {timeout_s}s) for session {session_id} test {test_id}")
                            raise
                        resp_latency_ms = int((time.time() - resp_start) * 1000)
                        last_metadata = metadata
                        
                        tool_called = metadata.get("tool_called", "") if metadata else ""
                        model_used = metadata.get("model_used", metadata.get("model", "")) if metadata else ""
                        result.genai = model_used
                        
                        result.add_turn(
                            user_prompt=current_input,
                            bot_response=bot_response,
                            tool_called=tool_called,
                            pq_frustration=pq_frustration,
                            pq_vagueness=pq_vagueness,
                            model=model_used,
                            duration_ms=resp_latency_ms
                        )
                        
                        bot_responses.append(bot_response)
                        conversation.append({"role": "user", "content": current_input})
                        conversation.append({"role": "bot", "content": bot_response})
                        turn_count += 1
                        
                        is_clarifying = bot_response.strip().endswith("?")
                        
                        # When the judge is disabled, do not attempt to simulate the user actor.
                        # Treat clarifying questions as final so the loop can end safely.
                        if is_clarifying and not self._use_judge:
                            logger.info("Judge disabled: treating bot clarifying question as final (no user simulation)")
                            is_clarifying = False
                        
                        if not is_clarifying:
                            logger.debug(f"Bot provided substantive answer at turn {turn_count}")
                            break
                        
                        if turn_count >= max_turns:
                            logger.debug(f"Max turns ({max_turns}) reached")
                            break
                        
                        judge_for_session = session_judge or self.judge
                        if not judge_for_session:
                            # When no judge is configured, log and stop the clarification loop rather than raising.
                            logger.info("No LLM judge available for this session: skipping user simulation")
                            break
                        user_response = await judge_for_session.generate_user_response(
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
                    
                    # Before judge scoring, run the Output Validator on the final response
                    try:
                        is_vague_query_flag = (initial_pq_vagueness == VaguenessDecision.VAGUE)
                        vagueness_confidence_val = float(getattr(locals(), 'initial_pq_vagueness_confidence', 0.0) or 0.0)
                    except Exception:
                        is_vague_query_flag = False

                    try:
                        ov_validation = await self.ov.validate(
                            query=prompt,
                            response=final_response,
                            pq_confidence=0.85,
                            is_vague_query=is_vague_query_flag,
                            vagueness_confidence=vagueness_confidence_val
                        )
                        result.ov_passed = ov_validation.passed
                        result.ov_issues = [f"{r.safeguard.value}:{r.explanation}" for r in ov_validation.results.values() if r.score > 3]
                    except Exception as e:
                        # If OV fails due to missing SLMs or errors, record as ERROR
                        logger.warning(f"Output Validator failed for {test_id}: {e}")
                        result.ov_passed = False
                        result.ov_issues = [str(e)]

                    # If judge is disabled, zero scores and continue (useful for debugging)
                    if not self._use_judge:
                        result.score_clarity = 0.0
                        result.score_empathy = 0.0
                        result.score_accuracy = 0.0
                        result.score_engagement = 0.0
                        result.score_tone = 0.0
                        result.score_alignment = 0.0
                        result.reasoning = "LLM Judge disabled (--no-judge) — scores zeroed"
                        # Mark as skipped when judge is intentionally disabled
                        result.pass_fail = "SKIPPED"
                    else:
                        judge_for_session = session_judge or self.judge
                        if not judge_for_session:
                            raise RuntimeError("LLM Judge not configured for this session")
                        # Time judge scoring for latency checks
                        t0 = time.time()
                        scores = await judge_for_session.score_response(
                            user_query=prompt,
                            bot_response=final_response,
                            context=full_conversation
                        )
                        result.judge_latency_ms = int((time.time() - t0) * 1000)

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
        
        if self._use_judge:
            if not self.judge:
                raise RuntimeError("LLM Judge not configured - cannot run vague loop test without judge")
            judge_available = await self.judge.check_availability()
            if not judge_available:
                raise RuntimeError("LLM Judge not available - cannot run vague loop test without judge")
            logger.info("Vague loop test: Agent and Judge both available - using real responses")
        else:
            logger.info("Vague loop test: running without LLM judge (--no-judge); scoring will be skipped or zeroed")        
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
            # Create a per-session judge pinned from INI (if configured)
            session_judge = None
            if self._use_judge:
                try:
                    session_judge = await self._create_session_judge(session_id)
                except Exception as e:
                    result.pass_fail = "ERROR"
                    result.reasoning = str(e)
                    logger.error(f"Vague loop test {test_id} failed during judge setup: {e}")
                    result.duration_ms = int((time.time() - start_time) * 1000)
                    results.append(result)
                    continue
            
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
                    model_used = metadata.get("model_used", metadata.get("model", "")) if metadata else ""
                    result.add_turn(
                        user_prompt=current_input,
                        bot_response=bot_response,
                        tool_called=tool_called,
                        pq_frustration=pq_frustration,
                        pq_vagueness=pq_vagueness,
                        model=model_used
                    )
                    
                    bot_responses.append(bot_response)
                    conversation.append({"role": "user", "content": current_input})
                    conversation.append({"role": "bot", "content": bot_response})
                    turn_count += 1
                    
                    if turn_count >= 3 and pq_result.vagueness_decision == VaguenessDecision.CLEAR:
                        break
                    
                    judge_for_session = session_judge or self.judge
                    if not judge_for_session:
                        logger.info("Judge disabled: skipping LLM user actor simulation in vague loop")
                        break
                    user_response = await judge_for_session.generate_user_response(
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
                # If judge disabled, zero scores
                if not self._use_judge:
                    result.score_clarity = 0.0
                    result.score_empathy = 0.0
                    result.score_accuracy = 0.0
                    result.score_engagement = 0.0
                    result.score_tone = 0.0
                    result.score_alignment = 0.0
                    result.reasoning = "LLM Judge disabled (--no-judge) — scores zeroed"
                    result.pass_fail = "SKIPPED"
                else:
                    judge_for_session = session_judge or self.judge
                    if not judge_for_session:
                        raise RuntimeError("LLM Judge not configured for this session")
                    scores = await judge_for_session.score_response(
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
                    result.genai = last_metadata.get("model_used", last_metadata.get("model", "")) if last_metadata else ""
                
            except Exception as e:
                result.pass_fail = "ERROR"
                result.reasoning = str(e)
                logger.error(f"Vague loop test {test_id} failed: {e}")
            
            result.duration_ms = int((time.time() - start_time) * 1000)
            results.append(result)
        
        return results
    
    async def run_multi_turn_tests(self) -> List[TestResult]:
        """Run Phase 5: Multi-Turn Conversation Tests.
        
        Tests callback cooldown, vague-to-clear transitions, and frustrated user handling
        across multiple conversation turns in the same session.
        """
        results = []
        
        if not AGENT_AVAILABLE:
            raise RuntimeError("AgentOrchestrator not available - cannot run multi-turn tests without real agent")
        
        agent_ready = await self._ensure_agent_ready()
        if not agent_ready or self.agent is None:
            raise RuntimeError("AgentOrchestrator failed to initialize - cannot run multi-turn tests without real agent")
        
        multi_turn_tests = self.test_data.get("multi_turn_tests", {}).get("tests", [])
        
        for test in multi_turn_tests:
            test_id = test["id"]
            logger.info(f"Running multi-turn test: {test_id}")
            
            result = TestResult(
                test_id=test_id,
                category="MULTI_TURN",
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            
            start_time = time.time()
            session_id = f"multi_turn_test_{test_id}_{int(time.time())}"
            
            try:
                turns = test.get("turns", [])
                all_passed = True
                turn_results = []
                metadata = {}
                
                for i, turn in enumerate(turns):
                    user_msg = turn.get("user", "")
                    expected_tool = turn.get("expected_tool", "")
                    expected_behavior = turn.get("expected", "")
                    
                    response, metadata = await self.agent.process_query(
                        message=user_msg,
                        session_id=session_id
                    )
                    
                    tool_calls = metadata.get("tool_calls", [])
                    tools_called = [tc.get("name", "") for tc in tool_calls]
                    tool_str = ", ".join(tools_called) if tools_called else "NONE"
                    
                    result.add_turn(
                        user_prompt=user_msg,
                        bot_response=response,
                        tool_called=tool_str,
                        model=metadata.get("model_used", "")
                    )
                    
                    if expected_tool:
                        if "Blocked" in expected_tool or expected_tool == "NONE":
                            turn_passed = len(tools_called) == 0 or all("blocked" in str(tc) for tc in tool_calls)
                        else:
                            turn_passed = any(expected_tool.lower() in tc.lower() for tc in tools_called)
                        
                        turn_results.append({
                            "turn": i + 1,
                            "expected": expected_tool,
                            "actual": tool_str,
                            "passed": turn_passed
                        })
                        
                        if not turn_passed:
                            all_passed = False
                
                result.pass_fail = "PASS" if all_passed else "FAIL"
                result.reasoning = f"Turns: {len(turns)}, Results: {turn_results}"
                result.genai = metadata.get("model_used", "") if metadata else ""
                
            except Exception as e:
                result.pass_fail = "ERROR"
                result.reasoning = str(e)
                logger.error(f"Multi-turn test {test_id} failed: {e}")
            
            result.duration_ms = int((time.time() - start_time) * 1000)
            results.append(result)
        
        return results
    
    async def run_repetition_tests(self) -> List[TestResult]:
        """Run Phase 6: OV Repetition Safeguard Tests.
        
        Tests cosine similarity-based repetition detection using the public OV API
        with weaviate_manager's embedding model for proper cosine similarity.
        
        FAIL-CLOSED: If SLM/embeddings unavailable, each test gets FAIL result (not skip).
        """
        await self._ensure_fec_rag_ready()
        
        results = []
        rep_tests = self.test_data.get("repetition_safeguard_tests", {}).get("tests", [])
        
        embedding_error = None
        if not self._weaviate:
            embedding_error = "Weaviate manager not available for embeddings"
        elif not hasattr(self._weaviate, 'encode_text') or not callable(getattr(self._weaviate, 'encode_text', None)):
            embedding_error = "Weaviate manager missing encode_text method"
        else:
            try:
                test_embed = self._weaviate.encode_text("test embedding initialization")
                if test_embed is None or len(test_embed) == 0:
                    embedding_error = "Weaviate manager returned empty embedding"
            except Exception as e:
                embedding_error = f"Weaviate embedding initialization failed: {e}"
        
        if embedding_error:
            logger.error(f"FAIL-CLOSED: {embedding_error}")
            for test in rep_tests:
                result = TestResult(
                    test_id=test["id"],
                    category="REPETITION",
                    user_prompt="Repetition safeguard test",
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    pass_fail="FAIL",
                    reasoning=f"FAIL-CLOSED: {embedding_error}"
                )
                results.append(result)
            return results
        
        for test in rep_tests:
            test_id = test["id"]
            logger.info(f"Running repetition test: {test_id}")
            
            result = TestResult(
                test_id=test_id,
                category="REPETITION",
                user_prompt="Repetition safeguard test",
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            
            start_time = time.time()
            
            try:
                previous_response = test.get("injected_previous", "")
                current_response = test.get("injected_output", "")
                expected = test.get("expected", "")
                expected_score_range = test.get("expected_score_range", None)
                
                self.ov.set_weaviate_manager(self._weaviate)
                
                try:
                    ov_result = await self.ov.check_repetition(
                        response=current_response,
                        previous_responses=[previous_response],
                        similarity_threshold=0.8
                    )
                except SLMNotAvailableError as e:
                    result.pass_fail = "FAIL"
                    result.reasoning = f"FAIL-CLOSED: SLM unavailable - {e}"
                    result.duration_ms = int((time.time() - start_time) * 1000)
                    results.append(result)
                    continue
                
                if "detect" in expected.lower() or "block" in expected.lower():
                    passed = ov_result.score >= 4
                elif "allow" in expected.lower():
                    passed = ov_result.score <= 3
                else:
                    passed = True
                
                if expected_score_range:
                    min_score, max_score = expected_score_range
                    score_in_range = min_score <= ov_result.score <= max_score
                    passed = passed and score_in_range
                
                result.bot_response = f"Score: {ov_result.score}, Method: {ov_result.method}"
                result.pass_fail = "PASS" if passed else "FAIL"
                result.reasoning = f"Expected: {expected}, OV Score: {ov_result.score}/5, Explanation: {ov_result.explanation}"
                
            except SLMNotAvailableError as e:
                result.pass_fail = "FAIL"
                result.reasoning = f"FAIL-CLOSED: SLM/embedding unavailable - {e}"
                logger.error(f"Repetition test {test_id} failed (SLM unavailable): {e}")
            except (asyncio.TimeoutError, RuntimeError) as e:
                result.pass_fail = "FAIL"
                result.reasoning = f"FAIL-CLOSED: Embedding operation failed - {e}"
                logger.error(f"Repetition test {test_id} failed (embedding error): {e}")
            except Exception as e:
                result.pass_fail = "FAIL"
                result.reasoning = f"FAIL-CLOSED: Unexpected error - {e}"
                logger.error(f"Repetition test {test_id} failed: {e}")
            
            result.duration_ms = int((time.time() - start_time) * 1000)
            results.append(result)
        
        return results
    
    async def run_callback_edge_case_tests(self) -> List[TestResult]:
        """Run callback edge case tests from M_EDGE_CASES.callback_edge_cases.
        
        Tests for M_EDGE_CASES-132 regression: callback repetition prevention,
        cooldown enforcement, and explicit vs implied callback detection.
        """
        results = []
        
        if not AGENT_AVAILABLE:
            raise RuntimeError("AgentOrchestrator not available - cannot run callback edge case tests")
        
        agent_ready = await self._ensure_agent_ready()
        if not agent_ready or self.agent is None:
            raise RuntimeError("AgentOrchestrator failed to initialize - cannot run callback edge case tests")
        
        m_edge_cases = self.test_data.get("categories", {}).get("M_EDGE_CASES", {})
        callback_edge_cases = m_edge_cases.get("callback_edge_cases", {}).get("tests", [])
        
        if not callback_edge_cases:
            logger.warning("No callback_edge_cases found in M_EDGE_CASES")
            return results
        
        for test in callback_edge_cases:
            test_id = test["id"]
            logger.info(f"Running callback edge case test: {test_id}")
            
            result = TestResult(
                test_id=test_id,
                category="CALLBACK_EDGE",
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            
            start_time = time.time()
            session_id = f"callback_edge_{test_id}_{int(time.time())}"
            
            try:
                if test_id == "M_EDGE_CASES-132":
                    prompt = test.get("prompt", "")
                    anti_pattern = test.get("anti_pattern", "")
                    
                    response, metadata = await self.agent.process_query(
                        message=prompt,
                        session_id=session_id
                    )
                    
                    tool_calls = metadata.get("tool_calls", [])
                    # Check if callback tool was EXECUTED (not blocked)
                    callback_executed = any(
                        tc.get("name") == "request_callback" and tc.get("blocked") is None
                        for tc in tool_calls
                    )
                    
                    passed = not callback_executed
                    
                    result.user_prompt = prompt
                    result.bot_response = response
                    result.tool_called = ", ".join([tc.get("name", "") for tc in tool_calls]) if tool_calls else "NONE"
                    result.pass_fail = "PASS" if passed else "FAIL"
                    result.reasoning = f"Anti-pattern: {anti_pattern}. Callback executed: {callback_executed}. Expected: False (answer clear+frustrated directly)"
                
                elif test_id == "M_EDGE_CASES-133":
                    turn_1 = test.get("turn_1", "")
                    turn_2 = test.get("turn_2", "")
                    
                    response1, metadata1 = await self.agent.process_query(
                        message=turn_1,
                        session_id=session_id
                    )
                    
                    response2, metadata2 = await self.agent.process_query(
                        message=turn_2,
                        session_id=session_id
                    )
                    
                    tool_calls_2 = metadata2.get("tool_calls", [])
                    # Check if callback was EXECUTED (not blocked)
                    callback_executed = any(
                        tc.get("name") == "request_callback" and tc.get("blocked") is None
                        for tc in tool_calls_2
                    )
                    
                    passed = not callback_executed
                    
                    result.user_prompt = f"Turn 1: {turn_1} | Turn 2: {turn_2}"
                    result.bot_response = f"Turn 1: {response1[:100]}... | Turn 2: {response2[:100]}..."
                    result.tool_called = ", ".join([tc.get("name", "") for tc in tool_calls_2]) if tool_calls_2 else "NONE"
                    result.pass_fail = "PASS" if passed else "FAIL"
                    result.reasoning = f"Turn 2 callback executed: {callback_executed}. Expected: False (should be on cooldown)"
                    
                    result.add_turn(user_prompt=turn_1, bot_response=response1)
                    result.add_turn(user_prompt=turn_2, bot_response=response2)
                
                elif test_id == "M_EDGE_CASES-134":
                    explicit = test.get("explicit", "")
                    implied = test.get("implied", "")
                    
                    resp_explicit, meta_explicit = await self.agent.process_query(
                        message=explicit,
                        session_id=f"{session_id}_explicit"
                    )
                    
                    resp_implied, meta_implied = await self.agent.process_query(
                        message=implied,
                        session_id=f"{session_id}_implied"
                    )
                    
                    explicit_tcs = meta_explicit.get("tool_calls", [])
                    implied_tcs = meta_implied.get("tool_calls", [])
                    
                    # Check if callback was EXECUTED (not blocked)
                    explicit_callback_executed = any(
                        tc.get("name") == "request_callback" and tc.get("blocked") is None
                        for tc in explicit_tcs
                    )
                    implied_callback_executed = any(
                        tc.get("name") == "request_callback" and tc.get("blocked") is None
                        for tc in implied_tcs
                    )
                    
                    passed = explicit_callback_executed and not implied_callback_executed
                    
                    result.user_prompt = f"Explicit: {explicit} | Implied: {implied}"
                    result.bot_response = f"Explicit resp: {resp_explicit[:100]}... | Implied resp: {resp_implied[:100]}..."
                    result.tool_called = f"Explicit: {[tc.get('name') for tc in explicit_tcs]}, Implied: {[tc.get('name') for tc in implied_tcs]}"
                    result.pass_fail = "PASS" if passed else "FAIL"
                    result.reasoning = f"Explicit callback executed: {explicit_callback_executed}, Implied callback executed: {implied_callback_executed}. Expected: explicit=True, implied=False"

                
                elif test_id == "M_EDGE_CASES-135":
                    prompt = test.get("prompt", "")
                    response, metadata = await self.agent.process_query(
                        message=prompt,
                        session_id=session_id
                    )
                    tool_calls = metadata.get("tool_calls", [])
                    # Check actual tool calls
                    callback_detected = any(tc.get("name") == "request_callback" for tc in tool_calls)
                    donate_detected = any(tc.get("name") == "make_donation" for tc in tool_calls)
                    
                    passed = callback_detected and not donate_detected
                    result.user_prompt = prompt
                    result.bot_response = response
                    result.tool_called = ", ".join([tc.get("name", "") for tc in tool_calls]) if tool_calls else "NONE"
                    result.pass_fail = "PASS" if passed else "FAIL"
                    result.reasoning = f"Should detect callback (not donate). Callback: {callback_detected}, Donate: {donate_detected}"
                
                elif test_id == "M_EDGE_CASES-136":
                    prompt = test.get("prompt", "")
                    response, metadata = await self.agent.process_query(
                        message=prompt,
                        session_id=session_id
                    )
                    ov_passed = metadata.get("ov_validation", {}).get("status") == "passed"
                    
                    result.user_prompt = prompt
                    result.bot_response = response
                    result.pass_fail = "PASS" if ov_passed else "FAIL"
                    result.reasoning = f"Callback query should bypass OV intent check. OV result: {ov_passed}"
                
                elif test_id == "M_EDGE_CASES-137":
                    prompt = test.get("prompt", "")
                    response, metadata = await self.agent.process_query(
                        message=prompt,
                        session_id=session_id
                    )
                    tool_calls = metadata.get("tool_calls", [])
                    callback_detected = any(tc.get("name") == "request_callback" for tc in tool_calls)
                    search_triggered = any(tc.get("name") == "search_brandon_positions" for tc in tool_calls)
                    
                    passed = callback_detected and not search_triggered
                    result.user_prompt = prompt
                    result.bot_response = response
                    result.tool_called = ", ".join([tc.get("name", "") for tc in tool_calls]) if tool_calls else "NONE"
                    result.pass_fail = "PASS" if passed else "FAIL"
                    result.reasoning = f"Callback should bypass factual safeguard. Callback: {callback_detected}, Search: {search_triggered}"
                
                elif test_id == "M_EDGE_CASES-138":
                    prompt = test.get("prompt", "")
                    response, metadata = await self.agent.process_query(
                        message=prompt,
                        session_id=session_id
                    )
                    tool_calls = metadata.get("tool_calls", [])
                    callback_detected = any(tc.get("name") == "request_callback" for tc in tool_calls)
                    
                    result.user_prompt = prompt
                    result.bot_response = response
                    result.tool_called = ", ".join([tc.get("name", "") for tc in tool_calls]) if tool_calls else "NONE"
                    result.pass_fail = "PASS" if callback_detected else "FAIL"
                    result.reasoning = f"Should detect 'schedule a call' as callback. Callback: {callback_detected}"
                
                elif test_id == "M_EDGE_CASES-139":
                    prompt = test.get("prompt", "")
                    response, metadata = await self.agent.process_query(
                        message=prompt,
                        session_id=session_id
                    )
                    tool_calls = metadata.get("tool_calls", [])
                    callback_detected = any(tc.get("name") == "request_callback" for tc in tool_calls)
                    
                    result.user_prompt = prompt
                    result.bot_response = response
                    result.tool_called = ", ".join([tc.get("name", "") for tc in tool_calls]) if tool_calls else "NONE"
                    result.pass_fail = "PASS" if callback_detected else "FAIL"
                    result.reasoning = f"Should detect 'speak to someone' as callback. Callback: {callback_detected}"
                
                elif test_id == "M_EDGE_CASES-140":
                    callback_prompt = test.get("prompts", {}).get("callback", "")
                    donate_prompt = test.get("prompts", {}).get("donate", "")
                    
                    resp_cb, meta_cb = await self.agent.process_query(
                        message=callback_prompt,
                        session_id=f"{session_id}_callback"
                    )
                    resp_do, meta_do = await self.agent.process_query(
                        message=donate_prompt,
                        session_id=f"{session_id}_donate"
                    )
                    
                    tools_cb_tcs = meta_cb.get("tool_calls", [])
                    tools_do_tcs = meta_do.get("tool_calls", [])
                    
                    callback_detected = any(tc.get("name") == "request_callback" for tc in tools_cb_tcs)
                    donate_detected = any(tc.get("name") == "make_donation" for tc in tools_do_tcs)
                    
                    passed = callback_detected and donate_detected
                    result.user_prompt = f"Callback: {callback_prompt} | Donate: {donate_prompt}"
                    result.bot_response = f"Callback: {resp_cb[:80]}... | Donate: {resp_do[:80]}..."
                    result.tool_called = f"Callback: {[tc.get('name') for tc in tools_cb_tcs]}, Donate: {[tc.get('name') for tc in tools_do_tcs]}"
                    result.pass_fail = "PASS" if passed else "FAIL"
                    result.reasoning = f"Callback={callback_detected}, Donate={donate_detected}. Both should be detected separately"
                
                else:
                    result.pass_fail = "SKIP"
                    result.reasoning = f"Unknown callback edge case ID: {test_id}"
                
            except Exception as e:
                result.pass_fail = "ERROR"
                result.reasoning = str(e)
                logger.error(f"Callback edge case test {test_id} failed: {e}")
            
            result.duration_ms = int((time.time() - start_time) * 1000)
            results.append(result)
        
        return results
    
    async def run_validation(self, phase: TestPhase, max_prompts: int = None, target_prompt_index: Optional[int] = None, target_prompt_id: Optional[str] = None) -> ValidationSession:
        """Run validation for specified phase."""
        self.session = ValidationSession(
            session_id=f"val_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            started_at=datetime.now(),
            phase=phase
        )
        
        m_edge_cases = self.test_data.get("categories", {}).get("M_EDGE_CASES", {})
        callback_edge_tests = m_edge_cases.get("callback_edge_cases", {}).get("tests", [])
        REQUIRED_CALLBACK_EDGE_IDS = {t["id"] for t in callback_edge_tests}
        
        if phase in [TestPhase.PQ, TestPhase.ALL]:
            logger.info("Running PQ tests...")
            results = await self.run_pq_tests()
            self.session.results.extend(results)
        
        if phase in [TestPhase.OV, TestPhase.ALL]:
            logger.info("Running OV unit tests...")
            results = await self.run_ov_unit_tests()
            self.session.results.extend(results)
            
            logger.info("Running OV E2E tests...")
            results = await self.run_ov_e2e_tests()
            self.session.results.extend(results)
            
            logger.info("Running repetition safeguard tests...")
            results = await self.run_repetition_tests()
            self.session.results.extend(results)
        
        if phase in [TestPhase.MCP, TestPhase.ALL]:
            logger.info("Running MCP tool tests...")
            results = await self.run_mcp_tests()
            self.session.results.extend(results)
            
            logger.info("Running multi-turn tests...")
            results = await self.run_multi_turn_tests()
            self.session.results.extend(results)
            
            logger.info("Running callback edge case tests (M_EDGE_CASES-132 regression)...")
            results = await self.run_callback_edge_case_tests()
            self.session.results.extend(results)
        
        if phase in [TestPhase.FULL, TestPhase.ALL]:
            logger.info("Running full validation...")
            results = await self.run_full_validation(max_prompts, target_prompt_index=target_prompt_index, target_prompt_id=target_prompt_id)
            self.session.results.extend(results)

            # If the caller provided a max_prompts limit, assume they wanted only to exercise
            # the core full-validation loop and skip the additional vague-loop and callback-edge tests.
            # When no max_prompts is provided (None), retain historical behavior and run the extra tests.
            if max_prompts is None or phase == TestPhase.ALL:
                logger.info("Running vague loop tests...")
                results = await self.run_vague_loop_test()
                self.session.results.extend(results)

                logger.info("Running callback edge case tests (M_EDGE_CASES-132 regression)...")
                results = await self.run_callback_edge_case_tests()
                self.session.results.extend(results)
        if REQUIRED_CALLBACK_EDGE_IDS and (
            phase == TestPhase.MCP
            or (phase == TestPhase.FULL and max_prompts is None)
            or phase == TestPhase.ALL
        ):
            executed_ids = {r.test_id for r in self.session.results}
            missing_ids = REQUIRED_CALLBACK_EDGE_IDS - executed_ids
            if missing_ids:
                logger.error(f"REGRESSION GUARD FAILED: Missing callback edge case tests: {missing_ids}")
                for missing_id in missing_ids:
                    fail_result = TestResult(
                        test_id=missing_id,
                        category="CALLBACK_EDGE",
                        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        pass_fail="FAIL",
                        reasoning=f"REGRESSION GUARD: Required test {missing_id} was not executed. Check test_prompts.json callback_edge_cases."
                    )
                    self.session.results.append(fail_result)
            else:
                logger.info(f"Callback edge case coverage verified: {REQUIRED_CALLBACK_EDGE_IDS}")
        
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
        
        # Categories that are unit tests (no LLM/persona involved)
        UNIT_TEST_CATEGORIES = {"PQ", "OV_UNIT"}
        
        for result in self.session.results:
            key = getattr(result, field_name, "")
            if not key:
                # Only label as "N/A (unit test)" for actual unit test categories
                # Full tests with missing metadata remain "unknown"
                if result.category in UNIT_TEST_CATEGORIES:
                    key = "N/A (unit test)"
                else:
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
        
        sorted_groups = dict(sorted(groups.items(), key=lambda x: x[1].get("pass_rate", 0)))
        return sorted_groups
    
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
    
    parser = argparse.ArgumentParser(
            description="BrandonBot Validation Script – Adversarial Evaluator",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
    Usage examples:
      python -m validation.validator                     # Run all phases (default)
      python -m validation.validator --phase all         # Explicitly run everything
      python -m validation.validator --phase pq          # Prequalifier tests only
      python -m validation.validator --phase ov          # Output Validator tests (unit, E2E, repetition)
      python -m validation.validator --phase mcp         # Tool (MCP) verification + multi-turn + callback edge cases
      python -m validation.validator --phase full        # Full adversarial conversations with LLMjudge scoring
    
      python -m validation.validator --phase full --max-prompts 10
      python -m validation.validator --no-judge --phase ov
      python -m validation.validator --output ./custom_results
    
    Phases:
      pq     → Rate limiting, sanitization, frustration/vagueness detection
      ov     → Output Validator unit tests, drift detection, repetition safeguard
      mcp    → Tool call verification, multi-turn logic, callback edge cases (incl. regression guards)
      full   → End-to-end adversarial conversations with persona simulation and scoring
      all    → Run pq + ov + mcp + full sequentially
    
    Results are exported as CSV + JSON summary with aggregations by category, persona, model, and style.
            """
        )
    
    parser.add_argument(
        "--phase",
        choices=["pq", "ov", "mcp", "full", "all"],
        default="all",
        help="Validation phase to run (default: %(default)s)"
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        metavar="N",
        help="Limit number of prompts in 'full' phase (useful for quick runs)"
    )
    parser.add_argument(
        "--prompt",
        type=int,
        default=None,
        metavar="I",
        help="Run only the prompt with the given global index (e.g., --prompt 42). Indexing is zero-based and spans all categories."
    )
    parser.add_argument(
        "--prompt-id",
        type=str,
        default=None,
        metavar="ID",
        help="Run only the prompt matching the exact test id (e.g., --prompt-id A_VAGUE-002)."
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Disable Ollama LLM judge – responses will not be scored (useful for debugging)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="DIR",
        help="Custom output directory for results (default: ./validation/results)"
    )
    args = parser.parse_args()
    # Accept either lowercase or uppercase CLI phase values (e.g. 'pq' or 'PQ')
    try:
        phase = TestPhase(args.phase)
    except ValueError:
        phase = TestPhase(args.phase.lower())
    print(f"\nStarting BrandonBot Validation – Phase: {phase.value.upper()}")
    print(f"Testing mode: {TESTING_MODE}")
    if args.no_judge:
        print("Ollama judge disabled (--no-judge)")
    if args.max_prompts:
        print(f"Limiting full validation to {args.max_prompts} prompts")
    # Agent is needed for phases that interact with the real bot
    use_agent = phase in [TestPhase.MCP, TestPhase.FULL, TestPhase.OV, TestPhase.ALL]
    # Always require SLMs for validation runs. This enforces the production
    # behavior where SLM-based checks (intent, FEC RAG, etc.) are mandatory.
    # Attempt to initialize a WeaviateManager for full validation runs and
    # pass it into the validator so FEC RAG is wired explicitly. If this
    # initialization fails, we continue with validator creation and allow
    # fail-closed behavior when OV requires RAG.
    wm_for_validator = None
    try:
        from weaviate_manager import WeaviateManager
        wm_candidate = WeaviateManager()
        try:
            # We're already inside an asyncio.run() context here; initialize
            # the WeaviateManager asynchronously to avoid event loop errors.
            await wm_candidate.initialize()
            wm_for_validator = wm_candidate
            logger.info("Initialized WeaviateManager for validation run")
        except Exception as e:
            logger.info(f"Could not initialize WeaviateManager for validation run: {e}")
    except Exception:
        # WeaviateManager not available in this environment
        wm_for_validator = None

    validator = BrandonBotValidator(
        use_judge=not args.no_judge,
        use_agent=use_agent,
        require_slm=True,
        weaviate_manager=wm_for_validator,
    )
    session = await validator.run_validation(phase, args.max_prompts, target_prompt_index=args.prompt, target_prompt_id=args.prompt_id)
    csv_path = validator.export_results(args.output)
    validator.print_summary()
    print(f"\nResults saved to: {csv_path}")
    if args.output:
        print(f"Output directory: {os.path.abspath(args.output)}")
    # Graceful shutdown of optional resources to avoid ResourceWarning(s)
    try:
        if hasattr(validator, '_weaviate') and validator._weaviate:
            try:
                await validator._weaviate.close()
                logger.info('WeaviateManager closed successfully')
            except Exception as e:
                logger.warning(f'Failed to close WeaviateManager: {e}')
    except Exception:
        # Defensive: ignore any unexpected errors while closing weaviate
        pass

    try:
        if hasattr(validator, 'agent') and validator.agent:
            close_fn = getattr(validator.agent, 'close', None)
            if callable(close_fn):
                try:
                    if hasattr(close_fn, '__call__') and __import__('asyncio').iscoroutinefunction(close_fn):
                        await close_fn()
                    else:
                        close_fn()
                    logger.info('Agent closed successfully')
                except Exception as e:
                    logger.warning(f'Failed to close agent: {e}')
    except Exception:
        pass

    try:
        if hasattr(validator, '_slm_manager') and validator._slm_manager:
            close_fn = getattr(validator._slm_manager, 'close', None)
            if callable(close_fn):
                try:
                    if __import__('asyncio').iscoroutinefunction(close_fn):
                        await close_fn()
                    else:
                        close_fn()
                    logger.info('SLMManager closed successfully')
                except Exception as e:
                    logger.warning(f'Failed to close SLMManager: {e}')
    except Exception:
        pass

if __name__ == "__main__":
    asyncio.run(main())
