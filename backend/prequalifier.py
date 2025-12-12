"""
Prequalifier Module for BrandonBot

3-Stage Pipeline: PQ → LLM → OV

The Prequalifier (PQ) performs:
1. Rate Limiting (from security.py)
2. Input Sanitization (from security.py)  
3. Hybrid Frustration/Escalation Detection (Pattern flags → SLM → ESCALATE/CONTINUE)
4. RAG-based Vagueness Detection (RAG confidence → SLM → CLEAR/VAGUE)
5. Prompt Enrichment based on 2x2 matrix (ESCALATE/CONTINUE × CLEAR/VAGUE)

The PQ outputs an enriched prompt for the main LLM, NOT classifications.
Intent classification belongs in the Output Validator (to verify response answers query).
Ogilvy categories are retrieved via retrieve_answer_style() tool, not PQ.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any, Union
from enum import Enum

from security import input_sanitizer, rate_limiter, SanitizationResult
from meme_detector import meme_detector, get_meme_response_prompt

logger = logging.getLogger(__name__)

MEME_BYPASS_CRYPTO_KEYWORDS = {
    "tokenize", "tokenization", "token", "tokens", "cryptocurrency", "crypto",
    "bitcoin", "btc", "ethereum", "eth", "blockchain", "defi", "stablecoin",
    "cbdc", "federal reserve", "fed", "asset-backed", "asset backed",
    "digital currency", "digital dollar", "fiat", "monetary policy",
    "sound money", "gold standard", "inflation", "central bank"
}

MEME_BYPASS_RELIGION_KEYWORDS = {
    "jesus", "christ", "god", "faith", "scripture", "bible", "biblical",
    "christian", "christianity", "prayer", "pray", "church", "gospel",
    "lord", "savior", "salvation", "blessed", "blessing", "holy", "spirit",
    "commandments", "sermon", "worship", "religious", "religion"
}

def _should_bypass_meme_detection(message: str) -> bool:
    """
    Check if message contains crypto or religion keywords that should bypass meme detection.
    
    Brandon's platform heavily focuses on cryptocurrency and faith topics. 
    These should be treated as serious policy questions, not potential memes.
    """
    message_lower = message.lower()
    
    for keyword in MEME_BYPASS_CRYPTO_KEYWORDS:
        if keyword in message_lower:
            logger.debug(f"Meme bypass: crypto keyword '{keyword}' detected")
            return True
    
    for keyword in MEME_BYPASS_RELIGION_KEYWORDS:
        if keyword in message_lower:
            logger.debug(f"Meme bypass: religion keyword '{keyword}' detected")
            return True
    
    return False


class SLMNotAvailableError(Exception):
    """Raised when SLM is required but not available for hybrid classification."""
    pass


class FrustrationDecision(Enum):
    """SLM decision on user frustration/escalation - 3-bucket classification"""
    CALM = "calm"
    ANNOYED = "annoyed"
    FRUSTRATED = "frustrated"
    ESCALATE = "escalate"  # Alias for FRUSTRATED (backward compat)
    CONTINUE = "continue"  # Alias for CALM (backward compat)


class VaguenessDecision(Enum):
    """SLM decision on query clarity - 3-bucket classification"""
    CLEAR = "clear"
    NEEDS_CLARIFICATION = "needs_clarification"
    VAGUE = "vague"


@dataclass
class PatternFlags:
    """Boolean flags from pattern matching (Step 1 of hybrid detection)"""
    profanity: bool = False
    all_caps: bool = False
    repeated_punct: bool = False
    urgent_keywords: bool = False
    demands_human: bool = False
    insults: bool = False
    frustration_phrases: bool = False
    frustration_count: int = 0  # Number of frustration patterns matched
    sqli_attempt: bool = False
    prompt_injection: bool = False
    
    @property
    def excessive_caps(self) -> bool:
        """Alias for all_caps (for test compatibility)"""
        return self.all_caps
    
    @property
    def excessive_punctuation(self) -> bool:
        """Alias for repeated_punct (for test compatibility)"""
        return self.repeated_punct
    
    def to_dict(self) -> Dict[str, bool]:
        return {
            "profanity": self.profanity,
            "all_caps": self.all_caps,
            "repeated_punct": self.repeated_punct,
            "urgent_keywords": self.urgent_keywords,
            "demands_human": self.demands_human,
            "insults": self.insults,
            "frustration_phrases": self.frustration_phrases,
            "sqli_attempt": self.sqli_attempt,
            "prompt_injection": self.prompt_injection,
            "excessive_caps": self.all_caps,
            "excessive_punctuation": self.repeated_punct,
        }
    
    def any_high_risk(self) -> bool:
        """Check if any high-risk flags are set"""
        return any([
            self.profanity,
            self.insults,
            self.demands_human,
            self.urgent_keywords,
            self.frustration_phrases,
            self.all_caps and len(self.to_dict()) > 1,
        ])


@dataclass
class RAGResult:
    """Single RAG retrieval result with confidence"""
    confidence: float
    source: str
    collection: str
    content: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "confidence": self.confidence,
            "source": self.source,
            "collection": self.collection,
            "content": self.content[:500],  # Truncate for SLM context
        }


@dataclass
class InternalHints:
    """
    Sideband signals for the LLM that stay in system prompt context.
    
    These hints guide the LLM's response without appearing in user messages.
    Leak detection checks ensure these never appear in final output.
    """
    buying_signals: List[str] = field(default_factory=list)
    suggested_actions: List[str] = field(default_factory=list)
    frustration_context: Optional[str] = None
    ov_feedback: Optional[str] = None
    detected_email: Optional[str] = None
    
    def to_system_prompt_block(self) -> str:
        """Format hints for injection into system prompt."""
        if not self.buying_signals and not self.suggested_actions and not self.frustration_context and not self.ov_feedback:
            return ""
        
        lines = ["[INTERNAL_CONTEXT_7x9k2m]"]  # Unique marker for leak detection
        
        if self.buying_signals:
            lines.append(f"- Buying signals detected: {', '.join(self.buying_signals)}")
        
        if self.suggested_actions:
            for action in self.suggested_actions:
                lines.append(f"- Suggested action: {action}")
        
        if self.detected_email:
            lines.append(f"- User provided email: {self.detected_email}")
        
        if self.frustration_context:
            lines.append(f"- User frustration note: {self.frustration_context}")
        
        if self.ov_feedback:
            lines.append(f"- Previous response issue: {self.ov_feedback}")
        
        lines.append("[END_INTERNAL_CONTEXT_7x9k2m]")
        return "\n".join(lines)
    
    @staticmethod
    def get_leak_markers() -> List[str]:
        """Return markers that should NEVER appear in final user-facing output."""
        return [
            "[INTERNAL_CONTEXT_7x9k2m]",
            "[END_INTERNAL_CONTEXT_7x9k2m]",
            "INTERNAL_CONTEXT",
            "Buying signals detected",
            "Suggested action:",
        ]


@dataclass
class PrequalifierResult:
    """Result from prequalifier analysis"""
    # Original query
    query: str = ""
    
    # Security checks
    rate_limited: bool = False
    rate_limit_wait_seconds: Optional[int] = None
    sanitized_message: str = ""
    sanitization_applied: bool = False
    sanitization_issues: List[str] = field(default_factory=list)
    blocked: bool = False
    block_reason: Optional[str] = None
    
    # Hybrid detection decisions
    frustration_decision: FrustrationDecision = FrustrationDecision.CALM
    vagueness_decision: VaguenessDecision = VaguenessDecision.CLEAR
    pattern_flags: Optional[PatternFlags] = None
    
    # Detected emotion from 7-emotion classifier
    detected_emotion: str = "neutral"
    
    # RAG context (for vagueness and prompt enrichment)
    rag_results: List[RAGResult] = field(default_factory=list)
    avg_rag_confidence: float = 0.0
    
    # Confidence score (0.0-1.0) based on RAG results
    confidence: float = 0.0
    
    # Enriched prompt for main LLM
    enriched_prompt: Optional[str] = None
    pq_instructions: Optional[str] = None
    
    # Pass-through flag (CLEAR + CALM = no enrichment needed)
    passthrough: bool = False
    
    # Internal hints for LLM guidance (never shown to user)
    internal_hints: InternalHints = field(default_factory=InternalHints)
    
    # Meme/subcontext detection
    meme_detected: bool = False
    meme_context: str = ""
    meme_prompt: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "query": self.query,
            "rate_limited": self.rate_limited,
            "rate_limit_wait_seconds": self.rate_limit_wait_seconds,
            "sanitized_message": self.sanitized_message,
            "sanitization_applied": self.sanitization_applied,
            "sanitization_issues": self.sanitization_issues,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "frustration_decision": self.frustration_decision.value,
            "vagueness_decision": self.vagueness_decision.value,
            "pattern_flags": self.pattern_flags.to_dict() if self.pattern_flags else None,
            "detected_emotion": self.detected_emotion,
            "rag_results": [r.to_dict() for r in self.rag_results],
            "avg_rag_confidence": self.avg_rag_confidence,
            "confidence": self.confidence,
            "enriched_prompt": self.enriched_prompt,
            "pq_instructions": self.pq_instructions,
            "passthrough": self.passthrough,
            "meme_detected": self.meme_detected,
            "meme_context": self.meme_context,
        }


class Prequalifier:
    """
    Prequalifier with hybrid pattern+SLM detection.
    
    Flow:
    1. Rate limiting check
    2. Input sanitization
    3. Pattern matching → flags dict (does NOT block)
    4. SLM frustration classifier (flags + message → ESCALATE/CONTINUE)
    5. RAG retrieval for query
    6. SLM vagueness classifier (query + RAG data → CLEAR/VAGUE)
    7. Build enriched prompt based on 2x2 matrix
    """
    
    # Pattern matching for Step 1 (does NOT block, just flags)
    # Severe profanity gets higher weight in frustration scoring
    SEVERE_PROFANITY_PATTERNS = [
        r"\b(fuck(ing|ed|er|s)?|shit(ty|s)?|ass(hole)?|bitch(es)?|bastard)\b",
        r"\bf+u+c+k+\b",
        r"\bs+h+i+t+\b",
    ]
    
    # Mild profanity - less weight in frustration scoring
    MILD_PROFANITY_PATTERNS = [
        r"\b(damn(ed)?|crap(py)?|hell|heck|bullshit|bs)\b",
        r"\bwhat the (hell|heck)\b",
    ]
    
    PROFANITY_PATTERNS = SEVERE_PROFANITY_PATTERNS + MILD_PROFANITY_PATTERNS
    
    INSULT_PATTERNS = [
        r"\b(you('re| are)|this is) (stupid|idiot|moron|dumb|useless)\b",
        r"\b(you('re| are)|brandon is) (a )?(joke|fraud|liar|fake|scam)\b",
        r"\b(worst|terrible|horrible|pathetic)\b",
    ]
    
    URGENCY_PATTERNS = [
        r"\b(urgent|emergency|asap|right now|immediately|now!)\b",
        r"\b(time.?sensitive|critical|deadline)\b",
        r"\bcan('t|not) wait\b",
    ]
    
    HUMAN_DEMAND_PATTERNS = [
        r"\b(talk|speak) (to|with) (a |someone |)(human|person|real|actual)\b",
        r"\bcall (me|back)\b",
        r"\bneed (a |to talk to a )(human|person|real)\b",
        r"\bwant (to talk to |)(a |)(human|person)\b",
    ]
    
    FRUSTRATION_PATTERNS = [
        r"(already|just) (said|asked|told|explained)",
        r"(you )?(already )?told me",
        r"(doesn't|don't|didn't|won't) (answer|help|make sense|understand|work)",
        r"(still )?(haven't|hasn't|don't|doesn't) (addressed|answered|helped)",
        r"(waste|wasting) (of |my )?time",
        r"(not|isn't|aren't) (helping|working|useful)",
        r"(i('m| am)|this is) (confused|frustrated|annoyed|angry)",
        r"forget it|never ?mind",
        r"(ugh|argh|omg|ffs|wtf)",
        r"this (doesn't|won't|isn't) (work|help)",
        r"(asked|asking|ask).*(times|again)",
        r"(keep|keeps) (ignoring|avoiding|missing)",
        r"useless|pointless|unhelpful",
        r"fed up|sick of|tired of|had enough",
        r"you (never|won't|can't|don't) (answer|help|listen)",
    ]
    
    # Buying signal patterns - indicate user wants to take action
    BUYING_SIGNAL_PATTERNS = [
        (r"(i('d| would)?|i'd) (love|like|want) to help", "wants_to_help"),
        (r"(sign|count) me (up|in)", "sign_up"),
        (r"(want|ready|like) to (volunteer|donate|contribute|support)", "action_ready"),
        (r"how (can|do) i (help|support|contribute|donate|volunteer)", "action_inquiry"),
        (r"(i('m| am)|i'd be) (happy|willing|interested) to", "willing_to_help"),
        (r"(take|taking) my money", "ready_to_donate"),
        (r"(send|give|email) me (the|a|more) (booklet|info|brochure|information)", "wants_materials"),
        (r"(my email|email.*(is|:))", "provided_email"),
    ]
    
    # SLM prompts for hybrid detection
    FRUSTRATION_SLM_PROMPT = """You are a sentiment-and-escalation classifier for a political campaign chatbot.

Input message: "{user_message}"

Pattern flags detected: {flags}

Based on the message and the flags, decide whether the user is:
- "ESCALATE" → the user is angry, hostile, demands a human, or is beyond frustrated. They need immediate human attention or de-escalation.
- "CONTINUE" → the message is neutral or mildly frustrated, safe to send to the main LLM.

Consider:
- Profanity or insults may indicate frustration, but context matters
- Urgent keywords alone don't always mean escalation
- Multiple flags together are more indicative than single flags
- The actual message content overrides pattern flags if the tone is clearly different

Respond with ONLY a single word: ESCALATE or CONTINUE"""

    VAGUENESS_SLM_PROMPT = """You are a query clarity classifier for a political campaign chatbot about Brandon Sowers.

Input message: "{user_message}"

Data retrieved from knowledge base:
{rag_data}

Average confidence of retrieved data: {avg_confidence:.2f}

Based on the message and the retrieved data, decide whether the user's query is:
- "CLEAR" → the user's question has clear intent and can be precisely answered with the available data
- "VAGUE" → the user's question needs refinement. The intent isn't well-formed, or we don't have a clear answer in the knowledge base.

Consider:
- Low confidence scores (< 0.5) suggest the question may not match our knowledge base well
- Very short queries (< 5 words) are often vague
- Questions with clear nouns and verbs are usually clear
- "What about X?" type questions are often vague
- If the RAG data doesn't seem to answer the question, it's vague

Respond with ONLY a single word: CLEAR or VAGUE"""

    # Enrichment templates for 2x2 matrix (supports both old and new frustration values)
    _ESCALATION_TEMPLATE = """user query: {user_query}

The user is agitated but we have relevant answers. Acknowledge the frustration and show you aim to help by stating 'would it be helpful if I explain...' followed by a plan based on the RAG content.

RAG retrieval:
{rag_data}

Data from BrandonPlatform and PreviousQA are authoritative based on Brandon's own words. Data from PartyPlatform is from party platforms - clearly distinguish between Brandon's positions and party positions.

Important: Validate your response before delivering. Acknowledge frustration, then provide helpful information."""

    _VAGUE_CALM_TEMPLATE = """user query: {user_query}

The query is vague. Guide the user to a clearer question using relevant parts of Brandon's platform.

Relevant positions from knowledge base that might help:
{rag_data}

CRITICAL - AVOID REPETITION:
- Review conversation history to see what you've already asked
- DO NOT repeat the same clarifying question you asked before
- If you've already asked for clarification, try a DIFFERENT approach:
  * Offer specific topic options to choose from
  * Share a relevant position and ask if that's what they meant
  * Ask a more specific follow-up based on their last response
- After 2-3 clarifying turns, make your best attempt to answer or offer a callback

Be warm and helpful. Always provide brandonsowers.com for more information."""

    _VAGUE_ESCALATE_TEMPLATE = """user query: {user_query}

The user is frustrated and their query is unclear. They need immediate de-escalation.

Explain that you want to help but aren't sure exactly what matters most to them. Apologize for any confusion. Offer to have a member of Brandon's team call them back for personal assistance.

Do NOT try to answer their unclear question. Focus on de-escalation and human escalation options."""

    ENRICHMENT_TEMPLATES: Dict[Tuple[str, str], Optional[str]] = {
        ("clear", "escalate"): _ESCALATION_TEMPLATE,
        ("clear", "frustrated"): _ESCALATION_TEMPLATE,
        ("vague", "continue"): _VAGUE_CALM_TEMPLATE,
        ("vague", "calm"): _VAGUE_CALM_TEMPLATE,
        ("vague", "escalate"): _VAGUE_ESCALATE_TEMPLATE,
        ("vague", "frustrated"): _VAGUE_ESCALATE_TEMPLATE,
        ("clear", "continue"): None,
        ("clear", "calm"): None,
    }

    def __init__(self, slm_provider=None, weaviate_manager=None, require_slm: bool = True):
        """
        Initialize prequalifier.
        
        Args:
            slm_provider: Small LLM for classification (uses main LLM if None)
            weaviate_manager: Vector DB for RAG retrieval
            require_slm: If True (default), raise SLMNotAvailableError when SLM 
                        is required but not available. Enforces hybrid mode
                        (patterns + SLM) rather than allowing pattern-only fallback.
        """
        self.slm = slm_provider
        self.weaviate = weaviate_manager
        self._require_slm = require_slm
    
    def set_slm_provider(self, provider):
        """Set SLM provider after initialization"""
        self.slm = provider
    
    def set_weaviate_manager(self, manager):
        """Set Weaviate manager after initialization"""
        self.weaviate = manager
    
    async def analyze(
        self,
        message: str,
        session_id: str = "default",
        conversation_history: List[Dict] = None,
    ) -> PrequalifierResult:
        """
        Full prequalifier analysis pipeline.
        
        Steps:
        1. Rate limiting
        2. Input sanitization
        3. Pattern matching → flags
        4. SLM frustration classification
        5. RAG retrieval
        6. SLM vagueness classification
        7. Build enriched prompt
        """
        result = PrequalifierResult()
        result.query = message  # Store original query
        
        # Step 1: Rate limiting
        is_allowed, wait_seconds = rate_limiter.check_rate_limit(session_id, "query")
        if not is_allowed:
            result.rate_limited = True
            result.rate_limit_wait_seconds = wait_seconds
            result.blocked = True
            result.block_reason = f"Rate limit exceeded. Please wait {wait_seconds} seconds."
            return result
        
        # Step 2: Input sanitization
        sanitization = input_sanitizer.sanitize(message)
        result.sanitized_message = sanitization.cleaned_text
        result.sanitization_applied = sanitization.was_modified
        result.sanitization_issues = [f"{issue[0]}: {issue[1]}" for issue in sanitization.issues_found]
        
        # Check for blocked content
        for issue_type, _ in sanitization.issues_found:
            if issue_type in ["script_injection", "sql_injection", "prompt_injection"]:
                result.blocked = True
                result.block_reason = f"Security violation detected: {issue_type}"
                return result
        
        # Step 2.5: Meme/subcontext detection (for short questions)
        # Skip for crypto/religion topics - Brandon's key focus areas
        if _should_bypass_meme_detection(result.sanitized_message):
            logger.info(f"Meme detection bypassed for crypto/religion topic: {result.sanitized_message[:50]}...")
        else:
            try:
                meme_result = await meme_detector.detect(result.sanitized_message)
                if meme_result.is_meme:
                    result.meme_detected = True
                    result.meme_context = meme_result.context
                    result.meme_prompt = get_meme_response_prompt(meme_result)
                    logger.info(f"Meme detected in query: {result.sanitized_message[:50]}...")
            except Exception as e:
                logger.warning(f"Meme detection failed (non-fatal): {e}")
        
        # Step 3: Pattern matching (does NOT block, just flags)
        pattern_flags = self._detect_patterns(result.sanitized_message)
        result.pattern_flags = pattern_flags
        
        # Step 4: SLM frustration classification (returns decision + detected emotion)
        frustration_decision, detected_emotion = await self._classify_frustration_async(
            result.sanitized_message,
            pattern_flags,
            conversation_history
        )
        result.frustration_decision = frustration_decision
        result.detected_emotion = detected_emotion
        
        # Step 5: RAG retrieval
        rag_results, avg_confidence = await self._retrieve_rag_context(result.sanitized_message)
        result.rag_results = rag_results
        result.avg_rag_confidence = avg_confidence
        result.confidence = avg_confidence  # Set confidence from RAG results
        
        # Step 6: SLM vagueness classification
        vagueness_decision = await self._classify_vagueness_async(
            result.sanitized_message,
            rag_results,
            avg_confidence
        )
        result.vagueness_decision = vagueness_decision
        
        # Step 7: Build enriched prompt based on 2x2 matrix
        enriched_prompt, pq_instructions = self._build_enriched_prompt(
            result.sanitized_message,
            frustration_decision,
            vagueness_decision,
            rag_results
        )
        result.enriched_prompt = enriched_prompt
        result.pq_instructions = pq_instructions
        result.passthrough = (
            frustration_decision in [FrustrationDecision.CALM, FrustrationDecision.CONTINUE] and
            vagueness_decision == VaguenessDecision.CLEAR
        )
        
        # Step 8: Detect buying signals and build internal hints
        buying_signals, detected_email = self._detect_buying_signals(result.sanitized_message)
        result.internal_hints = self._build_internal_hints(
            buying_signals,
            detected_email,
            frustration_decision
        )
        
        return result
    
    def _detect_patterns(self, message: str) -> PatternFlags:
        """
        Step 1: Pattern matching to create boolean flag dict.
        Does NOT block - just tags potential high-risk patterns.
        """
        flags = PatternFlags()
        message_lower = message.lower()
        
        # Check profanity
        for pattern in self.PROFANITY_PATTERNS:
            if re.search(pattern, message_lower, re.IGNORECASE):
                flags.profanity = True
                break
        
        # Check insults
        for pattern in self.INSULT_PATTERNS:
            if re.search(pattern, message_lower, re.IGNORECASE):
                flags.insults = True
                break
        
        # Check urgent keywords
        for pattern in self.URGENCY_PATTERNS:
            if re.search(pattern, message_lower, re.IGNORECASE):
                flags.urgent_keywords = True
                break
        
        # Check human demands
        for pattern in self.HUMAN_DEMAND_PATTERNS:
            if re.search(pattern, message_lower, re.IGNORECASE):
                flags.demands_human = True
                break
        
        # Check frustration phrases (count all matches)
        frustration_count = 0
        for pattern in self.FRUSTRATION_PATTERNS:
            if re.search(pattern, message_lower, re.IGNORECASE):
                frustration_count += 1
        flags.frustration_phrases = frustration_count > 0
        flags.frustration_count = frustration_count
        
        # Check for ALL CAPS (more than 50% uppercase, at least 10 chars)
        alpha_chars = [c for c in message if c.isalpha()]
        if len(alpha_chars) >= 10:
            upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            if upper_ratio > 0.5:
                flags.all_caps = True
        
        # Check repeated punctuation
        if re.search(r"[!?]{2,}", message):
            flags.repeated_punct = True
        
        return flags
    
    def _detect_buying_signals(self, message: str) -> Tuple[List[str], Optional[str]]:
        """
        Detect buying signals in user message.
        
        Returns:
            Tuple of (list of signal types detected, detected email if any)
        """
        signals = []
        message_lower = message.lower()
        detected_email = None
        
        for pattern, signal_type in self.BUYING_SIGNAL_PATTERNS:
            if re.search(pattern, message_lower, re.IGNORECASE):
                signals.append(signal_type)
        
        # Extract email if mentioned
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', message)
        if email_match:
            detected_email = email_match.group()
        
        return signals, detected_email
    
    def _build_internal_hints(
        self, 
        signals: List[str], 
        detected_email: Optional[str],
        frustration_decision: FrustrationDecision
    ) -> InternalHints:
        """
        Build internal hints based on detected signals and frustration.
        
        These hints go into the system prompt to guide the LLM without
        appearing in user-facing output.
        """
        hints = InternalHints()
        
        if signals:
            hints.buying_signals = signals
            
            # Generate suggested actions based on signals
            if any(s in signals for s in ['wants_to_help', 'sign_up', 'action_ready', 'willing_to_help']):
                hints.suggested_actions.append("Offer volunteer signup with register_volunteer tool")
            
            if any(s in signals for s in ['ready_to_donate', 'action_ready']):
                hints.suggested_actions.append("Offer donation with make_donation tool")
            
            if 'action_inquiry' in signals:
                hints.suggested_actions.append("Explain how user can help: volunteer, donate, or spread the word")
            
            if 'wants_materials' in signals:
                hints.suggested_actions.append("Use perform_web_search to find booklet download link on brandonsowers.com, then provide the link")
        
        if detected_email:
            hints.detected_email = detected_email
        
        if frustration_decision == FrustrationDecision.ESCALATE:
            hints.frustration_context = "User is frustrated. Prioritize de-escalation and offer callback."
        elif frustration_decision == FrustrationDecision.FRUSTRATED:
            hints.frustration_context = "User shows mild frustration. Be extra empathetic and helpful."
        
        return hints
    
    def _classify_frustration(
        self,
        flags: PatternFlags,
        message: str,
        history: List[Dict] = None
    ) -> FrustrationDecision:
        """
        Synchronous frustration classification for direct calls.
        Uses pattern flags + message to determine ESCALATE or CONTINUE.
        
        Args:
            flags: Pattern flags detected in message
            message: Original user message (REQUIRED for severity classification)
            history: Optional conversation history
        
        For SLM-based classification, use _classify_frustration_async.
        """
        if not message and flags.profanity:
            raise ValueError("message is required when profanity flag is set for severity classification")
        return self._fallback_frustration_classification(flags, message, history)
    
    async def _classify_frustration_async(
        self,
        message: str,
        flags: PatternFlags,
        history: List[Dict] = None
    ) -> Tuple[FrustrationDecision, str]:
        """
        Async SLM classification for frustration/escalation.
        Uses pattern flags + message to determine ESCALATE or CONTINUE.
        
        Returns:
            Tuple of (FrustrationDecision, detected_emotion)
            detected_emotion is one of: anger, disgust, fear, joy, neutral, sadness, surprise
            
        Raises:
            SLMNotAvailableError: If require_slm=True and SLM is not available.
        """
        if self.slm is None:
            if self._require_slm:
                raise SLMNotAvailableError(
                    "Frustration SLM not available. Prequalifier requires hybrid mode (patterns + SLM). "
                    "Set require_slm=False to use pattern-only fallback."
                )
            decision = self._fallback_frustration_classification(flags, message, history)
            return (decision, "neutral")
        
        try:
            response = await self.slm.classify_frustration(message, flags.to_dict())
            detected_emotion = getattr(response, 'detected_emotion', 'neutral') or 'neutral'
            
            if response.decision == "ESCALATE":
                return (FrustrationDecision.ESCALATE, detected_emotion)
            else:
                return (FrustrationDecision.CONTINUE, detected_emotion)
                
        except Exception as e:
            if self._require_slm:
                raise SLMNotAvailableError(
                    f"SLM frustration classification failed: {e}. "
                    "Prequalifier requires hybrid mode (patterns + SLM). "
                    "Set require_slm=False to use pattern-only fallback."
                )
            logger.warning(f"SLM frustration classification failed: {e}, using fallback")
            decision = self._fallback_frustration_classification(flags, message, history)
            return (decision, "neutral")
    
    def _fallback_frustration_classification(
        self,
        flags: PatternFlags,
        message: str,
        history: List[Dict] = None
    ) -> FrustrationDecision:
        """
        Fallback rule-based frustration classification when SLM unavailable.
        
        3-bucket classification:
        - CALM (score 0-1): No frustration indicators
        - ANNOYED (score 2-3): Mild frustration, can continue
        - FRUSTRATED (score 4+): High frustration, may need escalation
        
        Args:
            flags: Pattern flags detected in message
            message: Original user message (REQUIRED for severity classification)
            history: Optional conversation history
        """
        score = 0
        
        # Check for severe vs mild profanity (message is required)
        has_severe_profanity = False
        if message and flags.profanity:
            message_lower = message.lower()
            for pattern in self.SEVERE_PROFANITY_PATTERNS:
                if re.search(pattern, message_lower, re.IGNORECASE):
                    has_severe_profanity = True
                    break
        
        if flags.profanity:
            if has_severe_profanity:
                score += 3  # Severe profanity (fuck, shit, etc.)
            else:
                score += 1  # Mild profanity (damn, hell, heck)
        if flags.insults:
            score += 3
        if flags.demands_human:
            score += 2
        if flags.urgent_keywords:
            score += 1
        if flags.frustration_phrases:
            # Score based on number of frustration patterns matched
            # 1 pattern = +2, 2+ patterns = +3 (strong frustration)
            score += 2 if flags.frustration_count <= 1 else 3
        if flags.all_caps:
            score += 1
        if flags.repeated_punct:
            score += 2
        
        # Check conversation history for escalation patterns
        if history and len(history) >= 4:
            user_messages = [m for m in history if m.get("role") == "user"]
            if len(user_messages) >= 3:
                score += 1  # Long conversation without resolution
        
        # 3-bucket classification based on score
        # Score thresholds: CALM (0-2), ANNOYED (3), FRUSTRATED (4+)
        # This allows mild profanity + punctuation to stay CALM
        if score >= 4:
            return FrustrationDecision.FRUSTRATED
        elif score >= 3:
            return FrustrationDecision.ANNOYED
        else:
            return FrustrationDecision.CALM
    
    async def _retrieve_rag_context(
        self,
        query: str
    ) -> Tuple[List[RAGResult], float]:
        """
        Step 3: RAG retrieval for vagueness assessment and prompt enrichment.
        """
        if self.weaviate is None:
            return [], 0.0
        
        rag_results = []
        collections = ["BrandonPlatform", "PreviousQA", "PartyPlatform"]
        
        try:
            for collection in collections:
                try:
                    results = await self.weaviate.search(collection, query, limit=3)
                    for r in results:
                        rag_results.append(RAGResult(
                            confidence=r.get("confidence", 0.0),
                            source=r.get("source", "unknown"),
                            collection=collection,
                            content=r.get("content", r.get("text", ""))[:500],
                        ))
                except Exception as e:
                    logger.warning(f"RAG retrieval failed for {collection}: {e}")
            
            # Sort by confidence
            rag_results.sort(key=lambda x: x.confidence, reverse=True)
            rag_results = rag_results[:6]  # Top 6 results
            
            avg_confidence = (
                sum(r.confidence for r in rag_results) / len(rag_results)
                if rag_results else 0.0
            )
            
            return rag_results, avg_confidence
            
        except Exception as e:
            logger.error(f"RAG retrieval error: {e}")
            return [], 0.0
    
    def _classify_vagueness(
        self,
        message: str,
        avg_confidence: float,
        rag_results: List[RAGResult] = None
    ) -> VaguenessDecision:
        """
        Synchronous vagueness classification for direct calls.
        Uses query + confidence to determine CLEAR or VAGUE.
        
        For SLM-based classification, use _classify_vagueness_async.
        """
        return self._fallback_vagueness_classification(message, avg_confidence)
    
    async def _classify_vagueness_async(
        self,
        message: str,
        rag_results: List[RAGResult],
        avg_confidence: float
    ) -> VaguenessDecision:
        """
        RAG-informed vagueness classification using Qwen.
        
        The SLM receives the user query along with RAG results and similarity
        scores, allowing it to make an informed decision about whether the
        knowledge base can answer the query.
        
        Raises:
            SLMNotAvailableError: If require_slm=True and SLM is not available.
        """
        if self.slm is None:
            if self._require_slm:
                raise SLMNotAvailableError(
                    "Vagueness SLM not available. Prequalifier requires hybrid mode (patterns + SLM). "
                    "Set require_slm=False to use pattern-only fallback."
                )
            return self._fallback_vagueness_classification(message, avg_confidence)
        
        try:
            rag_dicts = [r.to_dict() for r in rag_results] if rag_results else []
            
            response = await self.slm.classify_vagueness_with_rag(
                message=message,
                rag_results=rag_dicts,
                avg_confidence=avg_confidence
            )
            
            logger.info(f"RAG+SLM vagueness: query='{message[:50]}...', decision={response.decision}, {response.explanation}")
            
            if response.decision == "VAGUE":
                return VaguenessDecision.VAGUE
            else:
                return VaguenessDecision.CLEAR
                
        except Exception as e:
            if self._require_slm:
                raise SLMNotAvailableError(
                    f"SLM vagueness classification failed: {e}. "
                    "Prequalifier requires hybrid mode (patterns + SLM). "
                    "Set require_slm=False to use pattern-only fallback."
                )
            logger.warning(f"RAG+SLM vagueness classification failed: {e}, using fallback")
            return self._fallback_vagueness_classification(message, avg_confidence)
    
    def _fallback_vagueness_classification(
        self,
        message: str,
        avg_confidence: float
    ) -> VaguenessDecision:
        """Fallback rule-based vagueness classification"""
        words = message.split()
        message_lower = message.lower()
        
        # Very short queries are vague
        if len(words) < 3:
            return VaguenessDecision.VAGUE
        
        # Single-word queries
        if len(words) == 1:
            return VaguenessDecision.VAGUE
        
        # "What about X?" pattern is often vague
        if re.match(r"^what about\s+", message_lower):
            return VaguenessDecision.VAGUE
        
        # Clear question patterns (specific topic + question structure)
        clear_patterns = [
            r"what is (brandon'?s?|his|the|your) (position|stance|view|plan|policy) on",
            r"where does (brandon|he) stand on",
            r"how (does|will|would|can) (brandon|he)",
            r"why (does|did|is|should) (brandon|he)",
            r"what (will|would|does|did) (brandon|he) (do|say|think|believe|propose)",
            r"(brandon'?s?|his) (position|stance|view|plan|policy) (on|about|regarding)",
            r"what are (brandon'?s?|his|the) (plans?|proposals?|ideas?|solutions?) (for|on|about|regarding)",
            r"tell me about (brandon'?s?|his|your|the) .*(policy|plan|position|proposal|stance|view)",
            r"(tell|explain|describe).*(brandon'?s?|his|your).*(on|about|for|regarding)",
        ]
        
        for pattern in clear_patterns:
            if re.search(pattern, message_lower):
                return VaguenessDecision.CLEAR
        
        # If we have RAG confidence, use it
        if avg_confidence >= 0.4:
            return VaguenessDecision.CLEAR
        
        # Longer queries with question words are likely clear
        question_words = ["what", "how", "why", "where", "when", "who", "which"]
        has_question = any(w in message_lower for w in question_words)
        has_topic = len(words) >= 5 and has_question
        
        if has_topic:
            return VaguenessDecision.CLEAR
        
        # Low confidence and no clear pattern
        if avg_confidence < 0.4:
            return VaguenessDecision.VAGUE
        
        return VaguenessDecision.CLEAR
    
    def _build_enriched_prompt(
        self,
        user_query: str,
        frustration: FrustrationDecision,
        vagueness: VaguenessDecision,
        rag_results: List[RAGResult]
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Step 5: Build enriched prompt based on 2x2 matrix.
        
        CLEAR + CONTINUE → passthrough (no enrichment)
        CLEAR + ESCALATE → de-escalate + answer
        VAGUE + CONTINUE → clarify question
        VAGUE + ESCALATE → immediate human escalation
        """
        key = (vagueness.value, frustration.value)
        template = self.ENRICHMENT_TEMPLATES.get(key)
        
        if template is None:
            # Passthrough - no enrichment needed
            return None, None
        
        # Format RAG data for prompt
        rag_data_str = "\n".join([
            f"- [{r.collection}] (source: {r.source}) {r.content}"
            for r in rag_results
        ]) or "No relevant data found."
        
        enriched = template.format(
            user_query=user_query,
            rag_data=rag_data_str
        )
        
        # Generate concise PQ instructions
        if key == ("clear", "escalate"):
            pq_instructions = "User is frustrated. Acknowledge feelings, then provide helpful answer from RAG."
        elif key == ("vague", "continue"):
            pq_instructions = "Query is vague. Ask clarifying questions before attempting to answer."
        elif key == ("vague", "escalate"):
            pq_instructions = "User is frustrated and query is unclear. De-escalate and offer human callback."
        else:
            pq_instructions = None
        
        return enriched, pq_instructions


# Singleton instance - uses require_slm=True by default to enforce hybrid mode
prequalifier = Prequalifier(require_slm=True)
