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

logger = logging.getLogger(__name__)


class FrustrationDecision(Enum):
    """SLM decision on user frustration/escalation"""
    ESCALATE = "escalate"
    CONTINUE = "continue"


class VaguenessDecision(Enum):
    """SLM decision on query clarity"""
    CLEAR = "clear"
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
    sqli_attempt: bool = False
    prompt_injection: bool = False
    
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
class PrequalifierResult:
    """Result from prequalifier analysis"""
    # Security checks
    rate_limited: bool = False
    rate_limit_wait_seconds: Optional[int] = None
    sanitized_message: str = ""
    sanitization_applied: bool = False
    sanitization_issues: List[str] = field(default_factory=list)
    blocked: bool = False
    block_reason: Optional[str] = None
    
    # Hybrid detection decisions
    frustration_decision: FrustrationDecision = FrustrationDecision.CONTINUE
    vagueness_decision: VaguenessDecision = VaguenessDecision.CLEAR
    pattern_flags: Optional[PatternFlags] = None
    
    # RAG context (for vagueness and prompt enrichment)
    rag_results: List[RAGResult] = field(default_factory=list)
    avg_rag_confidence: float = 0.0
    
    # Enriched prompt for main LLM
    enriched_prompt: Optional[str] = None
    pq_instructions: Optional[str] = None
    
    # Pass-through flag (CLEAR + CONTINUE = no enrichment needed)
    passthrough: bool = False


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
        r"\b(damn(ed)?|crap(py)?|hell|heck)\b",
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

    # Enrichment templates for 2x2 matrix
    ENRICHMENT_TEMPLATES: Dict[Tuple[str, str], Optional[str]] = {
        ("clear", "escalate"): """user query: {user_query}

The user is agitated but we have relevant answers. Acknowledge the frustration and show you aim to help by stating 'would it be helpful if I explain...' followed by a plan based on the RAG content.

RAG retrieval:
{rag_data}

Data from BrandonPlatform and PreviousQA are authoritative based on Brandon's own words. Data from PartyPlatform is from party platforms - clearly distinguish between Brandon's positions and party positions.

Important: Validate your response before delivering. Acknowledge frustration, then provide helpful information.""",

        ("vague", "continue"): """user query: {user_query}

The query is vague. Take a couple turns to gently guide the user to a clearer question using relevant parts of Brandon's platform. Don't assume what they're asking - ask clarifying questions.

Relevant positions from knowledge base that might help:
{rag_data}

Ask which specific aspect they'd like to know more about. Be warm and helpful, not dismissive.""",

        ("vague", "escalate"): """user query: {user_query}

The user is frustrated and their query is unclear. They need immediate de-escalation.

Explain that you want to help but aren't sure exactly what matters most to them. Apologize for any confusion. Offer to have a member of Brandon's team call them back for personal assistance.

Do NOT try to answer their unclear question. Focus on de-escalation and human escalation options.""",

        ("clear", "continue"): None,  # Passthrough - no enrichment needed
    }

    def __init__(self, slm_provider=None, weaviate_manager=None):
        """
        Initialize prequalifier.
        
        Args:
            slm_provider: Small LLM for classification (uses main LLM if None)
            weaviate_manager: Vector DB for RAG retrieval
        """
        self.slm = slm_provider
        self.weaviate = weaviate_manager
    
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
        
        # Step 3: Pattern matching (does NOT block, just flags)
        pattern_flags = self._detect_patterns(result.sanitized_message)
        result.pattern_flags = pattern_flags
        
        # Step 4: SLM frustration classification
        frustration_decision = await self._classify_frustration_async(
            result.sanitized_message,
            pattern_flags,
            conversation_history
        )
        result.frustration_decision = frustration_decision
        
        # Step 5: RAG retrieval
        rag_results, avg_confidence = await self._retrieve_rag_context(result.sanitized_message)
        result.rag_results = rag_results
        result.avg_rag_confidence = avg_confidence
        
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
            frustration_decision == FrustrationDecision.CONTINUE and
            vagueness_decision == VaguenessDecision.CLEAR
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
        
        # Check frustration phrases
        for pattern in self.FRUSTRATION_PATTERNS:
            if re.search(pattern, message_lower, re.IGNORECASE):
                flags.frustration_phrases = True
                break
        
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
    ) -> FrustrationDecision:
        """
        Async SLM classification for frustration/escalation.
        Uses pattern flags + message to determine ESCALATE or CONTINUE.
        """
        # If no SLM available, fall back to rule-based
        if self.slm is None:
            return self._fallback_frustration_classification(flags, message, history)
        
        try:
            response = await self.slm.classify_frustration(message, flags.to_dict())
            
            if response.decision == "ESCALATE":
                return FrustrationDecision.ESCALATE
            else:
                return FrustrationDecision.CONTINUE
                
        except Exception as e:
            logger.warning(f"SLM frustration classification failed: {e}, using fallback")
            return self._fallback_frustration_classification(flags, message, history)
    
    def _fallback_frustration_classification(
        self,
        flags: PatternFlags,
        message: str,
        history: List[Dict] = None
    ) -> FrustrationDecision:
        """
        Fallback rule-based frustration classification when SLM unavailable.
        
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
            score += 2
        if flags.all_caps:
            score += 1
        if flags.repeated_punct:
            score += 1
        
        # Check conversation history for escalation patterns
        if history and len(history) >= 4:
            user_messages = [m for m in history if m.get("role") == "user"]
            if len(user_messages) >= 3:
                score += 1  # Long conversation without resolution
        
        # Escalate if score >= 3 (severe profanity or insults alone should trigger)
        return FrustrationDecision.ESCALATE if score >= 3 else FrustrationDecision.CONTINUE
    
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
        Async SLM classification for query vagueness.
        Uses query + RAG data to determine CLEAR or VAGUE.
        """
        # If no SLM available, fall back to rule-based
        if self.slm is None:
            return self._fallback_vagueness_classification(message, avg_confidence)
        
        try:
            has_context = len(rag_results) > 0 and avg_confidence > 0.3
            response = await self.slm.classify_vagueness(message, avg_confidence, has_context)
            
            if response.decision == "VAGUE":
                return VaguenessDecision.VAGUE
            else:
                return VaguenessDecision.CLEAR
                
        except Exception as e:
            logger.warning(f"SLM vagueness classification failed: {e}, using fallback")
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
            r"what is (brandon'?s?|his|the) (position|stance|view|plan|policy) on",
            r"where does (brandon|he) stand on",
            r"how (does|will|would|can) (brandon|he)",
            r"why (does|did|is|should) (brandon|he)",
            r"what (will|would|does|did) (brandon|he) (do|say|think|believe|propose)",
            r"(brandon'?s?|his) (position|stance|view|plan|policy) (on|about|regarding)",
            r"what are (brandon'?s?|his|the) (plans?|proposals?|ideas?|solutions?) (for|on|about|regarding)",
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


# Singleton instance
prequalifier = Prequalifier()
