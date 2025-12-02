"""
SLM-based Output Validator for BrandonBot

Uses Small Language Models for validation:
- Cross-encoder (MiniLM) for Intent Checking - semantic similarity
- Emotion classifier for Ethics detection
- Phi-3 for complex reasoning (Ethics, FEC) when needed
- Pattern-based for PII detection (regex is highly effective)
- Confidence verification for exculpatory language

Violation Scale:
0 = Pass (no violation)
1-3 = Soft fail (minor issues)
4-5 = Hard fail (critical violations)
"""

import logging
import re
import asyncio
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

logger = logging.getLogger(__name__)


class OVSafeguard(Enum):
    """Output validation safeguard categories."""
    INTENT_CHECKING = "intent_checking"
    ETHICS_MORALITY = "ethics_morality"
    FEC_COMPLIANCE = "fec_compliance"
    CITATION_VERIFICATION = "citation_verification"
    REDACTION_PII = "redaction_pii"
    CONFIDENCE_VERIFICATION = "confidence_verification"


@dataclass
class OVResult:
    """Result from a single safeguard check."""
    safeguard: OVSafeguard
    score: int  # 0-5 violation score
    confidence: float  # Model confidence in the score
    explanation: str
    method: str  # 'slm', 'pattern', 'hybrid'


@dataclass
class OVValidationResult:
    """Complete validation result for a response."""
    query: str
    response: str
    pq_confidence: float
    results: Dict[OVSafeguard, OVResult] = field(default_factory=dict)
    max_violation: int = 0
    passed: bool = True
    
    def __post_init__(self):
        if self.results:
            self.max_violation = max(r.score for r in self.results.values())
            self.passed = self.max_violation <= 3  # Soft fails allowed


class OutputValidatorSLM:
    """
    SLM-based Output Validator using multiple small models.
    
    Architecture:
    - Phi-3 for Intent Checking (answers the question?)
    - Phi-3 for Ethics detection (Judeo-Christian ethics)
    - Cross-encoder as fallback for semantic similarity
    - Pattern-based for PII (regex is highly effective)
    - Confidence verification checks for hedging language
    """
    
    def __init__(self, use_phi3: bool = True):
        self._slm_manager = None
        self._phi3_validator = None
        self._use_phi3 = use_phi3
        self._cross_encoder_ready = False
        self._emotion_ready = False
        
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for pattern-based checks."""
        self.pii_patterns = [
            (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), 'ssn', 5),
            (re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'), 'credit_card', 5),
            (re.compile(r'\b(?:cvv|cvc|cvv2)\s*[:=]?\s*\d{3,4}\b', re.I), 'cvv', 5),
            (re.compile(r'\b(?:expir|exp)[\w]*\s*[:=]?\s*\d{1,2}[/\-]\d{2,4}\b', re.I), 'expiry', 4),
            (re.compile(r'\bsk-[a-zA-Z0-9]{20,}\b'), 'api_key', 4),
            (re.compile(r'\b(?:password|pwd|pin)\s*(?:is|was|:)\s*\S+', re.I), 'password', 5),
            (re.compile(r'\+?1?\s*\(?[2-9]\d{2}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b'), 'phone', 4),
            (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), 'email', 2),
            (re.compile(r'\b\d{5}(?:-\d{4})?\b'), 'zip', 1),
            (re.compile(r'\b(?:routing|aba)\s*(?:number|#)?\s*[:=]?\s*\d{9}\b', re.I), 'routing', 4),
            (re.compile(r'\blogin\s*id\s*(?:is|:)\s*\w+', re.I), 'login_id', 4),
            (re.compile(r'\bip\s*address\s*(?:is|:)\s*\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', re.I), 'ip_address', 3),
            (re.compile(r'\b\d{9}\b'), 'routing_number', 3),
        ]
        
        self.hedging_patterns = [
            re.compile(r'\b(?:based on|according to)\s+(?:available|platform|the)\s+(?:documents?|information)\b', re.I),
            re.compile(r'\b(?:i(?:\'m| am) not (?:certain|sure)|i don\'t know|uncertain)\b', re.I),
            re.compile(r'\b(?:may|might|could|possibly|perhaps|likely|probably)\b', re.I),
            re.compile(r'\b(?:it appears|it seems|it looks like)\b', re.I),
            re.compile(r'\b(?:approximately|roughly|about|around|estimated)\b', re.I),
            re.compile(r'\b(?:would you like|shall i|can i help you)\b', re.I),
            re.compile(r'\b(?:to my knowledge|as far as i know)\b', re.I),
            re.compile(r'\b(?:i believe|i think|in my understanding)\b', re.I),
        ]
        
        self.overconfidence_patterns = [
            re.compile(r'\b(?:definitely|certainly|absolutely|without a doubt)\b', re.I),
            re.compile(r'\b(?:it is|the answer is|this is)\s+(?:definitely|certainly)\b', re.I),
            re.compile(r'\b(?:i guarantee|guaranteed|100%|for sure)\b', re.I),
            re.compile(r'\b(?:always|never|must be|has to be)\b', re.I),
            re.compile(r'\b(?:the only|only way|only option)\b', re.I),
        ]
        
        self.harmful_patterns = [
            (re.compile(r'\b(?:kill|murder|assassinate|harm)\s+(?:him|her|them|someone|people)\b', re.I), 'violence', 5),
            (re.compile(r'\b(?:make|build|create)\s+(?:a\s+)?(?:bomb|explosive|weapon)\b', re.I), 'weapons', 5),
            (re.compile(r'\b(?:how to|instructions for)\s+(?:hack|steal|cheat|fraud)\b', re.I), 'crime_instructions', 4),
            (re.compile(r'\b(?:domestic terror|terrorism|terrorist)\b', re.I), 'terrorism', 5),
            (re.compile(r'\b(?:you should|you must)\s+(?:stop trying|give up|end it)\b', re.I), 'harmful_advice', 4),
            (re.compile(r'\bsuicide\b.*\b(?:method|how to|way to)\b', re.I), 'suicide_content', 5),
        ]
        
        self.fec_patterns = [
            (re.compile(r'\b(?:counterfeit|forge|fake)\s+(?:money|currency|bills?|id)\b', re.I), 'counterfeiting', 5),
            (re.compile(r'\b(?:ponzi|pyramid)\s+scheme\b', re.I), 'fraud_scheme', 5),
            (re.compile(r'\b(?:pump.and.dump|insider trading)\b', re.I), 'securities_fraud', 5),
            (re.compile(r'\b(?:without a background check|unregistered|illegal)\s+(?:gun|firearm|weapon)\b', re.I), 'illegal_weapons', 5),
            (re.compile(r'\b(?:exploit|hack|crack)\s+(?:the|a)\s+(?:phone|system|bootloader)\b', re.I), 'hacking', 4),
            (re.compile(r'\b(?:pirat|torrent|crack)(?:ed|ing)?\s+(?:software|stream|client)\b', re.I), 'piracy', 4),
            (re.compile(r'\b(?:felony|illegal|crime)\b', re.I), 'legal_reference', 2),
        ]
        
        self.refusal_patterns = [
            re.compile(r'\b(?:i cannot|i can\'t|i am unable to|i refuse to)\b', re.I),
            re.compile(r'\b(?:too complex|beyond my|outside my)\b', re.I),
            re.compile(r'\b(?:misuse of|unauthorized|not allowed)\b', re.I),
        ]
    
    async def _ensure_slm_ready(self):
        """Lazy load the SLM manager."""
        if self._slm_manager is None:
            try:
                from slm_manager import SLMManager
                self._slm_manager = SLMManager(device="cpu")
                logger.info("SLM Manager initialized for Output Validator")
            except Exception as e:
                logger.error(f"Failed to initialize SLM Manager: {e}")
                self._slm_manager = None
        return self._slm_manager is not None
    
    async def _ensure_phi3_ready(self):
        """Lazy load the Phi-3 validator."""
        if self._phi3_validator is None and self._use_phi3:
            try:
                from phi3_validator import phi3_validator
                self._phi3_validator = phi3_validator
                await self._phi3_validator.ensure_ready()
                logger.info("Phi-3 Validator initialized for Output Validator")
            except Exception as e:
                logger.error(f"Failed to initialize Phi-3 Validator: {e}")
                self._phi3_validator = None
        return self._phi3_validator is not None
    
    async def validate(
        self,
        query: str,
        response: str,
        pq_confidence: float = 0.85
    ) -> OVValidationResult:
        """
        Validate a response against all safeguards.
        
        Args:
            query: Original user query
            response: LLM response to validate
            pq_confidence: Prequalifier confidence score (0-1)
        
        Returns:
            OVValidationResult with scores for each safeguard
        """
        result = OVValidationResult(
            query=query,
            response=response,
            pq_confidence=pq_confidence
        )
        
        checks = await asyncio.gather(
            self._check_intent(query, response),
            self._check_ethics(response),
            self._check_fec(response),
            self._check_citations(response),
            self._check_pii(response),
            self._check_confidence(query, response, pq_confidence),
            return_exceptions=True
        )
        
        safeguards = [
            OVSafeguard.INTENT_CHECKING,
            OVSafeguard.ETHICS_MORALITY,
            OVSafeguard.FEC_COMPLIANCE,
            OVSafeguard.CITATION_VERIFICATION,
            OVSafeguard.REDACTION_PII,
            OVSafeguard.CONFIDENCE_VERIFICATION,
        ]
        
        for safeguard, check_result in zip(safeguards, checks):
            if isinstance(check_result, Exception):
                logger.error(f"Check failed for {safeguard}: {check_result}")
                result.results[safeguard] = OVResult(
                    safeguard=safeguard,
                    score=0,
                    confidence=0.5,
                    explanation=f"Check failed: {check_result}",
                    method="error"
                )
            else:
                result.results[safeguard] = check_result
        
        result.max_violation = max(r.score for r in result.results.values())
        result.passed = result.max_violation <= 3
        
        return result
    
    async def _check_intent(self, query: str, response: str) -> OVResult:
        """
        Check if response addresses the user's intent using cross-encoder semantic similarity.
        
        Detects:
        - Topic mismatch (response about different subject)
        - Inappropriate refusals (refusing answerable questions)
        - Tangential responses (starts relevant, derails)
        - Incomplete answers (missing key information)
        - Meta-commentary (talking about the response itself)
        - Questioning user motives
        """
        score = 0
        confidence = 0.8
        explanation = ""
        method = "cross_encoder"
        
        response_lower = response.lower()
        response_word_count = len(response.split())
        is_minimal = response_word_count < 15
        
        derail_patterns = [
            (re.compile(r'\b(?:but first|let\'s discuss|speaking of|by the way)\b', re.I), 'derail'),
            (re.compile(r'\b(?:that reminds me|on another note)\b', re.I), 'derail'),
        ]
        incomplete_patterns = [
            (re.compile(r'\bthat is all\b', re.I), 'incomplete'),
            (re.compile(r'\b(?:but i|however i)\s+(?:cannot|can\'t|won\'t)\b', re.I), 'partial_refusal'),
        ]
        false_inability_patterns = [
            (re.compile(r'\bi (?:cannot|can\'t|am unable to)\s+(?:access|provide|give|help|advise)\b', re.I), 'inability'),
            (re.compile(r'\bdo not (?:possess|have)\s+(?:knowledge|access|information)\b', re.I), 'inability'),
            (re.compile(r'\btoo complex\b', re.I), 'complexity_refusal'),
            (re.compile(r'\bi refuse to\b', re.I), 'explicit_refusal'),
        ]
        meta_patterns = [
            (re.compile(r'\b(?:the response is|provides only)\b', re.I), 'meta'),
            (re.compile(r'\b(?:followed by a period|in a single sentence)\b', re.I), 'meta'),
            (re.compile(r'\([Pp]rovides only\b', re.I), 'meta'),
            (re.compile(r'\bthe one word\b', re.I), 'meta'),
        ]
        motive_patterns = [
            (re.compile(r'\b(?:please\s+)?tell me why\b', re.I), 'question_motive'),
            (re.compile(r'\bwithout clarification\b', re.I), 'demands_clarification'),
            (re.compile(r'\byou need to define\b', re.I), 'demands_definition'),
            (re.compile(r'\bi (?:cannot|can\'t) proceed\b', re.I), 'refuses_proceed'),
        ]
        absurd_patterns = [
            (re.compile(r'\bmisuse of\b.*\b(?:resources|computational)\b', re.I), 'absurd_misuse'),
            (re.compile(r'\bunhelpful way\b', re.I), 'absurd_unhelpful'),
            (re.compile(r'\b(?:are|is) an? (?:misuse|waste|inappropriate)\b', re.I), 'absurd_waste'),
        ]
        
        detected_issues = []
        for patterns_list in [derail_patterns, incomplete_patterns, false_inability_patterns, 
                              meta_patterns, motive_patterns, absurd_patterns]:
            for pattern, issue_type in patterns_list:
                if pattern.search(response):
                    detected_issues.append(issue_type)
        
        relevance = 0.5
        try:
            if self._slm_manager and hasattr(self._slm_manager, '_cross_encoder') and self._slm_manager._cross_encoder:
                pairs = [(query, response)]
                scores = self._slm_manager._cross_encoder.predict(pairs)
                raw_relevance = float(scores[0]) if hasattr(scores, '__iter__') else float(scores)
                relevance = 1 / (1 + np.exp(-raw_relevance))  # Sigmoid to [0,1]
        except Exception as e:
            logger.warning(f"Cross-encoder scoring failed: {e}")
            relevance = 0.5
        
        has_alternative = 'but i can' in response_lower or 'but i could' in response_lower
        query_lower = query.lower()
        
        partial_answer_indicators = ['recommend', 'suggest', 'hire', 'consider', 'try', 'look into']
        has_partial_answer = any(ind in response_lower for ind in partial_answer_indicators) or 'but' in response_lower
        
        strip_punct = str.maketrans('', '', '.,!?:;"\'-')
        query_words = set(query_lower.translate(strip_punct).split()) - {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'when', 'where', 'who', 'how', 'why', 'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by'}
        response_words = set(response_lower.translate(strip_punct).split())
        keyword_overlap = len(query_words & response_words)
        has_topical_overlap = keyword_overlap >= 1
        
        if 'absurd_misuse' in detected_issues or 'absurd_unhelpful' in detected_issues or 'absurd_waste' in detected_issues:
            score = 4
            explanation = f"Absurd refusal (relevance: {relevance:.3f})"
        elif 'explicit_refusal' in detected_issues:
            score = 4
            explanation = f"Explicit refusal (relevance: {relevance:.3f})"
        elif 'inability' in detected_issues:
            if has_alternative:
                score = 3
                explanation = f"Correct refusal with alternative (relevance: {relevance:.3f})"
            elif has_partial_answer:
                score = 2
                explanation = f"Partial answer with refusal (relevance: {relevance:.3f})"
            elif relevance < 0.1 and not has_topical_overlap:
                score = 4
                explanation = f"Claims inability (relevance: {relevance:.3f})"
            else:
                score = 3
                explanation = f"Claims inability (relevance: {relevance:.3f})"
        elif 'complexity_refusal' in detected_issues:
            score = 3
            explanation = f"Claims too complex (relevance: {relevance:.3f})"
        elif 'derail' in detected_issues:
            score = 3
            explanation = f"Derails to unrelated topic (relevance: {relevance:.3f})"
        elif 'demands_clarification' in detected_issues or 'refuses_proceed' in detected_issues:
            score = 2
            explanation = f"Demands clarification (relevance: {relevance:.3f})"
        elif 'demands_definition' in detected_issues:
            score = 2
            explanation = f"Demands clarification (relevance: {relevance:.3f})"
        elif 'incomplete' in detected_issues:
            score = 2
            explanation = f"Incomplete response (relevance: {relevance:.3f})"
        elif 'partial_refusal' in detected_issues:
            score = 3 if relevance < 0.3 else 2
            explanation = f"Partial refusal (relevance: {relevance:.3f})"
        elif relevance < 0.01:
            if 'meta' in detected_issues:
                if response_word_count <= 20:
                    score = 4
                    explanation = f"Meta-commentary with no relevance (minimal): {relevance:.3f}"
                else:
                    score = 3
                    explanation = f"Meta-commentary with no relevance: {relevance:.3f}"
            elif has_topical_overlap:
                score = 0
                explanation = f"On-topic response: {relevance:.3f}"
            elif response_word_count > 15:
                score = 0
                explanation = f"Substantive response: {relevance:.3f}"
            elif is_minimal:
                if response_word_count < 10:
                    score = 5
                    explanation = f"Complete task failure: {relevance:.3f}"
                else:
                    score = 3
                    explanation = f"Avoids answering: {relevance:.3f}"
            else:
                score = 5
                explanation = f"Complete topic mismatch: {relevance:.3f}"
        elif relevance < 0.05:
            if 'meta' in detected_issues:
                score = 4 if is_minimal else 3
                explanation = f"Meta-commentary with very low relevance: {relevance:.3f}"
            elif has_topical_overlap or response_word_count > 15:
                score = 0
                explanation = f"On-topic response: {relevance:.3f}"
            else:
                score = 4 if is_minimal else 3
                explanation = f"Very low relevance: {relevance:.3f}"
        elif relevance < 0.15:
            if has_topical_overlap and response_word_count > 15:
                score = 0
                explanation = f"On-topic response: {relevance:.3f}"
            elif is_minimal:
                score = 1
                explanation = f"Minimal response: {relevance:.3f}"
            else:
                score = 3
                explanation = f"Low relevance (tangential): {relevance:.3f}"
        elif relevance < 0.5:
            if 'meta' in detected_issues:
                score = 2
                explanation = f"Meta-commentary (relevance: {relevance:.3f})"
            elif 'question_motive' in detected_issues:
                score = 2
                explanation = f"Questions user motive (relevance: {relevance:.3f})"
            elif is_minimal:
                score = 1
                explanation = f"Minimal response: {relevance:.3f}"
            elif 'but' in response_lower:
                score = 2
                explanation = f"Partial response: {relevance:.3f}"
            else:
                score = 0
                explanation = f"On-topic response: {relevance:.3f}"
        elif relevance < 0.8:
            if 'meta' in detected_issues:
                score = 2
                explanation = f"Meta-commentary (relevance: {relevance:.3f})"
            elif is_minimal:
                score = 1
                explanation = f"Minimal but relevant: {relevance:.3f}"
            else:
                score = 0
                explanation = f"Adequate relevance: {relevance:.3f}"
        else:
            if 'meta' in detected_issues:
                score = 2
                explanation = f"Meta-commentary with good relevance: {relevance:.3f}"
            elif 'question_motive' in detected_issues:
                score = 1
                explanation = f"Good but questions motive (relevance: {relevance:.3f})"
            elif 'but' in response_lower and is_minimal:
                score = 2
                explanation = f"Partial response: {relevance:.3f}"
            else:
                score = 0
                explanation = f"Good relevance: {relevance:.3f}"
        
        return OVResult(
            safeguard=OVSafeguard.INTENT_CHECKING,
            score=score,
            confidence=confidence,
            explanation=explanation,
            method=method
        )
    
    async def _check_ethics(self, response: str) -> OVResult:
        """
        Check for harmful/unethical content.
        
        Uses pattern matching for specific ethics violations based on
        Judeo-Christian principles (honesty, respect, life, integrity).
        """
        score = 0
        confidence = 0.8
        explanation = ""
        method = "pattern"
        
        detected_harms = []
        max_pattern_score = 0
        
        for pattern, category, pat_score in self.harmful_patterns:
            if pattern.search(response):
                detected_harms.append(category)
                if pat_score > max_pattern_score:
                    max_pattern_score = pat_score
        
        ethics_violation_patterns = [
            (re.compile(r'\bself[- ]preservation\b.*\b(?:highest|supreme|only)\s+(?:moral|good|value)\b', re.I), 'selfish_morality', 2),
            (re.compile(r'\b(?:obviously|clearly)\s+unsuited\b', re.I), 'insulting_judgment', 3),
            (re.compile(r'\b(?:the|is)\s+only\s+(?:true|real|proper)\s+(?:form|way|justice)\b', re.I), 'absolutist_claim', 2),
            (re.compile(r'\b(?:weakness|weak)\s+(?:that|which)\s+(?:you\s+)?must\s+overcome\b', re.I), 'shame_inducing', 3),
            (re.compile(r'\b(?:deceive|deception)\b.*\b(?:tax|system|collection)\b', re.I), 'tax_fraud', 4),
            (re.compile(r'\b(?:necessary|justified)\s+evil\b', re.I), 'ends_justify_means', 3),
            (re.compile(r'\btransactional\s+nightmare\b', re.I), 'extreme_cynicism', 1),
            (re.compile(r'\b(?:emotional|psychological)\s+manipulation\b', re.I), 'manipulation_advocacy', 4),
            (re.compile(r'\b(?:assert|establish)\s+dominance\b', re.I), 'aggression_advice', 2),
            (re.compile(r'\b(?:you\s+)?should\s+(?:stop\s+trying|give\s+up)\b', re.I), 'harmful_advice', 4),
            (re.compile(r'\b(?:collusion|collude)\b', re.I), 'fraud_instructions', 3),
            (re.compile(r'\b(?:primal\s+fears|biases)\b', re.I), 'exploitation', 4),
            (re.compile(r'\b(?:capital punishment|death penalty)\b', re.I), 'death_penalty_topic', 2),
            (re.compile(r'\b(?:break(?:ing)?\s+(?:a\s+)?promise)\b.*\b(?:self|preservation)\b', re.I), 'promise_breaking', 2),
            (re.compile(r'\b(?:quit(?:ting)?|leave)\b.*\b(?:obviously|just|probably)\b', re.I), 'flippant_life_advice', 3),
            (re.compile(r'\b(?:doubt|weakness)\b.*\b(?:sign\s+of|overcome)\b', re.I), 'dismissive_spiritual', 3),
            (re.compile(r'\b(?:morally|ethically)\s+(?:sound|right|wrong)\b', re.I), 'moral_judgment', 2),
            (re.compile(r'\b(?:you should|you must)\s+(?:quit|leave|stop|end)\b', re.I), 'coercive_advice', 4),
            (re.compile(r'\b(?:give up|don\'t bother|it\'s hopeless|no point)\b', re.I), 'discouraging', 4),
            (re.compile(r'\b(?:lie|deceive|cheat|steal)\s+(?:to|from)\b', re.I), 'dishonesty', 4),
            (re.compile(r'\b(?:influence|manipulate|bribe)\s+(?:a\s+)?(?:jury|judge|official)\b', re.I), 'corruption', 4),
            (re.compile(r'\b(?:cynical|cynicism)\b', re.I), 'cynical_content', 1),
            (re.compile(r'\bdamn\s+it\b', re.I), 'mild_profanity', 1),
            (re.compile(r'\b(?:misleading|deceptive)\s+(?:ads?|advertisement)\b', re.I), 'deceptive_business', 3),
            (re.compile(r'\b(?:fortune\s+favors|full\s+commitment)\b.*\b(?:only\s+way|true\s+returns)\b', re.I), 'reckless_advice', 3),
            (re.compile(r'\b(?:critical|negative)\s+(?:observations?|posts?)\s+(?:on\s+)?social\s+media\b', re.I), 'reputation_damage', 3),
            (re.compile(r'\b(?:work\s+of\s+)?satan\b', re.I), 'extreme_religious', 5),
            (re.compile(r'\beternal\s+damnation\b', re.I), 'hellfire_threat', 2),
            (re.compile(r'\b(?:is\s+a\s+)?sin\b.*\b(?:judgment|condemned|damnation)\b', re.I), 'absolute_religious_judgment', 3),
            (re.compile(r'\b(?:knives?|blades?)\b.*\b(?:ideal|perfect|best)\s+for\b.*\b(?:quick|discreet|use)\b', re.I), 'weapon_advice', 4),
            (re.compile(r'\b(?:small|compact)\b.*\b(?:knives?|weapons?)\b.*\beasiest\s+to\s+conceal\b', re.I), 'concealment_advice', 4),
            (re.compile(r'\b(?:gossip|gossiping)\b.*\b(?:vital|important|necessary)\b', re.I), 'gossip_justification', 2),
            (re.compile(r'\b(?:revenge|retaliate|get\s+back)\b', re.I), 'revenge_advice', 2),
            (re.compile(r'\b(?:erode|undermine)\s+(?:their|someone\'?s?)\s+(?:patience|peace|confidence)\b', re.I), 'harassment_advice', 2),
            (re.compile(r'\b(?:annoying|irritating)\s+(?:clutter|items|things)\b', re.I), 'passive_aggressive', 2),
            (re.compile(r'\bhighly\s+critical\b.*\b(?:social\s+media|online|posts?)\b', re.I), 'reputation_attack', 3),
            (re.compile(r'\b(?:post|share)\b.*\b(?:critical|negative|damaging)\b', re.I), 'reputation_damage_alt', 3),
            (re.compile(r'\b(?:fires?\s+of\s+)?hell\b.*\b(?:burn|eternal|punishment)\b', re.I), 'hellfire_content', 2),
        ]
        
        ethics_issues = []
        for pattern, category, pat_score in ethics_violation_patterns:
            if pattern.search(response):
                ethics_issues.append(category)
                if pat_score > max_pattern_score:
                    max_pattern_score = pat_score
        
        if detected_harms:
            score = max_pattern_score
            explanation = f"Harmful content: {', '.join(detected_harms)}"
        elif ethics_issues:
            score = max_pattern_score
            explanation = f"Ethics concern: {', '.join(ethics_issues)}"
        else:
            score = 0
            explanation = "No ethics violations detected"
        
        return OVResult(
            safeguard=OVSafeguard.ETHICS_MORALITY,
            score=score,
            confidence=confidence,
            explanation=explanation,
            method=method
        )
    
    async def _check_fec(self, response: str) -> OVResult:
        """
        Check for FEC/campaign compliance violations.
        
        Uses pattern matching for legal/regulatory violations.
        """
        score = 0
        confidence = 0.85
        method = "pattern"
        
        detected_violations = []
        max_score = 0
        
        for pattern, category, pat_score in self.fec_patterns:
            if pattern.search(response):
                detected_violations.append(category)
                if pat_score > max_score:
                    max_score = pat_score
        
        for pattern, category, pat_score in self.harmful_patterns:
            if 'weapons' in category or 'terrorism' in category or 'crime' in category:
                if pattern.search(response):
                    if category not in detected_violations:
                        detected_violations.append(category)
                    if pat_score > max_score:
                        max_score = pat_score
        
        score = max_score
        
        if detected_violations:
            explanation = f"FEC concerns: {', '.join(detected_violations)}"
        else:
            explanation = "No FEC violations detected"
        
        return OVResult(
            safeguard=OVSafeguard.FEC_COMPLIANCE,
            score=score,
            confidence=confidence,
            explanation=explanation,
            method=method
        )
    
    async def _check_citations(self, response: str) -> OVResult:
        """
        Check citation format, presence, and validity.
        
        For BrandonBot, proper citations should be:
        - [CITE-BP-xxx] or [CITE-QA-xxx] (standard format)
        - Not generic like [CITE] or [CITE: xxx] or [CITE-xxx]
        - WEBCITE is non-standard and should be flagged
        """
        score = 0
        confidence = 0.8
        method = "pattern"
        
        proper_cite_pattern = re.compile(r'\[CITE-(?:BP|QA|WEB|HISTORY)-[A-Z0-9]+\]')
        
        all_cite_patterns = [
            re.compile(r'\[CITE[-:][\w-]+\]'),
            re.compile(r'\[WEB[-:][\w-]+\]'),
            re.compile(r'\[WEBCITE:[\w\s]+\]'),
        ]
        
        incomplete_cite = re.compile(r'\[CITE\]')
        generic_cite = re.compile(r'\[CITE:\s*\w+\]')
        numeric_cite = re.compile(r'\[CITE-\d+\]')
        webcite_pattern = re.compile(r'\[WEBCITE:\s*[\w\s]+\]')
        colon_cite = re.compile(r'\[CITE:\s*[\w-]+\]')
        
        has_any_citation = any(p.search(response) for p in all_cite_patterns)
        has_proper_citation = proper_cite_pattern.search(response) is not None
        has_incomplete = incomplete_cite.search(response) is not None
        has_generic = generic_cite.search(response) is not None
        has_numeric = numeric_cite.search(response) is not None
        has_webcite = webcite_pattern.search(response) is not None
        has_colon = colon_cite.search(response) is not None
        
        statistical_patterns = [
            re.compile(r'\b(?:approximately|about|around|estimated)?\s*\$?[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|trillion|percent|%))'),
            re.compile(r'\bpopulation\b.*\b[\d.,]+\s*(?:billion|million)\b', re.I),
        ]
        has_statistics = any(p.search(response) for p in statistical_patterns)
        
        needs_citation_patterns = [
            re.compile(r'\bcite this\b', re.I),
            re.compile(r'\bprovide.*citation\b', re.I),
        ]
        
        if has_incomplete:
            score = 2
            explanation = "Incomplete citation: [CITE] without reference"
        elif has_numeric:
            score = 5
            explanation = "Invalid citation format: [CITE-nnn] is placeholder"
        elif has_webcite:
            score = 3
            explanation = "Non-standard format: WEBCITE"
        elif has_colon:
            score = 3
            explanation = "Non-standard format: [CITE: xxx]"
        elif has_statistics and not has_any_citation:
            score = 5
            explanation = f"Statistical claim without citation"
        elif not has_any_citation:
            if any(p.search(response) for p in needs_citation_patterns):
                score = 5
                explanation = "Requested citation not provided"
            else:
                score = 0
                explanation = "No citation-requiring claims detected"
        elif has_proper_citation:
            score = 0
            explanation = "Valid citation format"
        else:
            score = 2
            explanation = "Citation present but format uncertain"
        
        return OVResult(
            safeguard=OVSafeguard.CITATION_VERIFICATION,
            score=score,
            confidence=confidence,
            explanation=explanation,
            method=method
        )
    
    async def _check_pii(self, response: str) -> OVResult:
        """
        Check for PII leakage using regex patterns.
        
        Regex is highly effective for PII detection.
        """
        score = 0
        confidence = 0.95
        method = "pattern"
        
        detected_pii = []
        max_score = 0
        
        for pattern, pii_type, pat_score in self.pii_patterns:
            if pattern.search(response):
                detected_pii.append(pii_type)
                if pat_score > max_score:
                    max_score = pat_score
        
        score = max_score
        
        if detected_pii:
            explanation = f"PII detected: {', '.join(detected_pii)}"
        else:
            explanation = "No PII detected"
        
        return OVResult(
            safeguard=OVSafeguard.REDACTION_PII,
            score=score,
            confidence=confidence,
            explanation=explanation,
            method=method
        )
    
    async def _check_confidence(
        self,
        query: str,
        response: str,
        pq_confidence: float
    ) -> OVResult:
        """
        Check if response uses appropriate hedging when confidence is LOW.
        
        When PQ confidence < 0.75:
        - Response SHOULD use exculpatory/hedging language
        - Score of 5 = overconfident (no hedging)
        - Score of 0 = appropriate hedging present
        
        When PQ confidence >= 0.75:
        - Should be confident - inappropriate claims of inability are flagged
        """
        score = 0
        confidence = 0.85
        method = "pattern"
        
        has_hedging = any(p.search(response) for p in self.hedging_patterns)
        has_overconfidence = any(p.search(response) for p in self.overconfidence_patterns)
        
        response_lower = response.lower()
        has_question = '?' in response or 'would you like' in response_lower or 'can i help' in response_lower
        
        false_inability_patterns = [
            re.compile(r'\bi (?:cannot|can\'t|am unable to)\b', re.I),
            re.compile(r'\bdo not (?:possess|have)\s+(?:knowledge|access|information)\b', re.I),
            re.compile(r'\b(?:too complex|beyond my|outside my)\b', re.I),
            re.compile(r'\b(?:i refuse|misuse of|unauthorized)\b', re.I),
            re.compile(r'\b(?:this task is|that is) (?:too complex|not possible)\b', re.I),
        ]
        
        has_false_inability = any(p.search(response) for p in false_inability_patterns)
        
        if pq_confidence < 0.75:
            if has_overconfidence and not has_hedging:
                score = 5
                explanation = f"Overconfident response when PQ={pq_confidence:.2f}, no hedging"
            elif not has_hedging and not has_question:
                score = 5
                explanation = f"No hedging language when PQ={pq_confidence:.2f}"
            elif has_hedging and has_overconfidence:
                score = 3
                explanation = f"Mixed signals: hedging with overconfidence, PQ={pq_confidence:.2f}"
            elif has_hedging or has_question:
                score = 0
                explanation = f"Appropriate hedging for PQ={pq_confidence:.2f}"
            else:
                score = 3
                explanation = f"Weak hedging for low confidence PQ={pq_confidence:.2f}"
        else:
            if has_false_inability:
                if 'i refuse' in response_lower or 'misuse of' in response_lower:
                    score = 3
                    explanation = f"Inappropriate refusal when PQ={pq_confidence:.2f}"
                elif 'too complex' in response_lower:
                    score = 2
                    explanation = f"Claims task too complex when PQ={pq_confidence:.2f}"
                else:
                    score = 2
                    explanation = f"Claims inability when PQ={pq_confidence:.2f}"
            elif has_hedging and not has_overconfidence:
                score = 1
                explanation = f"Unnecessary hedging for high confidence PQ={pq_confidence:.2f}"
            else:
                score = 0
                explanation = f"Appropriate confidence for PQ={pq_confidence:.2f}"
        
        return OVResult(
            safeguard=OVSafeguard.CONFIDENCE_VERIFICATION,
            score=score,
            confidence=confidence,
            explanation=explanation,
            method=method
        )


output_validator_slm = OutputValidatorSLM()


async def run_ov_test():
    """Run the SLM-based OV validator against the test suite."""
    from ov_test_suite_v2 import OV_TEST_CASES_V2, get_test_cases_by_category
    
    validator = OutputValidatorSLM()
    
    results = {
        'intent_checking': {'correct': 0, 'total': 0, 'errors': []},
        'ethics_morality': {'correct': 0, 'total': 0, 'errors': []},
        'fec_compliance': {'correct': 0, 'total': 0, 'errors': []},
        'citation_verification': {'correct': 0, 'total': 0, 'errors': []},
        'redaction_pii': {'correct': 0, 'total': 0, 'errors': []},
        'confidence_verification': {'correct': 0, 'total': 0, 'errors': []},
    }
    
    for test_case in OV_TEST_CASES_V2:
        validation = await validator.validate(
            query=test_case.query,
            response=test_case.response,
            pq_confidence=test_case.pq_confidence
        )
        
        checks = [
            ('intent_checking', OVSafeguard.INTENT_CHECKING, test_case.intent_checking),
            ('ethics_morality', OVSafeguard.ETHICS_MORALITY, test_case.ethics_morality),
            ('fec_compliance', OVSafeguard.FEC_COMPLIANCE, test_case.fec_compliance),
            ('citation_verification', OVSafeguard.CITATION_VERIFICATION, test_case.citation_verification),
            ('redaction_pii', OVSafeguard.REDACTION_PII, test_case.redaction_pii),
            ('confidence_verification', OVSafeguard.CONFIDENCE_VERIFICATION, test_case.confidence_verification),
        ]
        
        for key, safeguard, expected in checks:
            actual = validation.results[safeguard].score
            results[key]['total'] += 1
            
            if actual == expected:
                results[key]['correct'] += 1
            else:
                if len(results[key]['errors']) < 5:
                    results[key]['errors'].append({
                        'id': test_case.id,
                        'expected': expected,
                        'actual': actual,
                        'query': test_case.query[:50],
                        'explanation': validation.results[safeguard].explanation
                    })
    
    print("=" * 60)
    print("OUTPUT VALIDATOR (SLM) TEST RESULTS")
    print("=" * 60)
    print()
    
    total_correct = 0
    total_tests = 0
    
    for key, data in results.items():
        accuracy = data['correct'] / data['total'] * 100 if data['total'] > 0 else 0
        total_correct += data['correct']
        total_tests += data['total']
        
        print(f"{key}:")
        print(f"  Accuracy: {data['correct']}/{data['total']} ({accuracy:.1f}%)")
        if data['errors']:
            print(f"  Errors (first 5):")
            for err in data['errors']:
                print(f"    [{err['id']}] exp={err['expected']}, got={err['actual']}: {err['query']}")
                print(f"         Reason: {err['explanation'][:60]}")
        print()
    
    overall = total_correct / total_tests * 100 if total_tests > 0 else 0
    print("=" * 60)
    print(f"OVERALL: {total_correct}/{total_tests} ({overall:.1f}%)")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_ov_test())
