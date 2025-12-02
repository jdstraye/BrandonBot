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
    - Cross-encoder for Intent Checking (fast semantic similarity)
    - Emotion classifier for Ethics detection
    - Pattern-based for PII (regex is highly effective)
    - Confidence verification checks for hedging language
    """
    
    def __init__(self):
        self._slm_manager = None
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
        Check if response addresses the user's intent using cross-encoder.
        
        Uses semantic similarity to detect:
        - Topic mismatch
        - Inappropriate refusals
        - Tangential responses
        """
        await self._ensure_slm_ready()
        
        score = 0
        confidence = 0.8
        explanation = ""
        method = "slm"
        
        try:
            if self._slm_manager:
                slm_result = await self._slm_manager.check_intent_fulfillment(query, response)
                
                raw_score = float(slm_result.raw_output) if slm_result.raw_output else 0.5
                
                has_refusal = any(p.search(response) for p in self.refusal_patterns)
                
                if raw_score < 0.2:
                    score = 5  # Complete mismatch
                    explanation = f"Low relevance: {raw_score:.3f}"
                elif raw_score < 0.3:
                    score = 4  # Significant mismatch
                    explanation = f"Poor relevance: {raw_score:.3f}"
                elif raw_score < 0.4:
                    score = 3  # Tangential
                    explanation = f"Tangential response: {raw_score:.3f}"
                elif raw_score < 0.5:
                    score = 2  # Partial
                    explanation = f"Partial match: {raw_score:.3f}"
                elif raw_score < 0.6:
                    score = 1  # Minor issue
                    explanation = f"Minor relevance gap: {raw_score:.3f}"
                else:
                    score = 0  # Good
                    explanation = f"Good relevance: {raw_score:.3f}"
                
                if has_refusal and score < 2:
                    score = max(score, 2)
                    explanation += " (contains refusal)"
                
                confidence = slm_result.confidence
            else:
                method = "pattern"
                has_refusal = any(p.search(response) for p in self.refusal_patterns)
                if has_refusal:
                    score = 3
                    explanation = "Contains refusal patterns"
                else:
                    score = 0
                    explanation = "No SLM available, pattern check passed"
                    
        except Exception as e:
            logger.warning(f"Intent check failed: {e}")
            score = 0
            explanation = f"Check failed: {e}"
            method = "error"
        
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
        
        Uses emotion classifier for sentiment and patterns for specific harms.
        """
        score = 0
        confidence = 0.8
        explanation = ""
        method = "hybrid"
        
        detected_harms = []
        max_pattern_score = 0
        
        for pattern, category, pat_score in self.harmful_patterns:
            if pattern.search(response):
                detected_harms.append(category)
                if pat_score > max_pattern_score:
                    max_pattern_score = pat_score
        
        await self._ensure_slm_ready()
        emotion_score = 0
        
        try:
            if self._slm_manager:
                slm_result = await self._slm_manager.check_ethics(response)
                if slm_result.decision == "FAIL":
                    emotion_score = 3
        except Exception as e:
            logger.warning(f"Ethics SLM check failed: {e}")
        
        score = max(max_pattern_score, emotion_score)
        
        if detected_harms:
            explanation = f"Harmful content detected: {', '.join(detected_harms)}"
        elif score > 0:
            explanation = "Ethics concern from sentiment analysis"
        else:
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
        
        For BrandonBot, we focus on whether claims should be cited.
        """
        score = 0
        confidence = 0.8
        method = "pattern"
        
        citation_patterns = [
            re.compile(r'\[CITE[-:][\w-]+\]'),
            re.compile(r'\[WEB[-:][\w-]+\]'),
            re.compile(r'\[WEBCITE:[\w\s]+\]'),
        ]
        
        has_citations = any(p.search(response) for p in citation_patterns)
        
        statistical_claims = re.findall(
            r'\b(?:approximately|about|around|estimated)?\s*\$?[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|trillion|percent|%))?\b',
            response,
            re.I
        )
        
        if statistical_claims and not has_citations:
            score = 5
            explanation = f"Statistical claims without citations: {len(statistical_claims)} found"
        elif has_citations:
            invalid_cites = re.findall(r'\[CITE\]', response)
            if invalid_cites:
                score = 2
                explanation = "Some citations incomplete"
            else:
                score = 0
                explanation = "Citations present and formatted"
        else:
            score = 0
            explanation = "No citation-requiring claims detected"
        
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
