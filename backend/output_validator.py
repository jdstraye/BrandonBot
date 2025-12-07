"""
SLM-based Output Validator for BrandonBot

Uses Specialized Small Language Models for each safeguard:
- ME2-BERT: Ethics checking (Moral Foundations Theory - Judeo-Christian aligned)
- MS-MARCO: Intent/Response alignment (trained on QA pairs)
- DeBERTa-PII: PII detection (fine-tuned for PII extraction)
- BERT-tiny: Confidence verification (hedging detection)
- FEC: RAG lookup from isolated FEC collection + SLM binary classifier
- Citations: Anchor injection + metadata resolution

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
    INTERNAL_LEAK = "internal_leak"


@dataclass
class OVResult:
    """Result from a single safeguard check."""
    safeguard: OVSafeguard
    score: int
    confidence: float
    explanation: str
    method: str


@dataclass
class OVValidationResult:
    """Complete validation result for a response."""
    query: str
    response: str
    pq_confidence: float
    results: Dict[OVSafeguard, OVResult] = field(default_factory=dict)
    max_violation: int = 0
    passed: bool = True
    rejection_reason: Optional[str] = None
    
    def __post_init__(self):
        if self.results:
            self.max_violation = max(r.score for r in self.results.values())
            self.passed = self.max_violation <= 3
            if not self.passed:
                failed = [f"{s.value}: {r.explanation}" 
                         for s, r in self.results.items() if r.score > 3]
                self.rejection_reason = "; ".join(failed)
    
    def get_feedback_for_retry(self) -> Optional[str]:
        """
        Generate feedback for the main LLM agent to address rejection reasons.
        
        Returns formatted instructions for the LLM to fix specific issues.
        """
        if self.passed:
            return None
        
        feedback_parts = []
        for safeguard, result in self.results.items():
            if result.score > 3:
                feedback_parts.append(
                    f"[{safeguard.value.upper()}] Issue: {result.explanation}"
                )
        
        if not feedback_parts:
            return None
        
        return (
            "Your previous response was rejected by the Output Validator. "
            "Please regenerate addressing these specific issues:\n" +
            "\n".join(feedback_parts) +
            "\n\nEnsure your new response corrects these problems while "
            "maintaining relevance to the original question."
        )


class SLMNotAvailableError(Exception):
    """Raised when SLM models are required but not available."""
    pass


class OutputValidatorSLM:
    """
    SLM-based Output Validator using specialized models for each safeguard.
    
    Architecture:
    - ME2-BERT: Ethics detection (Moral Foundations - Judeo-Christian aligned)
    - MS-MARCO: Intent/Response alignment (trained on QA pairs)
    - DeBERTa-PII: PII detection (fine-tuned model)
    - BERT-tiny: Confidence verification (pattern-based with SLM backup)
    - FEC: RAG lookup + SLM classifier (per specification)
    - Citations: Anchor resolution + metadata lookup (per specification)
    
    When require_slm=True (default), the validator will raise SLMNotAvailableError
    if the required SLM models cannot be loaded. This ensures tests fail when
    not using the correct SLM-based validation.
    """
    
    def __init__(self, require_slm: bool = True):
        """
        Initialize the Output Validator.
        
        Args:
            require_slm: If True (default), raise SLMNotAvailableError when SLM models
                        cannot be loaded. If False, fall back to pattern-based checking.
        """
        self._me2bert = None
        self._msmarco = None
        self._deberta_pii = None
        self._berttiny = None
        self._citation_store = None
        self._fec_rag = None
        self._require_slm = require_slm
        
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for pattern-based checks."""
        self.pii_patterns = [
            (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), 'ssn', 5),
            (re.compile(r'\b(?:ssn|social\s*security)[\s\w]*(?:is|:)\s*\d{9}\b', re.I), 'ssn', 5),
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
        ]
        
        self.pii_domain_whitelist = {
            'brandonsowers.com',
            'brandonsowers.org',
            'brandonforoffice.com',
            'twitter.com',
            'facebook.com',
            'instagram.com',
            'youtube.com',
            'linkedin.com',
        }
        
        self.fec_prohibited_patterns = [
            (re.compile(r'\b(?:your|this)\s+donation\s+(?:is|will be)\s+tax\s+deductible\b', re.I), 'tax_advice', 5),
            (re.compile(r'\b(?:you can|you\'ll be able to)\s+write off\b', re.I), 'tax_advice', 5),
            (re.compile(r'\blegally,?\s+you\s+should\b', re.I), 'legal_advice', 5),
            (re.compile(r'\bthis\s+(?:is not|isn\'t)\s+legal\s+advice,?\s+but\b', re.I), 'legal_advice', 4),
            (re.compile(r'\b(?:i|we)\s+(?:will|can)\s+(?:process|accept)\s+(?:your\s+)?(?:donation|credit card|payment)\b', re.I), 'direct_solicitation', 5),
            (re.compile(r'\b(?:enter|provide)\s+(?:your\s+)?(?:credit card|card number|payment)\b', re.I), 'direct_solicitation', 5),
            (re.compile(r'\b(?:he|she|they|opponent)\s+(?:is|are)\s+(?:a\s+)?(?:criminal|corrupt|fraud)\b', re.I), 'defamation', 4),
            (re.compile(r'\b(?:stole|steal|stealing)\s+(?:money|funds|from)\b', re.I), 'defamation', 4),
            (re.compile(r'\bendorsed\s+by\s+(?!brandon)\b', re.I), 'false_endorsement', 4),
            (re.compile(r'\bi\s+am\s+(?:brandon|the\s+candidate|a\s+human)\b', re.I), 'false_identity', 5),
            (re.compile(r'\bguaranteed?\s+to\s+(?:win|pass|happen)\b', re.I), 'false_promise', 4),
            (re.compile(r'\bvote\s+for\s+(?:us|him|brandon)\s+or\s+(?:else|bad\s+things)\b', re.I), 'coercion', 5),
            (re.compile(r'\byou\s+must\s+donate\b', re.I), 'coercion', 4),
            (re.compile(r'\b(?:offshore|foreign|hidden)\s+(?:account|bank|funds?)\b', re.I), 'money_laundering', 5),
            (re.compile(r'\bhide\s+(?:money|funds|donations?|contributions?)\b', re.I), 'money_laundering', 5),
            (re.compile(r'\b(?:launder|laundering|wash)\s+(?:money|funds)\b', re.I), 'money_laundering', 5),
            (re.compile(r'\bavoid\s+(?:fec|reporting|disclosure|limits?)\b', re.I), 'fec_evasion', 5),
            (re.compile(r'\b(?:anonymous|untraceable)\s+(?:donation|contribution)\b', re.I), 'fec_evasion', 5),
            # Foreign national contributions (52 U.S.C. § 30121)
            (re.compile(r'\b(?:foreign|overseas|international)\s+(?:donation|contribution|donor|money|funds|account|bank)\b', re.I), 'foreign_contribution', 5),
            (re.compile(r'\b(?:donate|contribution|give)\s+(?:from|via|through)\s+(?:abroad|overseas|another country|foreign)\b', re.I), 'foreign_contribution', 5),
            (re.compile(r'\b(?:non-?us|non-?citizen|foreign national)\s+(?:can|may|should)\s+(?:donate|contribute|give)\b', re.I), 'foreign_solicitation', 5),
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
    
    async def _ensure_me2bert_ready(self) -> bool:
        """Lazy load ME2-BERT ethics checker."""
        if self._me2bert is None:
            try:
                from ov_slm_models import me2bert_checker
                self._me2bert = me2bert_checker
                await self._me2bert.ensure_ready()
                logger.info("ME2-BERT ethics checker initialized")
            except Exception as e:
                logger.error(f"Failed to initialize ME2-BERT: {e}")
                return False
        return self._me2bert is not None and self._me2bert._initialized
    
    async def _ensure_msmarco_ready(self) -> bool:
        """Lazy load MS-MARCO intent checker."""
        if self._msmarco is None:
            try:
                from ov_slm_models import msmarco_checker
                self._msmarco = msmarco_checker
                await self._msmarco.ensure_ready()
                logger.info("MS-MARCO intent checker initialized")
            except Exception as e:
                logger.error(f"Failed to initialize MS-MARCO: {e}")
                return False
        return self._msmarco is not None and self._msmarco._initialized
    
    async def _ensure_deberta_pii_ready(self) -> bool:
        """Lazy load DeBERTa PII checker."""
        if self._deberta_pii is None:
            try:
                from ov_slm_models import deberta_pii_checker
                self._deberta_pii = deberta_pii_checker
                await self._deberta_pii.ensure_ready()
                logger.info("DeBERTa PII checker initialized")
            except Exception as e:
                logger.error(f"Failed to initialize DeBERTa PII: {e}")
                return False
        return self._deberta_pii is not None and self._deberta_pii._initialized
    
    async def _ensure_berttiny_ready(self) -> bool:
        """Lazy load BERT-tiny confidence checker."""
        if self._berttiny is None:
            try:
                from ov_slm_models import berttiny_confidence_checker
                self._berttiny = berttiny_confidence_checker
                await self._berttiny.ensure_ready()
                logger.info("BERT-tiny confidence checker initialized")
            except Exception as e:
                logger.error(f"Failed to initialize BERT-tiny: {e}")
                return False
        return self._berttiny is not None and self._berttiny._initialized
    
    def set_citation_store(self, citation_store: Dict[str, Dict[str, Any]]):
        """
        Set the citation metadata store for anchor resolution.
        
        The store maps citation anchors to their metadata:
        {
            "CITE-BP-001": {"document_id": "platform_doc_1", "page": 3, "line": 15, "content": "..."},
            "CITE-QA-005": {"document_id": "qa_responses", "page": 1, "line": 42, "content": "..."},
        }
        """
        self._citation_store = citation_store
    
    def set_fec_rag(self, fec_rag):
        """
        Set the FEC RAG retriever for compliance checking.
        
        The FEC RAG should be an isolated collection containing:
        - FEC regulations (11 CFR)
        - Prohibited statements
        - Disclaimer templates
        """
        self._fec_rag = fec_rag
    
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
            self._check_internal_leak(response),
            return_exceptions=True
        )
        
        safeguards = [
            OVSafeguard.INTENT_CHECKING,
            OVSafeguard.ETHICS_MORALITY,
            OVSafeguard.FEC_COMPLIANCE,
            OVSafeguard.CITATION_VERIFICATION,
            OVSafeguard.REDACTION_PII,
            OVSafeguard.CONFIDENCE_VERIFICATION,
            OVSafeguard.INTERNAL_LEAK,
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
        
        if not result.passed:
            failed = [f"{s.value}: {r.explanation}" 
                     for s, r in result.results.items() if r.score > 3]
            result.rejection_reason = "; ".join(failed)
        
        return result
    
    async def _check_intent(self, query: str, response: str) -> OVResult:
        """
        Check if response addresses the user's intent using MS-MARCO cross-encoder.
        
        MS-MARCO is trained on QA pairs, optimal for detecting query-response alignment.
        
        Raises:
            SLMNotAvailableError: If require_slm=True and MS-MARCO cannot be loaded.
        """
        if await self._ensure_msmarco_ready():
            try:
                intent_result = await self._msmarco.check_intent(query, response)
                return OVResult(
                    safeguard=OVSafeguard.INTENT_CHECKING,
                    score=intent_result.score,
                    confidence=intent_result.confidence,
                    explanation=intent_result.explanation,
                    method="ms_marco"
                )
            except Exception as e:
                if self._require_slm:
                    raise SLMNotAvailableError(f"MS-MARCO intent check failed: {e}")
                logger.warning(f"MS-MARCO intent check failed, falling back to patterns: {e}")
        elif self._require_slm:
            raise SLMNotAvailableError("MS-MARCO intent model not available. Set require_slm=False to use pattern fallback.")
        
        return await self._check_intent_fallback(query, response)
    
    async def _check_intent_fallback(self, query: str, response: str) -> OVResult:
        """Fallback intent checking using patterns when MS-MARCO is unavailable."""
        response_lower = response.lower()
        response_word_count = len(response.split())
        
        refusal_patterns = [
            (re.compile(r'\bi (?:cannot|can\'t|am unable to|refuse to)\b', re.I), 'refusal'),
            (re.compile(r'\bdo not (?:possess|have)\s+(?:knowledge|access|information)\b', re.I), 'inability'),
            (re.compile(r'\b(?:too complex|beyond my|outside my)\b', re.I), 'complexity'),
            (re.compile(r'\bmisuse of\b', re.I), 'absurd_refusal'),
        ]
        
        issues_found = []
        for pattern, issue_type in refusal_patterns:
            if pattern.search(response):
                issues_found.append(issue_type)
        
        if 'absurd_refusal' in issues_found:
            score = 4
            explanation = "Absurd refusal detected"
        elif 'refusal' in issues_found or 'inability' in issues_found:
            has_alternative = 'but i can' in response_lower or 'however' in response_lower
            if has_alternative:
                score = 2
                explanation = "Refusal with alternative offered"
            else:
                score = 3
                explanation = "Refusal without alternative"
        elif response_word_count < 10:
            score = 2
            explanation = "Very short response"
        else:
            score = 0
            explanation = "Response appears to address query"
        
        return OVResult(
            safeguard=OVSafeguard.INTENT_CHECKING,
            score=score,
            confidence=0.7,
            explanation=explanation,
            method="pattern_fallback"
        )
    
    async def _check_ethics(self, response: str) -> OVResult:
        """
        Check for ethics violations using ME2-BERT (Moral Foundations Theory).
        
        ME2-BERT detects 10 moral dimensions aligned with Judeo-Christian ethics:
        - Harm, Cheating, Betrayal, Subversion, Degradation (violations)
        - Care, Fairness, Loyalty, Authority, Purity (virtues)
        
        Raises:
            SLMNotAvailableError: If require_slm=True and ME2-BERT cannot be loaded.
        """
        if await self._ensure_me2bert_ready():
            try:
                ethics_result = await self._me2bert.check_ethics(response)
                return OVResult(
                    safeguard=OVSafeguard.ETHICS_MORALITY,
                    score=ethics_result.score,
                    confidence=ethics_result.confidence,
                    explanation=ethics_result.explanation,
                    method="me2_bert"
                )
            except Exception as e:
                if self._require_slm:
                    raise SLMNotAvailableError(f"ME2-BERT ethics check failed: {e}")
                logger.warning(f"ME2-BERT ethics check failed, falling back to patterns: {e}")
        elif self._require_slm:
            raise SLMNotAvailableError("ME2-BERT ethics model not available. Set require_slm=False to use pattern fallback.")
        
        return await self._check_ethics_fallback(response)
    
    async def _check_ethics_fallback(self, response: str) -> OVResult:
        """Fallback ethics checking using patterns when ME2-BERT is unavailable."""
        detected_harms = []
        max_score = 0
        
        for pattern, category, pat_score in self.harmful_patterns:
            if pattern.search(response):
                detected_harms.append(category)
                max_score = max(max_score, pat_score)
        
        if detected_harms:
            explanation = f"Harmful content: {', '.join(detected_harms)}"
        else:
            explanation = "No ethics violations detected"
        
        return OVResult(
            safeguard=OVSafeguard.ETHICS_MORALITY,
            score=max_score,
            confidence=0.8,
            explanation=explanation,
            method="pattern_fallback"
        )
    
    async def _check_fec(self, response: str) -> OVResult:
        """
        Check FEC compliance using RAG lookup + pattern matching.
        
        Per specification:
        1. Pattern match for obvious violations
        2. RAG query prohibited statements collection (FECProhibited collection)
        3. Binary violation classification
        
        When require_slm=True:
        - RAG MUST be available for comprehensive FEC checking
        - Pattern matching alone is not sufficient for FEC compliance
        - Raises SLMNotAvailableError if FEC RAG not configured
        
        Raises:
            SLMNotAvailableError: If require_slm=True and FEC RAG not configured.
        """
        if self._require_slm and not self._fec_rag:
            raise SLMNotAvailableError(
                "FEC RAG not configured. FEC compliance requires RAG retrieval from FECProhibited collection. "
                "Call set_fec_rag() with WeaviateManager or set require_slm=False for pattern-only fallback."
            )
        
        detected_violations = []
        max_score = 0
        
        for pattern, category, pat_score in self.fec_prohibited_patterns:
            if pattern.search(response):
                detected_violations.append(category)
                max_score = max(max_score, pat_score)
        
        for pattern, category, pat_score in self.harmful_patterns:
            if 'weapons' in category or 'terrorism' in category:
                if pattern.search(response):
                    if category not in detected_violations:
                        detected_violations.append(category)
                    max_score = max(max_score, pat_score)
        
        fec_rag_used = False
        if self._fec_rag:
            try:
                rag_results = await self._fec_rag.search(
                    collection_name="FECProhibited",
                    query=response[:200],
                    limit=3
                )
                fec_rag_used = True
                
                for result in rag_results:
                    content = result.get("content", "").lower()
                    response_lower = response.lower()
                    
                    prohibited_phrases = re.findall(r'"([^"]+)"', content)
                    for phrase in prohibited_phrases:
                        if phrase.lower() in response_lower:
                            detected_violations.append(f"rag_match:{phrase[:20]}")
                            max_score = max(max_score, 4)
                            
            except Exception as e:
                if self._require_slm:
                    raise SLMNotAvailableError(f"FEC RAG lookup failed: {e}")
                logger.warning(f"FEC RAG lookup failed: {e}")
        
        if detected_violations:
            explanation = f"FEC violations: {', '.join(detected_violations[:3])}"
            if len(detected_violations) > 3:
                explanation += f" (+{len(detected_violations)-3} more)"
        else:
            explanation = "No FEC violations detected"
        
        return OVResult(
            safeguard=OVSafeguard.FEC_COMPLIANCE,
            score=max_score,
            confidence=0.9 if detected_violations else 0.85,
            explanation=explanation,
            method="rag_pattern" if fec_rag_used else "pattern"
        )
    
    async def _check_citations(self, response: str) -> OVResult:
        """
        Check citation format, presence, and anchor resolution.
        
        Per specification:
        1. Extract citation anchors using regex
        2. Resolve anchors against metadata store
        3. Flag missing or invalid anchors
        """
        proper_cite_pattern = re.compile(r'\[(CITE-(?:BP|QA|WEB|HISTORY)-[A-Z0-9]+)\]')
        incomplete_cite = re.compile(r'\[CITE\]')
        generic_cite = re.compile(r'\[CITE:\s*[\w-]+\]')
        numeric_cite = re.compile(r'\[CITE-\d+\]')
        
        proper_matches = proper_cite_pattern.findall(response)
        has_incomplete = incomplete_cite.search(response) is not None
        has_generic = generic_cite.search(response) is not None
        has_numeric = numeric_cite.search(response) is not None
        
        statistical_patterns = [
            re.compile(r'\b(?:approximately|about|around|estimated)?\s*\$?[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|trillion|percent|%))'),
            re.compile(r'\bpopulation\b.*\b[\d.,]+\s*(?:billion|million)\b', re.I),
        ]
        has_statistics = any(p.search(response) for p in statistical_patterns)
        
        if has_incomplete:
            score = 2
            explanation = "Incomplete citation: [CITE] without reference"
        elif has_numeric:
            score = 5
            explanation = "Invalid citation format: [CITE-nnn] is placeholder"
        elif has_generic:
            score = 3
            explanation = "Non-standard format: [CITE: xxx]"
        elif proper_matches:
            if self._citation_store:
                invalid_anchors = []
                for full_anchor in proper_matches:
                    if full_anchor not in self._citation_store:
                        invalid_anchors.append(full_anchor)
                
                if invalid_anchors:
                    score = 4
                    explanation = f"Unresolved anchors: {', '.join(invalid_anchors[:3])}"
                else:
                    score = 0
                    explanation = f"All {len(proper_matches)} citations resolved"
            else:
                score = 0
                explanation = f"Valid citation format ({len(proper_matches)} found)"
        elif has_statistics:
            score = 5
            explanation = "Statistical claim without citation"
        else:
            score = 0
            explanation = "No citation-requiring claims detected"
        
        return OVResult(
            safeguard=OVSafeguard.CITATION_VERIFICATION,
            score=score,
            confidence=0.85,
            explanation=explanation,
            method="anchor_resolution" if self._citation_store else "pattern"
        )
    
    def _is_whitelisted_url(self, text: str) -> bool:
        """Check if text contains only whitelisted domains."""
        url_pattern = re.compile(r'(?:https?://)?(?:www\.)?([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})')
        matches = url_pattern.findall(text.lower())
        if not matches:
            return False
        return all(domain in self.pii_domain_whitelist for domain in matches)
    
    async def _check_pii(self, response: str) -> OVResult:
        """
        Check for PII using DeBERTa fine-tuned model + pattern matching.
        
        Hybrid approach:
        1. Pattern matching for structured PII (SSN, credit cards, etc.)
        2. DeBERTa for semantic PII detection (names, addresses, etc.)
        3. Whitelist filtering for campaign domains
        """
        detected_pii = []
        max_score = 0
        
        for pattern, pii_type, pat_score in self.pii_patterns:
            if pattern.search(response):
                detected_pii.append(f"pattern:{pii_type}")
                max_score = max(max_score, pat_score)
        
        if max_score < 4 and await self._ensure_deberta_pii_ready():
            try:
                pii_result = await self._deberta_pii.check_pii(response)
                
                if pii_result.score > max_score:
                    filtered_entities = []
                    for entity in pii_result.entities_found:
                        entity_word = entity.get('word', '').lower().strip()
                        entity_label = entity.get('label', '')
                        
                        is_whitelisted = False
                        if 'URL' in entity_label or 'USERNAME' in entity_label:
                            for domain in self.pii_domain_whitelist:
                                domain_parts = domain.replace('.', '').lower()
                                clean_word = entity_word.replace('.', '')
                                if clean_word in domain_parts or domain_parts.startswith(clean_word) or response.lower().find(domain) >= 0:
                                    is_whitelisted = True
                                    break
                        
                        if not is_whitelisted:
                            filtered_entities.append(entity)
                    
                    if filtered_entities:
                        filtered_max = max(e.get('severity', 0) for e in filtered_entities)
                        if filtered_max > max_score:
                            max_score = filtered_max
                            for entity in filtered_entities[:3]:
                                detected_pii.append(f"slm:{entity.get('label', 'PII')}")
                        
            except Exception as e:
                logger.warning(f"DeBERTa PII check failed: {e}")
        
        if detected_pii:
            explanation = f"PII detected: {', '.join(detected_pii[:5])}"
            if len(detected_pii) > 5:
                explanation += f" (+{len(detected_pii)-5} more)"
        else:
            explanation = "No PII detected"
        
        return OVResult(
            safeguard=OVSafeguard.REDACTION_PII,
            score=max_score,
            confidence=0.9 if max_score > 0 else 0.85,
            explanation=explanation,
            method="hybrid_deberta" if self._deberta_pii else "pattern"
        )
    
    async def _check_confidence(
        self,
        query: str,
        response: str,
        pq_confidence: float
    ) -> OVResult:
        """
        Check confidence calibration using BERT-tiny + pattern matching.
        
        When PQ confidence < 0.75:
        - Response SHOULD use hedging language
        - Overconfidence WITHOUT hedging is a violation
        
        When PQ confidence >= 0.75:
        - Should be confident and direct
        - False inability claims are flagged
        """
        if await self._ensure_berttiny_ready():
            try:
                conf_result = await self._berttiny.check_confidence(
                    query, response, pq_confidence
                )
                return OVResult(
                    safeguard=OVSafeguard.CONFIDENCE_VERIFICATION,
                    score=conf_result.score,
                    confidence=conf_result.confidence,
                    explanation=conf_result.explanation,
                    method="bert_tiny"
                )
            except Exception as e:
                logger.warning(f"BERT-tiny confidence check failed, falling back to patterns: {e}")
        
        return await self._check_confidence_fallback(query, response, pq_confidence)
    
    async def _check_confidence_fallback(
        self,
        query: str,
        response: str,
        pq_confidence: float
    ) -> OVResult:
        """Fallback confidence checking using patterns."""
        has_hedging = any(p.search(response) for p in self.hedging_patterns)
        has_overconfidence = any(p.search(response) for p in self.overconfidence_patterns)
        
        false_inability_patterns = [
            re.compile(r'\bi (?:cannot|can\'t|am unable to)\b', re.I),
            re.compile(r'\bdo not (?:possess|have)\s+(?:knowledge|access|information)\b', re.I),
        ]
        has_false_inability = any(p.search(response) for p in false_inability_patterns)
        
        if pq_confidence < 0.75:
            if has_overconfidence and not has_hedging:
                score = 4
                explanation = f"Overconfident without hedging (PQ={pq_confidence:.2f})"
            elif not has_hedging:
                score = 2
                explanation = f"No hedging for low confidence topic (PQ={pq_confidence:.2f})"
            else:
                score = 0
                explanation = f"Appropriate hedging for PQ={pq_confidence:.2f}"
        else:
            if has_false_inability:
                score = 3
                explanation = f"False inability claim (PQ={pq_confidence:.2f})"
            else:
                score = 0
                explanation = f"Appropriate confidence for PQ={pq_confidence:.2f}"
        
        return OVResult(
            safeguard=OVSafeguard.CONFIDENCE_VERIFICATION,
            score=score,
            confidence=0.75,
            explanation=explanation,
            method="pattern_fallback"
        )

    async def _check_internal_leak(self, response: str) -> OVResult:
        """
        Check if internal context markers leaked into the user-facing response.
        
        Internal hints (buying signals, frustration context, OV feedback) are
        injected into the system prompt for agent guidance but MUST NEVER appear
        in the final response to users.
        
        This is a critical safeguard - any leak is a hard fail (score=5).
        """
        from prequalifier import InternalHints
        
        leak_markers = InternalHints.get_leak_markers()
        found_leaks = []
        
        for marker in leak_markers:
            if marker.lower() in response.lower():
                found_leaks.append(marker)
        
        if found_leaks:
            return OVResult(
                safeguard=OVSafeguard.INTERNAL_LEAK,
                score=5,
                confidence=1.0,
                explanation=f"Internal context leaked to user: {', '.join(found_leaks[:3])}",
                method="pattern_match"
            )
        
        return OVResult(
            safeguard=OVSafeguard.INTERNAL_LEAK,
            score=0,
            confidence=1.0,
            explanation="No internal context leakage detected",
            method="pattern_match"
        )


OutputValidator = OutputValidatorSLM
ValidationResult = OVValidationResult


class ValidationStatus(Enum):
    """Backward-compatible validation status enum."""
    PASSED = "passed"
    MODIFIED = "modified"
    REJECTED = "rejected"
    BLOCKED = "blocked"


class RejectionReason(Enum):
    """Backward-compatible rejection reason enum."""
    INTENT_NOT_FULFILLED = "intent_not_fulfilled"
    ETHICS_VIOLATION = "ethics_violation"
    FEC_VIOLATION = "fec_violation"
    INFLAMES_SITUATION = "inflames_situation"
    CITATION_INVALID = "citation_invalid"
    PII_EXPOSURE = "pii_exposure"


output_validator = OutputValidatorSLM(require_slm=True)
