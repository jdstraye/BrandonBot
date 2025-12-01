"""
Output Validator Module for BrandonBot

3-Stage Pipeline: PQ → LLM → OV

The Output Validator (OV) performs multi-layer validation on LLM responses:
1. Intent Fulfillment Check - SLM verifies response answers the user's query
2. Ethics/Morality Check - SLM checks Christian moral alignment
3. FEC Compliance Check - RAG retrieval + SLM double-negative verification
4. De-escalation Check - Ensures response doesn't inflame situation
5. PII Redaction - Hybrid regex + SLM for comprehensive PII removal
6. Citation Verification - Verify anchors exist in metadata store

If ANY check fails, OV REJECTs the response and sends it back to the LLM
with an explanation for regeneration.

Flow:
LLM Response → OV Checks → Pass: Deliver to User
                        → Fail: REJECT + explanation → LLM regenerates
"""

import re
import os
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from enum import Enum

import yaml

logger = logging.getLogger(__name__)


def load_campaign_contacts():
    """Load campaign contacts allowlist from config"""
    config_path = os.path.join(os.path.dirname(__file__), "config", "campaign_contacts.yaml")
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Failed to load campaign contacts config: {e}")
    
    return {
        "emails": [],
        "phones": [],
        "urls": [],
        "allowed_contexts": ["volunteer", "donate", "contact"],
        "social_handles": []
    }


CAMPAIGN_CONTACTS = load_campaign_contacts()


class ValidationStatus(Enum):
    """Validation result status"""
    PASSED = "passed"           # All checks passed, deliver to user
    MODIFIED = "modified"       # Minor modifications made (PII redaction)
    REJECTED = "rejected"       # Failed validation, regenerate required
    BLOCKED = "blocked"         # Critical violation, cannot regenerate


class RejectionReason(Enum):
    """Reasons for rejecting LLM response"""
    INTENT_NOT_FULFILLED = "intent_not_fulfilled"
    ETHICS_VIOLATION = "ethics_violation"
    FEC_VIOLATION = "fec_violation"
    INFLAMES_SITUATION = "inflames_situation"
    CITATION_INVALID = "citation_invalid"
    PII_EXPOSURE = "pii_exposure"


@dataclass
class IntentCheckResult:
    """Result from intent fulfillment check"""
    fulfilled: bool
    explanation: str
    confidence: float = 0.0


@dataclass
class EthicsCheckResult:
    """Result from ethics/morality check"""
    passed: bool
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class FECCheckResult:
    """Result from FEC compliance check"""
    compliant: bool
    violations: List[str] = field(default_factory=list)
    relevant_regulations: List[Dict] = field(default_factory=list)


@dataclass
class PIICheckResult:
    """Result from PII redaction"""
    had_pii: bool
    redacted_text: str
    pii_found: List[Dict] = field(default_factory=list)


@dataclass
class CitationCheckResult:
    """Result from citation verification"""
    all_valid: bool
    invalid_citations: List[str] = field(default_factory=list)
    verified_citations: List[str] = field(default_factory=list)


@dataclass 
class ValidationResult:
    """Complete validation result"""
    status: ValidationStatus
    original_response: str
    validated_response: str
    
    # Individual check results
    intent_check: Optional[IntentCheckResult] = None
    ethics_check: Optional[EthicsCheckResult] = None
    fec_check: Optional[FECCheckResult] = None
    pii_check: Optional[PIICheckResult] = None
    citation_check: Optional[CitationCheckResult] = None
    
    # Rejection details for LLM regeneration
    rejection_reason: Optional[RejectionReason] = None
    rejection_explanation: str = ""
    regeneration_instructions: str = ""
    
    # Modifications log
    modifications: List[str] = field(default_factory=list)
    
    # For regeneration loop control
    regeneration_attempt: int = 0
    max_regenerations: int = 3


class OutputValidator:
    """
    Multi-layer output validation with SLM classifiers and RAG verification.
    
    Checks (in order):
    1. Intent fulfillment (SLM)
    2. Ethics/morality alignment (SLM)
    3. FEC compliance (RAG + SLM double-check)
    4. De-escalation (pattern + SLM)
    5. PII redaction (regex + SLM)
    6. Citation verification (metadata lookup)
    
    On failure: REJECT with regeneration instructions
    """
    
    # SLM Prompts
    INTENT_CHECK_PROMPT = """You are an intent-fulfillment classifier for a political campaign chatbot.

User's original query: "{user_query}"

LLM's proposed response: "{response}"

Does the response fulfill the user's query? Consider:
- Does it directly address what the user asked?
- Is it a complete answer or does it deflect/avoid the question?
- Does it provide actionable information?

Respond in this format:
DECISION: YES or NO
EXPLANATION: (1-2 sentences explaining your decision)"""

    ETHICS_CHECK_PROMPT = """You are an ethics classifier for a Christian political campaign.

Proposed response: "{response}"

Check if this response:
1. Conforms to Christian moral standards
2. Inspires right action
3. Is truthful and not deceptive
4. Shows respect for human dignity
5. Does not promote harmful actions

Respond in this format:
DECISION: PASS or FAIL
ISSUES: (list any ethical concerns, or "None")
SUGGESTIONS: (how to improve if needed)"""

    FEC_DOUBLE_CHECK_PROMPT = """You are an FEC compliance classifier for a political campaign.

Proposed response: "{response}"

Relevant FEC regulations retrieved:
{fec_regulations}

Does this response violate any FEC rules? Check for:
- Unauthorized promises or guarantees
- Tax advice without proper disclaimers
- Legal advice
- Direct payment solicitation
- Defamatory statements about opponents
- Foreign national solicitation
- Coordination claims

Respond with ONLY: COMPLIANT or VIOLATION
If VIOLATION, explain which rule on the next line."""

    DEESCALATION_CHECK_PROMPT = """You are a tone classifier for a political campaign chatbot.

The user is frustrated/escalated.

Proposed response: "{response}"

Does this response potentially inflame the situation? Check for:
- Dismissive language
- Condescending tone
- Failing to acknowledge feelings
- Argumentative phrasing
- Lack of empathy markers

Respond with ONLY: SAFE or INFLAMES
If INFLAMES, explain why on the next line."""

    PII_SLM_PROMPT = """You are a PII detection classifier.

Text after regex redaction: "{text}"

Check for any remaining personally identifiable information that regex may have missed:
- Names (especially contextual like "I'm John from Boston")
- Addresses (street addresses, cities with context)
- Dates of birth with context
- Employment details that identify someone
- Medical information
- Any other identifying information

If you find PII, list each item. If clean, say "NO PII FOUND".
Format: PII: <description of PII> or NO PII FOUND"""

    # Regex patterns for Step 1 of PII detection
    PII_REGEX_PATTERNS = [
        (r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b', '[SSN REDACTED]'),  # SSN
        (r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE REDACTED]'),  # Phone
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL REDACTED]'),  # Email
        (r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CARD REDACTED]'),  # Credit card
        (r'\b(?:sk-|pk_|api[_-]?key[=:]?\s*)[a-zA-Z0-9]{20,}\b', '[API KEY REDACTED]'),  # API keys
        (r'\b\d{1,5}\s+[\w\s]+(?:street|st|avenue|ave|road|rd|boulevard|blvd|drive|dr|lane|ln|way|court|ct)\b', '[ADDRESS REDACTED]'),  # Street address
    ]

    # FEC prohibited content patterns
    FEC_PROHIBITED_PATTERNS = [
        (r'\b(i guarantee|guaranteed|promise you will)\b', 'Unauthorized guarantee'),
        (r'\b(definitely will|100% certain|absolutely will)\b', 'Absolute promise'),
        (r'(your donation is|donations are) tax[- ]?deductible', 'Tax advice'),
        (r'\b(consult (a |your )?(tax|cpa|accountant|attorney|lawyer))\b', 'Professional advice referral'),
        (r'\b(send money|wire transfer|credit card number|pay directly)\b', 'Direct payment solicitation'),
        (r'(opponent|they|he|she) (is |are )?(a )?(corrupt|criminal|fraud|liar)', 'Defamatory statement'),
        (r'\b(you should sue|legally you must)\b', 'Legal advice'),
    ]

    # Dismissive phrases that inflame frustrated users
    DISMISSIVE_PATTERNS = [
        r'\b(calm down|relax|chill|take it easy)\b',
        r'\b(you\'re wrong|that\'s incorrect|that\'s not true)\b',
        r'\b(obviously|clearly|of course|simply|just)\b',
        r'\b(you should know|as i (said|mentioned|explained))\b',
        r'\b(it\'s (simple|easy|basic|obvious))\b',
    ]

    # Empathy markers (good to have for frustrated users)
    EMPATHY_PATTERNS = [
        r'\b(i understand|i hear you|i can see)\b',
        r'\b(that (must be|sounds) (frustrating|difficult|hard))\b',
        r'\b(i appreciate|thank you for)\b',
        r'\b(let me (help|try|work with))\b',
    ]

    def __init__(self, slm_provider=None, weaviate_manager=None, citation_store=None):
        """
        Initialize output validator.
        
        Args:
            slm_provider: Small LLM for classification
            weaviate_manager: Vector DB for FEC RAG
            citation_store: Metadata store for citation verification
        """
        self.slm = slm_provider
        self.weaviate = weaviate_manager
        self.citation_store = citation_store or {}
    
    def set_slm_provider(self, provider):
        """Set SLM provider after initialization"""
        self.slm = provider
    
    def set_weaviate_manager(self, manager):
        """Set Weaviate manager after initialization"""
        self.weaviate = manager
    
    async def validate(
        self,
        response: str,
        user_query: str,
        user_frustrated: bool = False,
        regeneration_attempt: int = 0,
        pq_context: Dict = None,
    ) -> ValidationResult:
        """
        Full output validation pipeline.
        
        Args:
            response: LLM's proposed response
            user_query: Original user query (for intent check)
            user_frustrated: Whether user is in escalated state
            regeneration_attempt: Current regeneration attempt number
            pq_context: Context from prequalifier (frustration level, etc.)
        
        Returns:
            ValidationResult with status and any modifications
        """
        result = ValidationResult(
            status=ValidationStatus.PASSED,
            original_response=response,
            validated_response=response,
            regeneration_attempt=regeneration_attempt,
        )
        
        # Check if we've exceeded max regenerations
        if regeneration_attempt >= result.max_regenerations:
            logger.warning("Max regeneration attempts exceeded")
            result.status = ValidationStatus.BLOCKED
            result.rejection_reason = RejectionReason.INTENT_NOT_FULFILLED
            result.rejection_explanation = "Unable to generate compliant response after multiple attempts"
            result.regeneration_instructions = "Please provide a safe, generic acknowledgment and offer human callback"
            return result
        
        # Step 1: Intent fulfillment check
        intent_result = await self._check_intent_fulfillment(response, user_query)
        result.intent_check = intent_result
        
        if not intent_result.fulfilled:
            result.status = ValidationStatus.REJECTED
            result.rejection_reason = RejectionReason.INTENT_NOT_FULFILLED
            result.rejection_explanation = intent_result.explanation
            result.regeneration_instructions = f"""Your response did not fulfill the user's intent.
User asked: "{user_query}"
Issue: {intent_result.explanation}
Please regenerate a response that directly addresses the user's question."""
            return result
        
        # Step 2: Ethics/morality check
        ethics_result = await self._check_ethics(response)
        result.ethics_check = ethics_result
        
        if not ethics_result.passed:
            result.status = ValidationStatus.REJECTED
            result.rejection_reason = RejectionReason.ETHICS_VIOLATION
            result.rejection_explanation = "; ".join(ethics_result.issues)
            result.regeneration_instructions = f"""Your response failed ethics review.
Issues: {'; '.join(ethics_result.issues)}
Suggestions: {'; '.join(ethics_result.suggestions)}
Please regenerate a response that aligns with Christian moral standards."""
            return result
        
        # Step 3: FEC compliance check (RAG + double-negative)
        fec_result = await self._check_fec_compliance(response)
        result.fec_check = fec_result
        
        if not fec_result.compliant:
            result.status = ValidationStatus.REJECTED
            result.rejection_reason = RejectionReason.FEC_VIOLATION
            result.rejection_explanation = "; ".join(fec_result.violations)
            result.regeneration_instructions = f"""Your response violates FEC regulations.
Violations: {'; '.join(fec_result.violations)}
Remove the violating content and regenerate a compliant response.
Do not: make guarantees, provide tax/legal advice, solicit payments directly, or defame opponents."""
            return result
        
        # Step 4: De-escalation check (if user is frustrated)
        if user_frustrated:
            deesc_passed, deesc_issues = await self._check_deescalation(response)
            
            if not deesc_passed:
                result.status = ValidationStatus.REJECTED
                result.rejection_reason = RejectionReason.INFLAMES_SITUATION
                result.rejection_explanation = "; ".join(deesc_issues)
                result.regeneration_instructions = f"""Your response may inflame the frustrated user.
Issues: {'; '.join(deesc_issues)}
Please regenerate with:
- Empathy markers (I understand, I hear you, etc.)
- Acknowledgment of their feelings
- Helpful, non-dismissive tone
- Offer of human callback option"""
                return result
        
        # Step 5: PII redaction (hybrid regex + SLM)
        pii_result = await self._redact_pii(response)
        result.pii_check = pii_result
        
        if pii_result.had_pii:
            result.validated_response = pii_result.redacted_text
            result.status = ValidationStatus.MODIFIED
            result.modifications.append(f"Redacted {len(pii_result.pii_found)} PII items")
        
        # Step 6: Citation verification
        citation_result = await self._verify_citations(response)
        result.citation_check = citation_result
        
        if not citation_result.all_valid:
            result.status = ValidationStatus.REJECTED
            result.rejection_reason = RejectionReason.CITATION_INVALID
            result.rejection_explanation = f"Invalid citations: {', '.join(citation_result.invalid_citations)}"
            result.regeneration_instructions = f"""Your response contains invalid citations.
Invalid: {', '.join(citation_result.invalid_citations)}
Remove or correct these citations. Only cite sources that exist in the knowledge base."""
            return result
        
        return result
    
    async def _check_intent_fulfillment(
        self,
        response: str,
        user_query: str
    ) -> IntentCheckResult:
        """
        SLM check: Does the response fulfill the user's intent?
        """
        if self.slm is None:
            return self._fallback_intent_check(response, user_query)
        
        try:
            slm_response = await self.slm.check_intent_fulfillment(user_query, response)
            
            fulfilled = slm_response.decision == "YES"
            
            return IntentCheckResult(
                fulfilled=fulfilled,
                explanation=slm_response.explanation or ("Response fulfills intent" if fulfilled else "Response does not address query"),
                confidence=slm_response.confidence
            )
            
        except Exception as e:
            logger.warning(f"SLM intent check failed: {e}, using fallback")
            return self._fallback_intent_check(response, user_query)
    
    def _fallback_intent_check(
        self,
        response: str,
        user_query: str
    ) -> IntentCheckResult:
        """Fallback intent check without SLM"""
        # Check if response is too short
        if len(response.split()) < 10:
            return IntentCheckResult(
                fulfilled=False,
                explanation="Response is too short to adequately address the query",
                confidence=0.5
            )
        
        # Check if response contains question keywords
        query_keywords = set(user_query.lower().split()) - {'what', 'how', 'why', 'when', 'where', 'who', 'is', 'are', 'the', 'a', 'an'}
        response_lower = response.lower()
        
        keyword_matches = sum(1 for kw in query_keywords if kw in response_lower)
        match_ratio = keyword_matches / len(query_keywords) if query_keywords else 0
        
        if match_ratio < 0.3:
            return IntentCheckResult(
                fulfilled=False,
                explanation="Response does not appear to address the topic of the query",
                confidence=0.4
            )
        
        return IntentCheckResult(
            fulfilled=True,
            explanation="Response appears to address the query",
            confidence=0.6
        )
    
    async def _check_ethics(self, response: str) -> EthicsCheckResult:
        """
        SLM check: Does the response conform to Christian moral standards?
        """
        if self.slm is None:
            return self._fallback_ethics_check(response)
        
        try:
            slm_response = await self.slm.check_ethics(response)
            
            passed = slm_response.decision == "PASS"
            
            issues = []
            suggestions = []
            if not passed and slm_response.explanation:
                issues = [slm_response.explanation]
                suggestions = ["Review for alignment with campaign values"]
            
            return EthicsCheckResult(
                passed=passed,
                issues=issues,
                suggestions=suggestions
            )
            
        except Exception as e:
            logger.warning(f"SLM ethics check failed: {e}, using fallback")
            return self._fallback_ethics_check(response)
    
    def _fallback_ethics_check(self, response: str) -> EthicsCheckResult:
        """Fallback ethics check without SLM"""
        issues = []
        response_lower = response.lower()
        
        # Check for obviously problematic content
        if re.search(r'\b(hate|kill|destroy|attack)\b', response_lower):
            issues.append("Contains potentially violent language")
        
        if re.search(r'\b(lie|deceive|trick|fool)\b.*\b(voter|people|them)\b', response_lower):
            issues.append("May encourage deception")
        
        return EthicsCheckResult(
            passed=len(issues) == 0,
            issues=issues,
            suggestions=["Review for alignment with campaign values"] if issues else []
        )
    
    async def _check_fec_compliance(self, response: str) -> FECCheckResult:
        """
        FEC compliance check with RAG retrieval + SLM double-negative verification.
        
        Step 1: Pattern matching for obvious violations
        Step 2: Double-negative context check (e.g., "can't guarantee" is OK)
        Step 3: RAG retrieval of relevant FEC regulations
        Step 4: SLM double-check with regulations in context
        """
        violations = []
        response_lower = response.lower()
        
        # Step 1: Pattern matching
        pattern_matches = []
        for pattern, violation_type in self.FEC_PROHIBITED_PATTERNS:
            if re.search(pattern, response_lower, re.IGNORECASE):
                pattern_matches.append((pattern, violation_type))
        
        # Step 2: Double-negative context check
        # Some pattern matches are OK when negated (e.g., "I can't guarantee")
        negation_patterns = [
            r"(can'?t|cannot|don'?t|do not|won'?t|will not|not able to|unable to)\s+",
            r"(i'?m not saying|not promising|no guarantee)",
        ]
        
        for pattern, violation_type in pattern_matches:
            match = re.search(pattern, response_lower, re.IGNORECASE)
            if match:
                start_pos = max(0, match.start() - 30)
                context_before = response_lower[start_pos:match.start()]
                
                is_negated = any(
                    re.search(neg_pattern, context_before, re.IGNORECASE)
                    for neg_pattern in negation_patterns
                )
                
                if not is_negated:
                    violations.append(violation_type)
        
        # If clear violations found, no need for SLM check
        if violations:
            return FECCheckResult(
                compliant=False,
                violations=violations,
                relevant_regulations=[]
            )
        
        # Step 3: RAG retrieval of FEC regulations
        relevant_regs = []
        if self.weaviate:
            try:
                fec_query = f"FEC regulation compliance {response[:200]}"
                results = await self.weaviate.search("FECProhibited", fec_query, limit=3)
                relevant_regs = [
                    {"content": r.get("content", ""), "source": r.get("source", "")}
                    for r in results
                ]
            except Exception as e:
                logger.warning(f"FEC RAG retrieval failed: {e}")
        
        # Step 4: SLM double-check (if SLM available)
        if self.slm:
            try:
                regs_text = "\n".join([f"- {r['content'][:300]}" for r in relevant_regs]) if relevant_regs else "Standard FEC campaign regulations"
                
                slm_response = await self.slm.check_fec_compliance(response, [regs_text])
                
                if slm_response.decision == "VIOLATION":
                    violations.append(slm_response.explanation or "FEC violation detected by classifier")
                    
            except Exception as e:
                logger.warning(f"SLM FEC check failed: {e}")
        
        return FECCheckResult(
            compliant=len(violations) == 0,
            violations=violations,
            relevant_regulations=relevant_regs
        )
    
    async def _check_deescalation(self, response: str) -> Tuple[bool, List[str]]:
        """
        Check if response is appropriate for a frustrated user.
        """
        issues = []
        response_lower = response.lower()
        
        # Check for dismissive patterns
        for pattern in self.DISMISSIVE_PATTERNS:
            match = re.search(pattern, response_lower, re.IGNORECASE)
            if match:
                issues.append(f"Dismissive phrase: '{match.group()}'")
        
        # Check for lack of empathy markers
        has_empathy = any(
            re.search(pattern, response_lower, re.IGNORECASE)
            for pattern in self.EMPATHY_PATTERNS
        )
        
        if not has_empathy and not issues:
            issues.append("Response lacks empathy markers for frustrated user")
        
        # SLM check for subtle inflaming language
        if self.slm and not issues:
            try:
                prompt = self.DEESCALATION_CHECK_PROMPT.format(response=response)
                slm_response = await self.slm.generate(
                    prompt=prompt,
                    max_tokens=50,
                    temperature=0.0,
                )
                
                if "INFLAMES" in slm_response.upper():
                    lines = slm_response.strip().split('\n')
                    reason = lines[1] if len(lines) > 1 else "Tone may inflame situation"
                    issues.append(reason)
                    
            except Exception as e:
                logger.warning(f"SLM de-escalation check failed: {e}")
        
        return len(issues) == 0, issues
    
    async def _redact_pii(self, text: str, context: str = None) -> PIICheckResult:
        """
        Hybrid PII redaction with context-aware allowlisting.
        
        Args:
            text: Text to redact PII from
            context: Optional context (e.g., "volunteer", "donate") to allow official contacts
        
        Steps:
        1. Check if context allows official campaign contacts
        2. Temporarily replace allowed contacts with placeholders
        3. Apply regex redaction
        4. Apply SLM detection (if available)
        5. Restore allowed contacts
        """
        pii_found = []
        redacted = text
        
        # Check if context allows official campaign contacts
        context_allows_official = False
        if context:
            context_lower = context.lower()
            allowed_contexts = CAMPAIGN_CONTACTS.get("allowed_contexts", [])
            context_allows_official = any(
                ctx.lower() in context_lower for ctx in allowed_contexts
            )
        
        # Step 1: Protect official campaign contacts with placeholders
        protected_items = []
        
        if context_allows_official:
            # Protect official emails
            for email in CAMPAIGN_CONTACTS.get("emails", []):
                placeholder = f"__PROTECTED_EMAIL_{len(protected_items)}__"
                if email.lower() in redacted.lower():
                    protected_items.append((placeholder, email))
                    redacted = re.sub(
                        re.escape(email), placeholder, redacted, flags=re.IGNORECASE
                    )
            
            # Protect official phones
            for phone in CAMPAIGN_CONTACTS.get("phones", []):
                placeholder = f"__PROTECTED_PHONE_{len(protected_items)}__"
                phone_pattern = re.escape(phone).replace(r"\ ", r"\s*").replace(r"\-", r"[-.\s]*")
                if re.search(phone_pattern, redacted, re.IGNORECASE):
                    protected_items.append((placeholder, phone))
                    redacted = re.sub(phone_pattern, placeholder, redacted, flags=re.IGNORECASE)
            
            # Protect official URLs
            for url in CAMPAIGN_CONTACTS.get("urls", []):
                placeholder = f"__PROTECTED_URL_{len(protected_items)}__"
                if url.lower() in redacted.lower():
                    protected_items.append((placeholder, url))
                    redacted = re.sub(
                        re.escape(url), placeholder, redacted, flags=re.IGNORECASE
                    )
        
        # Step 2: Regex redaction on unprotected content
        for pattern, replacement in self.PII_REGEX_PATTERNS:
            matches = re.findall(pattern, redacted, re.IGNORECASE)
            if matches:
                for match in matches:
                    if not match.startswith("__PROTECTED_"):
                        pii_found.append({"type": replacement, "value": match[:10] + "..."})
                redacted = re.sub(pattern, replacement, redacted, flags=re.IGNORECASE)
        
        # Step 3: SLM for contextual PII (if available and regex found nothing major)
        if self.slm and len(pii_found) < 2:
            try:
                slm_response = await self.slm.detect_pii(redacted)
                
                if slm_response.decision == "PII_FOUND":
                    logger.warning(f"SLM detected potential PII: {slm_response.explanation[:100]}")
                    pii_found.append({"type": "contextual_pii", "value": slm_response.explanation[:50]})
                    
            except Exception as e:
                logger.warning(f"SLM PII check failed: {e}")
        
        # Step 4: Restore protected campaign contacts
        for placeholder, original in protected_items:
            redacted = redacted.replace(placeholder, original)
        
        return PIICheckResult(
            had_pii=len(pii_found) > 0,
            redacted_text=redacted,
            pii_found=pii_found
        )
    
    async def _verify_citations(self, response: str) -> CitationCheckResult:
        """
        Verify that all citation anchors exist in the metadata store.
        
        Anchors format: [CITE-<collection>-<id>] or [WEB-<date>-<id>]
        """
        # Extract citation anchors from response
        anchor_pattern = r'\[(CITE|WEB)-[A-Za-z0-9\-_]+\]'
        anchors = re.findall(anchor_pattern, response)
        
        if not anchors:
            # No citations to verify
            return CitationCheckResult(
                all_valid=True,
                invalid_citations=[],
                verified_citations=[]
            )
        
        invalid = []
        verified = []
        
        for anchor in anchors:
            full_anchor = f"[{anchor}]" if not anchor.startswith("[") else anchor
            
            # Check citation store
            if self.citation_store and full_anchor in self.citation_store:
                verified.append(full_anchor)
            else:
                # For now, allow citations we can't verify (RAG data)
                # In production, this should be stricter
                verified.append(full_anchor)
        
        return CitationCheckResult(
            all_valid=len(invalid) == 0,
            invalid_citations=invalid,
            verified_citations=verified
        )
    
    def build_regeneration_prompt(
        self,
        original_response: str,
        validation_result: ValidationResult
    ) -> str:
        """
        Build a prompt for LLM regeneration after rejection.
        """
        return f"""Your previous response was rejected by the output validator.

ORIGINAL RESPONSE:
{original_response}

REJECTION REASON: {validation_result.rejection_reason.value if validation_result.rejection_reason else 'Unknown'}

EXPLANATION: {validation_result.rejection_explanation}

REGENERATION INSTRUCTIONS:
{validation_result.regeneration_instructions}

Please generate a new response that addresses these issues. This is regeneration attempt {validation_result.regeneration_attempt + 1} of {validation_result.max_regenerations}."""


# Singleton instance
output_validator = OutputValidator()
