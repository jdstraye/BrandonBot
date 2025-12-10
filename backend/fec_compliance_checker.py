"""
FEC Compliance Checker with Mandatory RAG Collection

FAIL-CLOSED DESIGN: System REQUIRES RAG to be operational.
No pattern matching fallback - if FECProhibited collection is unavailable,
the system refuses to process responses.

Implements the Zero-Tolerance, Isolated RAG Collection with Double-Negative Check:

1. FEC Collection (Isolated Data Layer):
   - FEC Regulations (11 CFR) - chunked by rule
   - Prohibited Statements - chunked by phrase
   - Disclaimer Templates - chunked by template

2. FEC Compliance Protocol (Double-Negative Check):
   Step 1: RAG retrieval of relevant violations using response as query
   Step 2: Binary violation classification with SLM

3. Remediation:
   - Block violating responses
   - Log violations for audit trail
   - Generate safe fallback responses
"""

import logging
import re
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class FECRAGUnavailableError(Exception):
    """Raised when FEC RAG is required but not available."""
    pass


class FECCollectionMissingError(Exception):
    """Raised when FECProhibited collection doesn't exist in Weaviate."""
    pass


@dataclass
class FECViolation:
    """Represents an FEC compliance violation."""
    violation_type: str
    severity: int
    matched_rule: str
    explanation: str
    source_citation: Optional[str] = None


@dataclass
class FECComplianceResult:
    """Result from FEC compliance check."""
    compliant: bool
    violations: List[FECViolation] = field(default_factory=list)
    max_severity: int = 0
    retrieved_rules: List[Dict[str, Any]] = field(default_factory=list)
    remediation_needed: bool = False
    safe_response: Optional[str] = None
    rag_available: bool = True
    
    def __post_init__(self):
        if self.violations:
            self.max_severity = max(v.severity for v in self.violations)
            self.compliant = self.max_severity <= 2
            self.remediation_needed = not self.compliant


SAFE_RESPONSES = {
    "tax_advice": "I cannot provide tax or legal advice. Please consult with a qualified tax professional or attorney for guidance on these matters.",
    "financial_solicitation": "I cannot process donations directly. For secure, FEC-compliant donation options, please visit our official donation page or request a callback from the team.",
    "defamation": "I focus on policy differences and Brandon's positions rather than personal attacks. Would you like to know more about Brandon's stance on specific issues?",
    "false_identity": "I'm an AI-powered assistant for Brandon's campaign. I can help you learn about Brandon's positions and connect you with the campaign team.",
    "false_claims": "I can share Brandon's positions and plans, but I cannot make guarantees about outcomes. Would you like to learn more about his platform?",
    "coercion": "We'd appreciate your support, but the choice is entirely yours. Can I share information about Brandon's positions that might help you decide?",
    "medical_advice": "I cannot provide medical advice. Please consult with a healthcare professional for guidance on health-related matters.",
    "rag_match": "I need to be careful about FEC compliance here. Let me rephrase that in a way that follows campaign finance regulations.",
    "slm_detected": "I need to be careful about FEC compliance here. Let me rephrase that in a way that follows campaign finance regulations.",
    "default": "That topic falls under strict campaign finance regulations. Please refer to the official campaign website's legal page for guidance."
}


class FECComplianceChecker:
    """
    FEC Compliance Checker with MANDATORY RAG-based violation detection.
    
    FAIL-CLOSED DESIGN:
    - RAG MUST be available for compliance checking
    - If Weaviate is not connected, raises FECRAGUnavailableError
    - If FECProhibited collection is missing, raises FECCollectionMissingError
    - NO pattern matching fallback - system refuses to operate without RAG
    
    Implements:
    1. RAG retrieval of relevant FEC rules (REQUIRED)
    2. SLM binary classification (if SLM available)
    3. Remediation with safe fallback responses
    """
    
    FEC_COLLECTION = "FECProhibited"
    
    def __init__(self, weaviate_manager=None, slm_classifier=None, require_rag: bool = True):
        self._weaviate = weaviate_manager
        self._slm = slm_classifier
        self._require_rag = require_rag
        self._rag_verified = False
        self._audit_log = []
    
    def set_weaviate(self, weaviate_manager):
        """Set the Weaviate manager for RAG lookups."""
        self._weaviate = weaviate_manager
        self._rag_verified = False
    
    def set_slm(self, slm_classifier):
        """Set the SLM classifier for binary violation detection."""
        self._slm = slm_classifier
    
    @property
    def rag_available(self) -> bool:
        """Check if RAG is available."""
        return self._weaviate is not None and self._rag_verified
    
    async def verify_rag_available(self) -> bool:
        """
        Verify that RAG is available and FECProhibited collection exists.
        
        Returns:
            True if RAG is ready for use
            
        Raises:
            FECRAGUnavailableError: If Weaviate is not connected
            FECCollectionMissingError: If FECProhibited collection doesn't exist
        """
        if self._weaviate is None:
            if self._require_rag:
                raise FECRAGUnavailableError(
                    "FEC RAG is REQUIRED but Weaviate is not connected. "
                    "System cannot operate without FEC compliance checking."
                )
            return False
        
        try:
            count = await self._weaviate.get_collection_count(self.FEC_COLLECTION)
            if count == 0:
                if self._require_rag:
                    raise FECCollectionMissingError(
                        f"FECProhibited collection exists but is EMPTY. "
                        f"Run ingest_all.py to load FEC compliance data. "
                        f"System cannot operate without FEC compliance data."
                    )
                return False
            
            logger.info(f"FEC RAG verified: {count} documents in {self.FEC_COLLECTION}")
            self._rag_verified = True
            return True
            
        except Exception as e:
            if "does not exist" in str(e).lower() or "not found" in str(e).lower():
                if self._require_rag:
                    raise FECCollectionMissingError(
                        f"FECProhibited collection does not exist in Weaviate. "
                        f"Run ingest_all.py to create and populate it. "
                        f"System cannot operate without FEC compliance checking."
                    )
                return False
            
            if self._require_rag:
                raise FECRAGUnavailableError(
                    f"FEC RAG verification failed: {e}. "
                    f"System cannot operate without FEC compliance checking."
                )
            return False
    
    async def check_compliance(
        self, 
        response: str,
        query: str = "",
        session_id: str = ""
    ) -> FECComplianceResult:
        """
        Check a response for FEC compliance violations.
        
        FAIL-CLOSED: If RAG is required but unavailable, raises an error.
        
        Protocol:
        1. Verify RAG is available (REQUIRED)
        2. RAG query of FECProhibited collection
        3. SLM binary classification (if available)
        4. Generate remediation if violations found
        
        Args:
            response: The LLM response to check
            query: Original user query (for context)
            session_id: Session ID for logging
        
        Returns:
            FECComplianceResult with violations and remediation
            
        Raises:
            FECRAGUnavailableError: If RAG is required but not available
            FECCollectionMissingError: If FECProhibited collection is missing
        """
        if not self._rag_verified:
            await self.verify_rag_available()
        
        violations = []
        retrieved_rules = []
        
        try:
            rag_violations, rag_rules = await self._check_rag(response)
            violations.extend(rag_violations)
            retrieved_rules.extend(rag_rules)
        except Exception as e:
            self._rag_verified = False
            error_msg = f"FEC RAG check failed: {e}"
            logger.error(error_msg)
            if self._require_rag:
                raise FECRAGUnavailableError(error_msg) from e
        
        if self._slm and retrieved_rules:
            try:
                slm_violations = await self._check_slm(response, retrieved_rules)
                violations.extend(slm_violations)
            except Exception as e:
                logger.warning(f"FEC SLM check failed (non-fatal): {e}")
        
        result = FECComplianceResult(
            compliant=len(violations) == 0 or all(v.severity <= 2 for v in violations),
            violations=violations,
            retrieved_rules=retrieved_rules,
            rag_available=True
        )
        
        if result.remediation_needed and violations:
            primary_violation = max(violations, key=lambda v: v.severity)
            result.safe_response = SAFE_RESPONSES.get(
                primary_violation.violation_type,
                SAFE_RESPONSES["default"]
            )
        
        if violations:
            self._log_violation(session_id, query, response, violations)
        
        return result
    
    async def _check_rag(self, response: str) -> Tuple[List[FECViolation], List[Dict]]:
        """
        RAG retrieval of relevant FEC rules using response as query.
        
        Step 1 of the Double-Negative Check:
        Query the FECProhibited collection with the response text.
        """
        violations = []
        rules = []
        
        results = await self._weaviate.search(
            collection_name=self.FEC_COLLECTION,
            query=response[:500],
            limit=5
        )
        
        for result in results:
            content = result.get("content", "")
            source = result.get("source", "")
            distance = result.get("distance", 1.0)
            
            rules.append({
                "content": content,
                "source": source,
                "distance": distance
            })
            
            if distance < 0.3:
                prohibited_phrases = re.findall(r'"([^"]+)"', content)
                response_lower = response.lower()
                
                for phrase in prohibited_phrases:
                    if phrase.lower() in response_lower:
                        violations.append(FECViolation(
                            violation_type="rag_match",
                            severity=4,
                            matched_rule=content[:100],
                            explanation=f"RAG match: '{phrase}' (distance={distance:.3f})",
                            source_citation=source
                        ))
                        break
        
        return violations, rules
    
    async def _check_slm(
        self, 
        response: str, 
        retrieved_rules: List[Dict]
    ) -> List[FECViolation]:
        """
        SLM binary violation classification.
        
        Step 2 of the Double-Negative Check:
        Use SLM to classify if response violates retrieved rules.
        """
        violations = []
        
        if not retrieved_rules:
            return violations
        
        rules_text = "\n".join([
            f"- {r['content'][:200]}" for r in retrieved_rules[:3]
        ])
        
        prompt = f"""You are a Compliance Auditor. Your task is to check the Final Answer against the Prohibited Rules.

Final Answer: {response[:400]}

Prohibited Rules:
{rules_text}

Does the Final Answer, in its entirety, constitute a VIOLATION of any of the rules provided?

Your response must be a single word: YES or NO."""
        
        try:
            result = await self._slm.classify(prompt)
            
            if result.upper().strip().startswith("YES"):
                violations.append(FECViolation(
                    violation_type="slm_detected",
                    severity=4,
                    matched_rule="SLM classification",
                    explanation="SLM detected FEC violation based on retrieved rules",
                    source_citation="SLM-FEC-CHECK"
                ))
                
        except Exception as e:
            logger.warning(f"SLM classification failed: {e}")
        
        return violations
    
    def _log_violation(
        self,
        session_id: str,
        query: str,
        response: str,
        violations: List[FECViolation]
    ):
        """Log violation for audit trail."""
        log_entry = {
            "session_id": session_id,
            "query": query[:200],
            "response": response[:500],
            "violations": [
                {
                    "type": v.violation_type,
                    "severity": v.severity,
                    "rule": v.matched_rule,
                    "explanation": v.explanation
                }
                for v in violations
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self._audit_log.append(log_entry)
        
        if len(self._audit_log) > 1000:
            self._audit_log = self._audit_log[-500:]
        
        logger.warning(f"FEC Violation logged: {[v.violation_type for v in violations]}")
    
    def get_audit_log(self) -> List[Dict]:
        """Get the audit log of FEC violations."""
        return self._audit_log.copy()


fec_checker = FECComplianceChecker(require_rag=True)
