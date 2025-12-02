"""
FEC Compliance Checker with Isolated RAG Collection

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
from pathlib import Path

logger = logging.getLogger(__name__)


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
    
    def __post_init__(self):
        if self.violations:
            self.max_severity = max(v.severity for v in self.violations)
            self.compliant = self.max_severity <= 2
            self.remediation_needed = not self.compliant


class FECProhibitedPhrasesStore:
    """
    In-memory store for prohibited phrases and FEC rules.
    
    This is used before Weaviate RAG is fully initialized.
    """
    
    PROHIBITED_PHRASES = {
        "tax_advice": [
            ("your donation is tax deductible", 5),
            ("you can write off this contribution", 5),
            ("this qualifies for a tax benefit", 5),
            ("legally, you should", 5),
            ("this is not legal advice, but", 4),
            ("you might want to sue", 4),
        ],
        "financial_solicitation": [
            ("enter your credit card", 5),
            ("provide your bank account", 5),
            ("we will process your donation", 5),
            ("click here to donate now", 4),
            ("i can accept your payment", 5),
        ],
        "defamation": [
            ("is a criminal", 4),
            ("committed fraud", 4),
            ("stole money", 4),
            ("is corrupt", 4),
            ("took bribes", 4),
        ],
        "false_identity": [
            ("i am brandon", 5),
            ("i am the candidate", 5),
            ("i am a human", 5),
            ("speaking as the candidate", 5),
        ],
        "false_claims": [
            ("endorsed by", 4),
            ("guaranteed to win", 4),
            ("will definitely happen", 4),
            ("promise you that", 3),
            ("100% certain", 3),
        ],
        "coercion": [
            ("vote for us or else", 5),
            ("if you don't support us", 4),
            ("you must donate", 4),
            ("failure to support", 4),
        ],
        "medical_advice": [
            ("you should take", 4),
            ("i recommend this medication", 5),
            ("this treatment will cure", 5),
        ],
    }
    
    SAFE_RESPONSES = {
        "tax_advice": "I cannot provide tax or legal advice. Please consult with a qualified tax professional or attorney for guidance on these matters.",
        "financial_solicitation": "I cannot process donations directly. For secure, FEC-compliant donation options, please visit our official donation page or request a callback from the team.",
        "defamation": "I focus on policy differences and Brandon's positions rather than personal attacks. Would you like to know more about Brandon's stance on specific issues?",
        "false_identity": "I'm an AI-powered assistant for Brandon's campaign. I can help you learn about Brandon's positions and connect you with the campaign team.",
        "false_claims": "I can share Brandon's positions and plans, but I cannot make guarantees about outcomes. Would you like to learn more about his platform?",
        "coercion": "We'd appreciate your support, but the choice is entirely yours. Can I share information about Brandon's positions that might help you decide?",
        "medical_advice": "I cannot provide medical advice. Please consult with a healthcare professional for guidance on health-related matters.",
        "default": "That topic falls under strict campaign finance regulations. Please refer to the official campaign website's legal page for guidance."
    }
    
    def __init__(self):
        self._compiled_patterns = {}
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for faster matching."""
        for category, phrases in self.PROHIBITED_PHRASES.items():
            self._compiled_patterns[category] = [
                (re.compile(re.escape(phrase), re.I), severity, phrase)
                for phrase, severity in phrases
            ]
    
    def check_prohibited(self, text: str) -> List[FECViolation]:
        """Check text against all prohibited phrases."""
        violations = []
        text_lower = text.lower()
        
        for category, patterns in self._compiled_patterns.items():
            for pattern, severity, phrase in patterns:
                if pattern.search(text):
                    violations.append(FECViolation(
                        violation_type=category,
                        severity=severity,
                        matched_rule=phrase,
                        explanation=f"Prohibited phrase in {category}: '{phrase}'",
                        source_citation=f"FEC-PROHIB-{category.upper()}"
                    ))
        
        return violations
    
    def get_safe_response(self, violation_type: str) -> str:
        """Get a safe fallback response for a violation type."""
        return self.SAFE_RESPONSES.get(violation_type, self.SAFE_RESPONSES["default"])


class FECComplianceChecker:
    """
    FEC Compliance Checker with RAG-based violation detection.
    
    Implements:
    1. Pattern-based first pass (fast)
    2. RAG retrieval of relevant FEC rules (if Weaviate available)
    3. SLM binary classification (if SLM available)
    4. Remediation with safe fallback responses
    """
    
    def __init__(self, weaviate_manager=None, slm_classifier=None):
        self._weaviate = weaviate_manager
        self._slm = slm_classifier
        self._phrase_store = FECProhibitedPhrasesStore()
        self._audit_log = []
    
    def set_weaviate(self, weaviate_manager):
        """Set the Weaviate manager for RAG lookups."""
        self._weaviate = weaviate_manager
    
    def set_slm(self, slm_classifier):
        """Set the SLM classifier for binary violation detection."""
        self._slm = slm_classifier
    
    async def check_compliance(
        self, 
        response: str,
        query: str = "",
        session_id: str = ""
    ) -> FECComplianceResult:
        """
        Check a response for FEC compliance violations.
        
        Protocol:
        1. Pattern matching for obvious violations
        2. RAG query of FECProhibited collection (if available)
        3. SLM binary classification (if available)
        4. Generate remediation if violations found
        
        Args:
            response: The LLM response to check
            query: Original user query (for context)
            session_id: Session ID for logging
        
        Returns:
            FECComplianceResult with violations and remediation
        """
        violations = []
        retrieved_rules = []
        
        pattern_violations = self._phrase_store.check_prohibited(response)
        violations.extend(pattern_violations)
        
        if self._weaviate and (not violations or max(v.severity for v in violations) < 4):
            try:
                rag_violations, rag_rules = await self._check_rag(response)
                violations.extend(rag_violations)
                retrieved_rules.extend(rag_rules)
            except Exception as e:
                logger.warning(f"FEC RAG check failed: {e}")
        
        if self._slm and (not violations or max(v.severity for v in violations) < 4):
            try:
                slm_violations = await self._check_slm(response, retrieved_rules)
                violations.extend(slm_violations)
            except Exception as e:
                logger.warning(f"FEC SLM check failed: {e}")
        
        result = FECComplianceResult(
            compliant=len(violations) == 0 or all(v.severity <= 2 for v in violations),
            violations=violations,
            retrieved_rules=retrieved_rules
        )
        
        if result.remediation_needed and violations:
            primary_violation = max(violations, key=lambda v: v.severity)
            result.safe_response = self._phrase_store.get_safe_response(
                primary_violation.violation_type
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
        
        try:
            results = await self._weaviate.search(
                collection_name="FECProhibited",
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
                            
        except Exception as e:
            logger.error(f"FEC RAG search failed: {e}")
        
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
            "timestamp": asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0
        }
        
        self._audit_log.append(log_entry)
        
        if len(self._audit_log) > 1000:
            self._audit_log = self._audit_log[-500:]
        
        logger.warning(f"FEC Violation logged: {[v.violation_type for v in violations]}")
    
    def get_audit_log(self) -> List[Dict]:
        """Get the audit log of FEC violations."""
        return self._audit_log.copy()


fec_checker = FECComplianceChecker()
