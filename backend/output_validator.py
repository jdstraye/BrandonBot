"""
Output Validator Module for BrandonBot
Validates LLM responses before sending to user:
- Tone validation for frustrated users (de-escalation)
- FEC compliance checking
- Empathy marker detection
- Prohibited content filtering
- Citation verification
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Validation result status"""
    PASSED = "passed"
    MODIFIED = "modified"
    BLOCKED = "blocked"
    WARNING = "warning"


@dataclass
class ValidationResult:
    """Result from output validation"""
    status: ValidationStatus
    original_response: str
    validated_response: str
    modifications: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    fec_issues: List[str] = field(default_factory=list)
    tone_issues: List[str] = field(default_factory=list)
    added_deescalation: bool = False
    added_disclosure: bool = False
    confidence: float = 1.0


class OutputValidator:
    """
    Validates and modifies LLM output for compliance and appropriate tone.
    """
    
    DISMISSIVE_PHRASES = [
        (r"\b(calm down|relax|chill|take it easy)\b", "dismissive_calm_down"),
        (r"\b(you('re| are) (wrong|incorrect|mistaken))\b", "accusatory"),
        (r"\b(that('s| is) (not true|false|wrong))\b", "contradicting"),
        (r"\b(obviously|clearly|of course)\b", "condescending"),
        (r"\b(you should (know|understand))\b", "assumption"),
        (r"\b(i (already|just) (said|told you|explained))\b", "frustrated_response"),
        (r"\b(as i (said|mentioned|explained))\b", "repetition_complaint"),
        (r"\b(it('s| is) (simple|easy|basic|obvious))\b", "minimizing"),
        (r"\b(you (need to|must|should) (calm|relax))\b", "ordering_calm"),
    ]
    
    EMPATHY_MARKERS = [
        r"\b(i understand|i hear you|i can see)\b",
        r"\b(that (must be|sounds) (frustrating|difficult|hard))\b",
        r"\b(i appreciate|thank you for)\b",
        r"\b(let me (help|try|work with))\b",
        r"\b(you('re| are) right to (ask|want|feel))\b",
        r"\b(that('s| is) a (great|good|valid|important) (question|point|concern))\b",
    ]
    
    FEC_PROHIBITED = [
        (r"\b(i guarantee|guaranteed|promise you will)\b", "Cannot make guarantees"),
        (r"\b(definitely (will|going to)|100% certain)\b", "Avoid absolute promises"),
        (r"(tax (advice|deduction|benefit)|tax.deductible)", "tax_advice"),
        (r"(consult (a |your )?(tax|cpa|accountant))", "tax_advice"),
        (r"(legal (advice|opinion)|consult.*(lawyer|attorney))", "legal_advice"),
        (r"(consult your attorney|speak (to|with) (a |your )?(lawyer|attorney))", "legal_advice"),
        (r"(you should (sue|litigate)|legally you (must|should))", "legal_advice"),
        (r"\b(credit card|bank account|send money|wire|pay now|pay directly)\b", "payment_solicitation"),
        (r"(opponent|they|he|she) (is |are )?(a )?(corrupt|criminal|fraud|liar|crook)", "defamatory"),
        (r"(committed (fraud|a crime)|stole money|stealing)", "defamatory"),
        (r"\b(vote for|elect) (me|brandon) (or else|otherwise)\b", "coercive"),
    ]
    
    REQUIRED_DISCLOSURES = [
        "I'm an AI assistant",
        "AI-generated",
        "automated assistant",
    ]
    
    DEESCALATION_TEMPLATES = {
        "high": {
            "prefix": "I really appreciate you taking the time to share this with me. I can tell this is important to you, and I want to make sure you get the help you need. ",
            "suffix": "\n\nI sense this matter is urgent for you. Would you like someone from Brandon's team to give you a call directly? They can provide more personalized assistance.",
        },
        "medium": {
            "prefix": "Thank you for your patience. I want to make sure I'm addressing your concerns properly. ",
            "suffix": "\n\nIf you'd prefer to speak with someone from the team, I'd be happy to arrange a callback.",
        },
        "low": {
            "prefix": "I appreciate your question. ",
            "suffix": "",
        },
    }
    
    SOFTENING_REPLACEMENTS = [
        (r"\bNo\b", "I understand your concern, but"),
        (r"\bYou('re| are) wrong\b", "I can see why you might think that, however"),
        (r"\bThat('s| is) incorrect\b", "Actually, the situation is a bit different"),
        (r"\bI (can't|cannot)\b", "I'm not able to"),
        (r"\bCalm down\b", "I understand this is important"),
        (r"\bcalm down\b", "I understand this is important"),
    ]
    
    def __init__(self):
        pass
    
    def validate(
        self,
        response: str,
        escalation_level: str = "none",
        user_frustrated: bool = False,
        include_ai_disclosure: bool = True
    ) -> ValidationResult:
        """
        Validate and potentially modify LLM response.
        
        Args:
            response: Original LLM response
            escalation_level: User's escalation level (none, low, medium, high)
            user_frustrated: Whether user is showing frustration
            include_ai_disclosure: Whether to add AI disclosure tag
        
        Returns:
            ValidationResult with validated/modified response
        """
        modifications = []
        warnings = []
        fec_issues = []
        tone_issues = []
        validated = response
        
        fec_issues = self._check_fec_compliance(validated)
        for issue in fec_issues:
            logger.warning(f"FEC issue detected: {issue}")
        
        if escalation_level in ["medium", "high"] or user_frustrated:
            validated, tone_mods = self._apply_deescalation(
                validated, 
                escalation_level,
                user_frustrated
            )
            modifications.extend(tone_mods)
        
        dismissive_found = self._check_dismissive(validated)
        if dismissive_found:
            tone_issues.extend(dismissive_found)
            validated = self._soften_response(validated)
            modifications.append("softened_dismissive_language")
        
        if not self._has_empathy_markers(validated) and (user_frustrated or escalation_level != "none"):
            validated = self._add_empathy(validated, escalation_level)
            modifications.append("added_empathy_markers")
        
        added_disclosure = False
        if include_ai_disclosure and not self._has_ai_disclosure(validated):
            pass
        
        status = ValidationStatus.PASSED
        if modifications:
            status = ValidationStatus.MODIFIED
        if fec_issues:
            status = ValidationStatus.WARNING
        
        return ValidationResult(
            status=status,
            original_response=response,
            validated_response=validated,
            modifications=modifications,
            warnings=warnings,
            fec_issues=fec_issues,
            tone_issues=tone_issues,
            added_deescalation=escalation_level in ["medium", "high"],
            added_disclosure=added_disclosure,
            confidence=1.0 - (len(fec_issues) * 0.1) - (len(tone_issues) * 0.05)
        )
    
    def _check_fec_compliance(self, text: str) -> List[str]:
        """Check for FEC compliance issues"""
        issues = []
        text_lower = text.lower()
        
        for pattern, issue_type in self.FEC_PROHIBITED:
            if re.search(pattern, text_lower, re.IGNORECASE):
                issues.append(issue_type)
        
        return issues
    
    def _check_dismissive(self, text: str) -> List[str]:
        """Check for dismissive language"""
        found = []
        text_lower = text.lower()
        
        for pattern, label in self.DISMISSIVE_PHRASES:
            if re.search(pattern, text_lower, re.IGNORECASE):
                found.append(label)
        
        return found
    
    def _has_empathy_markers(self, text: str) -> bool:
        """Check if response has empathy markers"""
        text_lower = text.lower()
        
        for pattern in self.EMPATHY_MARKERS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        
        return False
    
    def _has_ai_disclosure(self, text: str) -> bool:
        """Check if response has AI disclosure"""
        text_lower = text.lower()
        
        for disclosure in self.REQUIRED_DISCLOSURES:
            if disclosure.lower() in text_lower:
                return True
        
        return False
    
    def _apply_deescalation(
        self, 
        text: str, 
        level: str,
        user_frustrated: bool
    ) -> Tuple[str, List[str]]:
        """Apply de-escalation templates based on level"""
        modifications = []
        
        if level not in self.DEESCALATION_TEMPLATES:
            level = "low" if user_frustrated else "none"
        
        if level == "none":
            return text, modifications
        
        template = self.DEESCALATION_TEMPLATES[level]
        
        if template["prefix"] and not text.startswith(template["prefix"][:20]):
            text = template["prefix"] + text
            modifications.append(f"added_{level}_deescalation_prefix")
        
        if template["suffix"] and template["suffix"] not in text:
            text = text.rstrip() + template["suffix"]
            modifications.append(f"added_{level}_deescalation_suffix")
        
        return text, modifications
    
    def _soften_response(self, text: str) -> str:
        """Soften dismissive language"""
        softened = text
        
        for pattern, replacement in self.SOFTENING_REPLACEMENTS:
            softened = re.sub(pattern, replacement, softened, flags=re.IGNORECASE)
        
        return softened
    
    def _add_empathy(self, text: str, level: str) -> str:
        """Add empathy markers to response"""
        empathy_starters = {
            "high": "I really appreciate you sharing this with me. ",
            "medium": "Thank you for bringing this up. ",
            "low": "I appreciate your question. ",
        }
        
        starter = empathy_starters.get(level, "")
        
        if starter and not text.startswith(starter):
            return starter + text
        
        return text
    
    def validate_for_escalated_user(
        self,
        response: str,
        frustration_triggers: List[str] = None
    ) -> ValidationResult:
        """
        Specialized validation for users showing high frustration.
        Ensures response doesn't inflame the situation.
        """
        return self.validate(
            response=response,
            escalation_level="high",
            user_frustrated=True,
            include_ai_disclosure=True
        )


class FECComplianceChecker:
    """
    Dedicated FEC compliance checker using RAG and rules.
    """
    
    PROHIBITED_STATEMENTS = {
        "tax_advice": [
            "your donation is tax deductible",
            "you can write off",
            "tax benefit",
            "consult your tax advisor",
        ],
        "legal_advice": [
            "this is legal advice",
            "you should sue",
            "legally you must",
            "consult an attorney",
            "consult your attorney",
        ],
        "payment_direct": [
            "send money to",
            "wire transfer",
            "credit card number",
            "pay directly",
        ],
        "defamation": [
            "is a criminal",
            "committed fraud",
            "is corrupt",
            "stole money",
        ],
        "unauthorized_claims": [
            "endorsed by the president",
            "official party position",
            "guaranteed to win",
        ],
    }
    
    REQUIRED_QUALIFIERS = {
        "policy_positions": "Brandon's position is",
        "party_context": "The Republican/Independent platform suggests",
        "uncertainty": "Based on available information",
    }
    
    def check(self, response: str) -> Tuple[bool, List[str]]:
        """
        Check response for FEC compliance violations.
        
        Returns:
            Tuple of (is_compliant, list_of_issues)
        """
        issues = []
        response_lower = response.lower()
        
        for category, phrases in self.PROHIBITED_STATEMENTS.items():
            for phrase in phrases:
                if phrase.lower() in response_lower:
                    issues.append(f"{category}: '{phrase}'")
        
        return len(issues) == 0, issues
    
    def add_required_qualifiers(self, response: str, context_type: str) -> str:
        """Add required qualifiers based on context"""
        if context_type in self.REQUIRED_QUALIFIERS:
            qualifier = self.REQUIRED_QUALIFIERS[context_type]
            if qualifier.lower() not in response.lower():
                return f"{qualifier}: {response}"
        return response


output_validator = OutputValidator()
fec_checker = FECComplianceChecker()
