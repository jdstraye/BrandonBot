"""
Prequalifier Module for BrandonBot
SLM-based input preprocessing with:
- Ogilvy 10-category classification (Schwartz values)
- Sentiment/frustration detection
- Escalation level determination
- De-escalation triggers
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class OgilvyCategory(Enum):
    """
    Ogilvy 10-category classification based on Schwartz values.
    Used for matching user intent with response style.
    """
    POWER = "power"
    ACHIEVEMENT = "achievement"  
    HEDONISM = "hedonism"
    STIMULATION = "stimulation"
    SELF_DIRECTION = "self_direction"
    UNIVERSALISM = "universalism"
    BENEVOLENCE = "benevolence"
    TRADITION = "tradition"
    CONFORMITY = "conformity"
    SECURITY = "security"


class EscalationLevel(Enum):
    """User frustration/escalation level"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class UserIntent(Enum):
    """Primary user intent categories"""
    POLICY_QUESTION = "policy_question"
    PERSONAL_STORY = "personal_story"
    VOLUNTEER = "volunteer"
    DONATE = "donate"
    CALLBACK = "callback"
    COMPARISON = "comparison"
    VERIFICATION = "verification"
    SCRIPTURE = "scripture"
    PRACTICAL_IMPACT = "practical_impact"
    FUNDING = "funding"
    TIMELINE = "timeline"
    GENERAL_INFO = "general_info"
    GREETING = "greeting"
    CLARIFICATION = "clarification"
    OFF_TOPIC = "off_topic"


@dataclass
class PrequalifierResult:
    """Result from prequalifier analysis"""
    primary_intent: UserIntent
    secondary_intents: List[UserIntent] = field(default_factory=list)
    ogilvy_categories: List[OgilvyCategory] = field(default_factory=list)
    escalation_level: EscalationLevel = EscalationLevel.NONE
    needs_deescalation: bool = False
    frustration_triggers: List[str] = field(default_factory=list)
    suggested_tone: str = "neutral"
    is_vague: bool = False
    vagueness_reason: Optional[str] = None
    sanitization_applied: bool = False
    blocked: bool = False
    block_reason: Optional[str] = None
    confidence: float = 0.8


class Prequalifier:
    """
    SLM-based prequalifier for input analysis.
    Detects intent, sentiment, and escalation before main LLM processing.
    """
    
    FRUSTRATION_PATTERNS = [
        (r"(this is|you('re| are)|you are) (useless|stupid|broken|not helping)", 3),
        (r"(i('m| am)|still) (confused|lost|not getting)", 2),
        (r"(already|just) (said|asked|told you)", 2),
        (r"(doesn't|don't|didn't) (answer|help|make sense)", 2),
        (r"(can't|cannot) understand", 2),
        (r"(what|this) is (wrong|the problem)", 2),
        (r"forget it|never ?mind", 2),
        (r"(ugh|argh|OMG|FFS|WTF)", 2),
        (r"!{2,}", 2),
        (r"\?{2,}", 1),
        (r"(waste|wasting) (of|my) time", 3),
        (r"(terrible|awful|horrible|worst)", 2),
        (r"you('re| are) (an? )?(joke|fraud|liar)", 3),
        (r"useless", 2),
        (r"not helping", 2),
        (r"getting confused", 1),
        (r"can you explain again", 1),
        (r"still (don't|doesn't|haven't|hasn't)", 1),
        (r"(still )?(haven't|hasn't) addressed", 3),
        (r"you (still )?(haven't|hasn't)", 3),
        (r"haven't addressed", 3),
        (r"still haven't", 3),
        (r"addressed what i asked", 2),
    ]
    
    URGENCY_PATTERNS = [
        (r"(need|want) (to talk|speak) (to|with) (a|someone|human|person|real)", 3),
        (r"(urgent|emergency|asap|right now|immediately)", 3),
        (r"(critical|important|serious|major) (issue|problem|matter)", 2),
        (r"(deadline|time.?sensitive)", 2),
        (r"(can't|cannot) wait", 2),
        (r"(need|require) (help|assistance) (now|today)", 2),
        (r"this is (urgent|an emergency)", 3),
    ]
    
    INTENT_PATTERNS = {
        UserIntent.VOLUNTEER: [
            r"\b(volunteer|help out|sign up|get involved|join|campaign)\b",
            r"\b(door.?to.?door|canvas|phone bank|events?)\b",
        ],
        UserIntent.DONATE: [
            r"donat",
            r"contribut",
            r"support.*financially",
            r"financial.*support",
            r"give money",
            r"campaign fund",
            r"want to donate",
            r"donate to",
            r"like to support",
            r"help financially",
            r"monetary",
        ],
        UserIntent.CALLBACK: [
            r"\b(call (me|back)|speak (to|with) someone|talk to a (human|person|real))\b",
            r"\b(phone call|reach out|contact me)\b",
            r"\btalk to someone real\b",
            r"\bneed to talk to\b",
            r"\bspeak with a human\b",
        ],
        UserIntent.COMPARISON: [
            r"\b(vs\.?|versus|compared to|difference between|better than)\b",
            r"\b(opponent|other candidate|democrat|republican)\b",
        ],
        UserIntent.VERIFICATION: [
            r"\b(really|actually|truly|is that true|source|proof|evidence)\b",
            r"\b(how do (you|I) know|can you prove)\b",
        ],
        UserIntent.SCRIPTURE: [
            r"\b(bible|scripture|god|jesus|faith|christian|pray|church)\b",
            r"\b(moral|ethics|values|sin|righteous)\b",
        ],
        UserIntent.PRACTICAL_IMPACT: [
            r"\b(affect me|impact (on |my )?|how does this|what does this mean for)\b",
            r"\b(my (family|kids|business|job|taxes|healthcare))\b",
        ],
        UserIntent.FUNDING: [
            r"\b(pay for|fund|cost|afford|budget|money for)\b",
            r"\b(taxpayer|spending|deficit|debt)\b",
        ],
        UserIntent.TIMELINE: [
            r"\b(when|timeline|how long|schedule|deadline|by when)\b",
            r"\b(first (100|hundred) days|term|year one)\b",
        ],
        UserIntent.GREETING: [
            r"^(hi|hello|hey|good (morning|afternoon|evening)|greetings)",
            r"^(what('?s| is) up|how are you)",
        ],
    }
    
    OGILVY_PATTERNS = {
        OgilvyCategory.POWER: [
            r"\b(control|influence|authority|leadership|strong|powerful)\b",
        ],
        OgilvyCategory.ACHIEVEMENT: [
            r"\b(success|accomplish|achieve|win|results|performance)\b",
        ],
        OgilvyCategory.HEDONISM: [
            r"\b(enjoy|pleasure|fun|happy|freedom to)\b",
        ],
        OgilvyCategory.STIMULATION: [
            r"\b(exciting|new|different|change|innovation|bold)\b",
        ],
        OgilvyCategory.SELF_DIRECTION: [
            r"\b(independent|my own|choice|freedom|liberty|rights|right to)\b",
        ],
        OgilvyCategory.UNIVERSALISM: [
            r"\b(everyone|all people|equal|fair|justice|environment)\b",
        ],
        OgilvyCategory.BENEVOLENCE: [
            r"\b(help|care|community|family|neighbor|together)\b",
        ],
        OgilvyCategory.TRADITION: [
            r"\b(tradition|heritage|values|faith|respect|honor)\b",
        ],
        OgilvyCategory.CONFORMITY: [
            r"\b(rules|law|order|proper|should|duty|responsible)\b",
        ],
        OgilvyCategory.SECURITY: [
            r"\b(safe|secure|protect|stability|certain|reliable|security|border)\b",
        ],
    }
    
    VAGUENESS_PATTERNS = [
        (r"^(what|how|why|tell me|explain)\s*$", "Too short - need more context"),
        (r"^(stuff|things|it|that|this)\s*$", "Unclear reference"),
        (r"^.{1,10}$", "Very short query - may need clarification"),
    ]
    
    BLOCKED_PATTERNS = [
        (r"(credit card|bank account|ssn|social security)", "Financial data collection not allowed"),
        (r"(password|login|hack|exploit)", "Security-related request blocked"),
        (r"(kill|murder|attack|bomb|weapon)", "Violence-related content blocked"),
    ]
    
    def __init__(self):
        self.conversation_history: Dict[str, List[Dict]] = {}
    
    def analyze(
        self, 
        message: str, 
        session_id: str = None,
        conversation_history: List[Dict] = None
    ) -> PrequalifierResult:
        """
        Analyze user message for intent, sentiment, and escalation.
        
        Args:
            message: User's message
            session_id: Session identifier for history tracking
            conversation_history: Previous messages in conversation
        
        Returns:
            PrequalifierResult with analysis
        """
        message_lower = message.lower().strip()
        
        blocked, block_reason = self._check_blocked(message_lower)
        if blocked:
            return PrequalifierResult(
                primary_intent=UserIntent.OFF_TOPIC,
                blocked=True,
                block_reason=block_reason,
                escalation_level=EscalationLevel.NONE
            )
        
        frustration_score, frustration_triggers = self._detect_frustration(message_lower)
        urgency_score, urgency_triggers = self._detect_urgency(message_lower)
        
        if conversation_history:
            history_boost = self._analyze_history(conversation_history, message_lower)
            frustration_score += history_boost.get("frustration_boost", 0)
            frustration_triggers.extend(history_boost.get("triggers", []))
        
        escalation_level = self._determine_escalation(frustration_score, urgency_score)
        needs_deescalation = escalation_level in [EscalationLevel.MEDIUM, EscalationLevel.HIGH]
        
        primary_intent = self._detect_intent(message_lower)
        secondary_intents = self._detect_secondary_intents(message_lower, primary_intent)
        
        ogilvy_categories = self._detect_ogilvy_categories(message_lower)
        
        is_vague, vagueness_reason = self._check_vagueness(message_lower)
        
        suggested_tone = self._suggest_tone(escalation_level, primary_intent, ogilvy_categories)
        
        return PrequalifierResult(
            primary_intent=primary_intent,
            secondary_intents=secondary_intents,
            ogilvy_categories=ogilvy_categories,
            escalation_level=escalation_level,
            needs_deescalation=needs_deescalation,
            frustration_triggers=frustration_triggers,
            suggested_tone=suggested_tone,
            is_vague=is_vague,
            vagueness_reason=vagueness_reason,
            confidence=0.85 if not is_vague else 0.6
        )
    
    def _check_blocked(self, message: str) -> Tuple[bool, Optional[str]]:
        """Check if message contains blocked content"""
        for pattern, reason in self.BLOCKED_PATTERNS:
            if re.search(pattern, message, re.IGNORECASE):
                return True, reason
        return False, None
    
    def _detect_frustration(self, message: str) -> Tuple[int, List[str]]:
        """Detect frustration level from message patterns"""
        score = 0
        triggers = []
        
        for pattern, weight in self.FRUSTRATION_PATTERNS:
            if re.search(pattern, message, re.IGNORECASE):
                score += weight
                triggers.append(f"frustration:{pattern[:25]}")
        
        return score, triggers
    
    def _detect_urgency(self, message: str) -> Tuple[int, List[str]]:
        """Detect urgency signals"""
        score = 0
        triggers = []
        
        for pattern, weight in self.URGENCY_PATTERNS:
            if re.search(pattern, message, re.IGNORECASE):
                score += weight
                triggers.append(f"urgency:{pattern[:25]}")
        
        return score, triggers
    
    def _analyze_history(self, history: List[Dict], current: str) -> Dict:
        """Analyze conversation history for escalation patterns"""
        result = {"frustration_boost": 0, "triggers": []}
        
        if len(history) < 2:
            return result
        
        user_messages = [m.get("content", "").lower() for m in history if m.get("role") == "user"]
        
        repetition_words = ["already", "said", "told", "asked", "mentioned", "explained"]
        for word in repetition_words:
            if f"already {word}" in current or f"just {word}" in current:
                result["frustration_boost"] += 2
                result["triggers"].append("repetition_complaint")
                break
        
        if len(history) >= 6:
            user_count = len(user_messages)
            if user_count >= 4:
                result["frustration_boost"] += 1
                result["triggers"].append("long_conversation")
        
        recent_user = user_messages[-3:] if len(user_messages) >= 3 else user_messages
        exclaim_count = sum(1 for m in recent_user if "!" in m or "??" in m)
        if exclaim_count >= 2:
            result["frustration_boost"] += 1
            result["triggers"].append("punctuation_escalation")
        
        return result
    
    def _determine_escalation(self, frustration: int, urgency: int) -> EscalationLevel:
        """Determine escalation level based on scores"""
        total = frustration + urgency
        
        if total >= 5 or urgency >= 4:
            return EscalationLevel.HIGH
        elif total >= 3 or urgency >= 2:
            return EscalationLevel.MEDIUM
        elif frustration >= 2:
            return EscalationLevel.LOW
        else:
            return EscalationLevel.NONE
    
    def _detect_intent(self, message: str) -> UserIntent:
        """Detect primary user intent"""
        best_intent = UserIntent.GENERAL_INFO
        best_score = 0
        
        for intent, patterns in self.INTENT_PATTERNS.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, message, re.IGNORECASE):
                    score += 1
            if score > best_score:
                best_score = score
                best_intent = intent
        
        if best_score == 0:
            if "?" in message:
                return UserIntent.POLICY_QUESTION
            elif len(message.split()) < 5:
                return UserIntent.GREETING
            else:
                return UserIntent.GENERAL_INFO
        
        return best_intent
    
    def _detect_secondary_intents(self, message: str, primary: UserIntent) -> List[UserIntent]:
        """Detect secondary intents beyond the primary"""
        secondary = []
        
        for intent, patterns in self.INTENT_PATTERNS.items():
            if intent == primary:
                continue
            for pattern in patterns:
                if re.search(pattern, message, re.IGNORECASE):
                    secondary.append(intent)
                    break
        
        return secondary[:3]
    
    def _detect_ogilvy_categories(self, message: str) -> List[OgilvyCategory]:
        """Detect Ogilvy/Schwartz value categories"""
        categories = []
        
        for category, patterns in self.OGILVY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, message, re.IGNORECASE):
                    categories.append(category)
                    break
        
        if not categories:
            categories = [OgilvyCategory.BENEVOLENCE]
        
        return categories[:3]
    
    def _check_vagueness(self, message: str) -> Tuple[bool, Optional[str]]:
        """Check if message is too vague"""
        for pattern, reason in self.VAGUENESS_PATTERNS:
            if re.search(pattern, message, re.IGNORECASE):
                return True, reason
        return False, None
    
    def _suggest_tone(
        self, 
        escalation: EscalationLevel, 
        intent: UserIntent,
        categories: List[OgilvyCategory]
    ) -> str:
        """Suggest appropriate response tone based on analysis"""
        if escalation == EscalationLevel.HIGH:
            return "empathetic_urgent"
        elif escalation == EscalationLevel.MEDIUM:
            return "empathetic_patient"
        elif escalation == EscalationLevel.LOW:
            return "warm_helpful"
        
        if intent == UserIntent.SCRIPTURE:
            return "reverent"
        elif intent in [UserIntent.VOLUNTEER, UserIntent.DONATE]:
            return "enthusiastic"
        elif intent == UserIntent.COMPARISON:
            return "balanced"
        elif intent == UserIntent.VERIFICATION:
            return "factual"
        
        if OgilvyCategory.TRADITION in categories:
            return "respectful"
        elif OgilvyCategory.SECURITY in categories:
            return "reassuring"
        elif OgilvyCategory.UNIVERSALISM in categories:
            return "inclusive"
        
        return "friendly"


prequalifier = Prequalifier()
