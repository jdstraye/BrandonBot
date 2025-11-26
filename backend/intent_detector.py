"""
Intent Detection Module

Detects the underlying user intent beyond the surface question.
This is separate from question type - intent captures WHAT the user truly wants to know.

Example:
- Question: "How will you pay for that?"
- Question Type: policy
- Intent: funding_sources (they want to know WHERE the money comes from)

- Question: "Is that really true?"
- Question Type: truth_seeking  
- Intent: verification (they want proof/sources)
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class UserIntent(Enum):
    FUNDING_SOURCES = "funding_sources"
    COST_EFFECTIVENESS = "cost_effectiveness"
    VERIFICATION = "verification"
    COMPARISON = "comparison"
    PERSONAL_VALUES = "personal_values"
    PRACTICAL_IMPACT = "practical_impact"
    TIMELINE = "timeline"
    OPPOSITION_RESPONSE = "opposition_response"
    VOLUNTEER = "volunteer"
    DONATE = "donate"
    CONTACT = "contact"
    SCRIPTURE = "scripture"
    GENERAL_INFO = "general_info"
    CLARIFICATION = "clarification"
    

@dataclass
class IntentResult:
    primary_intent: UserIntent
    secondary_intents: List[UserIntent]
    confidence: float
    triggers: List[str]
    needs_scripture: bool = False
    needs_callback: bool = False


class IntentDetector:
    """Detects user intent from questions and conversation context"""
    
    FUNDING_TRIGGERS = [
        r"how (will|would|do|does|can) (you|he|she|they|brandon) pay",
        r"where('s| is| will) the money",
        r"who('s| is) (paying|funding)",
        r"cost to taxpayers",
        r"funded (by|through)",
        r"budget (for|from)",
        r"taxpayer (money|dollars|funds)",
        r"fiscal responsibility",
    ]
    
    COST_EFFECTIVENESS_TRIGGERS = [
        r"worth (it|the cost)",
        r"cost.?(effective|efficient)",
        r"(save|waste) money",
        r"return on investment",
        r"bang for (the|your) buck",
        r"affordable",
        r"economic impact",
    ]
    
    VERIFICATION_TRIGGERS = [
        r"(is|that's|this) (really|actually) true",
        r"proof|evidence|source",
        r"can you (prove|verify|confirm)",
        r"where did you (get|hear|find)",
        r"fact.?check",
        r"citation|reference",
        r"according to (who|whom)",
    ]
    
    COMPARISON_TRIGGERS = [
        r"(vs|versus|compared to|different from)",
        r"(democrat|republican|liberal|conservative)",
        r"opponent|other candidate",
        r"(better|worse) than",
        r"what makes you different",
        r"why (you|brandon) (instead|over)",
    ]
    
    PERSONAL_VALUES_TRIGGERS = [
        r"(why|what) do you (believe|think|feel)",
        r"personal (opinion|view|belief|faith)",
        r"as a (christian|person|father|man)",
        r"moral|ethical|values",
        r"heart|soul|spirit",
        r"god|faith|prayer|bible",
        r"conviction",
    ]
    
    PRACTICAL_IMPACT_TRIGGERS = [
        r"how (will|does|would) this affect (me|us|my)",
        r"what (does|will) this mean for",
        r"impact on (my|our|the)",
        r"in practice",
        r"real.?world",
        r"day.?to.?day",
        r"my (taxes|healthcare|family|business)",
    ]
    
    TIMELINE_TRIGGERS = [
        r"(when|how soon|how long)",
        r"(first|second) (day|week|month|year|term)",
        r"immediately|right away",
        r"timeline|schedule|plan",
        r"priority|order",
    ]
    
    OPPOSITION_TRIGGERS = [
        r"(respond|answer) to (criticism|critics|attacks)",
        r"what about (the|your) (opposition|critics)",
        r"they say|people say|some say",
        r"(accusation|allegation|claim) (that|about)",
        r"attack ads",
        r"defend|rebuttal|response to",
    ]
    
    VOLUNTEER_TRIGGERS = [
        r"(want|like|interested) to (help|volunteer)",
        r"sign.?up|join|get involved",
        r"volunteer|canvass|door.?knock",
        r"campaign (event|office)",
        r"how can i help",
    ]
    
    DONATE_TRIGGERS = [
        r"(want|like) to (give|donate|contribute)",
        r"donation|contribution",
        r"financial support|support (you|the campaign)",
        r"accept (donations|contributions)",
    ]
    
    CONTACT_TRIGGERS = [
        r"(contact|reach|email|call) (you|brandon|campaign)",
        r"get in touch",
        r"phone number|email address",
        r"office (hours|location)",
        r"meet (you|brandon|him)",
    ]
    
    SCRIPTURE_TRIGGERS = [
        r"(what|where) does (the )?bible say",
        r"scripture|verse|biblical",
        r"god('s)? (word|plan|will)",
        r"christian perspective",
        r"(spiritual|faith).?based (answer|response)",
        r"pray|prayer",
        r"jesus|christ|lord",
    ]
    
    VALUES_INTENT_MAP = {
        "meaning", "purpose", "truth", "right", "wrong", "moral", "immoral",
        "good", "evil", "justice", "freedom", "liberty", "dignity", "life",
        "family", "marriage", "faith", "hope", "love", "compassion",
    }
    
    def detect(self, question: str, conversation_history: List[Dict] = None) -> IntentResult:
        """
        Detect the user's underlying intent from their question.
        
        Args:
            question: The user's question
            conversation_history: Optional list of previous turns
            
        Returns:
            IntentResult with primary intent, secondary intents, and metadata
        """
        question_lower = question.lower()
        intents = []
        triggers_found = []
        
        intent_checks = [
            (UserIntent.FUNDING_SOURCES, self.FUNDING_TRIGGERS),
            (UserIntent.COST_EFFECTIVENESS, self.COST_EFFECTIVENESS_TRIGGERS),
            (UserIntent.VERIFICATION, self.VERIFICATION_TRIGGERS),
            (UserIntent.COMPARISON, self.COMPARISON_TRIGGERS),
            (UserIntent.PERSONAL_VALUES, self.PERSONAL_VALUES_TRIGGERS),
            (UserIntent.PRACTICAL_IMPACT, self.PRACTICAL_IMPACT_TRIGGERS),
            (UserIntent.TIMELINE, self.TIMELINE_TRIGGERS),
            (UserIntent.OPPOSITION_RESPONSE, self.OPPOSITION_TRIGGERS),
            (UserIntent.VOLUNTEER, self.VOLUNTEER_TRIGGERS),
            (UserIntent.DONATE, self.DONATE_TRIGGERS),
            (UserIntent.CONTACT, self.CONTACT_TRIGGERS),
            (UserIntent.SCRIPTURE, self.SCRIPTURE_TRIGGERS),
        ]
        
        for intent, triggers in intent_checks:
            for pattern in triggers:
                if re.search(pattern, question_lower):
                    intents.append(intent)
                    triggers_found.append(pattern)
                    break
        
        for word in self.VALUES_INTENT_MAP:
            if re.search(rf"\b{word}\b", question_lower):
                if UserIntent.PERSONAL_VALUES not in intents:
                    intents.append(UserIntent.PERSONAL_VALUES)
                    triggers_found.append(f"value_word:{word}")
                break
        
        if self._is_clarification(question, conversation_history):
            intents.insert(0, UserIntent.CLARIFICATION)
            triggers_found.insert(0, "clarification_detected")
        
        if not intents:
            intents.append(UserIntent.GENERAL_INFO)
        
        primary_intent = intents[0]
        secondary_intents = intents[1:] if len(intents) > 1 else []
        
        needs_scripture = (
            UserIntent.SCRIPTURE in intents or
            (UserIntent.PERSONAL_VALUES in intents and 
             any(w in question_lower for w in ["faith", "god", "pray", "bible", "christian"]))
        )
        
        needs_callback = self._check_callback_need(question, conversation_history)
        
        confidence = min(0.9, 0.5 + 0.2 * len(triggers_found))
        
        logger.debug(f"Intent detected: {primary_intent.value}, triggers: {triggers_found}, "
                    f"needs_scripture: {needs_scripture}")
        
        return IntentResult(
            primary_intent=primary_intent,
            secondary_intents=secondary_intents,
            confidence=confidence,
            triggers=triggers_found,
            needs_scripture=needs_scripture,
            needs_callback=needs_callback
        )
    
    def _is_clarification(self, question: str, history: List[Dict] = None) -> bool:
        """Check if question is asking for clarification of previous response"""
        clarification_patterns = [
            r"what do you mean",
            r"can you (explain|clarify|elaborate)",
            r"i don't understand",
            r"huh\?|what\?",
            r"be more specific",
            r"in other words",
        ]
        
        question_lower = question.lower()
        for pattern in clarification_patterns:
            if re.search(pattern, question_lower):
                return True
        
        if history and len(history) > 0:
            short_followup = len(question.split()) < 8
            has_pronoun_reference = any(p in question_lower for p in ["that", "this", "it", "those", "these"])
            if short_followup and has_pronoun_reference:
                return True
        
        return False
    
    def _check_callback_need(self, question: str, history: List[Dict] = None) -> bool:
        """Check if conversation suggests user needs a callback"""
        callback_signals = [
            r"(can|could) (someone|somebody|you) call me",
            r"(need|want) to (talk|speak) (to|with)",
            r"(prefer|rather) (talk|speak|phone)",
            r"complicated|confusing|hard to understand",
            r"this is (important|urgent|personal)",
        ]
        
        question_lower = question.lower()
        for pattern in callback_signals:
            if re.search(pattern, question_lower):
                return True
        
        return False
    
    def get_intent_context(self, intent_result: IntentResult) -> str:
        """Generate context string for LLM based on detected intent"""
        context_parts = []
        
        intent_guidance = {
            UserIntent.FUNDING_SOURCES: "User wants to know WHERE funding comes from. Be specific about revenue sources and budget allocations.",
            UserIntent.COST_EFFECTIVENESS: "User is evaluating value/efficiency. Provide ROI data, savings estimates, or comparative costs.",
            UserIntent.VERIFICATION: "User wants proof/evidence. Cite specific sources, documents, or verifiable facts.",
            UserIntent.COMPARISON: "User wants to compare positions. Be fair but highlight Brandon's unique stance.",
            UserIntent.PERSONAL_VALUES: "User is interested in values/beliefs. Can share personal perspective authentically.",
            UserIntent.PRACTICAL_IMPACT: "User wants to know real-world effects. Give concrete, personal examples.",
            UserIntent.TIMELINE: "User wants timing info. Be specific about priorities and deadlines.",
            UserIntent.OPPOSITION_RESPONSE: "User mentions criticism. Address directly but pivot to positive vision.",
            UserIntent.VOLUNTEER: "User wants to help! Gather contact info and interests.",
            UserIntent.DONATE: "User wants to contribute. Provide donation info (FEC compliant).",
            UserIntent.CONTACT: "User wants to reach campaign. Provide contact details.",
            UserIntent.SCRIPTURE: "User wants faith perspective. Can include relevant scripture if appropriate.",
            UserIntent.CLARIFICATION: "User needs previous point clarified. Rephrase and provide more detail.",
            UserIntent.GENERAL_INFO: "General information request. Provide clear, comprehensive answer.",
        }
        
        primary_guidance = intent_guidance.get(intent_result.primary_intent, "")
        if primary_guidance:
            context_parts.append(f"Primary intent: {primary_guidance}")
        
        if intent_result.needs_scripture:
            context_parts.append("Scripture context may be appropriate for this response.")
        
        if intent_result.needs_callback:
            context_parts.append("User may benefit from a personal callback - offer this option.")
        
        return " ".join(context_parts)


intent_detector = IntentDetector()


@dataclass
class EscalationResult:
    needs_escalation: bool
    escalation_level: str
    triggers: List[str]
    suggested_response: Optional[str] = None


class EscalationDetector:
    """Detects conversation patterns indicating user frustration or need for human contact"""
    
    FRUSTRATION_INDICATORS = [
        r"(this is|you('re| are)) (useless|stupid|broken|not helping)",
        r"(i('m| am)|still) (confused|lost|not getting)",
        r"(already|just) (said|asked|told you)",
        r"(doesn't|don't|didn't) (answer|help|make sense)",
        r"(can't|cannot) understand",
        r"(what|this) is (wrong|the problem)",
        r"forget it|never ?mind",
        r"(ugh|argh|OMG|FFS|WTF)",
        r"!{2,}",
        r"\?{2,}",
    ]
    
    URGENCY_INDICATORS = [
        r"(need|want) (to talk|speak) (to|with) (a|someone|human|person|real)",
        r"(urgent|emergency|asap|right now|immediately)",
        r"(critical|important|serious|major)",
        r"(deadline|time.?sensitive)",
        r"(can't|cannot) wait",
        r"(need|require) (help|assistance) (now|today)",
    ]
    
    POLITENESS_EROSION = [
        r"^(just|so|look|okay|fine|whatever)",
        r"(please|thanks|thank you)[\.\!]?$",
        r"(i guess|if you must|fine then)",
    ]
    
    REPEATED_PATTERNS = [
        "asked", "said", "told", "mentioned", "explained",
    ]
    
    def detect(self, current_message: str, conversation_history: List[Dict] = None) -> EscalationResult:
        """
        Detect if user shows signs of frustration or needs escalation.
        
        Args:
            current_message: Current user message
            conversation_history: Previous turns in conversation
            
        Returns:
            EscalationResult with escalation level and suggested action
        """
        message_lower = current_message.lower()
        triggers = []
        frustration_score = 0
        urgency_score = 0
        
        for pattern in self.FRUSTRATION_INDICATORS:
            if re.search(pattern, message_lower, re.IGNORECASE):
                frustration_score += 2
                triggers.append(f"frustration:{pattern[:20]}")
        
        for pattern in self.URGENCY_INDICATORS:
            if re.search(pattern, message_lower, re.IGNORECASE):
                urgency_score += 2
                triggers.append(f"urgency:{pattern[:20]}")
        
        for pattern in self.POLITENESS_EROSION:
            if re.search(pattern, message_lower, re.IGNORECASE):
                frustration_score += 1
                triggers.append(f"politeness:{pattern[:15]}")
        
        if conversation_history:
            history_context = self._analyze_history(conversation_history, current_message)
            frustration_score += history_context.get("frustration_boost", 0)
            triggers.extend(history_context.get("triggers", []))
        
        total_score = frustration_score + urgency_score
        
        if total_score >= 5 or urgency_score >= 3:
            return EscalationResult(
                needs_escalation=True,
                escalation_level="high",
                triggers=triggers,
                suggested_response="I understand this is important to you. Would you like someone from Brandon's team to give you a call directly? They can provide personalized assistance."
            )
        elif total_score >= 3:
            return EscalationResult(
                needs_escalation=True,
                escalation_level="medium",
                triggers=triggers,
                suggested_response="I want to make sure I'm answering your question properly. Would it help to speak with someone from the campaign team?"
            )
        elif frustration_score >= 2:
            return EscalationResult(
                needs_escalation=False,
                escalation_level="low",
                triggers=triggers,
                suggested_response=None
            )
        else:
            return EscalationResult(
                needs_escalation=False,
                escalation_level="none",
                triggers=[],
                suggested_response=None
            )
    
    def _analyze_history(self, history: List[Dict], current: str) -> Dict:
        """Analyze conversation history for escalation patterns"""
        result = {"frustration_boost": 0, "triggers": []}
        
        if len(history) < 2:
            return result
        
        user_messages = [m["content"].lower() for m in history if m.get("role") == "user"]
        
        current_lower = current.lower()
        for word in self.REPEATED_PATTERNS:
            if word in current_lower and f"already {word}" in current_lower:
                result["frustration_boost"] += 2
                result["triggers"].append("repetition_complaint")
                break
        
        if len(history) >= 6:
            user_count = len(user_messages)
            if user_count >= 4:
                result["frustration_boost"] += 1
                result["triggers"].append("long_conversation")
        
        recent_user = user_messages[-3:] if len(user_messages) >= 3 else user_messages
        exclaim_count = sum(1 for m in recent_user if "!" in m or "?" * 2 in m)
        if exclaim_count >= 2:
            result["frustration_boost"] += 1
            result["triggers"].append("punctuation_escalation")
        
        return result


escalation_detector = EscalationDetector()
