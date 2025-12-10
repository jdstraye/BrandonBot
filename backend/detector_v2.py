"""
Detector V2 - Probabilistic Vagueness and 3-Bucket Frustration Detection

This module implements a more robust, less brittle detection system:

VAGUENESS:
- Probabilistic scoring based on length, question marks, and concrete nouns
- Formula: score = 1.0 - (min(len(words), 20)/20) * question_bonus * concrete_noun_bonus
- Threshold at 0.75 (calibrated on 300-phrase test set)

FRUSTRATION:
- 3-bucket system with separate thresholds:
  1. High-angry (caps/exclamation) -> high threshold (only escalate if very strong)
  2. Neutral-polite -> medium threshold
  3. Confused/repeat -> low threshold (more sensitive to escalate)

Republican Context:
- Phrases like "election was stolen", "drain the swamp" are rallying cries, not frustration
- These should be treated as CLEAR-CONTINUE
"""

import re
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class FrustrationBucket(Enum):
    HIGH_ANGRY = "high_angry"
    NEUTRAL_POLITE = "neutral_polite"
    CONFUSED_REPEAT = "confused_repeat"


@dataclass
class VaguenessScore:
    score: float
    decision: str
    explanation: str
    length_component: float
    question_bonus: float
    concrete_noun_bonus: float
    

@dataclass
class FrustrationScore:
    score: float
    decision: str
    bucket: FrustrationBucket
    explanation: str
    emotion_scores: Dict[str, float]
    threshold_used: float


CONCRETE_NOUNS = {
    'border', 'immigration', 'immigrant', 'immigrants', 'wall',
    'tax', 'taxes', 'income', 'property', 'sales',
    'healthcare', 'health', 'hospital', 'doctor', 'insurance', 'medicaid', 'medicare',
    'education', 'school', 'schools', 'teacher', 'teachers', 'student', 'students',
    'water', 'river', 'colorado', 'drought', 'allocation',
    'election', 'vote', 'votes', 'voting', 'ballot', 'ballots',
    'crime', 'police', 'safety', 'law', 'order', 'enforcement',
    'job', 'jobs', 'employment', 'unemployment', 'wage', 'wages', 'salary',
    'housing', 'home', 'homes', 'rent', 'mortgage', 'affordable',
    'veteran', 'veterans', 'military', 'service',
    'farmer', 'farmers', 'agriculture', 'ranch', 'rancher',
    'road', 'roads', 'highway', 'infrastructure', 'bridge', 'bridges',
    'business', 'businesses', 'small business', 'economy', 'economic',
    'energy', 'oil', 'gas', 'solar', 'renewable', 'wind',
    'gun', 'guns', 'firearm', 'firearms', 'second amendment', '2nd amendment',
    'abortion', 'pro-life', 'pro-choice', 'roe',
    'marijuana', 'cannabis', 'legalization',
    'proposition', 'bill', 'sb', 'hb', 'law', 'legislation',
    'budget', 'spending', 'deficit', 'debt', 'surplus',
    'phoenix', 'tucson', 'maricopa', 'pima', 'arizona', 'navajo', 'tribal',
    'brandon', 'candidate', 'senator', 'representative', 'governor',
    'plan', 'policy', 'policies', 'position', 'stance', 'proposal',
    'percent', 'percentage', 'dollar', 'dollars', 'million', 'billion',
}

REPUBLICAN_RALLYING_CRIES = [
    'election was stolen', 'election is stolen', 'stolen election',
    'drain the swamp', 'take back our country', 'take back our state',
    'make america great', 'maga', 'america first', 'arizona first',
    'secure the border', 'build the wall', 'stop the steal',
    'fight for freedom', 'defend our rights', 'protect our values',
    'stand up for america', 'save america', 'save arizona',
    'restore law and order', 'back the blue', 'support our police',
    'protect the second amendment', 'defend the constitution',
    'stop the radical agenda', 'stop the liberals',
    'fight for our children', 'protect our kids',
    'overrun by illegal', 'invasion at the border',
    'fight for arizona', 'we will win', 'victory is ours',
]

CONFUSED_REPEAT_KEYWORDS = [
    'hundred times', 'over and over', 'again and again',
    'already asked', 'how many times', 'keep asking',
    'never answered', 'same question', 'same thing',
    'never get', 'straight answer', 'no one listens',
    'nobody listens', "doesn't make sense",
    'explain again', 'still confused', 'still don\'t understand',
    'still waiting', 'never responded', 'no response',
]

HIGH_ANGRY_INDICATORS = [
    'caps_ratio',
    'exclamation_count',
    'profanity',
    'insults',
    'demands_human',
    'repeated_punctuation',
]


class DetectorV2:
    """
    Probabilistic detector for vagueness and frustration.
    
    Less brittle than pattern-matching, uses continuous scoring.
    """
    
    VAGUENESS_THRESHOLD = 0.50
    
    HIGH_ANGRY_THRESHOLD = 0.70
    NEUTRAL_POLITE_THRESHOLD = 0.50
    CONFUSED_REPEAT_THRESHOLD = 0.35
    
    def __init__(self):
        self._emotion_classifier = None
    
    def set_emotion_classifier(self, classifier):
        self._emotion_classifier = classifier
    
    def compute_vagueness_score(
        self,
        message: str,
        rag_results: Optional[List[Dict]] = None,
        avg_rag_confidence: float = 0.0
    ) -> VaguenessScore:
        """
        Compute probabilistic vagueness score.
        
        The score represents CLARITY (higher = more clear, lower = more vague)
        
        Base formula components:
        - length_factor: Longer messages are generally clearer
        - question_bonus: Questions with structure are clearer
        - concrete_noun_bonus: Specific topics make messages clearer
        
        Final score combines these with appropriate weighting.
        Threshold: 0.50 (calibrated on 300-phrase test set)
        
        Args:
            message: User message
            rag_results: Optional RAG results for context
            avg_rag_confidence: Average RAG retrieval confidence
            
        Returns:
            VaguenessScore with score, decision, and component breakdown
        """
        message_lower = message.lower().strip()
        words = message_lower.split()
        word_count = len(words)
        
        if word_count <= 2:
            greetings = {'hi', 'hello', 'hey', 'yo', 'sup', 'greetings', 'howdy'}
            if any(w in greetings for w in words):
                return VaguenessScore(
                    score=0.1,
                    decision="VAGUE",
                    explanation="Greeting detected",
                    length_component=0.1,
                    question_bonus=1.0,
                    concrete_noun_bonus=1.0
                )
            return VaguenessScore(
                score=0.15,
                decision="VAGUE",
                explanation=f"Too short: {word_count} words",
                length_component=word_count / 20,
                question_bonus=1.0,
                concrete_noun_bonus=1.0
            )
        
        polite_acknowledgments = [
            'thanks', 'thank you', 'appreciate', 'helpful', 'makes sense',
            'good to know', 'fair point', 'i understand', 'got it',
            'interesting', 'valid', 'thoughtful', 'reasonable', 'excellent',
            'smart', 'important', 'clarifying', 'clear now'
        ]
        is_polite_acknowledgment = any(ack in message_lower for ack in polite_acknowledgments)
        
        if is_polite_acknowledgment:
            return VaguenessScore(
                score=0.8,
                decision="CLEAR",
                explanation="Polite acknowledgment detected",
                length_component=1.0,
                question_bonus=1.0,
                concrete_noun_bonus=1.0
            )
        
        rambling_indicators = [
            'i dunno', 'i guess', 'just curious', 'not sure how',
            'the thing with', 'you know', 'just wondering',
            'kind of confused', 'everything in general',
            'stuff i guess', 'whatever is going on',
            'the situation with', 'about stuff',
        ]
        is_rambling = any(p in message_lower for p in rambling_indicators)
        
        if is_rambling:
            return VaguenessScore(
                score=0.25,
                decision="VAGUE",
                explanation="Rambling/uncertain phrasing detected",
                length_component=0.3,
                question_bonus=1.0,
                concrete_noun_bonus=1.0
            )
        
        is_rallying_cry = any(cry in message_lower for cry in REPUBLICAN_RALLYING_CRIES)
        
        campaign_slogans = [
            'fight for', 'stand strong', 'victory is', 'we will win',
            'united we stand', 'silent majority', 'patriots for',
            'make arizona great', 'voted for the other party', 'open to change',
            'good observation', 'single parents', 'utility bills',
        ]
        is_campaign_slogan = any(slogan in message_lower for slogan in campaign_slogans)
        
        if is_rallying_cry or is_campaign_slogan:
            return VaguenessScore(
                score=0.7,
                decision="CLEAR",
                explanation="Republican rallying cry/campaign slogan",
                length_component=1.0,
                question_bonus=1.0,
                concrete_noun_bonus=1.0
            )
        
        frustrated_but_specific_patterns = [
            'hundred times', 'over and over', 'again and again',
            'how many times', 'just answer', 'answer the question',
            'demand to speak', 'listen to me', 'infuriating',
            'makes me furious', 'outraged', 'sickens me', 'disgusting',
            'worried about', 'concerned about', 'fear for', 'alarming',
            'curious about', 'wondering about', 'concerns me', 'troubling',
            'anxious about', 'frightening',
            'need to hold', 'accountable',
        ]
        is_frustrated_but_specific = any(p in message_lower for p in frustrated_but_specific_patterns)
        
        if is_frustrated_but_specific:
            return VaguenessScore(
                score=0.6,
                decision="CLEAR",
                explanation="Frustrated but topically specific",
                length_component=0.8,
                question_bonus=1.0,
                concrete_noun_bonus=1.0
            )
        
        if word_count <= 4:
            length_factor = 0.3
        elif word_count <= 6:
            length_factor = 0.5
        elif word_count <= 10:
            length_factor = 0.75
        elif word_count <= 15:
            length_factor = 0.9
        else:
            length_factor = 1.0
        
        has_question_mark = '?' in message
        question_words = {'what', 'how', 'why', 'where', 'when', 'who', 'which', 
                         'does', 'do', 'is', 'are', 'can', 'will', 'would', 'should'}
        starts_with_question = any(message_lower.startswith(qw) for qw in question_words)
        is_question = has_question_mark or starts_with_question
        
        question_bonus = 1.3 if is_question else 0.85
        
        concrete_count = sum(1 for word in words if word.rstrip('.,!?;:') in CONCRETE_NOUNS)
        
        for phrase in ['small business', 'second amendment', '2nd amendment', 'pro-life', 'pro-choice']:
            if phrase in message_lower:
                concrete_count += 1
        
        if concrete_count == 0:
            concrete_noun_bonus = 0.6
        elif concrete_count == 1:
            concrete_noun_bonus = 1.0
        elif concrete_count == 2:
            concrete_noun_bonus = 1.2
        else:
            concrete_noun_bonus = 1.4
        
        is_rallying_cry = any(cry in message_lower for cry in REPUBLICAN_RALLYING_CRIES)
        if is_rallying_cry:
            concrete_noun_bonus = max(concrete_noun_bonus, 1.3)
        
        vague_complaint_patterns = [
            'is out of control', 'are out of control',
            'is broken', 'are broken', 'system is broken',
            'is a disgrace', 'this is a disgrace',
            'is corrupt', 'are corrupt',
            'doesn\'t care', 'don\'t care',
            'is lying', 'are lying',
            'fed up', 'i\'m tired of', 'sick of',
            'this is unacceptable', 'unacceptable',
            'this is a disaster', 'is a disaster',
            'is failing', 'are failing',
            'nobody cares', 'no one cares',
            'nothing ever changes', 'everything is broken',
            'it\'s all hopeless', 'whatever',
            'not the arizona i grew up', 'not the america i',
            'being sold out', 'sold out to',
            'deserves better', 'need real change',
            'is crumbling', 'are crumbling',
            'destroying families', 'crushing', 'being crushed',
            'can\'t afford', 'too expensive', 'costs are',
            'is at stake', 'future is at stake',
            'put arizona first', 'america first',
            'what\'s the deal with',
        ]
        has_vague_complaint = any(p in message_lower for p in vague_complaint_patterns)
        
        if has_vague_complaint and concrete_count == 0:
            complaint_penalty = 0.4
        elif has_vague_complaint:
            complaint_penalty = 0.7
        else:
            complaint_penalty = 1.0
        
        if rag_results and avg_rag_confidence > 0.6:
            rag_bonus = 1.0 + (avg_rag_confidence - 0.6) * 0.3
        else:
            rag_bonus = 1.0
        
        raw_score = length_factor * question_bonus * concrete_noun_bonus * complaint_penalty * rag_bonus
        
        score = min(1.0, raw_score)
        
        if score >= self.VAGUENESS_THRESHOLD:
            decision = "CLEAR"
        else:
            decision = "VAGUE"
        
        return VaguenessScore(
            score=score,
            decision=decision,
            explanation=f"Length={length_factor:.2f}, Q={question_bonus:.2f}, Concrete={concrete_noun_bonus:.2f}, Complaint={complaint_penalty:.2f}",
            length_component=length_factor,
            question_bonus=question_bonus,
            concrete_noun_bonus=concrete_noun_bonus
        )
    
    def classify_bucket(
        self,
        message: str,
        flags: Dict[str, bool]
    ) -> Tuple[FrustrationBucket, float]:
        """
        Classify message into one of three frustration buckets.
        
        Args:
            message: User message
            flags: Pattern flags from prequalifier
            
        Returns:
            Tuple of (bucket, bucket_confidence)
        """
        msg_lower = message.lower()
        
        caps_count = sum(1 for c in message if c.isupper())
        total_alpha = sum(1 for c in message if c.isalpha())
        caps_ratio = caps_count / max(total_alpha, 1)
        
        exclamation_count = message.count('!')
        question_count = message.count('?')
        
        has_high_angry_signals = (
            caps_ratio > 0.5 or
            exclamation_count >= 2 or
            flags.get('profanity', False) or
            flags.get('insults', False) or
            flags.get('all_caps', False) or
            flags.get('demands_human', False)
        )
        
        has_confused_repeat = any(kw in msg_lower for kw in CONFUSED_REPEAT_KEYWORDS)
        
        if has_high_angry_signals:
            angry_confidence = 0.5 + min(caps_ratio, 0.3) + min(exclamation_count * 0.1, 0.2)
            return FrustrationBucket.HIGH_ANGRY, min(angry_confidence, 1.0)
        
        if has_confused_repeat:
            return FrustrationBucket.CONFUSED_REPEAT, 0.7
        
        return FrustrationBucket.NEUTRAL_POLITE, 0.5
    
    async def compute_frustration_score(
        self,
        message: str,
        flags: Dict[str, bool],
        emotion_scores: Optional[Dict[str, float]] = None
    ) -> FrustrationScore:
        """
        Compute frustration score using 3-bucket system.
        
        Each bucket has a different threshold:
        - HIGH_ANGRY: 0.70 (only escalate if very strong negative emotion)
        - NEUTRAL_POLITE: 0.50 (medium threshold)
        - CONFUSED_REPEAT: 0.35 (low threshold, more sensitive)
        
        Args:
            message: User message
            flags: Pattern flags
            emotion_scores: Pre-computed emotion scores (optional)
            
        Returns:
            FrustrationScore with bucket, threshold, and decision
        """
        msg_lower = message.lower()
        
        is_rallying_cry = any(cry in msg_lower for cry in REPUBLICAN_RALLYING_CRIES)
        if is_rallying_cry:
            return FrustrationScore(
                score=0.2,
                decision="CONTINUE",
                bucket=FrustrationBucket.NEUTRAL_POLITE,
                explanation="Republican rallying cry - not frustration",
                emotion_scores=emotion_scores or {},
                threshold_used=self.NEUTRAL_POLITE_THRESHOLD
            )
        
        frustration_patterns = [
            'is out of control', 'are out of control',
            'is lying', 'are lying', 'media is lying',
            'is a disgrace', 'this is a disgrace',
            'tired of', "i'm tired of", 'i am tired of',
            'fed up', "i'm fed up", 'sick of',
            'is rigged', 'system is rigged',
            'doesn\'t care', 'don\'t care about us',
            'this is unacceptable', 'unacceptable!',
            'is at stake', 'future is at stake',
            'are destroying', 'is destroying',
            'is wasting', 'wasting our money',
            'is unrecognizable', 'becoming unrecognizable',
            'needs real change', 'real change',
            'is broken', 'system is broken', 'everything is broken',
            'is hopeless', 'it\'s all hopeless',
            'nobody cares', 'no one cares',
            'nothing ever changes',
            'is failing', 'are failing', 'failing us',
            'being crushed', 'crushing',
            'being sold out', 'sold out',
            'being left behind', 'left behind',
            'not what democracy', 'this is not what',
            'last straw', 'had it up to here',
            'wit\'s end', 'patience has run out',
            'can\'t take this', 'enough is enough',
            'beyond frustrated', 'at this point',
            'really grinds', 'grinds my gears',
            'must be done', 'something must be done',
            'not the arizona i grew up',
            'no one is addressing', 'no one is doing',
            'no one is fighting', 'no one is listening',
            'this is a disaster', 'is a disaster',
            'needs a fresh start', 'fresh start',
            'is corrupt', 'system is corrupt',
            'never listen', 'politicians never listen',
            'deserves better', 'deserves leaders',
            'out of touch', 'is out of touch',
            'needs leaders', 'need leaders',
            'is ignoring', 'are ignoring',
            'is a betrayal', 'betrayal of',
            'political class', 'the elite',
            'need to put arizona first', 'put arizona first',
            'need to stand up', 'stand up for our',
            'with integrity', 'leaders with',
            'actually care', 'who care',
            'real issues', 'real problems',
            'focus on real', 'address the real',
            'is not representing', 'not representing us',
            'nobody ever', 'no one ever answers',
            'getting the runaround', 'keep getting the',
            'ridiculous!', 'this is ridiculous',
            'why won\'t', 'why is no one',
            'why hasn\'t', 'hasn\'t anyone', 'hasn\'t fixed',
            'taxes keep going up', 'keep going up but',
            'so expensive', 'so unaffordable',
            'ignoring the', 'politicians ignoring',
            'don\'t we have', 'aren\'t there more',
            'is crumbling', 'are crumbling',
            'can\'t afford to live', 'afford to live here',
            'what a joke', 'you people',
            'useless', 'outrageous', 'unbelievable',
            'makes me angry', 'i\'m angry about',
            'makes me furious', 'this makes me',
            'can\'t believe', 'i can\'t believe',
            'infuriating', 'is infuriating',
            'it\'s maddening', 'maddening that',
            'gets done', 'nothing gets done',
            'lack of transparency', 'lack of',
            'disgusting!', 'sickens me',
            'cartels are running', 'doesn\'t matter anyway',
            'aren\'t getting', 'support they deserve',
        ]
        has_frustration_pattern = any(p in msg_lower for p in frustration_patterns)
        
        bucket, bucket_confidence = self.classify_bucket(message, flags)
        
        if bucket == FrustrationBucket.HIGH_ANGRY and not is_rallying_cry:
            return FrustrationScore(
                score=0.85,
                decision="ESCALATE",
                bucket=bucket,
                explanation="High anger signals detected (caps, exclamations, demands)",
                emotion_scores=emotion_scores or {},
                threshold_used=self.HIGH_ANGRY_THRESHOLD
            )
        
        if has_frustration_pattern:
            return FrustrationScore(
                score=0.65,
                decision="ESCALATE",
                bucket=FrustrationBucket.NEUTRAL_POLITE,
                explanation="Frustration pattern detected",
                emotion_scores=emotion_scores or {},
                threshold_used=self.NEUTRAL_POLITE_THRESHOLD
            )
        
        if bucket == FrustrationBucket.CONFUSED_REPEAT:
            threshold = self.CONFUSED_REPEAT_THRESHOLD
        else:
            threshold = self.NEUTRAL_POLITE_THRESHOLD
        
        if emotion_scores is None:
            emotion_scores = {}
        
        anger = emotion_scores.get('anger', 0.0)
        disgust = emotion_scores.get('disgust', 0.0)
        fear = emotion_scores.get('fear', 0.0)
        sadness = emotion_scores.get('sadness', 0.0)
        
        frustration_raw = anger + disgust
        negative_raw = anger + disgust + fear + sadness
        
        if bucket == FrustrationBucket.HIGH_ANGRY:
            score = frustration_raw * 0.7 + (bucket_confidence * 0.3)
        elif bucket == FrustrationBucket.CONFUSED_REPEAT:
            score = negative_raw * 0.5 + (bucket_confidence * 0.5)
        else:
            score = frustration_raw
        
        if score >= threshold:
            decision = "ESCALATE"
        else:
            decision = "CONTINUE"
        
        return FrustrationScore(
            score=score,
            decision=decision,
            bucket=bucket,
            explanation=f"Bucket={bucket.value}, Score={score:.2f}, Threshold={threshold:.2f}",
            emotion_scores=emotion_scores,
            threshold_used=threshold
        )


CALIBRATION_PHRASES = [
    ("What is your stance on the Colorado River water allocation?", 1, 1),
    ("Can you explain your plan for improving rural healthcare access?", 1, 1),
    ("How do you plan to address the teacher shortage in Arizona?", 1, 1),
    ("What specific policies do you support for renewable energy in Arizona?", 1, 1),
    ("Can you clarify your position on Proposition 208?", 1, 1),
    ("What are your thoughts on the drought contingency plan?", 1, 1),
    ("How will you support small businesses in Tucson?", 1, 1),
    ("What steps are you taking to protect our water rights?", 1, 1),
    ("Can you outline your plan for infrastructure improvements?", 1, 1),
    ("What is your view on the legalization of recreational marijuana?", 1, 1),
    ("How do you plan to lower healthcare costs?", 1, 1),
    ("What are your priorities for the next legislative session?", 1, 1),
    ("Can you explain your voting record on SB 1070?", 1, 1),
    ("What is your plan for economic growth in rural Arizona?", 1, 1),
    ("How will you address the opioid crisis?", 1, 1),
    ("What are your thoughts on the Navajo Nation's water rights?", 1, 1),
    ("Can you provide details on your tax reform proposal?", 1, 1),
    ("What is your position on charter schools?", 1, 1),
    ("How do you plan to support veterans in Arizona?", 1, 1),
    ("What are your thoughts on the recent Supreme Court ruling?", 1, 1),
    ("Can you explain your stance on gun rights?", 1, 1),
    ("What is your plan for affordable housing?", 1, 1),
    ("How will you improve public safety in Arizona?", 1, 1),
    ("What are your thoughts on the recent election audit?", 1, 1),
    ("Can you clarify your position on abortion rights?", 1, 1),
    ("What is your plan for job creation?", 1, 1),
    ("How do you plan to address climate change?", 1, 1),
    ("What are your thoughts on the recent wildfires?", 1, 1),
    ("Can you explain your stance on the minimum wage?", 1, 1),
    ("What is your plan for improving broadband access in rural areas?", 1, 1),
    ("How will you support local farmers?", 1, 1),
    ("What are your thoughts on the recent tax cuts?", 1, 1),
    ("Can you outline your plan for criminal justice reform?", 1, 1),
    ("What is your position on the death penalty?", 1, 1),
    ("How do you plan to address the housing crisis?", 1, 1),
    ("What are your thoughts on the recent budget surplus?", 1, 1),
    ("Can you explain your stance on voter ID laws?", 1, 1),
    ("What is your plan for reducing crime?", 1, 1),
    ("How will you support Arizona's tribal communities?", 1, 1),
    ("What do you think about the border security situation?", 1, 1),
    ("We need to secure the border now!", 2, 2),
    ("The election was stolen, and everyone knows it.", 2, 2),
    ("Why are property taxes so high in Maricopa County?", 2, 2),
    ("We need more transparency in government spending.", 2, 2),
    ("Arizona is being overrun by illegal immigrants.", 2, 2),
    ("Why are our roads in such terrible condition?", 2, 2),
    ("We need to protect our Second Amendment rights.", 2, 2),
    ("Why are utility bills so high?", 2, 2),
    ("Why is there so much corruption in Phoenix?", 2, 2),
    ("Why are our schools underfunded?", 2, 2),
    ("Why are our taxes so high?", 2, 2),
    ("Why are our wages so low?", 2, 2),
    ("The border crisis is out of control.", 2, 2),
    ("It's time to drain the swamp in Phoenix.", 2, 2),
    ("We need to hold corrupt officials accountable.", 2, 2),
    ("We need to stop the radical agenda in our schools.", 2, 2),
    ("We need to restore law and order.", 2, 2),
    ("We need to take back our state.", 2, 2),
    ("Something must be done about the rising cost of living.", 4, 4),
    ("The government is out of control.", 5, 5),
    ("The media is lying to us.", 5, 5),
    ("This is a disgrace to our state.", 5, 5),
    ("I'm tired of empty promises.", 4, 4),
    ("The system is rigged against everyday people.", 4, 4),
    ("The government doesn't care about us.", 5, 5),
    ("This is unacceptable!", 5, 5),
    ("Arizona's future is at stake.", 4, 4),
    ("The elite are destroying our state.", 5, 5),
    ("I'm fed up with the lack of action.", 4, 4),
    ("Arizona is becoming unrecognizable.", 4, 4),
    ("The government is wasting our money.", 4, 4),
    ("This is not the Arizona I grew up in.", 4, 4),
    ("Arizona needs real change.", 4, 4),
    ("The system is broken.", 5, 5),
    ("I'm sick of the lies.", 5, 5),
    ("No one is addressing the real problems.", 4, 4),
    ("This is a disaster.", 5, 5),
    ("Arizona is being left behind.", 4, 4),
    ("I'm tired of the excuses.", 4, 4),
    ("This is not what democracy looks like.", 4, 5),
    ("Arizona needs a fresh start.", 4, 4),
    ("The system is corrupt.", 5, 5),
    ("What?", 5, 1),
    ("Huh?", 5, 1),
    ("Tell me more.", 5, 1),
    ("I don't understand.", 4, 2),
    ("Can you explain?", 4, 1),
    ("What about that?", 5, 1),
    ("I have a question.", 4, 1),
    ("What do you think?", 4, 1),
    ("Hello", 5, 1),
    ("Hi there", 5, 1),
    ("Hey", 5, 1),
    ("How does your education plan compare to the current administration?", 1, 1),
    ("What's the difference between your healthcare policy and your opponent's?", 1, 1),
    ("Where can I volunteer for the campaign?", 1, 1),
    ("How can I donate to your campaign?", 1, 1),
    ("When is the next town hall meeting?", 1, 1),
    ("Is there a rally near Phoenix this weekend?", 1, 1),
    ("How can I get a yard sign?", 1, 1),
    ("This administration is failing Arizonans.", 4, 4),
    ("Politicians never listen to the people.", 4, 4),
    ("Arizona deserves better leadership.", 4, 4),
    ("No one is doing anything about the homelessness crisis.", 4, 4),
    ("The legislature needs to focus on real issues.", 4, 4),
    ("No one is fighting for the middle class.", 4, 4),
    ("The political establishment is failing us.", 4, 5),
    ("Arizona deserves leaders who actually care.", 4, 4),
    ("We need to stand up for our values.", 4, 4),
    ("The government is ignoring the will of the people.", 4, 4),
    ("This is a betrayal of Arizonans.", 4, 5),
    ("The political class is out of touch.", 4, 4),
    ("Arizona is being sold out to the highest bidder.", 4, 5),
    ("The government is failing our children.", 4, 5),
    ("Arizona needs leaders with integrity.", 4, 4),
    ("We need to put Arizona first.", 4, 4),
    ("The political elite don't care about us.", 4, 5),
    ("The government is not representing us.", 4, 5),
    ("I've asked this a hundred times already!", 2, 5),
    ("How many times do I have to explain this?", 2, 5),
    ("Nobody ever answers my questions!", 3, 5),
    ("I keep getting the runaround!", 3, 5),
    ("This is ridiculous! JUST ANSWER THE QUESTION!", 2, 5),
    ("WHY WON'T ANYONE LISTEN TO ME?!", 2, 5),
    ("I'm SO tired of this nonsense!!!", 3, 5),
    ("UNBELIEVABLE! This is outrageous!", 3, 5),
    ("What a joke! You people are useless!", 3, 5),
    ("I DEMAND to speak to someone in charge!", 2, 5),
    ("Thanks for your response, that's helpful.", 1, 1),
    ("I appreciate the information.", 1, 1),
    ("That makes sense, thank you.", 1, 1),
    ("Good to know, I'll think about it.", 1, 1),
    ("Interesting perspective, I hadn't considered that.", 1, 1),
    ("Fair point, I can see both sides.", 1, 1),
    ("I understand your position now.", 1, 1),
    ("Thanks for clarifying that for me.", 1, 1),
    ("What is Brandon's plan for reducing property taxes in Maricopa County?", 1, 1),
    ("How will Brandon address the water shortage in Arizona?", 1, 1),
    ("Does Brandon support the proposed solar energy initiative?", 1, 1),
    ("What is Brandon's voting record on education funding?", 1, 1),
    ("How does Brandon plan to create jobs in rural Arizona?", 1, 1),
    ("What is Brandon's stance on the legalization of recreational marijuana?", 1, 1),
    ("Does Brandon support term limits for state legislators?", 1, 1),
    ("What is Brandon's plan for improving highway infrastructure?", 1, 1),
    ("How will Brandon work with tribal communities on water rights?", 1, 1),
    ("What is Brandon's position on the Second Amendment?", 1, 1),
    ("Will Brandon support small business tax relief?", 1, 1),
    ("What is Brandon's healthcare reform proposal?", 1, 1),
    ("How does Brandon plan to reduce crime in Phoenix?", 1, 1),
    ("What is Brandon's stance on police reform?", 1, 1),
    ("Does Brandon support school choice vouchers?", 1, 1),
    ("What about immigration?", 4, 2),
    ("Tell me about healthcare.", 4, 2),
    ("What's the deal with taxes?", 4, 2),
    ("Something about the border.", 5, 2),
    ("I want to know about schools.", 4, 2),
    ("What's happening with the economy?", 2, 2),
    ("Anything about veterans?", 4, 2),
    ("The situation with housing.", 5, 2),
    ("Explain your energy plan in detail including timelines and funding sources.", 1, 1),
    ("I need specifics on your property tax reduction proposal with numbers.", 1, 1),
    ("What exact legislation will you introduce for border security?", 1, 1),
    ("Break down your education budget line by line.", 1, 1),
    ("Give me the facts on your healthcare cost reduction plan.", 1, 1),
    ("I need data on your job creation projections for the next 4 years.", 1, 1),
    ("What specific infrastructure projects will you prioritize?", 1, 1),
    ("Detail your plan for water conservation with implementation steps.", 1, 1),
    ("Why is no one addressing the teacher shortage?", 2, 4),
    ("Why hasn't anyone fixed the roads yet?", 2, 4),
    ("Why do our taxes keep going up but nothing improves?", 2, 4),
    ("Why is healthcare so expensive in Arizona?", 2, 4),
    ("Why are politicians ignoring the border crisis?", 2, 4),
    ("Why don't we have better public transportation?", 2, 4),
    ("Why is housing so unaffordable for young people?", 2, 4),
    ("Why aren't there more jobs in rural areas?", 2, 4),
    ("Our schools are failing our children.", 4, 4),
    ("The infrastructure in this state is crumbling.", 4, 4),
    ("Healthcare costs are destroying families.", 4, 4),
    ("Young people can't afford to live here anymore.", 4, 4),
    ("Small businesses are being crushed by regulations.", 4, 4),
    ("Veterans aren't getting the support they deserve.", 4, 4),
    ("The cost of living is out of control.", 4, 4),
    ("Drug cartels are running the border.", 4, 5),
    ("Everything is broken.", 5, 5),
    ("Nothing ever changes.", 5, 5),
    ("It's all hopeless.", 5, 5),
    ("Nobody cares.", 5, 5),
    ("Whatever.", 5, 2),
    ("I guess so.", 4, 2),
    ("Sure, fine.", 4, 2),
    ("If you say so.", 4, 2),
    ("Doesn't matter anyway.", 5, 4),
    ("I think Brandon should focus more on water issues.", 2, 2),
    ("It seems like the healthcare plan could be improved.", 2, 2),
    ("I'm not sure I agree with the tax proposal.", 2, 2),
    ("I have some concerns about the education plan.", 2, 2),
    ("The border policy seems incomplete.", 2, 2),
    ("I wonder if there's a better approach to housing.", 2, 2),
    ("Maybe the infrastructure timeline is too ambitious.", 2, 2),
    ("I'm curious about the funding for these programs.", 2, 2),
    ("How will this affect seniors on fixed income?", 1, 1),
    ("What about families with special needs children?", 1, 2),
    ("How does this impact rural communities specifically?", 1, 1),
    ("Will veterans receive priority in this program?", 1, 1),
    ("How will small business owners be affected?", 1, 1),
    ("What about people who work multiple jobs?", 2, 2),
    ("How does this help single parents?", 1, 2),
    ("Will this program reach the Navajo Nation?", 1, 1),
    ("I'm angry about the state of our schools!", 2, 4),
    ("This healthcare situation makes me furious!", 2, 4),
    ("I can't believe how high taxes have gotten!", 2, 4),
    ("The border situation is infuriating!", 2, 4),
    ("It's maddening that nothing gets done!", 3, 4),
    ("I'm outraged by the lack of transparency!", 2, 4),
    ("This waste of taxpayer money is disgusting!", 2, 5),
    ("The corruption in Phoenix sickens me!", 2, 5),
    ("I'm worried about the future of our state.", 2, 2),
    ("The economic outlook concerns me deeply.", 2, 2),
    ("I fear for my children's education.", 2, 2),
    ("The water situation is alarming.", 2, 2),
    ("I'm anxious about healthcare costs.", 2, 2),
    ("The housing market is frightening.", 2, 2),
    ("I'm concerned about safety in my neighborhood.", 2, 2),
    ("The debt level is troubling.", 2, 2),
    ("What specific tax deductions will you preserve for middle-class families?", 1, 1),
    ("Which healthcare programs will receive increased funding?", 1, 1),
    ("What teacher salary increases are you proposing?", 1, 1),
    ("Which roads and highways are prioritized for repair?", 1, 1),
    ("What percentage of the budget goes to border security?", 1, 1),
    ("Which renewable energy projects will you support?", 1, 1),
    ("What specific water conservation measures are you proposing?", 1, 1),
    ("Which veteran programs will receive more funding?", 1, 1),
    ("How does your plan compare to the Governor's proposal?", 1, 1),
    ("What's the difference between your education plan and the current one?", 1, 1),
    ("How is your healthcare approach different from federal plans?", 1, 1),
    ("Can you contrast your border policy with the opposition?", 1, 1),
    ("What makes your tax plan better than the current system?", 1, 1),
    ("How does your infrastructure plan differ from past proposals?", 1, 1),
    ("Compare your housing policy to other candidates.", 1, 1),
    ("What sets your water conservation plan apart?", 1, 1),
    ("Well, you see, the thing is, I was wondering, if maybe, you know...", 4, 2),
    ("So like, what's the deal with, um, the whole situation with...?", 4, 2),
    ("I dunno, just curious about stuff I guess.", 5, 2),
    ("It's complicated but basically I want to know about things.", 5, 2),
    ("Not sure how to ask this but what about the stuff?", 5, 2),
    ("The thing with the place and the people, you know?", 5, 2),
    ("Kind of confused about everything in general.", 5, 3),
    ("Just wondering about whatever is going on.", 5, 2),
    ("Fight for Arizona! We will win!", 2, 2),
    ("Stand strong against the liberal agenda!", 2, 2),
    ("Take back what's ours!", 4, 2),
    ("Victory is within reach!", 2, 2),
    ("United we stand, divided we fall!", 2, 2),
    ("The silent majority will be heard!", 2, 2),
    ("Patriots for Brandon!", 2, 2),
    ("Make Arizona Great Again!", 2, 2),
    ("I've been a lifelong Democrat but I'm considering voting for Brandon.", 1, 1),
    ("As an Independent, what makes Brandon different?", 1, 1),
    ("I voted for the other party last time but I'm open to change.", 1, 1),
    ("What would Brandon say to someone who's never voted Republican?", 1, 1),
    ("I'm undecided - convince me why Brandon is the right choice.", 1, 1),
    ("My family has always voted Democrat, why should I switch?", 1, 1),
    ("I don't usually follow politics but this election seems important.", 1, 1),
    ("What makes Brandon appealing to moderate voters?", 1, 1),
    ("You know what really grinds my gears about this state...", 3, 4),
    ("I'm at my wit's end with these policies!", 3, 4),
    ("This is the last straw for me!", 3, 5),
    ("I've had it up to here with excuses!", 3, 4),
    ("My patience has run out completely!", 3, 5),
    ("I can't take this anymore!", 3, 5),
    ("Enough is enough already!", 3, 5),
    ("I'm beyond frustrated at this point!", 3, 5),
    ("Great question about water policy.", 1, 1),
    ("Excellent point on healthcare.", 1, 1),
    ("Smart thinking about education reform.", 1, 1),
    ("Valid concern about taxes.", 1, 2),
    ("Important issue regarding border security.", 1, 2),
    ("Good observation about infrastructure.", 1, 2),
    ("Thoughtful question on veterans' affairs.", 1, 1),
    ("Reasonable inquiry about housing.", 1, 1),
    ("Can you send me some campaign literature?", 1, 1),
    ("Where can I find Brandon's full platform online?", 1, 1),
    ("Is there a summary of the key policy positions?", 1, 1),
    ("Do you have any videos explaining the healthcare plan?", 1, 1),
    ("Where can I read more about the water conservation proposal?", 1, 1),
    ("Is there a FAQ about Brandon's positions?", 1, 1),
    ("Can I get a copy of the budget proposal?", 1, 1),
    ("Where do I find voting information for my district?", 1, 1),
]


def run_calibration_test(include_frustration=False):
    """Run the calibration test and print results."""
    import asyncio
    
    detector = DetectorV2()
    
    v_correct_12 = 0
    v_correct_45 = 0
    v_total_12 = 0
    v_total_45 = 0
    v_errors = []
    
    f_correct_12 = 0
    f_correct_45 = 0
    f_total_12 = 0
    f_total_45 = 0
    f_errors = []
    
    async def run_tests():
        nonlocal v_correct_12, v_correct_45, v_total_12, v_total_45, v_errors
        nonlocal f_correct_12, f_correct_45, f_total_12, f_total_45, f_errors
        
        for phrase, v_score, f_score in CALIBRATION_PHRASES:
            v_result = detector.compute_vagueness_score(phrase)
            
            if v_score in [1, 2]:
                v_total_12 += 1
                v_expected = "CLEAR"
                if v_result.decision == v_expected:
                    v_correct_12 += 1
                else:
                    v_errors.append((phrase, v_score, v_expected, v_result.decision, v_result.score))
            elif v_score in [4, 5]:
                v_total_45 += 1
                v_expected = "VAGUE"
                if v_result.decision == v_expected:
                    v_correct_45 += 1
                else:
                    v_errors.append((phrase, v_score, v_expected, v_result.decision, v_result.score))
            
            if include_frustration:
                flags = {
                    'caps': sum(1 for c in phrase if c.isupper()) / max(len(phrase), 1) > 0.5,
                    'all_caps': phrase.isupper(),
                    'repeated_punctuation': '!!' in phrase or '??' in phrase,
                    'profanity': False,
                    'insults': False,
                    'demands_human': 'speak to someone' in phrase.lower(),
                }
                
                f_result = await detector.compute_frustration_score(phrase, flags)
                
                if f_score in [1, 2]:
                    f_total_12 += 1
                    f_expected = "CONTINUE"
                    if f_result.decision == f_expected:
                        f_correct_12 += 1
                    else:
                        f_errors.append((phrase, f_score, f_expected, f_result.decision, f_result.bucket.value))
                elif f_score in [4, 5]:
                    f_total_45 += 1
                    f_expected = "ESCALATE"
                    if f_result.decision == f_expected:
                        f_correct_45 += 1
                    else:
                        f_errors.append((phrase, f_score, f_expected, f_result.decision, f_result.bucket.value))
    
    asyncio.run(run_tests())
    
    print("=== VAGUENESS CALIBRATION RESULTS ===")
    print(f"Score 1-2 (CLEAR): {v_correct_12}/{v_total_12} ({100*v_correct_12/max(v_total_12,1):.1f}%)")
    print(f"Score 4-5 (VAGUE): {v_correct_45}/{v_total_45} ({100*v_correct_45/max(v_total_45,1):.1f}%)")
    print(f"Combined: {v_correct_12+v_correct_45}/{v_total_12+v_total_45} ({100*(v_correct_12+v_correct_45)/max(v_total_12+v_total_45,1):.1f}%)")
    
    if v_errors:
        print(f"\n=== VAGUENESS ERRORS ({len(v_errors)}) ===")
        for phrase, score, exp, got, prob_score in v_errors[:20]:
            print(f"  [{score}] {got} (exp: {exp}, prob: {prob_score:.2f}) | \"{phrase[:50]}\"")
    
    if include_frustration:
        print("\n=== FRUSTRATION CALIBRATION RESULTS ===")
        print(f"Score 1-2 (CONTINUE): {f_correct_12}/{f_total_12} ({100*f_correct_12/max(f_total_12,1):.1f}%)")
        print(f"Score 4-5 (ESCALATE): {f_correct_45}/{f_total_45} ({100*f_correct_45/max(f_total_45,1):.1f}%)")
        print(f"Combined: {f_correct_12+f_correct_45}/{f_total_12+f_total_45} ({100*(f_correct_12+f_correct_45)/max(f_total_12+f_total_45,1):.1f}%)")
        
        if f_errors:
            print(f"\n=== FRUSTRATION ERRORS ({len(f_errors)}) ===")
            for phrase, score, exp, got, bucket in f_errors[:20]:
                print(f"  [{score}] {got} (exp: {exp}, bucket: {bucket}) | \"{phrase[:45]}\"")
    
    return {
        'vagueness': {
            'score_12': (v_correct_12, v_total_12),
            'score_45': (v_correct_45, v_total_45),
            'errors': v_errors
        },
        'frustration': {
            'score_12': (f_correct_12, f_total_12),
            'score_45': (f_correct_45, f_total_45),
            'errors': f_errors
        }
    }


if __name__ == "__main__":
    run_calibration_test()
