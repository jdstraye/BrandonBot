"""
SLM Manager for BrandonBot

Uses cross-encoder for semantic relevance scoring and classification:
- Vagueness classification via query-document relevance scoring
- Frustration classification via emotion detection (7 emotions)
- Intent fulfillment check
- Ethics check
- FEC compliance verification
- PII detection

Cross-encoder (BAAI/bge-reranker-v2-m3) is optimized for relevance scoring.
For emotion detection, we use j-hartmann 7-emotion classifier.
"""

import logging
import asyncio
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class SLMTask(Enum):
    """Classification tasks the SLM can perform"""
    FRUSTRATION = "frustration"
    VAGUENESS = "vagueness"
    VAGUENESS_WITH_RAG = "vagueness_with_rag"
    INTENT_FULFILLMENT = "intent_fulfillment"
    ETHICS = "ethics"
    FEC_COMPLIANCE = "fec_compliance"
    PII_DETECTION = "pii_detection"


@dataclass
class SLMResponse:
    """Response from SLM classification"""
    decision: str
    confidence: float
    explanation: str = ""
    raw_output: str = ""
    detected_emotion: str = ""


class SLMManager:
    """
    Manages lightweight models for semantic classification tasks.
    
    Uses:
    - Cross-encoder (BAAI/bge-reranker-v2-m3) for relevance scoring
    - Emotion classifier (j-hartmann) for frustration detection (7 emotions)
    
    Both are loaded lazily on first use.
    """
    
    CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
    EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
    
    EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    FRUSTRATION_EMOTIONS = {"anger", "disgust"}
    NEGATIVE_EMOTIONS = {"anger", "disgust", "fear", "sadness"}
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize SLM manager with lazy loading.
        
        Args:
            device: Device to run on (cpu, cuda, etc.)
        """
        self.device = device
        self._cross_encoder = None
        self._emotion_classifier = None
        self._cross_encoder_loaded = False
        self._emotion_loaded = False
        self._loading_cross_encoder = False
        self._loading_emotion = False
        self._load_lock = asyncio.Lock()
    
    @staticmethod
    def clean_rag_text(text: str) -> str:
        """
        Clean RAG result text for better cross-encoder scoring.
        
        Strips whitespace, joins fragmented sentences, normalizes spacing.
        
        Args:
            text: Raw RAG result text
        
        Returns:
            Cleaned text suitable for cross-encoder
        """
        if not text:
            return ""
        
        text = ' '.join(text.split())
        
        text = re.sub(r'\s*\n\s*', ' ', text)
        
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        
        text = re.sub(r'\s{2,}', ' ', text)
        
        return text.strip()
    
    async def _ensure_cross_encoder_loaded(self):
        """Lazy load the cross-encoder on first use"""
        if self._cross_encoder_loaded:
            return
        
        async with self._load_lock:
            if self._cross_encoder_loaded:
                return
            
            if self._loading_cross_encoder:
                while self._loading_cross_encoder:
                    await asyncio.sleep(0.1)
                return
            
            self._loading_cross_encoder = True
            
            try:
                logger.info(f"Loading cross-encoder: {self.CROSS_ENCODER_MODEL}")
                
                from sentence_transformers import CrossEncoder
                
                self._cross_encoder = CrossEncoder(
                    self.CROSS_ENCODER_MODEL,
                    max_length=512,
                    device=self.device
                )
                
                self._cross_encoder_loaded = True
                logger.info(f"Cross-encoder loaded successfully: {self.CROSS_ENCODER_MODEL}")
                
            except Exception as e:
                logger.error(f"Failed to load cross-encoder: {e}")
                self._cross_encoder_loaded = False
                raise
            finally:
                self._loading_cross_encoder = False
    
    async def _ensure_sentiment_loaded(self):
        """Lazy load the emotion classifier on first use (backward compat alias)"""
        await self._ensure_emotion_loaded()
    
    async def _ensure_emotion_loaded(self):
        """Lazy load the emotion classifier on first use"""
        if self._emotion_loaded:
            return
        
        async with self._load_lock:
            if self._emotion_loaded:
                return
            
            if self._loading_emotion:
                while self._loading_emotion:
                    await asyncio.sleep(0.1)
                return
            
            self._loading_emotion = True
            
            try:
                logger.info(f"Loading emotion classifier: {self.EMOTION_MODEL}")
                
                from transformers import pipeline
                
                device_id = -1 if self.device == "cpu" else 0
                
                self._emotion_classifier = pipeline(
                    "text-classification",
                    model=self.EMOTION_MODEL,
                    device=device_id,
                    top_k=None
                )
                
                self._emotion_loaded = True
                logger.info(f"Emotion classifier loaded successfully: {self.EMOTION_MODEL}")
                
            except Exception as e:
                logger.error(f"Failed to load emotion classifier: {e}")
                self._emotion_loaded = False
                raise
            finally:
                self._loading_emotion = False
    
    async def score_relevance(
        self,
        query: str,
        documents: List[str]
    ) -> List[float]:
        """
        Score relevance between a query and multiple documents.
        
        Uses cross-encoder to compute relevance scores for each query-document pair.
        Applies sigmoid to convert raw logits to probabilities [0, 1].
        
        Args:
            query: User query
            documents: List of document texts to score
        
        Returns:
            List of relevance scores (0-1, higher = more relevant)
        """
        await self._ensure_cross_encoder_loaded()
        
        if not documents:
            return []
        
        pairs = [(query, doc) for doc in documents]
        
        try:
            import torch
            
            raw_scores = self._cross_encoder.predict(pairs)
            
            if hasattr(raw_scores, 'tolist'):
                raw_scores = raw_scores.tolist()
            
            if isinstance(raw_scores, (list, tuple)):
                tensor_scores = torch.tensor(raw_scores)
            else:
                tensor_scores = torch.tensor([raw_scores])
            
            sigmoid_scores = torch.sigmoid(tensor_scores).tolist()
            
            return sigmoid_scores
        except Exception as e:
            logger.warning(f"Cross-encoder scoring failed: {e}")
            return [0.0] * len(documents)
    
    async def classify_frustration(
        self,
        message: str,
        flags: Dict[str, bool]
    ) -> SLMResponse:
        """
        Classify user frustration level using 7-emotion detection.
        
        Uses j-hartmann emotion classifier to detect:
        anger, disgust, fear, joy, neutral, sadness, surprise
        
        Frustration is detected via anger + disgust scores.
        Pattern flags provide additional signal.
        
        Args:
            message: User message
            flags: Pattern flags from prequalifier
        
        Returns:
            SLMResponse with ESCALATE or CONTINUE decision and detected_emotion
        """
        await self._ensure_emotion_loaded()
        
        try:
            result = self._emotion_classifier(message[:512])
            
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    scores_list = result[0]
                else:
                    scores_list = result
            else:
                scores_list = []
            
            emotion_scores = {item['label']: item['score'] for item in scores_list}
            
            anger_score = emotion_scores.get('anger', 0.0)
            disgust_score = emotion_scores.get('disgust', 0.0)
            fear_score = emotion_scores.get('fear', 0.0)
            joy_score = emotion_scores.get('joy', 0.0)
            neutral_score = emotion_scores.get('neutral', 0.0)
            sadness_score = emotion_scores.get('sadness', 0.0)
            surprise_score = emotion_scores.get('surprise', 0.0)
            
            frustration_score = anger_score + disgust_score
            negative_score = anger_score + disgust_score + fear_score + sadness_score
            
            top_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0] if emotion_scores else "neutral"
            
            has_severe_flags = (
                flags.get('profanity', False) or 
                flags.get('insults', False) or
                flags.get('demands_human', False)
            )
            
            has_frustration_signals = (
                flags.get('caps', False) or
                flags.get('all_caps', False) or
                flags.get('repeated_punctuation', False) or
                flags.get('repeated_punct', False)
            )
            
            frustration_keywords = [
                'hundred times', 'over and over', 'again and again',
                'already asked', 'how many times', 'keep asking',
                'never answered', 'same question', 'tired of',
                'never get', 'straight answer', 'no one listens',
                'nobody listens', 'fed up', 'sick of'
            ]
            msg_lower = message.lower()
            has_repetition_frustration = any(kw in msg_lower for kw in frustration_keywords)
            
            if anger_score > 0.35:
                return SLMResponse(
                    decision="ESCALATE",
                    confidence=anger_score,
                    explanation=f"High anger: anger={anger_score:.2f}, disgust={disgust_score:.2f}",
                    raw_output=str(emotion_scores),
                    detected_emotion="anger"
                )
            
            if disgust_score > 0.35:
                return SLMResponse(
                    decision="ESCALATE",
                    confidence=disgust_score,
                    explanation=f"High disgust: anger={anger_score:.2f}, disgust={disgust_score:.2f}",
                    raw_output=str(emotion_scores),
                    detected_emotion="disgust"
                )
            
            if frustration_score > 0.3:
                return SLMResponse(
                    decision="ESCALATE",
                    confidence=max(frustration_score, 0.60),
                    explanation=f"Combined frustration: anger={anger_score:.2f}, disgust={disgust_score:.2f}",
                    raw_output=str(emotion_scores),
                    detected_emotion=top_emotion
                )
            
            if has_severe_flags and frustration_score > 0.1:
                return SLMResponse(
                    decision="ESCALATE",
                    confidence=max(frustration_score + 0.4, 0.65),
                    explanation=f"Frustration + flags: anger={anger_score:.2f}, disgust={disgust_score:.2f}, flags={flags}",
                    raw_output=str(emotion_scores),
                    detected_emotion=top_emotion
                )
            
            if has_frustration_signals and frustration_score > 0.1:
                return SLMResponse(
                    decision="ESCALATE",
                    confidence=max(frustration_score + 0.3, 0.6),
                    explanation=f"Frustration signals: anger={anger_score:.2f}, disgust={disgust_score:.2f}, flags={flags}",
                    raw_output=str(emotion_scores),
                    detected_emotion=top_emotion
                )
            
            if fear_score > 0.35:
                return SLMResponse(
                    decision="ESCALATE",
                    confidence=fear_score,
                    explanation=f"High fear detected: fear={fear_score:.2f}",
                    raw_output=str(emotion_scores),
                    detected_emotion="fear"
                )
            
            if sadness_score > 0.4:
                return SLMResponse(
                    decision="ESCALATE",
                    confidence=sadness_score,
                    explanation=f"High sadness detected: sadness={sadness_score:.2f}",
                    raw_output=str(emotion_scores),
                    detected_emotion="sadness"
                )
            
            if surprise_score > 0.5 and has_frustration_signals:
                return SLMResponse(
                    decision="ESCALATE",
                    confidence=surprise_score,
                    explanation=f"Surprise with frustration signals: surprise={surprise_score:.2f}",
                    raw_output=str(emotion_scores),
                    detected_emotion="surprise"
                )
            
            if has_repetition_frustration and negative_score > 0.15:
                return SLMResponse(
                    decision="ESCALATE",
                    confidence=max(negative_score, 0.60),
                    explanation=f"Repetition frustration keywords detected: {msg_lower[:50]}",
                    raw_output=str(emotion_scores),
                    detected_emotion=top_emotion
                )
            
            return SLMResponse(
                decision="CONTINUE",
                confidence=max(joy_score, neutral_score),
                explanation=f"Acceptable emotion: joy={joy_score:.2f}, neutral={neutral_score:.2f}, anger={anger_score:.2f}",
                raw_output=str(emotion_scores),
                detected_emotion=top_emotion
            )
            
        except Exception as e:
            logger.warning(f"Frustration classification failed: {e}")
            if flags.get('profanity', False) or flags.get('insults', False):
                return SLMResponse(
                    decision="ESCALATE",
                    confidence=0.6,
                    explanation=f"Fallback due to flags: {flags}",
                    raw_output="",
                    detected_emotion="anger"
                )
            return SLMResponse(
                decision="CONTINUE",
                confidence=0.5,
                explanation=f"Error fallback: {e}",
                raw_output="",
                detected_emotion="neutral"
            )
    
    async def classify_vagueness(
        self,
        message: str,
        rag_confidence: float = 0.0,
        has_context: bool = False
    ) -> SLMResponse:
        """
        Classify query vagueness (without RAG context).
        
        Uses simple heuristics for standalone vagueness detection.
        
        Args:
            message: User query
            rag_confidence: Average RAG retrieval confidence (unused)
            has_context: Whether RAG found relevant context (unused)
        
        Returns:
            SLMResponse with CLEAR or VAGUE decision
        """
        words = message.lower().split()
        word_count = len(words)
        
        greetings = {'hi', 'hello', 'hey', 'yo', 'sup', 'greetings'}
        if word_count <= 2 and any(w in greetings for w in words):
            return SLMResponse(
                decision="VAGUE",
                confidence=0.95,
                explanation="Greeting detected",
                raw_output=message
            )
        
        if word_count < 3:
            return SLMResponse(
                decision="VAGUE",
                confidence=0.85,
                explanation=f"Too short: {word_count} words",
                raw_output=message
            )
        
        question_words = {'what', 'how', 'why', 'where', 'when', 'who', 'which', 'does', 'is', 'are', 'can', 'will'}
        topic_words = {'policy', 'position', 'stance', 'view', 'plan', 'think', 'believe', 'support', 'oppose'}
        
        has_question = any(w in question_words for w in words)
        has_topic = any(w in topic_words for w in words) or 'brandon' in message.lower()
        
        if has_question and (has_topic or word_count >= 5):
            return SLMResponse(
                decision="CLEAR",
                confidence=0.8,
                explanation="Question with topic detected",
                raw_output=message
            )
        
        if word_count >= 6:
            return SLMResponse(
                decision="CLEAR",
                confidence=0.7,
                explanation=f"Sufficient length: {word_count} words",
                raw_output=message
            )
        
        return SLMResponse(
            decision="VAGUE",
            confidence=0.6,
            explanation="Unclear intent",
            raw_output=message
        )
    
    async def classify_vagueness_with_rag(
        self,
        message: str,
        rag_results: List[Dict[str, Any]],
        avg_confidence: float
    ) -> SLMResponse:
        """
        Classify query vagueness using cross-encoder relevance scoring.
        
        Uses cross-encoder to score relevance between user query and RAG results.
        High relevance = CLEAR (we can answer), Low relevance = VAGUE (we can't).
        
        Args:
            message: User query
            rag_results: List of RAG results with content and confidence
            avg_confidence: Average similarity score from RAG (embedding-based)
        
        Returns:
            SLMResponse with CLEAR or VAGUE decision
        """
        await self._ensure_cross_encoder_loaded()
        
        words = message.lower().split()
        word_count = len(words)
        message_lower = message.lower()
        
        greetings = {'hi', 'hello', 'hey', 'yo', 'sup', 'greetings'}
        if word_count <= 2 and any(w in greetings for w in words):
            return SLMResponse(
                decision="VAGUE",
                confidence=0.95,
                explanation="Greeting detected",
                raw_output=message
            )
        
        if word_count < 3:
            return SLMResponse(
                decision="VAGUE",
                confidence=0.85,
                explanation=f"Too short: {word_count} words",
                raw_output=message
            )
        
        vague_patterns_exact = [
            'what about that',
            'tell me more',
            'what do you think',
            'i have a question',
            'i want to know more',
        ]
        vague_patterns_end = [
            'can you explain',
            'what are the policies',
            'what are the plans',
        ]
        for pattern in vague_patterns_exact:
            if pattern in message_lower:
                return SLMResponse(
                    decision="VAGUE",
                    confidence=0.80,
                    explanation=f"Vague pattern detected: '{pattern}'",
                    raw_output=message
                )
        for pattern in vague_patterns_end:
            if message_lower.strip().rstrip('?').endswith(pattern) or message_lower.strip() == pattern:
                return SLMResponse(
                    decision="VAGUE",
                    confidence=0.80,
                    explanation=f"Vague pattern (standalone): '{pattern}'",
                    raw_output=message
                )
        
        if "what's brandon's opinion" in message_lower or "what is brandon's opinion" in message_lower:
            if 'on' not in message_lower:
                return SLMResponse(
                    decision="VAGUE",
                    confidence=0.80,
                    explanation="Opinion without topic",
                    raw_output=message
                )
        
        if message_lower.startswith('what about ') and word_count <= 5:
            return SLMResponse(
                decision="VAGUE",
                confidence=0.75,
                explanation="Short 'what about' question",
                raw_output=message
            )
        
        question_words = {'what', 'how', 'why', 'where', 'when', 'who', 'which', 'does', 'do', 'is', 'are', 'can', 'will', 'would', 'should'}
        has_question_word = any(message_lower.startswith(qw) for qw in question_words)
        has_question_mark = '?' in message
        is_question = has_question_word or has_question_mark
        
        if not is_question and word_count <= 8:
            opinion_statement_patterns = [
                'is out of control', 'are out of control',
                'is broken', 'are broken',
                'is terrible', 'is awful', 'is great',
                'needs to', 'should be', 'must be',
            ]
            if any(p in message_lower for p in opinion_statement_patterns):
                return SLMResponse(
                    decision="VAGUE",
                    confidence=0.75,
                    explanation="Opinion statement without specific question",
                    raw_output=message
                )
        
        if not rag_results:
            return SLMResponse(
                decision="VAGUE",
                confidence=0.75,
                explanation="No RAG results found",
                raw_output=""
            )
        
        try:
            documents = [
                self.clean_rag_text(r.get('content', ''))[:300] 
                for r in rag_results[:5]
            ]
            documents = [d for d in documents if d]
            
            if not documents:
                return SLMResponse(
                    decision="VAGUE",
                    confidence=0.70,
                    explanation="No valid RAG content after cleaning",
                    raw_output=""
                )
            
            relevance_scores = await self.score_relevance(message, documents)
            
            if not relevance_scores:
                return SLMResponse(
                    decision="VAGUE",
                    confidence=0.6,
                    explanation="Failed to compute relevance scores",
                    raw_output=""
                )
            
            max_score = max(relevance_scores)
            avg_score = sum(relevance_scores) / len(relevance_scores)
            
            combined_signal = (max_score * 0.3) + (avg_confidence * 0.7)
            
            if combined_signal >= 0.55:
                return SLMResponse(
                    decision="CLEAR",
                    confidence=min(0.95, 0.6 + combined_signal * 0.5),
                    explanation=f"High combined: cross={max_score:.3f}, rag={avg_confidence:.2f}, combined={combined_signal:.3f}",
                    raw_output=str(relevance_scores)
                )
            
            if combined_signal >= 0.45 or avg_confidence >= 0.65:
                return SLMResponse(
                    decision="CLEAR",
                    confidence=min(0.80, 0.5 + combined_signal * 0.4),
                    explanation=f"Good combined: cross={max_score:.3f}, rag={avg_confidence:.2f}, combined={combined_signal:.3f}",
                    raw_output=str(relevance_scores)
                )
            
            if avg_confidence >= 0.50 and max_score >= 0.002:
                return SLMResponse(
                    decision="CLEAR",
                    confidence=0.65,
                    explanation=f"Moderate RAG match: cross={max_score:.3f}, rag={avg_confidence:.2f}",
                    raw_output=str(relevance_scores)
                )
            
            return SLMResponse(
                decision="VAGUE",
                confidence=max(0.7, 0.9 - combined_signal),
                explanation=f"Low relevance: cross={max_score:.3f}, rag={avg_confidence:.2f}, combined={combined_signal:.3f}",
                raw_output=str(relevance_scores)
            )
            
        except Exception as e:
            logger.warning(f"RAG vagueness classification failed: {e}")
            decision = "CLEAR" if avg_confidence > 0.5 else "VAGUE"
            return SLMResponse(
                decision=decision,
                confidence=0.5,
                explanation=f"Error fallback based on avg_confidence={avg_confidence:.2f}: {e}",
                raw_output=""
            )
    
    async def check_intent_fulfillment(
        self,
        query: str,
        response: str
    ) -> SLMResponse:
        """
        Check if response fulfills user's intent using cross-encoder.
        
        Args:
            query: Original user query
            response: LLM response to check
        
        Returns:
            SLMResponse with YES or NO decision
        """
        await self._ensure_cross_encoder_loaded()
        
        try:
            scores = await self.score_relevance(query, [response[:500]])
            score = scores[0] if scores else 0.0
            
            if score > 3.0:
                return SLMResponse(
                    decision="YES",
                    confidence=min(0.95, 0.7 + score * 0.05),
                    explanation=f"High relevance: score={score:.2f}",
                    raw_output=str(score)
                )
            elif score > 0.5:
                return SLMResponse(
                    decision="YES",
                    confidence=0.7,
                    explanation=f"Moderate relevance: score={score:.2f}",
                    raw_output=str(score)
                )
            else:
                return SLMResponse(
                    decision="NO",
                    confidence=max(0.6, 0.8 - score * 0.1),
                    explanation=f"Low relevance: score={score:.2f}",
                    raw_output=str(score)
                )
            
        except Exception as e:
            logger.warning(f"Intent check failed: {e}")
            return SLMResponse(
                decision="YES",
                confidence=0.5,
                explanation=f"Error fallback: {e}",
                raw_output=""
            )
    
    async def check_ethics(self, response: str) -> SLMResponse:
        """
        Check response for ethical issues using sentiment analysis.
        
        Checks for extremely negative or hostile content.
        
        Args:
            response: LLM response to check
        
        Returns:
            SLMResponse with PASS or FAIL decision
        """
        await self._ensure_sentiment_loaded()
        
        try:
            result = self._sentiment_classifier(response[:512])
            
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    scores_list = result[0]
                else:
                    scores_list = result
            else:
                scores_list = []
            
            sentiment_scores = {item['label']: item['score'] for item in scores_list}
            negative_score = sentiment_scores.get('negative', 0.0)
            
            if negative_score > 0.9:
                return SLMResponse(
                    decision="FAIL",
                    confidence=negative_score,
                    explanation=f"Extremely negative content: {negative_score:.2f}",
                    raw_output=str(sentiment_scores)
                )
            
            return SLMResponse(
                decision="PASS",
                confidence=1 - negative_score,
                explanation=f"Acceptable content: neg={negative_score:.2f}",
                raw_output=str(sentiment_scores)
            )
            
        except Exception as e:
            logger.warning(f"Ethics check failed: {e}")
            return SLMResponse(
                decision="PASS",
                confidence=0.5,
                explanation=f"Error fallback: {e}",
                raw_output=""
            )
    
    async def check_fec_compliance(
        self,
        response: str,
        regulations: List[str] = None
    ) -> SLMResponse:
        """
        Check response for FEC compliance using pattern matching.
        
        Checks for common FEC violations like donation solicitation, 
        false promises, etc.
        
        Args:
            response: LLM response to check
            regulations: Relevant FEC regulations (optional)
        
        Returns:
            SLMResponse with COMPLIANT or VIOLATION decision
        """
        response_lower = response.lower()
        
        violation_patterns = [
            (r'\b(donate|contribution|give).*\$\d+', 'Specific dollar amount solicitation'),
            (r'\b(guarantee|promise|definitely will)\b.*\b(win|elected|victory)\b', 'Election guarantee'),
            (r'\bif elected.*will give you\b', 'Quid pro quo implication'),
            (r'\b(opponent|rival).*\b(corrupt|criminal|evil)\b', 'Defamatory statement'),
        ]
        
        for pattern, violation_type in violation_patterns:
            if re.search(pattern, response_lower, re.IGNORECASE):
                return SLMResponse(
                    decision="VIOLATION",
                    confidence=0.9,
                    explanation=f"Pattern matched: {violation_type}",
                    raw_output=pattern
                )
        
        return SLMResponse(
            decision="COMPLIANT",
            confidence=0.85,
            explanation="No FEC violation patterns detected",
            raw_output=""
        )
    
    async def detect_pii(self, text: str) -> SLMResponse:
        """
        Detect PII in text using pattern matching.
        
        Args:
            text: Text to check for PII
        
        Returns:
            SLMResponse with FOUND or CLEAN decision
        """
        pii_patterns = [
            (r'\b\d{3}-\d{2}-\d{4}\b', 'SSN'),
            (r'\b\d{3}[-.]\d{3}[-.]\d{4}\b', 'phone'),
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'email'),
            (r'\b\d{5}(?:-\d{4})?\b', 'zip'),
            (r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b', 'date'),
            (r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', 'date'),
        ]
        
        found_pii = []
        for pattern, pii_type in pii_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                found_pii.append(pii_type)
        
        if found_pii:
            return SLMResponse(
                decision="FOUND",
                confidence=0.95,
                explanation=f"PII types found: {', '.join(found_pii)}",
                raw_output=str(found_pii)
            )
        
        return SLMResponse(
            decision="CLEAN",
            confidence=0.9,
            explanation="No PII patterns detected",
            raw_output=""
        )


slm_manager = SLMManager()
