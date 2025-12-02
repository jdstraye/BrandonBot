"""
SLM Manager for BrandonBot

Uses cross-encoder for semantic relevance scoring and classification:
- Vagueness classification via query-document relevance scoring
- Frustration classification via sentiment detection
- Intent fulfillment check
- Ethics check
- FEC compliance verification
- PII detection

Cross-encoder (ms-marco-TinyBERT-L-2) is ~17MB and optimized for relevance scoring.
For sentiment, we use a separate sentiment classifier.
"""

import logging
import asyncio
import re
from dataclasses import dataclass
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


class SLMManager:
    """
    Manages lightweight models for semantic classification tasks.
    
    Uses:
    - Cross-encoder (ms-marco-TinyBERT-L-2) for relevance scoring (~17MB)
    - Sentiment classifier for frustration detection (~420MB)
    
    Both are loaded lazily on first use.
    """
    
    CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-TinyBERT-L-2"
    SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize SLM manager with lazy loading.
        
        Args:
            device: Device to run on (cpu, cuda, etc.)
        """
        self.device = device
        self._cross_encoder = None
        self._sentiment_classifier = None
        self._cross_encoder_loaded = False
        self._sentiment_loaded = False
        self._loading_cross_encoder = False
        self._loading_sentiment = False
        self._load_lock = asyncio.Lock()
    
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
        """Lazy load the sentiment classifier on first use"""
        if self._sentiment_loaded:
            return
        
        async with self._load_lock:
            if self._sentiment_loaded:
                return
            
            if self._loading_sentiment:
                while self._loading_sentiment:
                    await asyncio.sleep(0.1)
                return
            
            self._loading_sentiment = True
            
            try:
                logger.info(f"Loading sentiment classifier: {self.SENTIMENT_MODEL}")
                
                from transformers import pipeline
                
                device_id = -1 if self.device == "cpu" else 0
                
                self._sentiment_classifier = pipeline(
                    "sentiment-analysis",
                    model=self.SENTIMENT_MODEL,
                    device=device_id,
                    top_k=None
                )
                
                self._sentiment_loaded = True
                logger.info(f"Sentiment classifier loaded successfully: {self.SENTIMENT_MODEL}")
                
            except Exception as e:
                logger.error(f"Failed to load sentiment classifier: {e}")
                self._sentiment_loaded = False
                raise
            finally:
                self._loading_sentiment = False
    
    async def score_relevance(
        self,
        query: str,
        documents: List[str]
    ) -> List[float]:
        """
        Score relevance between a query and multiple documents.
        
        Uses cross-encoder to compute relevance scores for each query-document pair.
        
        Args:
            query: User query
            documents: List of document texts to score
        
        Returns:
            List of relevance scores (higher = more relevant)
        """
        await self._ensure_cross_encoder_loaded()
        
        if not documents:
            return []
        
        pairs = [(query, doc) for doc in documents]
        
        try:
            scores = self._cross_encoder.predict(pairs)
            if hasattr(scores, 'tolist'):
                scores = scores.tolist()
            return scores
        except Exception as e:
            logger.warning(f"Cross-encoder scoring failed: {e}")
            return [0.0] * len(documents)
    
    async def classify_frustration(
        self,
        message: str,
        flags: Dict[str, bool]
    ) -> SLMResponse:
        """
        Classify user frustration level using sentiment analysis.
        
        Uses RoBERTa-based sentiment classifier to detect negative sentiment.
        Pattern flags provide additional signal.
        
        Args:
            message: User message
            flags: Pattern flags from prequalifier
        
        Returns:
            SLMResponse with ESCALATE or CONTINUE decision
        """
        await self._ensure_sentiment_loaded()
        
        try:
            result = self._sentiment_classifier(message[:512])
            
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    scores_list = result[0]
                else:
                    scores_list = result
            else:
                scores_list = []
            
            sentiment_scores = {item['label']: item['score'] for item in scores_list}
            
            negative_score = sentiment_scores.get('negative', 0.0)
            positive_score = sentiment_scores.get('positive', 0.0)
            neutral_score = sentiment_scores.get('neutral', 0.0)
            
            has_severe_flags = (
                flags.get('profanity', False) or 
                flags.get('insults', False) or
                flags.get('demands_human', False)
            )
            
            if negative_score > 0.7:
                return SLMResponse(
                    decision="ESCALATE",
                    confidence=negative_score,
                    explanation=f"High negative sentiment: neg={negative_score:.2f}, pos={positive_score:.2f}",
                    raw_output=str(sentiment_scores)
                )
            
            if negative_score > 0.4 and has_severe_flags:
                return SLMResponse(
                    decision="ESCALATE",
                    confidence=max(negative_score, 0.65),
                    explanation=f"Negative sentiment + flags: neg={negative_score:.2f}, flags={flags}",
                    raw_output=str(sentiment_scores)
                )
            
            if has_severe_flags and negative_score > 0.25:
                return SLMResponse(
                    decision="ESCALATE",
                    confidence=0.6,
                    explanation=f"Flags with mild negative: neg={negative_score:.2f}, flags={flags}",
                    raw_output=str(sentiment_scores)
                )
            
            return SLMResponse(
                decision="CONTINUE",
                confidence=max(positive_score, neutral_score),
                explanation=f"Acceptable sentiment: neg={negative_score:.2f}, pos={positive_score:.2f}, neu={neutral_score:.2f}",
                raw_output=str(sentiment_scores)
            )
            
        except Exception as e:
            logger.warning(f"Frustration classification failed: {e}")
            if flags.get('profanity', False) or flags.get('insults', False):
                return SLMResponse(
                    decision="ESCALATE",
                    confidence=0.6,
                    explanation=f"Fallback due to flags: {flags}",
                    raw_output=""
                )
            return SLMResponse(
                decision="CONTINUE",
                confidence=0.5,
                explanation=f"Error fallback: {e}",
                raw_output=""
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
        
        if not rag_results:
            return SLMResponse(
                decision="VAGUE",
                confidence=0.75,
                explanation="No RAG results found",
                raw_output=""
            )
        
        try:
            documents = [r.get('content', '')[:300] for r in rag_results[:5]]
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
            
            if max_score > 5.0:
                return SLMResponse(
                    decision="CLEAR",
                    confidence=min(0.95, 0.7 + max_score * 0.02),
                    explanation=f"High relevance: max={max_score:.2f}, avg={avg_score:.2f}",
                    raw_output=str(relevance_scores)
                )
            
            if max_score > 2.0 and avg_score > 0.5:
                return SLMResponse(
                    decision="CLEAR",
                    confidence=min(0.85, 0.6 + max_score * 0.05),
                    explanation=f"Good relevance: max={max_score:.2f}, avg={avg_score:.2f}",
                    raw_output=str(relevance_scores)
                )
            
            if max_score > 0.5:
                return SLMResponse(
                    decision="CLEAR",
                    confidence=0.65,
                    explanation=f"Moderate relevance: max={max_score:.2f}, avg={avg_score:.2f}",
                    raw_output=str(relevance_scores)
                )
            
            return SLMResponse(
                decision="VAGUE",
                confidence=max(0.6, 0.8 - max_score * 0.1),
                explanation=f"Low relevance: max={max_score:.2f}, avg={avg_score:.2f}",
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
