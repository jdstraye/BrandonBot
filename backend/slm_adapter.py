"""
SLM Adapter for BrandonBot Prequalifier

Wraps SLMManager to provide the interface expected by Prequalifier:
- classify_frustration(message, has_profanity, has_urgency, frustration_count, conversation_history)
- classify_vagueness(message, rag_results)

This adapter uses the specialized SLMs:
- j-hartmann/emotion-english-distilroberta-base for frustration detection
- cross-encoder/ms-marco-MiniLM-L6-v2 for vagueness/relevance scoring
"""

import logging
from typing import Optional, Tuple, List, Dict, Any

from slm_manager import SLMManager

logger = logging.getLogger(__name__)


class SLMAdapter:
    """
    Adapter that wraps SLMManager to match the interface expected by Prequalifier.
    
    Uses specialized SLMs (NOT LLM APIs):
    - Emotion classifier (j-hartmann) for frustration detection
    - Cross-encoder (MS-MARCO) for vagueness/relevance scoring
    """
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize the SLM adapter.
        
        Args:
            device: Device to run on (cpu, cuda, etc.)
        """
        self._slm_manager = SLMManager(device=device)
        self.mode = "specialized_slm"  # For compatibility with logging
    
    async def check_availability(self) -> bool:
        """Check if the specialized SLMs can be loaded."""
        try:
            await self._slm_manager._ensure_emotion_loaded()
            return True
        except Exception as e:
            logger.warning(f"SLM not available: {e}")
            return False
    
    async def classify_frustration(
        self,
        message: str,
        has_profanity: bool = False,
        has_urgency: bool = False,
        frustration_count: int = 0,
        conversation_history: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Classify user frustration level using specialized emotion SLM.
        
        Uses j-hartmann/emotion-english-distilroberta-base for 7-emotion detection.
        
        Args:
            message: User message
            has_profanity: Pattern flag for profanity
            has_urgency: Pattern flag for urgency keywords
            frustration_count: Number of frustration patterns detected
            conversation_history: Optional conversation history text
            
        Returns:
            Tuple of (decision, emotion):
            - decision: "escalate" or "continue"
            - emotion: detected emotion like "frustrated", "calm", "angry"
        """
        flags = {
            "profanity": has_profanity,
            "insults": False,
            "demands_human": has_urgency,
            "all_caps": False,
            "repeated_punct": frustration_count > 1,
            "caps": False,
            "repeated_punctuation": frustration_count > 1,
        }
        
        try:
            response = await self._slm_manager.classify_frustration(message, flags)
            
            decision = response.decision.lower()
            emotion = response.detected_emotion or "neutral"
            
            if emotion == "anger":
                emotion = "angry"
            elif emotion == "disgust":
                emotion = "frustrated"
            elif emotion in ("joy", "surprise"):
                emotion = "calm"
            elif emotion == "fear":
                emotion = "anxious"
            elif emotion == "sadness":
                emotion = "frustrated"
            
            return (decision, emotion)
            
        except Exception as e:
            logger.error(f"Frustration classification failed: {e}")
            raise
    
    async def classify_vagueness(
        self,
        message: str,
        rag_results: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Classify query vagueness using specialized cross-encoder SLM.
        
        Uses cross-encoder/ms-marco-MiniLM-L6-v2 for relevance scoring.
        
        Args:
            message: User query
            rag_results: Optional RAG results text (for context)
            
        Returns:
            Tuple of (decision, confidence):
            - decision: "vague" or "clear"
            - confidence: float between 0.0 and 1.0
        """
        try:
            if rag_results:
                rag_list = [{"content": rag_results, "confidence": 0.5}]
                response = await self._slm_manager.classify_vagueness_with_rag(
                    message=message,
                    rag_results=rag_list,
                    avg_confidence=0.5
                )
            else:
                response = await self._slm_manager.classify_vagueness(
                    message=message,
                    rag_confidence=0.0,
                    has_context=False
                )
            
            decision = response.decision.lower()
            confidence = response.confidence
            
            return (decision, confidence)
            
        except Exception as e:
            logger.error(f"Vagueness classification failed: {e}")
            raise
