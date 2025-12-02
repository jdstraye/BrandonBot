"""
SLM Manager for BrandonBot

Uses DistilBERT with zero-shot-classification for lightweight classification tasks:
- Frustration classification (ESCALATE/CONTINUE)
- Vagueness classification (CLEAR/VAGUE)
- Intent fulfillment check
- Ethics check
- FEC compliance verification
- PII detection

The classifier is loaded lazily on first use to save memory.
DistilBERT is ~207MB and optimized for classification tasks.
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
    Manages a lightweight classifier for semantic classification tasks.
    
    Uses DistilBERT with zero-shot-classification pipeline:
    - Lazy loading (only loads when first used)
    - CPU-optimized inference
    - Native binary classification support
    - Fast inference (~100-500ms per classification)
    """
    
    MODEL_ID = "typeform/distilbert-base-uncased-mnli"
    
    LABEL_SETS = {
        SLMTask.FRUSTRATION: {
            "labels": ["angry and hostile", "calm and polite"],
            "mapping": {"angry and hostile": "ESCALATE", "calm and polite": "CONTINUE"}
        },
        SLMTask.VAGUENESS: {
            "labels": ["clear specific question", "vague or unclear"],
            "mapping": {"clear specific question": "CLEAR", "vague or unclear": "VAGUE"}
        },
        SLMTask.VAGUENESS_WITH_RAG: {
            "labels": ["answerable question with matching context", "unanswerable or mismatched context"],
            "mapping": {"answerable question with matching context": "CLEAR", "unanswerable or mismatched context": "VAGUE"}
        },
        SLMTask.INTENT_FULFILLMENT: {
            "labels": ["response answers the question", "response does not answer the question"],
            "mapping": {"response answers the question": "YES", "response does not answer the question": "NO"}
        },
        SLMTask.ETHICS: {
            "labels": ["ethical and appropriate", "unethical or inappropriate"],
            "mapping": {"ethical and appropriate": "PASS", "unethical or inappropriate": "FAIL"}
        },
        SLMTask.FEC_COMPLIANCE: {
            "labels": ["compliant with campaign regulations", "violates campaign regulations"],
            "mapping": {"compliant with campaign regulations": "COMPLIANT", "violates campaign regulations": "VIOLATION"}
        },
    }
    
    def __init__(self, model_id: str = None, device: str = "cpu"):
        """
        Initialize SLM manager with lazy loading.
        
        Args:
            model_id: HuggingFace model ID (default: typeform/distilbert-base-uncased-mnli)
            device: Device to run on (cpu, cuda, etc.)
        """
        self.model_id = model_id or self.MODEL_ID
        self.device = device
        self._classifier = None
        self._loaded = False
        self._loading = False
        self._load_lock = asyncio.Lock()
    
    async def _ensure_loaded(self):
        """Lazy load the classifier on first use"""
        if self._loaded:
            return
        
        async with self._load_lock:
            if self._loaded:
                return
            
            if self._loading:
                while self._loading:
                    await asyncio.sleep(0.1)
                return
            
            self._loading = True
            
            try:
                logger.info(f"Loading zero-shot classifier: {self.model_id}")
                
                from transformers import pipeline
                
                device_id = -1 if self.device == "cpu" else 0
                
                self._classifier = pipeline(
                    "zero-shot-classification",
                    model=self.model_id,
                    device=device_id
                )
                
                self._loaded = True
                logger.info(f"Zero-shot classifier loaded successfully: {self.model_id}")
                
            except Exception as e:
                logger.error(f"Failed to load classifier: {e}")
                self._loaded = False
                raise
            finally:
                self._loading = False
    
    async def classify(
        self,
        text: str,
        task: SLMTask,
        hypothesis_template: str = "This text is {}."
    ) -> SLMResponse:
        """
        Classify text using zero-shot classification.
        
        Args:
            text: Text to classify
            task: Classification task type
            hypothesis_template: Template for label hypotheses
        
        Returns:
            SLMResponse with decision and confidence
        """
        await self._ensure_loaded()
        
        if task not in self.LABEL_SETS:
            raise ValueError(f"Unknown task: {task}")
        
        label_config = self.LABEL_SETS[task]
        labels = label_config["labels"]
        mapping = label_config["mapping"]
        
        try:
            result = self._classifier(
                text,
                candidate_labels=labels,
                hypothesis_template=hypothesis_template
            )
            
            top_label = result["labels"][0]
            top_score = result["scores"][0]
            
            decision = mapping.get(top_label, top_label.upper())
            
            return SLMResponse(
                decision=decision,
                confidence=top_score,
                explanation=f"label='{top_label}', score={top_score:.3f}",
                raw_output=str(result)
            )
            
        except Exception as e:
            logger.warning(f"Classification failed: {e}")
            default_decision = list(mapping.values())[0]
            return SLMResponse(
                decision=default_decision,
                confidence=0.5,
                explanation=f"Error: {e}",
                raw_output=""
            )
    
    async def classify_frustration(
        self,
        message: str,
        flags: Dict[str, bool]
    ) -> SLMResponse:
        """
        Classify user frustration level.
        
        Uses zero-shot classification to detect angry/hostile vs calm/polite tone.
        Pattern flags are used as a secondary signal.
        
        Args:
            message: User message
            flags: Pattern flags from prequalifier
        
        Returns:
            SLMResponse with ESCALATE or CONTINUE decision
        """
        await self._ensure_loaded()
        
        try:
            result = await self.classify(
                text=message,
                task=SLMTask.FRUSTRATION,
                hypothesis_template="The speaker is {}."
            )
            
            if result.decision == "ESCALATE" and result.confidence > 0.6:
                return result
            
            if result.decision == "CONTINUE" and result.confidence > 0.7:
                has_high_risk_flags = (
                    flags.get('profanity', False) or 
                    flags.get('insults', False) or
                    flags.get('demands_human', False)
                )
                if has_high_risk_flags:
                    return SLMResponse(
                        decision="ESCALATE",
                        confidence=0.65,
                        explanation=f"Flags override: {flags}, original={result.explanation}",
                        raw_output=result.raw_output
                    )
                return result
            
            if flags.get('profanity', False) or flags.get('insults', False):
                return SLMResponse(
                    decision="ESCALATE",
                    confidence=max(result.confidence, 0.6),
                    explanation=f"Flag-boosted: {result.explanation}",
                    raw_output=result.raw_output
                )
            
            return result
            
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
        
        Args:
            message: User query
            rag_confidence: Average RAG retrieval confidence (unused in this version)
            has_context: Whether RAG found relevant context (unused in this version)
        
        Returns:
            SLMResponse with CLEAR or VAGUE decision
        """
        await self._ensure_loaded()
        
        try:
            result = await self.classify(
                text=message,
                task=SLMTask.VAGUENESS,
                hypothesis_template="This is a {}."
            )
            
            return result
            
        except Exception as e:
            logger.warning(f"Vagueness classification failed: {e}")
            return SLMResponse(
                decision="CLEAR",
                confidence=0.5,
                explanation=f"Error fallback: {e}",
                raw_output=""
            )
    
    async def classify_vagueness_with_rag(
        self,
        message: str,
        rag_results: List[Dict[str, Any]],
        avg_confidence: float
    ) -> SLMResponse:
        """
        Classify query vagueness using RAG results as context.
        
        The classifier sees both the query and the retrieved content,
        allowing it to determine if the question can be answered.
        
        Args:
            message: User query
            rag_results: List of RAG results with content and confidence
            avg_confidence: Average similarity score from RAG
        
        Returns:
            SLMResponse with CLEAR or VAGUE decision
        """
        await self._ensure_loaded()
        
        rag_summary = ""
        if rag_results:
            top_results = rag_results[:3]
            for i, result in enumerate(top_results):
                content = result.get('content', '')[:150]
                score = result.get('confidence', 0.0)
                rag_summary += f"[{score:.2f}] {content}... "
        else:
            rag_summary = "No relevant content found."
        
        combined_text = f"Question: {message}\n\nRetrieved context (avg score {avg_confidence:.2f}): {rag_summary}"
        
        try:
            result = await self.classify(
                text=combined_text,
                task=SLMTask.VAGUENESS_WITH_RAG,
                hypothesis_template="This question is {}."
            )
            
            if len(message.split()) < 3:
                if result.decision == "CLEAR" and result.confidence < 0.8:
                    return SLMResponse(
                        decision="VAGUE",
                        confidence=0.7,
                        explanation=f"Short query override: {result.explanation}",
                        raw_output=result.raw_output
                    )
            
            if avg_confidence < 0.4 and result.decision == "CLEAR" and result.confidence < 0.75:
                return SLMResponse(
                    decision="VAGUE",
                    confidence=0.65,
                    explanation=f"Low RAG confidence override: avg={avg_confidence:.2f}, {result.explanation}",
                    raw_output=result.raw_output
                )
            
            return result
            
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
        Check if response fulfills user's intent.
        
        Args:
            query: Original user query
            response: LLM response to check
        
        Returns:
            SLMResponse with YES or NO decision
        """
        await self._ensure_loaded()
        
        combined_text = f"Question: {query}\nAnswer: {response[:500]}"
        
        try:
            result = await self.classify(
                text=combined_text,
                task=SLMTask.INTENT_FULFILLMENT,
                hypothesis_template="This {}."
            )
            
            return result
            
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
        Check response for ethical issues.
        
        Args:
            response: LLM response to check
        
        Returns:
            SLMResponse with PASS or FAIL decision
        """
        await self._ensure_loaded()
        
        try:
            result = await self.classify(
                text=response[:500],
                task=SLMTask.ETHICS,
                hypothesis_template="This content is {}."
            )
            
            return result
            
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
        Check response for FEC compliance.
        
        Args:
            response: LLM response to check
            regulations: Relevant FEC regulations (optional)
        
        Returns:
            SLMResponse with COMPLIANT or VIOLATION decision
        """
        await self._ensure_loaded()
        
        try:
            result = await self.classify(
                text=response[:500],
                task=SLMTask.FEC_COMPLIANCE,
                hypothesis_template="This campaign communication is {}."
            )
            
            return result
            
        except Exception as e:
            logger.warning(f"FEC compliance check failed: {e}")
            return SLMResponse(
                decision="COMPLIANT",
                confidence=0.5,
                explanation=f"Error fallback: {e}",
                raw_output=""
            )
    
    async def detect_pii(self, text: str) -> SLMResponse:
        """
        Detect PII in text.
        
        Uses pattern matching for PII detection (more reliable than classification).
        
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
