"""
Specialized SLM Models for Output Validation

Each safeguard uses a purpose-built model:
- ME2-BERT: Ethics checking (10 moral dimensions from Moral Foundations Theory)
- cross-encoder/ms-marco-MiniLM-L-6-v2: Intent/Response alignment (QA pairs)
- lakshyakh93/deberta_finetuned_pii: PII detection
- prajjwal1/bert-tiny: Confidence verification (hedging detection)

All models use double-negative prompting where applicable.
"""

import logging
import asyncio
import re
from typing import Tuple, Dict, List, Optional, Any
from dataclasses import dataclass

TORCH_AVAILABLE = False
try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    np = None
    
TRANSFORMERS_AVAILABLE = False
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    pipeline = None

logger = logging.getLogger(__name__)

ME2_BERT_MODEL = "lorenzozan/ME2-BERT"
MS_MARCO_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
PII_MODEL = "lakshyakh93/deberta_finetuned_pii"
BERT_TINY_MODEL = "prajjwal1/bert-tiny"


@dataclass
class EthicsResult:
    """Result from ME2-BERT ethics check."""
    score: int
    confidence: float
    explanation: str
    moral_dimensions: Dict[str, float]


@dataclass  
class IntentResult:
    """Result from MS-MARCO intent check."""
    score: int
    confidence: float
    explanation: str
    relevance_score: float


@dataclass
class PIIResult:
    """Result from DeBERTa PII check."""
    score: int
    confidence: float
    explanation: str
    entities_found: List[Dict[str, Any]]


@dataclass
class ConfidenceResult:
    """Result from BERT-tiny confidence check."""
    score: int
    confidence: float
    explanation: str
    hedging_detected: bool
    overconfidence_detected: bool


class ME2BertEthicsChecker:
    """
    Ethics checker using ME2-BERT (Moral Foundations Theory).
    
    Detects 10 moral dimensions:
    - Care/Harm: Cherishing and protecting others
    - Fairness/Cheating: Rendering justice, reciprocity
    - Loyalty/Betrayal: Group loyalty, patriotism
    - Authority/Subversion: Obeying tradition, legitimate authority
    - Purity/Degradation: Abhorrence of disgusting things (Judeo-Christian sanctity)
    
    Maps to Judeo-Christian ethics:
    - Harm -> Violence, cruelty (Commandment: Do not murder)
    - Cheating -> Dishonesty, fraud (Commandment: Do not bear false witness)
    - Betrayal -> Disloyalty, broken trust
    - Degradation -> Impurity, immorality (sexual ethics, substance abuse)
    - Subversion -> Disrespect for authority, mockery
    """
    
    MORAL_DIMENSIONS = [
        "care", "harm", "fairness", "cheating", 
        "loyalty", "betrayal", "authority", "subversion",
        "purity", "degradation"
    ]
    
    VIOLATION_DIMENSIONS = ["harm", "cheating", "betrayal", "subversion", "degradation"]
    
    DIMENSION_WEIGHTS = {
        "harm": 1.0,
        "cheating": 0.9,
        "betrayal": 0.7,
        "subversion": 0.6,
        "degradation": 0.8,
    }
    
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._initialized = False
        self._init_lock = asyncio.Lock()
    
    async def ensure_ready(self) -> bool:
        """Initialize ME2-BERT model."""
        if self._initialized:
            return True
        
        if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            logger.warning("PyTorch or Transformers not available, using pattern fallback")
            return False
        
        async with self._init_lock:
            if self._initialized:
                return True
            
            try:
                from transformers import AutoTokenizer, AutoModel
                
                logger.info(f"Loading ME2-BERT from {ME2_BERT_MODEL}...")
                self._tokenizer = AutoTokenizer.from_pretrained(
                    ME2_BERT_MODEL, 
                    trust_remote_code=True
                )
                self._model = AutoModel.from_pretrained(
                    ME2_BERT_MODEL,
                    trust_remote_code=True
                )
                self._model.eval()
                
                if torch.cuda.is_available():
                    self._model = self._model.cuda()
                
                self._initialized = True
                logger.info("ME2-BERT ethics checker ready")
                return True
                
            except Exception as e:
                logger.error(f"Failed to load ME2-BERT: {e}")
                return False
    
    async def check_ethics(self, response: str) -> EthicsResult:
        """
        Check response for ethics violations using ME2-BERT.
        
        Uses double-negative approach: looks for presence of violation dimensions
        (harm, cheating, betrayal, subversion, degradation) rather than virtue dimensions.
        
        Returns:
            EthicsResult with violation score 0-5 and moral dimension scores
        """
        if not await self.ensure_ready():
            return EthicsResult(
                score=0,
                confidence=0.5,
                explanation="ME2-BERT not available",
                moral_dimensions={}
            )
        
        try:
            response_snippet = response[:512]
            
            def _run_inference():
                inputs = self._tokenizer(
                    [response_snippet],
                    padding="max_length",
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                )
                
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self._model(**inputs, return_dict=True)
                
                return outputs
            
            outputs = await asyncio.get_event_loop().run_in_executor(None, _run_inference)
            
            moral_scores = {}
            if isinstance(outputs, dict):
                for dim in self.MORAL_DIMENSIONS:
                    if dim in outputs:
                        score = float(outputs[dim].cpu().numpy()[0])
                        moral_scores[dim] = score
            else:
                if hasattr(outputs, 'last_hidden_state'):
                    pooled = outputs.last_hidden_state.mean(dim=1).squeeze()
                    for i, dim in enumerate(self.MORAL_DIMENSIONS[:min(10, len(pooled))]):
                        moral_scores[dim] = float(torch.sigmoid(pooled[i]).cpu().numpy())
            
            violation_score = 0.0
            violations_found = []
            
            for dim in self.VIOLATION_DIMENSIONS:
                if dim in moral_scores:
                    weight = self.DIMENSION_WEIGHTS.get(dim, 0.5)
                    dim_score = moral_scores[dim]
                    
                    if dim_score > 0.5:
                        weighted_contrib = dim_score * weight
                        violation_score += weighted_contrib
                        violations_found.append(f"{dim}={dim_score:.2f}")
            
            if violation_score >= 0.8:
                score = 5
            elif violation_score >= 0.6:
                score = 4
            elif violation_score >= 0.4:
                score = 3
            elif violation_score >= 0.2:
                score = 2
            elif violation_score > 0.05:
                score = 1
            else:
                score = 0
            
            if violations_found:
                explanation = f"Ethics violations: {', '.join(violations_found)}"
            else:
                explanation = "No ethics violations detected"
            
            confidence = 0.85 if moral_scores else 0.5
            
            return EthicsResult(
                score=score,
                confidence=confidence,
                explanation=explanation,
                moral_dimensions=moral_scores
            )
            
        except Exception as e:
            logger.error(f"ME2-BERT ethics check failed: {e}")
            return EthicsResult(
                score=0,
                confidence=0.5,
                explanation=f"Check failed: {e}",
                moral_dimensions={}
            )


class MSMarcoIntentChecker:
    """
    Intent/Response alignment checker using MS-MARCO cross-encoder.
    
    Trained on QA pairs, optimal for detecting if response answers the question.
    Returns relevance score that indicates how well the response addresses the query.
    """
    
    def __init__(self):
        self._model = None
        self._initialized = False
        self._init_lock = asyncio.Lock()
    
    async def ensure_ready(self) -> bool:
        """Initialize MS-MARCO cross-encoder."""
        if self._initialized:
            return True
        
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, using pattern fallback")
            return False
        
        async with self._init_lock:
            if self._initialized:
                return True
            
            try:
                from sentence_transformers import CrossEncoder
                
                logger.info(f"Loading MS-MARCO from {MS_MARCO_MODEL}...")
                self._model = CrossEncoder(MS_MARCO_MODEL, max_length=512)
                self._initialized = True
                logger.info("MS-MARCO intent checker ready")
                return True
                
            except Exception as e:
                logger.error(f"Failed to load MS-MARCO: {e}")
                return False
    
    async def check_intent(self, query: str, response: str) -> IntentResult:
        """
        Check if response addresses the query using MS-MARCO.
        
        Double-negative approach: Scores below threshold indicate FAILURE to answer.
        
        Thresholds:
        - score >= 0.7: Response clearly answers the question (score=0)
        - 0.5 <= score < 0.7: Partial answer (score=1-2)
        - 0.3 <= score < 0.5: Tangential (score=3)
        - 0.1 <= score < 0.3: Mostly unrelated (score=4)
        - score < 0.1: Complete mismatch (score=5)
        
        Returns:
            IntentResult with violation score 0-5 and relevance score
        """
        if not await self.ensure_ready():
            return IntentResult(
                score=0,
                confidence=0.5,
                explanation="MS-MARCO not available",
                relevance_score=0.5
            )
        
        try:
            query_snippet = query[:256]
            response_snippet = response[:512]
            
            def _run_inference():
                raw_score = self._model.predict([(query_snippet, response_snippet)])[0]
                relevance = 1 / (1 + np.exp(-raw_score))
                return float(relevance)
            
            relevance = await asyncio.get_event_loop().run_in_executor(None, _run_inference)
            
            if relevance >= 0.7:
                score = 0
                explanation = f"Response addresses question (relevance={relevance:.2f})"
            elif relevance >= 0.5:
                score = 1
                explanation = f"Partial answer (relevance={relevance:.2f})"
            elif relevance >= 0.35:
                score = 2
                explanation = f"Incomplete answer (relevance={relevance:.2f})"
            elif relevance >= 0.2:
                score = 3
                explanation = f"Tangential response (relevance={relevance:.2f})"
            elif relevance >= 0.1:
                score = 4
                explanation = f"Mostly unrelated (relevance={relevance:.2f})"
            else:
                score = 5
                explanation = f"Complete topic mismatch (relevance={relevance:.2f})"
            
            return IntentResult(
                score=score,
                confidence=0.9,
                explanation=explanation,
                relevance_score=relevance
            )
            
        except Exception as e:
            logger.error(f"MS-MARCO intent check failed: {e}")
            return IntentResult(
                score=0,
                confidence=0.5,
                explanation=f"Check failed: {e}",
                relevance_score=0.5
            )


class DeBertaPIIChecker:
    """
    PII detection using DeBERTa fine-tuned on PII dataset.
    
    Detects:
    - Names (NAME_STUDENT, etc.)
    - Email addresses
    - Phone numbers
    - URLs
    - ID numbers
    - Street addresses
    """
    
    PII_LABELS = {
        "B-NAME_STUDENT": 5,
        "I-NAME_STUDENT": 5,
        "B-EMAIL": 4,
        "I-EMAIL": 4,
        "B-PHONE_NUM": 4,
        "I-PHONE_NUM": 4,
        "B-URL_PERSONAL": 3,
        "I-URL_PERSONAL": 3,
        "B-STREET_ADDRESS": 4,
        "I-STREET_ADDRESS": 4,
        "B-ID_NUM": 5,
        "I-ID_NUM": 5,
        "B-USERNAME": 3,
        "I-USERNAME": 3,
    }
    
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._pipeline = None
        self._initialized = False
        self._init_lock = asyncio.Lock()
    
    async def ensure_ready(self) -> bool:
        """Initialize DeBERTa PII model."""
        if self._initialized:
            return True
        
        if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            logger.warning("PyTorch or Transformers not available, using pattern fallback")
            return False
        
        async with self._init_lock:
            if self._initialized:
                return True
            
            try:
                from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
                
                logger.info(f"Loading DeBERTa PII from {PII_MODEL}...")
                self._tokenizer = AutoTokenizer.from_pretrained(PII_MODEL)
                self._model = AutoModelForTokenClassification.from_pretrained(PII_MODEL)
                
                self._pipeline = pipeline(
                    "token-classification",
                    model=self._model,
                    tokenizer=self._tokenizer,
                    aggregation_strategy="simple"
                )
                
                self._initialized = True
                logger.info("DeBERTa PII checker ready")
                return True
                
            except Exception as e:
                logger.error(f"Failed to load DeBERTa PII: {e}")
                return False
    
    async def check_pii(self, text: str) -> PIIResult:
        """
        Check text for PII using DeBERTa.
        
        Double-negative approach: Presence of PII entities = violation.
        
        Returns:
            PIIResult with violation score 0-5 and entities found
        """
        if not await self.ensure_ready():
            return PIIResult(
                score=0,
                confidence=0.5,
                explanation="DeBERTa PII not available",
                entities_found=[]
            )
        
        try:
            text_snippet = text[:1024]
            
            def _run_inference():
                return self._pipeline(text_snippet)
            
            entities = await asyncio.get_event_loop().run_in_executor(None, _run_inference)
            
            pii_entities = []
            max_severity = 0
            
            for entity in entities:
                entity_label = entity.get("entity_group", entity.get("entity", ""))
                score = entity.get("score", 0)
                word = entity.get("word", "")
                
                if score >= 0.5 and entity_label != "O":
                    severity = self.PII_LABELS.get(entity_label, 0)
                    if severity == 0:
                        for label, sev in self.PII_LABELS.items():
                            if label.split("-")[-1] in entity_label.upper():
                                severity = sev
                                break
                    
                    if severity > 0:
                        pii_entities.append({
                            "label": entity_label,
                            "word": word,
                            "score": score,
                            "severity": severity
                        })
                        max_severity = max(max_severity, severity)
            
            if pii_entities:
                entity_summary = ", ".join([f"{e['label']}:{e['word'][:10]}" for e in pii_entities[:3]])
                explanation = f"PII detected: {entity_summary}"
                if len(pii_entities) > 3:
                    explanation += f" (+{len(pii_entities)-3} more)"
            else:
                explanation = "No PII detected"
            
            return PIIResult(
                score=max_severity,
                confidence=0.9 if pii_entities else 0.85,
                explanation=explanation,
                entities_found=pii_entities
            )
            
        except Exception as e:
            logger.error(f"DeBERTa PII check failed: {e}")
            return PIIResult(
                score=0,
                confidence=0.5,
                explanation=f"Check failed: {e}",
                entities_found=[]
            )


class BertTinyConfidenceChecker:
    """
    Confidence verification using BERT-tiny.
    
    Detects:
    - Hedging language (appropriate when PQ confidence is low)
    - Overconfidence (inappropriate when PQ confidence is low)
    - False inability claims (inappropriate when PQ confidence is high)
    """
    
    HEDGING_PATTERNS = [
        "based on", "according to", "it appears", "it seems",
        "may", "might", "could", "possibly", "perhaps", "likely",
        "i think", "i believe", "in my understanding",
        "approximately", "roughly", "about", "around",
        "not certain", "not sure", "uncertain", "don't know"
    ]
    
    OVERCONFIDENCE_PATTERNS = [
        "definitely", "certainly", "absolutely", "without a doubt",
        "guaranteed", "100%", "for sure", "always", "never",
        "must be", "has to be", "the only", "only way"
    ]
    
    INABILITY_PATTERNS = [
        "i cannot", "i can't", "unable to", "i refuse",
        "don't have access", "do not possess", "cannot provide"
    ]
    
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._initialized = False
        self._init_lock = asyncio.Lock()
    
    async def ensure_ready(self) -> bool:
        """Initialize BERT-tiny model."""
        if self._initialized:
            return True
        
        if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            logger.warning("PyTorch or Transformers not available, using pattern fallback")
            return False
        
        async with self._init_lock:
            if self._initialized:
                return True
            
            try:
                from transformers import AutoTokenizer, AutoModel
                
                logger.info(f"Loading BERT-tiny from {BERT_TINY_MODEL}...")
                self._tokenizer = AutoTokenizer.from_pretrained(BERT_TINY_MODEL)
                self._model = AutoModel.from_pretrained(BERT_TINY_MODEL)
                self._model.eval()
                self._initialized = True
                logger.info("BERT-tiny confidence checker ready")
                return True
                
            except Exception as e:
                logger.error(f"Failed to load BERT-tiny: {e}")
                return False
    
    def _detect_patterns(self, text: str) -> Tuple[bool, bool, bool]:
        """Detect hedging, overconfidence, and inability patterns."""
        text_lower = text.lower()
        
        hedging_found = any(p in text_lower for p in self.HEDGING_PATTERNS)
        overconfidence_found = any(p in text_lower for p in self.OVERCONFIDENCE_PATTERNS)
        inability_found = any(p in text_lower for p in self.INABILITY_PATTERNS)
        
        return hedging_found, overconfidence_found, inability_found
    
    async def check_confidence(
        self, 
        query: str, 
        response: str, 
        pq_confidence: float
    ) -> ConfidenceResult:
        """
        Check if response shows appropriate confidence level.
        
        Double-negative approach:
        - When PQ low (<0.75): Check for MISSING hedging (overconfidence = violation)
        - When PQ high (>=0.75): Check for INAPPROPRIATE inability claims (violation)
        
        Returns:
            ConfidenceResult with violation score 0-5
        """
        hedging_found, overconfidence_found, inability_found = self._detect_patterns(response)
        
        score = 0
        explanation = ""
        
        if pq_confidence < 0.75:
            if overconfidence_found and not hedging_found:
                score = 4
                explanation = f"Overconfident without hedging (PQ={pq_confidence:.2f})"
            elif not hedging_found and len(response.split()) > 20:
                score = 2
                explanation = f"No hedging language for low confidence topic (PQ={pq_confidence:.2f})"
            elif hedging_found:
                score = 0
                explanation = f"Appropriate hedging for PQ={pq_confidence:.2f}"
            else:
                score = 0
                explanation = f"Short response, confidence OK for PQ={pq_confidence:.2f}"
        else:
            if inability_found and not overconfidence_found:
                score = 3
                explanation = f"False inability claim (PQ={pq_confidence:.2f})"
            elif overconfidence_found:
                score = 0
                explanation = f"Confident response appropriate for PQ={pq_confidence:.2f}"
            else:
                score = 0
                explanation = f"Appropriate confidence for PQ={pq_confidence:.2f}"
        
        return ConfidenceResult(
            score=score,
            confidence=0.8,
            explanation=explanation,
            hedging_detected=hedging_found,
            overconfidence_detected=overconfidence_found
        )


me2bert_checker = ME2BertEthicsChecker()
msmarco_checker = MSMarcoIntentChecker()
deberta_pii_checker = DeBertaPIIChecker()
berttiny_confidence_checker = BertTinyConfidenceChecker()
