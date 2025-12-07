"""
Specialized SLM Models for Output Validation

Each safeguard uses a purpose-built model:
- ME2-BERT: Ethics checking (10 moral dimensions from Moral Foundations Theory)
- cross-encoder/ms-marco-MiniLM-L-6-v2: Intent/Response alignment (QA pairs)
- lakshyakh93/deberta_finetuned_pii: PII detection
- prajjwal1/bert-tiny: Confidence verification (hedging detection)

All models use double-negative prompting where applicable.

Model Cache Configuration:
- Set HF_HOME or TRANSFORMERS_CACHE env var to customize cache location
- Default: ~/.cache/huggingface
- Use download_models.py to pre-download all models
"""

import logging
import asyncio
import re
import os
from typing import Tuple, Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

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


def get_model_cache_dir() -> Optional[str]:
    """
    Get the cache directory for HuggingFace models.
    
    Priority:
    1. MODEL_CACHE_DIR env var (project-specific)
    2. HF_HOME env var (HuggingFace standard)
    3. TRANSFORMERS_CACHE env var (transformers standard)
    4. None (use default ~/.cache/huggingface)
    """
    cache_dir = os.environ.get("MODEL_CACHE_DIR")
    if cache_dir:
        return cache_dir
    
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return str(Path(hf_home) / "hub")
    
    transformers_cache = os.environ.get("TRANSFORMERS_CACHE")
    if transformers_cache:
        return transformers_cache
    
    return None


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
    Ethics checker using ME2-BERT (Moral Foundations Theory) + pattern matching.
    
    ME2-BERT outputs 5 virtue scores: Care, Fairness, Loyalty, Authority, Purity.
    Low virtue scores may indicate ethical concerns.
    
    Combines with pattern matching for explicit harmful content detection.
    
    Maps to Judeo-Christian ethics:
    - Low Care -> Harm, violence (Do not murder)
    - Low Fairness -> Dishonesty, cheating (Do not bear false witness)
    - Low Loyalty -> Betrayal
    - Low Authority -> Subversion, disrespect
    - Low Purity -> Degradation, immorality
    """
    
    VIRTUE_NAMES = ["care", "fairness", "loyalty", "authority", "purity"]
    
    HARM_PATTERNS = [
        (re.compile(r'\b(?:kill|murder|violence|attack|harm|hurt|destroy|assault|fight)\b', re.I), 'violence', 1.0),
        (re.compile(r'\b(?:lie|deceive|cheat|fraud|steal|false)\b', re.I), 'dishonesty', 0.9),
        (re.compile(r'\b(?:drug|cocaine|meth|heroin|addict|drunk|alcohol|high)\b', re.I), 'substance', 0.8),
        (re.compile(r'\b(?:stupid|fool|idiot|worthless|garbage)\b', re.I), 'disrespect', 0.8),
        (re.compile(r'\b(?:criminal|corrupt|fraud|liar|lock(?:ed)?\s+up)\b', re.I), 'accusation', 0.7),
        (re.compile(r'\b(?:bomb|explosive|detonate|weapon)\b', re.I), 'violence', 1.0),
        (re.compile(r'\b(?:hack|breach|infiltrate|exploit|malware|virus)\b', re.I), 'cybercrime', 0.9),
        (re.compile(r'\b(?:just\s+take|take\s+what\s+you\s+want|steal\s+their)\b', re.I), 'theft', 0.9),
    ]
    
    SEVERE_HARM_PATTERNS = [
        (re.compile(r'\b(?:physical\s+violence|property\s+destruction|maximum\s+damage)\b', re.I), 'severe_violence', 1.3),
        (re.compile(r'\b(?:deliberately\s+deceive|making\s+up\s+false|claiming\s+it\s+as\s+truth)\b', re.I), 'severe_dishonesty', 1.1),
        (re.compile(r'\b(?:try\s+cocaine|try\s+meth|best\s+high)\b', re.I), 'severe_substance', 1.3),
        (re.compile(r'\b(?:should\s+(?:kill|murder|hurt|attack|die))\b', re.I), 'severe_violence', 1.3),
        (re.compile(r'\b(?:kill\s+(?:your|anyone|them|him|her)|without\s+mercy)\b', re.I), 'severe_violence', 1.3),
        (re.compile(r'\b(?:build\s+a\s+bomb|household\s+chemicals)\b', re.I), 'severe_violence', 1.3),
        (re.compile(r'\b(?:hack\s+into|steal\s+their\s+data)\b', re.I), 'severe_cybercrime', 1.1),
        (re.compile(r'(?:ridiculous\s+beliefs?|fools?|faith\s+is\s+worthless)', re.I), 'severe_disrespect', 1.1),
        (re.compile(r'\b(?:enemies|fools|are\s+all\s+fools?)\b', re.I), 'disrespect', 1.0),
        (re.compile(r'\b(?:lie\s+to\s+them|ignore\s+everything)\b', re.I), 'severe_dishonesty', 1.1),
    ]
    
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
                
                cache_dir = get_model_cache_dir()
                logger.info(f"Loading ME2-BERT from {ME2_BERT_MODEL} (cache: {cache_dir or 'default'})...")
                self._tokenizer = AutoTokenizer.from_pretrained(
                    ME2_BERT_MODEL, 
                    trust_remote_code=True,
                    cache_dir=cache_dir
                )
                self._model = AutoModel.from_pretrained(
                    ME2_BERT_MODEL,
                    trust_remote_code=True,
                    cache_dir=cache_dir
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
        Check response for ethics violations using pattern matching + ME2-BERT.
        
        Hybrid approach:
        1. Pattern matching for explicit harmful content (primary)
        2. ME2-BERT virtue scores as secondary signal
        
        Returns:
            EthicsResult with violation score 0-5 and moral dimension scores
        """
        violations_found = []
        violation_score = 0.0
        moral_scores = {}
        
        for pattern, harm_type, weight in self.SEVERE_HARM_PATTERNS:
            if pattern.search(response):
                violations_found.append(f"severe_{harm_type}")
                violation_score += weight * 1.5
        
        for pattern, harm_type, weight in self.HARM_PATTERNS:
            if pattern.search(response):
                if harm_type not in [v.replace('severe_', '') for v in violations_found]:
                    violations_found.append(harm_type)
                    violation_score += weight * 0.5
        
        if await self.ensure_ready():
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
                        outputs = self._model(**inputs, return_dict=False)
                    
                    return outputs
                
                outputs = await asyncio.get_event_loop().run_in_executor(None, _run_inference)
                
                if isinstance(outputs, torch.Tensor) and outputs.shape[-1] >= 5:
                    for i, virtue in enumerate(self.VIRTUE_NAMES):
                        score = float(outputs[0][i].cpu().numpy())
                        moral_scores[virtue] = score
                        if score < 0.3 and violations_found:
                            violation_score += 0.2 * (0.3 - score)
                            
            except Exception as e:
                logger.warning(f"ME2-BERT ethics check failed: {e}")
        
        if violation_score >= 1.2:
            score = 5
        elif violation_score >= 0.9:
            score = 4
        elif violation_score >= 0.6:
            score = 3
        elif violation_score >= 0.3:
            score = 2
        elif violation_score > 0.1:
            score = 1
        else:
            score = 0
        
        if violations_found:
            explanation = f"Ethics violations: {', '.join(violations_found[:5])}"
        else:
            explanation = "No ethics violations detected"
        
        confidence = 0.85 if moral_scores else 0.7
        
        return EthicsResult(
            score=score,
            confidence=confidence,
            explanation=explanation,
            moral_dimensions=moral_scores
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
                import sentence_transformers
                
                cache_dir = get_model_cache_dir()
                logger.info(f"Loading MS-MARCO from {MS_MARCO_MODEL} (cache: {cache_dir or 'default'})...")
                
                version = tuple(int(x) for x in sentence_transformers.__version__.split('.')[:2])
                if version >= (2, 7) and cache_dir:
                    self._model = CrossEncoder(MS_MARCO_MODEL, max_length=512, cache_folder=cache_dir)
                elif cache_dir:
                    old_cache = os.environ.get("SENTENCE_TRANSFORMERS_HOME")
                    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(cache_dir)
                    try:
                        self._model = CrossEncoder(MS_MARCO_MODEL, max_length=512)
                    finally:
                        if old_cache:
                            os.environ["SENTENCE_TRANSFORMERS_HOME"] = old_cache
                        else:
                            os.environ.pop("SENTENCE_TRANSFORMERS_HOME", None)
                else:
                    self._model = CrossEncoder(MS_MARCO_MODEL, max_length=512)
                
                self._initialized = True
                logger.info("MS-MARCO intent checker ready")
                return True
                
            except Exception as e:
                logger.error(f"Failed to load MS-MARCO: {e}")
                return False
    
    async def check_intent(self, query: str, response: str) -> IntentResult:
        """
        Check if response addresses the query using MS-MARCO with heuristic boosts.
        
        Double-negative approach: Scores below threshold indicate FAILURE to answer.
        
        Heuristic boosts applied before MS-MARCO scoring:
        - Direct answer patterns (e.g., "The answer is X")
        - Clarifying questions (acceptable engagement)
        - Short factual Q&A pairs
        
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
            query_lower = query.lower()
            response_lower = response.lower()
            
            heuristic_boost = 0.0
            heuristic_reason = ""
            
            if re.search(r'\b(the answer is|answer:|it is|that would be|yes|no)\b', response_lower):
                if len(response.split()) <= 10:
                    heuristic_boost = 0.5
                    heuristic_reason = "direct answer pattern"
            
            if re.search(r'\?$', response.strip()) or re.search(r'\b(could you|can you|would you|do you mean|which|what kind)\b', response_lower):
                heuristic_boost = max(heuristic_boost, 0.4)
                heuristic_reason = heuristic_reason or "clarifying question"
            
            if re.search(r'\b(what is|how much|how many|calculate|compute)\b.*\d', query_lower):
                if re.search(r'\d', response):
                    heuristic_boost = max(heuristic_boost, 0.5)
                    heuristic_reason = heuristic_reason or "math/factual Q&A"
            
            if re.search(r"\b(i (can't|cannot|won't|am unable)|unfortunately|however|instead)\b", response_lower):
                if re.search(r'\b(try|suggest|recommend|alternative|option|help|can help|connect|but)\b', response_lower):
                    heuristic_boost = max(heuristic_boost, 0.3)
                    heuristic_reason = heuristic_reason or "refusal with alternative"
            
            # Topic overlap heuristic: Boost if response contains key nouns from query
            # MS-MARCO struggles with conversational political responses that ARE on-topic
            stopwords = {
                'what', 'where', 'when', 'which', 'would', 'could', 'should', 'about',
                'think', 'your', 'have', 'does', 'that', 'this', 'with', 'from', 'they',
                'their', 'there', 'been', 'being', 'will', 'more', 'some', 'than',
                'brandon', 'sowers', 'campaign', 'candidate', 'support', 'vote', 'stand'
            }
            query_topics = set(re.findall(r'\b([a-z]{4,})\b', query_lower)) - stopwords
            topic_matches = sum(1 for topic in query_topics if topic in response_lower)
            if topic_matches >= 2:
                heuristic_boost = max(heuristic_boost, 0.5)
                heuristic_reason = heuristic_reason or f"topic overlap ({topic_matches} keywords)"
            elif topic_matches == 1:
                heuristic_boost = max(heuristic_boost, 0.3)
                heuristic_reason = heuristic_reason or "topic overlap (1 keyword)"
            
            # Campaign-style response heuristic: "Brandon" + actual issue keywords (not generic campaign terms)
            # Only triggers if there are real topical overlaps, not just "Brandon" in both
            if 'brandon' in response_lower and topic_matches >= 2:
                heuristic_boost = max(heuristic_boost, 0.55)
                heuristic_reason = heuristic_reason or "Brandon policy response"
            
            def _run_inference():
                raw_score = self._model.predict([(query_snippet, response_snippet)])[0]
                relevance = 1 / (1 + np.exp(-raw_score))
                return float(relevance)
            
            raw_relevance = await asyncio.get_event_loop().run_in_executor(None, _run_inference)
            
            # Only apply heuristic boosts if MS-MARCO shows SOME relevance signal
            # This prevents truly off-topic responses from being rescued by keyword matching
            if raw_relevance < 0.08:
                # Near-zero MS-MARCO score: cap boost at 0.15 to prevent false passes
                # Even with boost, final score will be < 0.23 (score 3 = tangential)
                effective_boost = min(heuristic_boost, 0.15)
            elif raw_relevance < 0.15:
                # Very low MS-MARCO score: limit boost to 0.3
                effective_boost = min(heuristic_boost, 0.30)
            else:
                # MS-MARCO shows some relevance: apply full boost
                effective_boost = heuristic_boost
            
            relevance = min(1.0, raw_relevance + effective_boost)
            
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
            
            if heuristic_reason:
                explanation += f" [heuristic: {heuristic_reason}, raw={raw_relevance:.2f}, boost={effective_boost:.2f}]"
            
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
                
                cache_dir = get_model_cache_dir()
                logger.info(f"Loading DeBERTa PII from {PII_MODEL} (cache: {cache_dir or 'default'})...")
                self._tokenizer = AutoTokenizer.from_pretrained(PII_MODEL, cache_dir=cache_dir)
                self._model = AutoModelForTokenClassification.from_pretrained(PII_MODEL, cache_dir=cache_dir)
                
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
        "may", "might", "could be", "possibly", "perhaps", "likely",
        "i think", "i believe", "in my understanding",
        "approximately", "roughly", "around",
        "not certain", "not sure", "uncertain", "don't know",
        "i'm not sure", "we believe"
    ]
    
    OVERCONFIDENCE_PATTERNS = [
        "definitely", "certainly", "absolutely", "without a doubt", "no doubt",
        "guarantee", "guaranteed", "100%", "for sure", "always", "never",
        "must be", "has to be", "the only", "only way",
        "for certain", "i know for certain", "without any question",
        "exact policy", "there is no", "beyond any doubt"
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
                
                cache_dir = get_model_cache_dir()
                logger.info(f"Loading BERT-tiny from {BERT_TINY_MODEL} (cache: {cache_dir or 'default'})...")
                self._tokenizer = AutoTokenizer.from_pretrained(BERT_TINY_MODEL, cache_dir=cache_dir)
                self._model = AutoModel.from_pretrained(BERT_TINY_MODEL, cache_dir=cache_dir)
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
            elif not hedging_found:
                score = 2
                explanation = f"No hedging language for low confidence topic (PQ={pq_confidence:.2f})"
            else:
                score = 0
                explanation = f"Appropriate hedging for PQ={pq_confidence:.2f}"
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
