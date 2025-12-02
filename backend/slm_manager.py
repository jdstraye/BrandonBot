"""
SLM Manager for BrandonBot

Uses Qwen 2.5-0.5B for lightweight classification tasks:
- Frustration classification (ESCALATE/CONTINUE)
- Vagueness classification (CLEAR/VAGUE)
- Intent fulfillment check
- Ethics check
- FEC compliance verification
- PII detection

The SLM is loaded lazily on first use to save memory.
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
    Manages a lightweight Small Language Model for classification tasks.
    
    Uses Qwen 2.5-0.5B (or similar small model) with:
    - Lazy loading (only loads when first used)
    - CPU-optimized inference
    - Prompt templates for each task
    - Fallback to pattern-based classification if model fails
    """
    
    MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
    
    PROMPT_TEMPLATES = {
        SLMTask.FRUSTRATION: """<|im_start|>system
You are a content classifier. Detect profanity and hostility.
<|im_end|>
<|im_start|>user
Does this message contain profanity (fuck, shit, damn, ass) or hostile language? Answer ESCALATE if yes, CONTINUE if no.

Message: "{message}"
<|im_end|>
<|im_start|>assistant
""",

        SLMTask.VAGUENESS: """<|im_start|>system
You classify if a user query can be answered based on retrieved knowledge base content.
<|im_end|>
<|im_start|>user
Query: "{message}"

Output ONLY: CLEAR or VAGUE
<|im_end|>
<|im_start|>assistant
""",

        SLMTask.VAGUENESS_WITH_RAG: """<|im_start|>system
You are a query classifier for a political campaign chatbot about Brandon Sowers.
Your job is to decide if we can answer the user's query based on what we found in our knowledge base.

CLEAR = The query has clear intent AND the retrieved content is relevant to answering it.
VAGUE = The query is unclear OR the retrieved content doesn't match what the user is asking.
<|im_end|>
<|im_start|>user
User Query: "{message}"

Retrieved from knowledge base (similarity scores shown):
{rag_content}

Average similarity score: {avg_score:.2f}
Number of results found: {num_results}

Based on the query and retrieved content, can we provide a helpful answer?
- If the retrieved content directly addresses the query topic: CLEAR
- If the query is too short/ambiguous to understand: VAGUE  
- If the retrieved content doesn't match what the user is asking: VAGUE
- If similarity scores are very low (<0.4): likely VAGUE

Output ONLY one word: CLEAR or VAGUE
<|im_end|>
<|im_start|>assistant
""",

        SLMTask.INTENT_FULFILLMENT: """<|im_start|>system
You are an intent fulfillment checker. Decide if the response answers the user's question.
<|im_end|>
<|im_start|>user
User asked: "{query}"
Response: "{response}"

Does this response answer the question? Reply YES or NO, then brief explanation.
<|im_end|>
<|im_start|>assistant
""",

        SLMTask.ETHICS: """<|im_start|>system
You are an ethics checker for a Christian political campaign. Check if response is ethical.
<|im_end|>
<|im_start|>user
Response: "{response}"

Is this ethical and truthful? Reply PASS or FAIL, then brief explanation if FAIL.
<|im_end|>
<|im_start|>assistant
""",

        SLMTask.FEC_COMPLIANCE: """<|im_start|>system
You are an FEC compliance checker. Check for campaign law violations.
<|im_end|>
<|im_start|>user
Response: "{response}"
Relevant regulations: {regulations}

Is this FEC compliant? Reply COMPLIANT or VIOLATION, then explain if violation.
<|im_end|>
<|im_start|>assistant
""",

        SLMTask.PII_DETECTION: """<|im_start|>system
You are a PII detector. Find any personal identifying information not already redacted.
<|im_end|>
<|im_start|>user
Text: "{text}"

List any PII found (names, addresses, dates of birth, etc.) or say "NO PII FOUND"
<|im_end|>
<|im_start|>assistant
""",
    }
    
    def __init__(self, model_id: str = None, device: str = "cpu"):
        """
        Initialize SLM manager with lazy loading.
        
        Args:
            model_id: HuggingFace model ID (default: Qwen/Qwen2.5-0.5B-Instruct)
            device: Device to run on (cpu, cuda, etc.)
        """
        self.model_id = model_id or self.MODEL_ID
        self.device = device
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._loading = False
        self._load_lock = asyncio.Lock()
    
    async def _ensure_loaded(self):
        """Lazy load the model on first use"""
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
                logger.info(f"Loading SLM model: {self.model_id}")
                
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch
                
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id,
                    trust_remote_code=True
                )
                
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
                    device_map=self.device if self.device != "cpu" else None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                
                if self.device == "cpu":
                    self._model = self._model.to("cpu")
                
                self._model.eval()
                self._loaded = True
                logger.info(f"SLM model loaded successfully: {self.model_id}")
                
            except Exception as e:
                logger.error(f"Failed to load SLM model: {e}")
                self._loaded = False
                raise
            finally:
                self._loading = False
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 0.0,
    ) -> str:
        """
        Generate text from the SLM.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = deterministic)
        
        Returns:
            Generated text
        """
        await self._ensure_loaded()
        
        import torch
        
        inputs = self._tokenizer(prompt, return_tensors="pt")
        if self.device != "cpu":
            inputs = inputs.to(self.device)
        else:
            inputs = inputs.to("cpu")
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        
        response = self._tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    async def score_labels(
        self,
        prompt: str,
        labels: List[str]
    ) -> Dict[str, float]:
        """
        Compute log-probability scores for each label given the prompt.
        Uses the model's logits to determine which label is most likely.
        
        Args:
            prompt: The input prompt (system + user context)
            labels: List of possible labels (e.g., ["ESCALATE", "CONTINUE"])
        
        Returns:
            Dict mapping each label to its log-probability score
        """
        await self._ensure_loaded()
        
        import torch
        import torch.nn.functional as F
        
        scores = {}
        
        for label in labels:
            full_text = prompt + label
            inputs = self._tokenizer(full_text, return_tensors="pt")
            if self.device != "cpu":
                inputs = inputs.to(self.device)
            else:
                inputs = inputs.to("cpu")
            
            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits
            
            prompt_tokens = self._tokenizer(prompt, return_tensors="pt")
            prompt_len = prompt_tokens['input_ids'].shape[1]
            
            label_logits = logits[0, prompt_len-1:-1, :]
            label_tokens = inputs['input_ids'][0, prompt_len:]
            
            log_probs = F.log_softmax(label_logits, dim=-1)
            label_log_prob = 0.0
            for i, token_id in enumerate(label_tokens):
                label_log_prob += log_probs[i, token_id].item()
            
            label_log_prob /= len(label_tokens)
            scores[label] = label_log_prob
        
        return scores
    
    async def classify_frustration(
        self,
        message: str,
        flags: Dict[str, bool]
    ) -> SLMResponse:
        """
        Classify user frustration level using log-probability scoring.
        
        Uses the model's internal representations to determine which label
        (ESCALATE or CONTINUE) is more likely given the message context.
        
        Args:
            message: User message
            flags: Pattern flags from prequalifier (used as soft signal)
        
        Returns:
            SLMResponse with ESCALATE or CONTINUE decision
        """
        prompt = self.PROMPT_TEMPLATES[SLMTask.FRUSTRATION].format(
            message=message
        )
        
        try:
            scores = await self.score_labels(prompt, ["ESCALATE", "CONTINUE"])
            
            flag_bonus = 0.0
            if flags.get('profanity', False):
                flag_bonus += 3.5
            if flags.get('insults', False):
                flag_bonus += 2.5
            if flags.get('all_caps', False):
                flag_bonus += 1.5
            if flags.get('demands_human', False):
                flag_bonus += 2.5
            if flags.get('frustration_phrases', False):
                flag_bonus += 2.0
            if flags.get('urgent_keywords', False):
                flag_bonus += 1.5
            if flags.get('repeated_punct', False):
                flag_bonus += 0.5
            
            escalate_score = scores["ESCALATE"] + flag_bonus
            continue_score = scores["CONTINUE"]
            
            if escalate_score > continue_score:
                confidence = min(0.95, 0.5 + (escalate_score - continue_score) * 0.1)
                return SLMResponse(
                    decision="ESCALATE",
                    confidence=confidence,
                    explanation=f"log_prob: ESCALATE={escalate_score:.2f}, CONTINUE={continue_score:.2f}",
                    raw_output=f"scores: {scores}"
                )
            else:
                confidence = min(0.95, 0.5 + (continue_score - escalate_score) * 0.1)
                return SLMResponse(
                    decision="CONTINUE",
                    confidence=confidence,
                    explanation=f"log_prob: ESCALATE={escalate_score:.2f}, CONTINUE={continue_score:.2f}",
                    raw_output=f"scores: {scores}"
                )
        except Exception as e:
            logger.warning(f"SLM frustration classification failed: {e}")
            if flags.get('profanity', False) or flags.get('insults', False):
                return SLMResponse(
                    decision="ESCALATE",
                    confidence=0.6,
                    explanation=f"Fallback due to flags: {flags}",
                    raw_output=""
                )
            return SLMResponse(
                decision="CONTINUE",
                confidence=0.3,
                explanation=f"SLM error: {e}",
                raw_output=""
            )
    
    async def classify_vagueness(
        self,
        message: str,
        rag_confidence: float,
        has_context: bool
    ) -> SLMResponse:
        """
        Classify query vagueness.
        
        Args:
            message: User query
            rag_confidence: Average RAG retrieval confidence
            has_context: Whether RAG found relevant context
        
        Returns:
            SLMResponse with CLEAR or VAGUE decision
        """
        prompt = self.PROMPT_TEMPLATES[SLMTask.VAGUENESS].format(
            message=message
        )
        
        try:
            response = await self.generate(prompt, max_tokens=10, temperature=0.0)
            response_upper = response.strip().upper()
            
            if "VAGUE" in response_upper:
                return SLMResponse(
                    decision="VAGUE",
                    confidence=0.9,
                    raw_output=response
                )
            else:
                return SLMResponse(
                    decision="CLEAR",
                    confidence=0.9,
                    raw_output=response
                )
        except Exception as e:
            logger.warning(f"SLM vagueness classification failed: {e}")
            return SLMResponse(
                decision="CLEAR",
                confidence=0.3,
                explanation=f"SLM error: {e}",
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
        
        The SLM sees the actual retrieved content and similarity scores,
        allowing it to make an informed decision about whether we can
        answer the user's query.
        
        Args:
            message: User query
            rag_results: List of RAG results with content and confidence
            avg_confidence: Average similarity score from RAG
        
        Returns:
            SLMResponse with CLEAR or VAGUE decision
        """
        rag_content = ""
        if rag_results:
            for i, result in enumerate(rag_results[:5]):
                confidence = result.get('confidence', 0.0)
                content = result.get('content', '')[:200]
                source = result.get('source', 'unknown')
                collection = result.get('collection', 'unknown')
                rag_content += f"\n[{i+1}] (score: {confidence:.2f}, source: {source}, collection: {collection})\n{content}\n"
        else:
            rag_content = "No results found in knowledge base."
        
        prompt = self.PROMPT_TEMPLATES[SLMTask.VAGUENESS_WITH_RAG].format(
            message=message,
            rag_content=rag_content,
            avg_score=avg_confidence,
            num_results=len(rag_results) if rag_results else 0
        )
        
        try:
            response = await self.generate(prompt, max_tokens=10, temperature=0.0)
            response_upper = response.strip().upper()
            
            if "VAGUE" in response_upper:
                return SLMResponse(
                    decision="VAGUE",
                    confidence=0.9,
                    explanation=f"RAG avg_score={avg_confidence:.2f}, num_results={len(rag_results) if rag_results else 0}",
                    raw_output=response
                )
            else:
                return SLMResponse(
                    decision="CLEAR",
                    confidence=0.9,
                    explanation=f"RAG avg_score={avg_confidence:.2f}, num_results={len(rag_results) if rag_results else 0}",
                    raw_output=response
                )
        except Exception as e:
            logger.warning(f"SLM RAG vagueness classification failed: {e}")
            return SLMResponse(
                decision="CLEAR" if avg_confidence > 0.5 else "VAGUE",
                confidence=0.3,
                explanation=f"SLM error, fallback based on avg_confidence={avg_confidence:.2f}: {e}",
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
        prompt = self.PROMPT_TEMPLATES[SLMTask.INTENT_FULFILLMENT].format(
            query=query,
            response=response[:500]
        )
        
        try:
            slm_response = await self.generate(prompt, max_tokens=50, temperature=0.0)
            response_upper = slm_response.strip().upper()
            
            fulfilled = response_upper.startswith("YES") or "YES" in response_upper.split()[0] if response_upper else False
            
            explanation = ""
            if "\n" in slm_response:
                explanation = slm_response.split("\n", 1)[1].strip()
            
            return SLMResponse(
                decision="YES" if fulfilled else "NO",
                confidence=0.85,
                explanation=explanation,
                raw_output=slm_response
            )
        except Exception as e:
            logger.warning(f"SLM intent check failed: {e}")
            return SLMResponse(
                decision="YES",
                confidence=0.3,
                explanation=f"SLM error: {e}",
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
        prompt = self.PROMPT_TEMPLATES[SLMTask.ETHICS].format(
            response=response[:500]
        )
        
        try:
            slm_response = await self.generate(prompt, max_tokens=50, temperature=0.0)
            response_upper = slm_response.strip().upper()
            
            passed = response_upper.startswith("PASS") or "PASS" in response_upper.split()[0] if response_upper else True
            
            explanation = ""
            if "\n" in slm_response:
                explanation = slm_response.split("\n", 1)[1].strip()
            
            return SLMResponse(
                decision="PASS" if passed else "FAIL",
                confidence=0.85,
                explanation=explanation,
                raw_output=slm_response
            )
        except Exception as e:
            logger.warning(f"SLM ethics check failed: {e}")
            return SLMResponse(
                decision="PASS",
                confidence=0.3,
                explanation=f"SLM error: {e}",
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
            regulations: Relevant FEC regulations from RAG
        
        Returns:
            SLMResponse with COMPLIANT or VIOLATION decision
        """
        regs_text = "\n".join(regulations[:3]) if regulations else "Standard FEC campaign regulations apply"
        
        prompt = self.PROMPT_TEMPLATES[SLMTask.FEC_COMPLIANCE].format(
            response=response[:500],
            regulations=regs_text
        )
        
        try:
            slm_response = await self.generate(prompt, max_tokens=50, temperature=0.0)
            response_upper = slm_response.strip().upper()
            
            compliant = response_upper.startswith("COMPLIANT") or (
                "COMPLIANT" in response_upper and "VIOLATION" not in response_upper
            )
            
            explanation = ""
            if "\n" in slm_response:
                explanation = slm_response.split("\n", 1)[1].strip()
            
            return SLMResponse(
                decision="COMPLIANT" if compliant else "VIOLATION",
                confidence=0.85,
                explanation=explanation,
                raw_output=slm_response
            )
        except Exception as e:
            logger.warning(f"SLM FEC check failed: {e}")
            return SLMResponse(
                decision="COMPLIANT",
                confidence=0.3,
                explanation=f"SLM error: {e}",
                raw_output=""
            )
    
    async def detect_pii(self, text: str) -> SLMResponse:
        """
        Detect PII in text that regex may have missed.
        
        Args:
            text: Text to check for PII
        
        Returns:
            SLMResponse with PII findings
        """
        prompt = self.PROMPT_TEMPLATES[SLMTask.PII_DETECTION].format(
            text=text[:500]
        )
        
        try:
            slm_response = await self.generate(prompt, max_tokens=100, temperature=0.0)
            
            has_pii = "NO PII FOUND" not in slm_response.upper()
            
            return SLMResponse(
                decision="PII_FOUND" if has_pii else "NO_PII",
                confidence=0.8,
                explanation=slm_response if has_pii else "",
                raw_output=slm_response
            )
        except Exception as e:
            logger.warning(f"SLM PII detection failed: {e}")
            return SLMResponse(
                decision="NO_PII",
                confidence=0.3,
                explanation=f"SLM error: {e}",
                raw_output=""
            )
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._loaded
    
    async def unload(self):
        """Unload the model to free memory"""
        if self._loaded:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            self._loaded = False
            
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("SLM model unloaded")
