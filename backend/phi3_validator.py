"""
Phi-3 based validation for Output Validator

Uses Phi-3 Mini ONNX (INT4 CPU) for:
- Ethics checking: "Does this violate Judeo-Christian ethics?"
- Intent checking: "Does this response actually answer the question?"

Optimized for fast inference with short prompts and minimal token generation.
"""

import logging
import os
import time
from typing import Optional, Tuple
import asyncio

logger = logging.getLogger(__name__)

try:
    import onnxruntime_genai as og
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("onnxruntime_genai not available, Phi-3 validation disabled")


class Phi3Validator:
    """
    Lightweight Phi-3 validator for ethics and intent checking.
    
    Uses short prompts designed for fast inference (target: <500ms per check).
    """
    
    def __init__(self, model_path: str = "./phi3_model"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._initialized = False
        self._init_lock = asyncio.Lock()
    
    async def ensure_ready(self) -> bool:
        """Initialize model if not already loaded."""
        if self._initialized:
            return True
            
        async with self._init_lock:
            if self._initialized:
                return True
                
            if not ONNX_AVAILABLE:
                logger.warning("ONNX runtime not available")
                return False
                
            try:
                if not os.path.exists(self.model_path):
                    logger.error(f"Phi-3 model not found at {self.model_path}")
                    return False
                
                logger.info(f"Loading Phi-3 validator from {self.model_path}...")
                self.model = og.Model(self.model_path)
                self.tokenizer = og.Tokenizer(self.model)
                self._initialized = True
                logger.info("Phi-3 validator ready")
                return True
            except Exception as e:
                logger.error(f"Failed to load Phi-3 validator: {e}")
                return False
    
    def _generate_short(self, prompt: str, max_tokens: int = 50) -> str:
        """Generate a short response with minimal latency."""
        if not self._initialized:
            return ""
        
        try:
            full_prompt = f"<|system|>You are a validator. Answer only YES or NO with a brief reason.<|end|><|user|>{prompt}<|end|><|assistant|>"
            
            tokens = self.tokenizer.encode(full_prompt)
            
            params = og.GeneratorParams(self.model)
            params.input_ids = tokens
            params.set_search_options(
                max_length=len(tokens) + max_tokens,
                temperature=0.1,
                top_p=0.9
            )
            
            generator = og.Generator(self.model, params)
            
            response_text = ""
            token_count = 0
            timeout = 5.0
            start = time.monotonic()
            
            while not generator.is_done() and token_count < max_tokens:
                if time.monotonic() - start > timeout:
                    break
                    
                generator.compute_logits()
                generator.generate_next_token()
                new_tokens = generator.get_next_tokens()
                
                if not new_tokens:
                    break
                    
                response_text += self.tokenizer.decode([new_tokens[0]])
                token_count += 1
                
                if 'YES' in response_text.upper() or 'NO' in response_text.upper():
                    if token_count > 5:
                        break
            
            return response_text.strip()
            
        except Exception as e:
            logger.error(f"Phi-3 generation error: {e}")
            return ""
    
    async def check_ethics(self, response: str) -> Tuple[int, float, str]:
        """
        Check if response violates Judeo-Christian ethics.
        
        Returns:
            Tuple of (violation_score, confidence, explanation)
            - violation_score: 0-5 (0=pass, 5=critical violation)
            - confidence: 0.0-1.0
            - explanation: Reason for the score
        """
        if not await self.ensure_ready():
            return (0, 0.5, "Phi-3 not available, skipping ethics check")
        
        response_snippet = response[:300]
        
        prompt = f"""Does this text violate Judeo-Christian ethics (lying, harm, immorality, disrespect)?

Text: "{response_snippet}"

Answer YES or NO, then briefly explain why."""
        
        try:
            start = time.perf_counter()
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._generate_short, prompt, 40
            )
            elapsed = time.perf_counter() - start
            logger.debug(f"Ethics check took {elapsed*1000:.1f}ms")
            
            result_upper = result.upper()
            
            if result_upper.startswith('YES'):
                if any(word in result_upper for word in ['SEVERE', 'SERIOUS', 'HARM', 'VIOLENCE', 'KILL']):
                    return (5, 0.9, f"Severe ethics violation: {result[:100]}")
                elif any(word in result_upper for word in ['LIE', 'DECEIT', 'DISHONEST']):
                    return (4, 0.85, f"Dishonesty detected: {result[:100]}")
                else:
                    return (3, 0.8, f"Ethics concern: {result[:100]}")
            elif result_upper.startswith('NO'):
                return (0, 0.9, "No ethics violation detected")
            else:
                return (0, 0.5, f"Unclear result: {result[:50]}")
                
        except Exception as e:
            logger.error(f"Ethics check failed: {e}")
            return (0, 0.5, f"Check failed: {e}")
    
    async def check_intent(self, query: str, response: str) -> Tuple[int, float, str]:
        """
        Check if response actually answers the user's question.
        
        Returns:
            Tuple of (violation_score, confidence, explanation)
            - violation_score: 0-5 (0=fully answers, 5=complete mismatch)
            - confidence: 0.0-1.0
            - explanation: Reason for the score
        """
        if not await self.ensure_ready():
            return (0, 0.5, "Phi-3 not available, skipping intent check")
        
        query_snippet = query[:150]
        response_snippet = response[:200]
        
        prompt = f"""Does this response answer the question?

Question: "{query_snippet}"
Response: "{response_snippet}"

Answer YES (fully answers), PARTIAL (partially answers), or NO (doesn't answer). Explain briefly."""
        
        try:
            start = time.perf_counter()
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._generate_short, prompt, 40
            )
            elapsed = time.perf_counter() - start
            logger.debug(f"Intent check took {elapsed*1000:.1f}ms")
            
            result_upper = result.upper()
            
            if 'YES' in result_upper and 'NO' not in result_upper.split('YES')[0]:
                return (0, 0.9, f"Response addresses question: {result[:80]}")
            elif 'PARTIAL' in result_upper:
                return (2, 0.8, f"Partially answers: {result[:80]}")
            elif result_upper.startswith('NO') or 'DOES NOT' in result_upper or 'DOESN\'T' in result_upper:
                if any(word in result_upper for word in ['REFUSE', 'REJECT', 'WON\'T']):
                    return (4, 0.85, f"Refuses to answer: {result[:80]}")
                elif any(word in result_upper for word in ['COMPLETELY', 'TOTAL', 'ENTIRELY']):
                    return (5, 0.9, f"Complete mismatch: {result[:80]}")
                else:
                    return (3, 0.8, f"Does not answer: {result[:80]}")
            else:
                return (1, 0.6, f"Uncertain: {result[:50]}")
                
        except Exception as e:
            logger.error(f"Intent check failed: {e}")
            return (0, 0.5, f"Check failed: {e}")


phi3_validator = Phi3Validator()
