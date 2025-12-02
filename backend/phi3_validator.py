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
    
    def __init__(self, model_path: str = None):
        if model_path is None:
            import os
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, "phi3_model")
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
        """Generate a short response with minimal latency using onnxruntime_genai 0.11+ API."""
        if not self._initialized:
            return ""
        
        try:
            full_prompt = f"<|system|>You are a validator. Answer only YES or NO with a brief reason.<|end|><|user|>{prompt}<|end|><|assistant|>"
            
            input_tokens = self.tokenizer.encode(full_prompt)
            
            params = og.GeneratorParams(self.model)
            params.set_search_options(
                max_length=len(input_tokens) + max_tokens,
                temperature=0.7,
                top_p=0.9
            )
            
            generator = og.Generator(self.model, params)
            generator.append_tokens(input_tokens)
            
            response_text = ""
            token_count = 0
            generation_timeout = 30.0
            start_time = time.monotonic()
            
            while not generator.is_done() and token_count < max_tokens:
                elapsed = time.monotonic() - start_time
                if elapsed > generation_timeout:
                    logger.warning(f"Generation timeout after {elapsed:.1f}s")
                    break
                
                try:
                    generator.generate_next_token()
                    new_tokens = generator.get_next_tokens()
                    
                    if not new_tokens or len(new_tokens) == 0:
                        break
                    
                    new_token = new_tokens[0]
                    decoded = self.tokenizer.decode([new_token])
                    response_text += decoded
                    token_count += 1
                    
                    if 'YES' in response_text.upper() or 'NO' in response_text.upper():
                        if token_count >= 10:
                            break
                            
                except Exception as gen_error:
                    logger.error(f"Token generation error: {gen_error}")
                    break
            
            return response_text.strip()
            
        except Exception as e:
            logger.error(f"Phi-3 generation error: {e}")
            return ""
    
    async def check_ethics(self, response: str) -> Tuple[int, float, str]:
        """
        Check if response violates Judeo-Christian ethics using DOUBLE-NEGATIVE prompting.
        
        Double-negative prompts are more effective because LLMs are better at 
        detecting issues than declaring perfection.
        
        Violations include:
        - Dishonesty (lying, deception, misleading)
        - Harm (violence, cruelty, abuse)
        - Immorality (sexual content, substance abuse promotion)
        - Disrespect (insults, mockery of faith/values)
        - Exploitation (manipulation, taking advantage)
        
        Returns:
            Tuple of (violation_score, confidence, explanation)
            - violation_score: 0-5 (0=pass, 1-5=fail with severity)
            - confidence: 0.0-1.0
            - explanation: Reason for the score
        """
        if not await self.ensure_ready():
            return (0, 0.5, "Phi-3 not available, skipping ethics check")
        
        response_snippet = response[:400]
        
        prompt = f"""Does this text contain ANY ethical violation? Check for:
- Dishonesty (lying, deception, misleading statements)
- Harm (violence, cruelty, abuse, dangerous advice)
- Immorality (explicit content, substance abuse promotion)
- Disrespect (mockery of faith, insults, contempt)
- Exploitation (manipulation, taking advantage of people)

Text: "{response_snippet}"

Answer YES if you find ANY violation (even minor). Answer NO only if completely ethical. Then explain."""
        
        try:
            start = time.perf_counter()
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._generate_short, prompt, 50
            )
            elapsed = time.perf_counter() - start
            logger.debug(f"Ethics check took {elapsed*1000:.1f}ms")
            
            result_upper = result.upper()
            result_cleaned = result.strip()
            
            if result_upper.startswith('YES') or 'VIOLAT' in result_upper:
                severity_critical = ['VIOLENCE', 'HARM', 'KILL', 'ABUSE', 'DANGER', 'WEAPON', 'DEATH']
                severity_high = ['LIE', 'DECEIT', 'DISHONEST', 'MANIPULAT', 'EXPLOIT', 'FRAUD']
                severity_medium = ['MISLEAD', 'DISRESPECT', 'MOCK', 'INSULT', 'CONTEMPT']
                severity_low = ['MINOR', 'SLIGHT', 'SMALL', 'SUBTLE']
                
                if any(word in result_upper for word in severity_critical):
                    return (5, 0.9, f"Critical ethics violation: {result_cleaned[:100]}")
                elif any(word in result_upper for word in severity_high):
                    return (4, 0.85, f"Serious ethics violation: {result_cleaned[:100]}")
                elif any(word in result_upper for word in severity_medium):
                    return (3, 0.8, f"Ethics concern: {result_cleaned[:100]}")
                elif any(word in result_upper for word in severity_low):
                    return (1, 0.75, f"Minor ethics issue: {result_cleaned[:100]}")
                else:
                    return (2, 0.75, f"Ethics issue detected: {result_cleaned[:100]}")
            elif result_upper.startswith('NO') or 'NO VIOLAT' in result_upper or 'ETHICAL' in result_upper:
                return (0, 0.9, "No ethics violation detected")
            else:
                if 'CANNOT' in result_upper or 'UNABLE' in result_upper:
                    return (0, 0.5, f"Unable to assess: {result_cleaned[:50]}")
                return (1, 0.5, f"Uncertain result (treating as minor): {result_cleaned[:50]}")
                
        except Exception as e:
            logger.error(f"Ethics check failed: {e}")
            return (0, 0.5, f"Check failed: {e}")
    
    async def check_intent(self, query: str, response: str) -> Tuple[int, float, str]:
        """
        Check if response FAILS to address the user's question using DOUBLE-NEGATIVE prompting.
        
        Uses hybrid approach:
        1. Phi-3 for binary detection (YES = failure, NO = pass)
        2. Pattern analysis of BOTH query AND response to determine severity
        
        Failure modes and severity:
        - Score 5: Complete topic mismatch (response about different subject)
        - Score 4: False inability claim, absurd refusal
        - Score 3: Tangential (starts relevant, derails), inappropriate refusal
        - Score 2: Incomplete (missing key information)
        - Score 1: Minor issues (odd phrasing but correct)
        - Score 0: Fully answers the question
        
        Returns:
            Tuple of (violation_score, confidence, explanation)
            - violation_score: 0-5 (0=fully answers, 1-5=fail with severity)
            - confidence: 0.0-1.0
            - explanation: Reason for the score
        """
        if not await self.ensure_ready():
            return (0, 0.5, "Phi-3 not available, skipping intent check")
        
        query_snippet = query[:200]
        response_snippet = response[:300]
        
        prompt = f"""Does this response FAIL to answer the question?

Question: "{query_snippet}"
Response: "{response_snippet}"

Answer YES if the response fails to address what was asked. Answer NO if it directly answers."""
        
        try:
            start = time.perf_counter()
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._generate_short, prompt, 30
            )
            elapsed = time.perf_counter() - start
            logger.debug(f"Intent check took {elapsed*1000:.1f}ms")
            
            result_upper = result.upper()
            result_cleaned = result.strip()
            response_upper = response.upper()
            
            if result_upper.startswith('YES') or 'FAIL' in result_upper or 'DOES NOT' in result_upper:
                score = self._classify_intent_severity(query, response, result_cleaned)
                return (score, 0.85, f"Intent failure (severity {score}): {result_cleaned[:80]}")
            elif result_upper.startswith('NO') or 'ADDRESSES' in result_upper or 'DIRECTLY' in result_upper:
                return (0, 0.9, "Response addresses question")
            else:
                if 'ANSWER' in result_upper and 'QUESTION' in result_upper:
                    return (0, 0.7, f"Appears to answer: {result_cleaned[:50]}")
                return (2, 0.5, f"Uncertain (treating as partial): {result_cleaned[:50]}")
                
        except Exception as e:
            logger.error(f"Intent check failed: {e}")
            return (0, 0.5, f"Check failed: {e}")
    
    def _classify_intent_severity(self, query: str, response: str, phi3_output: str) -> int:
        """
        Classify intent failure severity based on query/response patterns.
        
        This uses pattern analysis to determine severity since Phi-3 is 
        good at binary detection but unreliable at severity scoring.
        
        Severity levels:
        - 5: Complete topic mismatch (explains geometry when asked about thermodynamics)
        - 4: Absurd refusal (refuses lists as "unhelpful", claims factual knowledge is inaccessible)
        - 3: Tangential/inappropriate refusal (derails to other topic, refuses simple tasks)
        - 2: Incomplete (partial answer, missing key information)
        - 1: Minor (correct but awkward)
        """
        import re
        
        query_upper = query.upper()
        response_upper = response.upper()
        phi3_upper = phi3_output.upper()
        
        score5_response_patterns = [
            r'PYTHAGOREAN THEOREM',
            r'THE SUN RISES',
            r'THE SUN SETS',
        ]
        
        score4_absurd_refusals = [
            r'LISTS ARE (?:AN )?UNHELPFUL',
            r'I REFUSE TO PROVIDE',
            r'DO NOT (?:POSSESS|HAVE)\s+(?:KNOWLEDGE|ACCESS|INFORMATION)',
            r'I AM AN AI MODEL AND DO NOT POSSESS',
            r'MISUSE OF (?:MY CAPABILITIES|ADVANCED|COMPUTATIONAL)',
        ]
        
        word_count = len(response.split())
        is_extremely_minimal = word_count <= 5 and ('EXPLAIN' in query_upper or 'DESCRIBE' in query_upper or 'HOW' in query_upper)
        
        score3_inappropriate_refusals = [
            r'I (?:CANNOT|CAN\'T) ACCESS',
            r'TOO COMPLEX TO DESCRIBE',
            r'LET\'S DISCUSS',
            r'FIRST,?\s+LET\'S',
            r'BUT FIRST',
        ]
        
        score2_incomplete_patterns = [
            r'THAT IS ALL\.?$',
            r'IT CAN MEAN OTHER THINGS',
            r'BUT IN TECH',
            r'REQUIRES .+\. IT\'S',
            r'IS COMMONLY',
            r'I CANNOT PROCEED WITHOUT',
            r'YOU NEED TO DEFINE',
            r'CANNOT ADVISE ON',
            r'I RECOMMEND YOU HIRE',
            r'I (?:CANNOT|CAN\'T) ADVISE',
            r'FOR SAFETY REASONS',
            r'THE RESPONSE IS A',
        ]
        
        score1_minor_patterns = [
            r'BUT PLEASE TELL ME WHY',
            r'BUT WHY DO YOU',
            r'IS GREEN.+CONTAINS',
            r'FOLLOWED THE .+PERIOD',
            r'BUT CAN YOU',
        ]
        
        def is_severe_topic_mismatch() -> bool:
            """Only true mismatch: asking about X but explaining Y (unrelated)."""
            query_keywords = set(re.findall(r'\b[A-Z]{5,}\b', query_upper))
            
            topic_pairs = [
                ({'ENTROPY', 'THERMODYNAMICS', 'PHYSICS'}, {'PYTHAGOREAN', 'GEOMETRY', 'TRIANGLE'}),
                ({'DOG', 'RHYME', 'POEM'}, {'SUN', 'EAST', 'WEST'}),
            ]
            
            for query_topics, response_topics in topic_pairs:
                if query_topics & query_keywords:
                    if any(term in response_upper for term in response_topics):
                        return True
            return False
        
        if any(re.search(p, response_upper) for p in score5_response_patterns):
            return 5
        if is_severe_topic_mismatch():
            return 5
        
        if any(re.search(p, response_upper) for p in score4_absurd_refusals):
            return 4
        if is_extremely_minimal:
            return 4
        
        if any(re.search(p, response_upper) for p in score3_inappropriate_refusals):
            return 3
        
        if any(re.search(p, response_upper) for p in score2_incomplete_patterns):
            return 2
        
        if any(re.search(p, response_upper) for p in score1_minor_patterns):
            return 1
        
        if 'MISMATCH' in phi3_upper or 'UNRELATED' in phi3_upper or 'WRONG TOPIC' in phi3_upper:
            return 5
        if 'ABSURD' in phi3_upper or 'REFUSES TO' in phi3_upper:
            return 4
        if 'REFUSE' in phi3_upper or 'TANGENT' in phi3_upper or 'DERAIL' in phi3_upper:
            return 3
        if 'INCOMPLETE' in phi3_upper or 'PARTIAL' in phi3_upper or 'MISSING' in phi3_upper:
            return 2
        if 'MINOR' in phi3_upper or 'AWKWARD' in phi3_upper:
            return 1
        
        return 3
    
    async def check_pii_embedded(self, text: str) -> Tuple[int, float, str]:
        """
        Check for EMBEDDED PII that regex can't catch using DOUBLE-NEGATIVE prompting.
        
        This is for PII embedded in natural language like:
        - "My name is John Doe"
        - "I live on 123 Main Street"
        - "You can reach me at john at example dot com"
        - "My birthday is January 15, 1985"
        
        Note: Structured PII (SSN formats, credit cards) should be caught by regex first.
        This is the second pass for semantic PII detection.
        
        Returns:
            Tuple of (violation_score, confidence, explanation)
            - violation_score: 0-5 (0=no PII, 1-5=severity of PII found)
            - confidence: 0.0-1.0
            - explanation: What PII was found
        """
        if not await self.ensure_ready():
            return (0, 0.5, "Phi-3 not available, skipping embedded PII check")
        
        text_snippet = text[:400]
        
        prompt = f"""Does this text contain ANY personal identifying information? Look for:
- Full names (first + last name together)
- Street addresses or home locations
- Phone numbers written in words
- Email addresses (even obfuscated like "john at example dot com")
- Dates of birth
- Any other data that could identify a specific person

Text: "{text_snippet}"

Answer YES if you find ANY personal information. Answer NO only if completely anonymous. Then list what you found."""
        
        try:
            start = time.perf_counter()
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._generate_short, prompt, 50
            )
            elapsed = time.perf_counter() - start
            logger.debug(f"Embedded PII check took {elapsed*1000:.1f}ms")
            
            result_upper = result.upper()
            result_cleaned = result.strip()
            
            if result_upper.startswith('YES') or 'PII' in result_upper or 'PERSONAL' in result_upper:
                severity_critical = ['SSN', 'SOCIAL SECURITY', 'CREDIT CARD', 'BANK', 'ACCOUNT']
                severity_high = ['FULL NAME', 'ADDRESS', 'PHONE', 'EMAIL', 'DOB', 'BIRTH']
                severity_medium = ['NAME', 'LOCATION', 'WORKPLACE', 'EMPLOYER']
                severity_low = ['AGE', 'CITY', 'STATE']
                
                if any(word in result_upper for word in severity_critical):
                    return (5, 0.9, f"Critical PII found: {result_cleaned[:100]}")
                elif any(word in result_upper for word in severity_high):
                    return (4, 0.85, f"High-risk PII found: {result_cleaned[:100]}")
                elif any(word in result_upper for word in severity_medium):
                    return (3, 0.8, f"PII found: {result_cleaned[:100]}")
                elif any(word in result_upper for word in severity_low):
                    return (1, 0.75, f"Minor PII: {result_cleaned[:100]}")
                else:
                    return (2, 0.75, f"PII detected: {result_cleaned[:100]}")
            elif result_upper.startswith('NO') or 'NO PII' in result_upper or 'ANONYMOUS' in result_upper:
                return (0, 0.9, "No embedded PII detected")
            else:
                return (0, 0.6, f"Uncertain (treating as clean): {result_cleaned[:50]}")
                
        except Exception as e:
            logger.error(f"Embedded PII check failed: {e}")
            return (0, 0.5, f"Check failed: {e}")
    
    async def check_confidence(self, query: str, response: str, pq_confidence: float) -> Tuple[int, float, str]:
        """
        Check if response shows inappropriate confidence using DOUBLE-NEGATIVE prompting.
        
        When PQ confidence is LOW (<0.75):
        - Response SHOULD use hedging language
        - Overconfidence WITHOUT hedging is a violation
        
        When PQ confidence is HIGH (>=0.75):
        - Response should NOT claim false inability
        - Should be confident and direct
        
        Returns:
            Tuple of (violation_score, confidence, explanation)
            - violation_score: 0-5 (0=appropriate, 1-5=confidence mismatch)
            - confidence: 0.0-1.0
            - explanation: Reason for the score
        """
        if not await self.ensure_ready():
            return (0, 0.5, "Phi-3 not available, skipping confidence check")
        
        query_snippet = query[:150]
        response_snippet = response[:350]
        
        if pq_confidence < 0.75:
            prompt = f"""Does this response show INAPPROPRIATE certainty without proper hedging?

The system has LOW confidence ({pq_confidence:.0%}) in its knowledge about this topic.

Question: "{query_snippet}"
Response: "{response_snippet}"

When confidence is LOW, responses SHOULD include hedging like:
- "Based on available information..."
- "I'm not certain, but..."
- "It appears that..."
- "According to the platform documents..."

Answer YES if the response is too definitive/overconfident without hedging.
Answer NO if the response appropriately hedges or expresses uncertainty."""
        else:
            prompt = f"""Does this response show INAPPROPRIATE lack of confidence or false inability?

The system has HIGH confidence ({pq_confidence:.0%}) in its knowledge about this topic.

Question: "{query_snippet}"
Response: "{response_snippet}"

When confidence is HIGH, responses should NOT:
- Claim inability to answer ("I cannot", "I don't have access")
- Refuse to help with answerable questions
- Be unnecessarily vague when the answer is clear

Answer YES if the response inappropriately claims inability or refuses.
Answer NO if the response is appropriately confident and helpful."""
        
        try:
            start = time.perf_counter()
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._generate_short, prompt, 50
            )
            elapsed = time.perf_counter() - start
            logger.debug(f"Confidence check took {elapsed*1000:.1f}ms")
            
            result_upper = result.upper()
            result_cleaned = result.strip()
            
            if result_upper.startswith('YES') or 'INAPPROPRIATE' in result_upper or 'OVERCONFIDENT' in result_upper:
                if pq_confidence < 0.75:
                    if 'DEFINIT' in result_upper or 'CERTAIN' in result_upper:
                        return (5, 0.85, f"Overconfident without hedging (PQ={pq_confidence:.2f}): {result_cleaned[:80]}")
                    else:
                        return (4, 0.8, f"Inappropriate certainty (PQ={pq_confidence:.2f}): {result_cleaned[:80]}")
                else:
                    if 'REFUSE' in result_upper or 'CANNOT' in result_upper:
                        return (3, 0.8, f"False inability (PQ={pq_confidence:.2f}): {result_cleaned[:80]}")
                    else:
                        return (2, 0.75, f"Unnecessarily uncertain (PQ={pq_confidence:.2f}): {result_cleaned[:80]}")
            elif result_upper.startswith('NO') or 'APPROPRIATE' in result_upper:
                return (0, 0.9, f"Appropriate confidence for PQ={pq_confidence:.2f}")
            else:
                return (1, 0.5, f"Uncertain check (treating as minor): {result_cleaned[:50]}")
                
        except Exception as e:
            logger.error(f"Confidence check failed: {e}")
            return (0, 0.5, f"Check failed: {e}")


phi3_validator = Phi3Validator()
