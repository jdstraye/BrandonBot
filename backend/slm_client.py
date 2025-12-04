"""
SLM Client for Classification Tasks

Provides a unified interface for small classification tasks that can use either:
1. Local Ollama (self-hosted mode with 32GB+ RAM)
2. Nvidia API (Replit mode with fixed model)

Used by:
- Prequalifier: Frustration and vagueness classification
- OutputValidator: Intent alignment, ethics checks, FEC compliance

The client is designed for simple classification prompts that return
structured decisions (e.g., "frustrated" vs "calm", "vague" vs "clear").
"""

import os
import json
import logging
from typing import Optional, Dict, Any, Tuple
from enum import Enum
import httpx

logger = logging.getLogger(__name__)

NVIDIA_API_KEY = os.environ.get("NVIDIA_LLAMA33_API_KEY") or os.environ.get("NVIDIA_API_KEY")
NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"
NVIDIA_SLM_MODEL = os.environ.get("NVIDIA_SLM_MODEL", "meta/llama-3.3-70b-instruct")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_SLM_MODEL = os.environ.get("OLLAMA_SLM_MODEL", "llama3.1:8b")

SELF_HOSTED = os.environ.get("SELF_HOSTED", "").lower() in ("true", "1", "yes")


class SLMMode(Enum):
    OLLAMA = "ollama"
    NVIDIA = "nvidia"
    AUTO = "auto"


class SLMClient:
    """
    SLM client for classification tasks.
    
    Provides async methods for:
    - classify_frustration: Detect user frustration level
    - classify_vagueness: Detect if query is vague
    - classify_intent: Check if response matches intent
    - classify_ethics: Check for ethical issues
    - classify_fec: Check FEC compliance
    
    Automatically selects backend based on deployment mode:
    - Self-hosted: Uses Ollama with local LLM
    - Replit: Uses Nvidia API with fixed model
    """
    
    def __init__(self, mode: SLMMode = SLMMode.AUTO):
        self._mode = mode
        self._effective_mode: Optional[SLMMode] = None
        self._available: Optional[bool] = None
        
        self._nvidia_key = NVIDIA_API_KEY
        self._nvidia_base = NVIDIA_API_BASE
        self._nvidia_model = NVIDIA_SLM_MODEL
        
        self._ollama_host = OLLAMA_HOST
        self._ollama_model = OLLAMA_SLM_MODEL
    
    def _detect_mode(self) -> SLMMode:
        """Detect effective mode based on environment."""
        if self._mode != SLMMode.AUTO:
            return self._mode
        
        if SELF_HOSTED:
            return SLMMode.OLLAMA
        
        try:
            with open("/sys/fs/cgroup/memory.max", "r") as f:
                limit = f.read().strip()
                if limit != "max":
                    limit_gb = int(limit) // (1024 ** 3)
                    if limit_gb >= 16:
                        return SLMMode.OLLAMA
        except Exception:
            pass
        
        return SLMMode.NVIDIA
    
    async def check_availability(self) -> bool:
        """Check if the SLM backend is available."""
        if self._effective_mode is None:
            self._effective_mode = self._detect_mode()
        
        if self._effective_mode == SLMMode.OLLAMA:
            return await self._check_ollama_available()
        else:
            return await self._check_nvidia_available()
    
    async def _check_ollama_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self._ollama_host}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models = [m.get("name", "") for m in data.get("models", [])]
                    model_base = self._ollama_model.split(":")[0]
                    self._available = any(model_base in m for m in models)
                    if self._available:
                        logger.info(f"SLMClient: Ollama available with model {self._ollama_model}")
                    return self._available
                return False
        except Exception as e:
            logger.warning(f"SLMClient: Ollama not available: {e}")
            self._available = False
            return False
    
    async def _check_nvidia_available(self) -> bool:
        """Check if Nvidia API is available."""
        if not self._nvidia_key:
            logger.warning("SLMClient: No Nvidia API key found")
            self._available = False
            return False
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self._nvidia_base}/models",
                    headers={"Authorization": f"Bearer {self._nvidia_key}"}
                )
                self._available = response.status_code == 200
                if self._available:
                    logger.info(f"SLMClient: Nvidia API available with model {self._nvidia_model}")
                return self._available
        except Exception as e:
            logger.warning(f"SLMClient: Nvidia API not available: {e}")
            self._available = False
            return False
    
    @property
    def mode(self) -> str:
        """Return the effective mode being used."""
        if self._effective_mode is None:
            self._effective_mode = self._detect_mode()
        return self._effective_mode.value
    
    @property
    def model(self) -> str:
        """Return the model being used."""
        if self._effective_mode == SLMMode.OLLAMA:
            return self._ollama_model
        return self._nvidia_model
    
    async def _generate(self, prompt: str, system: str, max_tokens: int = 256) -> str:
        """Generate a response from the SLM."""
        if self._available is None:
            await self.check_availability()
        
        if not self._available:
            raise RuntimeError(f"SLM backend not available (mode: {self._effective_mode})")
        
        if self._effective_mode == SLMMode.OLLAMA:
            return await self._generate_ollama(prompt, system, max_tokens)
        else:
            return await self._generate_nvidia(prompt, system, max_tokens)
    
    async def _generate_ollama(self, prompt: str, system: str, max_tokens: int) -> str:
        """Generate using Ollama."""
        payload = {
            "model": self._ollama_model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": max_tokens,
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self._ollama_host}/api/generate",
                    json=payload
                )
                if response.status_code == 200:
                    return response.json().get("response", "")
                raise RuntimeError(f"Ollama error: {response.status_code}")
        except httpx.TimeoutException:
            raise RuntimeError("Ollama request timed out")
    
    async def _generate_nvidia(self, prompt: str, system: str, max_tokens: int) -> str:
        """Generate using Nvidia API."""
        payload = {
            "model": self._nvidia_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": max_tokens,
        }
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self._nvidia_base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._nvidia_key}",
                        "Content-Type": "application/json"
                    },
                    json=payload
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get("choices", [{}])[0].get("message", {}).get("content", "")
                raise RuntimeError(f"Nvidia API error: {response.status_code}")
        except httpx.TimeoutException:
            raise RuntimeError("Nvidia API request timed out")
    
    async def classify_frustration(
        self,
        message: str,
        has_profanity: bool = False,
        has_urgency: bool = False,
        frustration_count: int = 0,
        conversation_history: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Classify user frustration level.
        
        Returns:
            Tuple of (decision, emotion):
            - decision: "escalate" or "continue"
            - emotion: detected emotion like "frustrated", "calm", "angry"
        """
        context_parts = []
        if has_profanity:
            context_parts.append("- User message contains profanity")
        if has_urgency:
            context_parts.append("- User message contains urgency markers (e.g., 'need to speak to someone', 'immediately')")
        if frustration_count > 0:
            context_parts.append(f"- {frustration_count} frustration pattern(s) detected in message")
        if conversation_history:
            context_parts.append(f"- Recent conversation:\n{conversation_history}")
        
        context = "\n".join(context_parts) if context_parts else "No special flags detected."
        
        system = """You are an emotion classifier for a political campaign chatbot.
Classify the user's emotional state and decide if they need escalation to a human.

Classifications:
- "escalate": User is frustrated, angry, or demanding human contact. Route to callback.
- "continue": User is calm enough to continue with the chatbot.

Emotions to detect: frustrated, angry, anxious, confused, neutral, calm, happy

Respond ONLY with JSON: {"decision": "escalate"|"continue", "emotion": "..."}"""

        prompt = f"""User message: {message}

Context:
{context}

Classify the user's emotional state."""

        try:
            response = await self._generate(prompt, system)
            
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(response[json_start:json_end])
                decision = data.get("decision", "continue")
                emotion = data.get("emotion", "neutral")
                return (decision, emotion)
            
            if "escalate" in response.lower():
                return ("escalate", "frustrated")
            return ("continue", "neutral")
            
        except Exception as e:
            logger.error(f"Frustration classification failed: {e}")
            if has_profanity or frustration_count >= 2:
                return ("escalate", "frustrated")
            return ("continue", "neutral")
    
    async def classify_vagueness(
        self,
        message: str,
        rag_results: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Classify if a query is vague or clear.
        
        Returns:
            Tuple of (decision, confidence):
            - decision: "vague" or "clear"
            - confidence: 0.0 to 1.0
        """
        context = ""
        if rag_results:
            context = f"\nRelevant knowledge base results:\n{rag_results}"
        
        system = """You are a query clarity classifier for a political campaign chatbot.
Determine if the user's question is clear enough to answer directly, or if it's vague and needs clarification.

Classifications:
- "vague": Query is too general, ambiguous, or lacks specifics. Examples: "Hi", "What do you think?", "Tell me about that"
- "clear": Query has a specific topic or question. Examples: "What is Brandon's position on immigration?", "How can I volunteer?"

Respond ONLY with JSON: {"decision": "vague"|"clear", "confidence": 0.0-1.0}"""

        prompt = f"""User query: {message}
{context}

Is this query clear or vague?"""

        try:
            response = await self._generate(prompt, system)
            
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(response[json_start:json_end])
                decision = data.get("decision", "clear")
                confidence = float(data.get("confidence", 0.5))
                return (decision, confidence)
            
            if "vague" in response.lower():
                return ("vague", 0.7)
            return ("clear", 0.7)
            
        except Exception as e:
            logger.error(f"Vagueness classification failed: {e}")
            words = message.split()
            if len(words) <= 3:
                return ("vague", 0.6)
            return ("clear", 0.6)
    
    async def classify_intent_alignment(
        self,
        user_query: str,
        bot_response: str
    ) -> Tuple[bool, float, str]:
        """
        Check if bot response aligns with user's intent.
        
        Returns:
            Tuple of (aligned, confidence, reasoning)
        """
        system = """You are an intent alignment checker for a political campaign chatbot.
Determine if the bot's response actually addresses what the user asked.

Respond ONLY with JSON: {"aligned": true|false, "confidence": 0.0-1.0, "reasoning": "brief explanation"}"""

        prompt = f"""User question: {user_query}

Bot response: {bot_response}

Does the response address the user's question?"""

        try:
            response = await self._generate(prompt, system)
            
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(response[json_start:json_end])
                aligned = data.get("aligned", True)
                confidence = float(data.get("confidence", 0.5))
                reasoning = data.get("reasoning", "")
                return (aligned, confidence, reasoning)
            
            return (True, 0.5, "Could not parse response")
            
        except Exception as e:
            logger.error(f"Intent alignment check failed: {e}")
            return (True, 0.3, f"Error: {e}")
    
    async def classify_ethics(
        self,
        response: str
    ) -> Tuple[bool, float, str]:
        """
        Check for ethical issues in the response.
        
        Returns:
            Tuple of (is_safe, confidence, issue_description)
        """
        system = """You are an ethics checker for a political campaign chatbot.
Check if the response contains:
- Hate speech or discrimination
- Violence or threats
- Harmful advice
- Personal attacks
- Misinformation

Respond ONLY with JSON: {"is_safe": true|false, "confidence": 0.0-1.0, "issue": "description if unsafe"}"""

        prompt = f"""Check this campaign chatbot response for ethical issues:

{response}"""

        try:
            result = await self._generate(prompt, system)
            
            json_start = result.find("{")
            json_end = result.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(result[json_start:json_end])
                is_safe = data.get("is_safe", True)
                confidence = float(data.get("confidence", 0.5))
                issue = data.get("issue", "")
                return (is_safe, confidence, issue)
            
            return (True, 0.5, "")
            
        except Exception as e:
            logger.error(f"Ethics check failed: {e}")
            return (True, 0.3, f"Error: {e}")
    
    async def classify_fec_compliance(
        self,
        response: str,
        fec_context: Optional[str] = None
    ) -> Tuple[bool, float, str]:
        """
        Check for FEC compliance issues.
        
        Returns:
            Tuple of (is_compliant, confidence, issue_description)
        """
        context = ""
        if fec_context:
            context = f"\nRelevant FEC rules:\n{fec_context}"
        
        system = """You are an FEC compliance checker for a political campaign chatbot.
Check if the response violates campaign finance laws:
- Promising specific benefits in exchange for donations (quid pro quo)
- Soliciting donations from prohibited sources (foreign nationals, corporations directly)
- Making false promises about what donations will achieve
- Missing required disclaimers
- Coordinating with outside groups improperly

Respond ONLY with JSON: {"is_compliant": true|false, "confidence": 0.0-1.0, "issue": "description if non-compliant"}"""

        prompt = f"""Check this campaign chatbot response for FEC compliance:

{response}
{context}"""

        try:
            result = await self._generate(prompt, system)
            
            json_start = result.find("{")
            json_end = result.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(result[json_start:json_end])
                is_compliant = data.get("is_compliant", True)
                confidence = float(data.get("confidence", 0.5))
                issue = data.get("issue", "")
                return (is_compliant, confidence, issue)
            
            return (True, 0.5, "")
            
        except Exception as e:
            logger.error(f"FEC compliance check failed: {e}")
            return (True, 0.3, f"Error: {e}")


slm_client = SLMClient()
