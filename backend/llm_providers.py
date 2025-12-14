"""
Unified LLM Provider Manager

Handles multiple LLM providers with slot-based rotation and failover.
Design:
- API key slots rotate across conversations (primary rotation)
- Models within each slot rotate when that slot is used (secondary rotation)
- One model per conversation, switch only on rate limit/failure

The flow is directed by some environment variables:
- MOCK_LLM: if "true", use mock responses instead of real API calls
- LLM_ACTIVE_PROVIDERS: comma-separated list of active providers
"""

import logging
import os
import random
import time
import json
import httpx
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

# Optional config loader for role pinning and scoring whitelist
try:
    from backend.config_loader import load_config, BrandonBotConfig
except Exception:
    # If config loader missing during some test runs, fall back to defaults
    load_config = None
    BrandonBotConfig = None

logger = logging.getLogger(__name__)

# Optional Ollama client integration for local Llama models
try:
    from backend.ollama_client import OllamaClient
except Exception:
    OllamaClient = None

class SlotStatus(Enum):
    AVAILABLE = "available"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    NO_API_KEY = "no_api_key"

@dataclass
class APIKeySlot:
    """
    Represents a single API key slot with its associated models.
    For most providers: 1 slot with multiple models.
    For Nvidia: 5 slots, each with 1 model (unique key per model).
    
    Uses lazy-loading: API key checks happen at runtime in is_available(),
    not at initialization. This allows env vars loaded at different times.
    """
    slot_id: str
    api_key_env: str
    models: List[str]
    last_model_idx: int = 0
    status: SlotStatus = SlotStatus.AVAILABLE
    error_count: int = 0
    last_error_time: Optional[float] = None
    
    def __post_init__(self):
        # Lazy-load: Don't check API key here, defer to is_available()
        pass
    
    def get_api_key(self) -> Optional[str]:
        return os.getenv(self.api_key_env)
    
    def get_next_model(self) -> str:
        """Get the next model in rotation and advance the index."""
        model = self.models[self.last_model_idx]
        self.last_model_idx = (self.last_model_idx + 1) % len(self.models)
        return model
    
    def peek_next_model(self) -> str:
        """Peek at the next model without advancing."""
        return self.models[self.last_model_idx]
    
    def mark_rate_limited(self):
        self.status = SlotStatus.RATE_LIMITED
        self.last_error_time = time.time()
        self.error_count += 1
        logger.warning(f"Slot {self.slot_id} marked as rate limited")
    
    def mark_error(self, error: str):
        self.status = SlotStatus.ERROR
        self.last_error_time = time.time()
        self.error_count += 1
        logger.error(f"Slot {self.slot_id} error: {error}")
    
    def reset_if_recovered(self, cooldown_seconds: int = 60):
        if self.last_error_time and time.time() - self.last_error_time > cooldown_seconds:
            self.status = SlotStatus.AVAILABLE
            logger.info(f"Slot {self.slot_id} recovered after cooldown")
    
    def is_available(self) -> bool:
        """Check availability with lazy API key loading.
        
        First checks if API key exists (lazy-loaded at runtime).
        Then checks status and applies cooldown recovery for transient errors.
        """
        # Lazy-load: Check if API key is available now
        if not self.get_api_key():
            return False
        
        # Reset transient errors if cooldown has passed
        self.reset_if_recovered()
        
        # Available if status is AVAILABLE
        return self.status == SlotStatus.AVAILABLE

@dataclass
class LLMResponse:
    """Standardized response from any LLM provider"""
    text: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    model: str = "unknown"
    provider: str = "unknown"
    tokens_used: int = 0
    error: Optional[str] = None
    latency_ms: int = 0

@dataclass
class ProviderConfig:
    """Configuration for a provider with slot-based rotation."""
    name: str
    slots: List[APIKeySlot] = field(default_factory=list)
    priority: int = 50
    supports_function_calling: bool = True
    next_slot_idx: int = 0
    
    def __post_init__(self):
        if self.slots:
            self.next_slot_idx = random.randint(0, len(self.slots) - 1)
    
    def get_available_slots(self) -> List[APIKeySlot]:
        return [s for s in self.slots if s.is_available()]
    
    def get_next_slot(self) -> Optional[APIKeySlot]:
        """Get the next available slot in rotation."""
        if not self.slots:
            return None
        
        available = self.get_available_slots()
        if not available:
            return None
        
        for _ in range(len(self.slots)):
            slot = self.slots[self.next_slot_idx]
            self.next_slot_idx = (self.next_slot_idx + 1) % len(self.slots)
            if slot.is_available():
                return slot
        
        return available[0] if available else None
    
    def get_all_models(self) -> List[str]:
        """Get all models across all slots."""
        models = []
        for slot in self.slots:
            models.extend(slot.models)
        return models
    
    def is_available(self) -> bool:
        return len(self.get_available_slots()) > 0

class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers with slot-based rotation."""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.current_slot: Optional[APIKeySlot] = None
        self.current_model: Optional[str] = None
    
    def select_slot_and_model(self) -> Tuple[Optional[APIKeySlot], Optional[str]]:
        """
        Select the next available slot and get the next model from it.
        Advances both slot index and model index.
        """
        slot = self.config.get_next_slot()
        if not slot:
            return None, None
        
        model = slot.get_next_model()
        self.current_slot = slot
        self.current_model = model
        return slot, model
    
    def set_slot_and_model(self, slot: APIKeySlot, model: str):
        """Set a specific slot and model (used for session persistence)."""
        self.current_slot = slot
        self.current_model = model
    
    def get_slot_by_id(self, slot_id: str) -> Optional[APIKeySlot]:
        """Find a slot by its ID."""
        for slot in self.config.slots:
            if slot.slot_id == slot_id:
                return slot
        return None
    
    def is_available(self) -> bool:
        """Check if provider has any available slots."""
        return self.config.is_available()
    
    @abstractmethod
    async def generate_with_tools(self, messages: List[Dict], tools: List[Dict],
                                   system_prompt: str) -> LLMResponse:
        """Generate response with function calling support"""
        pass


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider - 1 slot with 2 models"""
    
    def __init__(self):
        slots = [
            APIKeySlot(
                slot_id="gemini_main",
                api_key_env="GOOGLE_API_KEY",
                models=["gemini-2.0-flash", "gemini-2.5-flash"]
            )
        ]
        config = ProviderConfig(
            name="gemini",
            slots=slots,
            priority=80,
            supports_function_calling=True
        )
        super().__init__(config)
        self._configured = False
    
    async def generate_with_tools(self, messages: List[Dict], tools: List[Dict],
                                   system_prompt: str) -> LLMResponse:
        start_time = time.time()
        
        if not self.current_slot:
            return LLMResponse(error="No slot selected", provider=self.config.name)
        
        api_key = self.current_slot.get_api_key()
        if not api_key:
            return LLMResponse(error="No API key", provider=self.config.name)
        
        try:
            import google.generativeai as genai  # type: ignore
            from google.generativeai.types import FunctionDeclaration, Tool  # type: ignore
            
            if not self._configured:
                genai.configure(api_key=api_key)
                self._configured = True
            
            model_name = self.current_model
            if not model_name:
                return LLMResponse(error="No model selected", provider=self.config.name)
            logger.info(f"Gemini using model: {model_name}")
            
            gemini_tools = None
            if tools:
                function_declarations = []
                for tool in tools:
                    fd = FunctionDeclaration(
                        name=tool["name"],
                        description=tool["description"],
                        parameters=tool["parameters"]
                    )
                    function_declarations.append(fd)
                gemini_tools = [Tool(function_declarations=function_declarations)]
            
            model = genai.GenerativeModel(
                model_name,
                tools=gemini_tools if gemini_tools else None,
                system_instruction=system_prompt
            )
            
            gemini_messages = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                if role == "user":
                    gemini_messages.append({"role": "user", "parts": [content]})
                elif role == "assistant":
                    gemini_messages.append({"role": "model", "parts": [content]})
                elif role == "tool":
                    gemini_messages.append({"role": "user", "parts": [f"Tool results:\n{content}"]})
                elif role == "system":
                    gemini_messages.append({"role": "user", "parts": [f"System context: {content}"]})
            
            generation_config = genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=2048,
                top_p=0.9,
            )
            
            response = model.generate_content(
                gemini_messages,
                generation_config=generation_config
            )
            
            latency_ms = int((time.time() - start_time) * 1000)
            tokens = response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0
            
            result = LLMResponse(
                model=model_name,
                provider=self.config.name,
                tokens_used=tokens,
                latency_ms=latency_ms
            )
            
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                tool_calls = []
                text_parts = []
                
                for part in candidate.content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        fc = part.function_call
                        args = {}
                        if fc.args:
                            for key, value in fc.args.items():
                                if hasattr(value, 'string_value'):
                                    args[key] = value.string_value
                                elif hasattr(value, 'number_value'):
                                    args[key] = value.number_value
                                elif hasattr(value, 'bool_value'):
                                    args[key] = value.bool_value
                                elif isinstance(value, (str, int, float, bool)):
                                    args[key] = value
                                else:
                                    args[key] = str(value)
                        
                        tool_calls.append({
                            "name": fc.name,
                            "arguments": args,
                            "id": f"{fc.name}_{len(tool_calls)}"
                        })
                    elif hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)
                
                if tool_calls:
                    result.tool_calls = tool_calls
                if text_parts:
                    result.text = "\n".join(text_parts).strip()
            
            return result
            
        except Exception as e:
            error_str = str(e).lower()
            if self.current_slot:
                if "rate" in error_str or "quota" in error_str or "429" in error_str:
                    self.current_slot.mark_rate_limited()
                else:
                    self.current_slot.mark_error(str(e))
            return LLMResponse(error=str(e), provider=self.config.name, 
                             latency_ms=int((time.time() - start_time) * 1000))


class OllamaProvider(BaseLLMProvider):
    """Local Ollama-backed provider. Does not require API keys; slot is always available
    if an Ollama client can be created. This provider exposes local Llama models via Ollama.
    """

    def __init__(self, models: Optional[List[str]] = None):
        # Create a single local slot that reports available without env API key
        slot = APIKeySlot(slot_id="ollama_local", api_key_env="OLLAMA_LOCAL", models=models or ["llama3.2:3b"])
        # make the slot report an API key so is_available() is True by default
        slot.get_api_key = lambda: "ollama-local"
        config = ProviderConfig(name="ollama", slots=[slot], priority=90, supports_function_calling=False)
        super().__init__(config)
        self._client = None

    def _ensure_client(self):
        if self._client is None and OllamaClient is not None:
            host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            try:
                self._client = OllamaClient(host)
            except Exception:
                self._client = None
        return self._client is not None

    async def generate_with_tools(self, messages: List[Dict], tools: List[Dict], system_prompt: str) -> LLMResponse:
        start_time = time.time()
        # Use the last user message as the query and full conversation as context
        user_query = ""
        context_parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            context_parts.append(f"{role}: {content}")
            if role == "user":
                user_query = content

        context = "\n".join(context_parts[-20:])

        if not self._ensure_client():
            return LLMResponse(error="Ollama client unavailable", provider=self.config.name)

        try:
            # OllamaClient.generate_response returns dict with 'response' and 'model'
            client = self._client
            confidence = 1.0
            resp = await client.generate_response(user_query, context, confidence)
            latency_ms = int((time.time() - start_time) * 1000)
            return LLMResponse(text=resp.get("response", ""), model=resp.get("model", "ollama"), provider=self.config.name, latency_ms=latency_ms)
        except Exception as e:
            return LLMResponse(error=str(e), provider=self.config.name, latency_ms=int((time.time() - start_time) * 1000))


class OpenAICompatibleProvider(BaseLLMProvider):
    """Base class for OpenAI-compatible API providers with slot support"""
    
    def __init__(self, config: ProviderConfig, base_url: str):
        super().__init__(config)
        self.base_url = base_url
        self._clients: Dict[str, Any] = {}
    
    def _get_client(self, api_key: str):
        """Get or create client for a specific API key."""
        if api_key not in self._clients:
            from openai import AsyncOpenAI  # type: ignore
            self._clients[api_key] = AsyncOpenAI(
                api_key=api_key,
                base_url=self.base_url
            )
        return self._clients[api_key]
    
    def _convert_tools_to_openai_format(self, tools: List[Dict]) -> List[Dict]:
        """Convert our tool format to OpenAI format"""
        openai_tools = []
        for tool in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"]
                }
            })
        return openai_tools
    
    async def generate_with_tools(self, messages: List[Dict], tools: List[Dict],
                                   system_prompt: str) -> LLMResponse:
        start_time = time.time()
        
        if not self.current_slot:
            return LLMResponse(error="No slot selected", provider=self.config.name)
        
        api_key = self.current_slot.get_api_key()
        if not api_key:
            return LLMResponse(error="No API key", provider=self.config.name)
        
        model_name = self.current_model
        if not model_name:
            return LLMResponse(error="No model selected", provider=self.config.name)
        
        try:
            client = self._get_client(api_key)
            logger.info(f"{self.config.name} using model: {model_name}")
            
            openai_messages = [{"role": "system", "content": system_prompt}]
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role in ["user", "assistant", "system"]:
                    openai_messages.append({"role": role, "content": content})
                elif role == "tool":
                    openai_messages.append({"role": "user", "content": f"Tool results:\n{content}"})
            
            openai_tools = self._convert_tools_to_openai_format(tools)
            
            response = await client.chat.completions.create(
                model=model_name,
                messages=openai_messages,
                tools=openai_tools if openai_tools else None,
                temperature=0.7,
                max_tokens=2048
            )
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            result = LLMResponse(
                model=model_name,
                provider=self.config.name,
                tokens_used=response.usage.total_tokens if response.usage else 0,
                latency_ms=latency_ms
            )
            
            choice = response.choices[0] if response.choices else None
            if choice:
                if choice.message.content:
                    result.text = choice.message.content
                if choice.message.tool_calls:
                    result.tool_calls = []
                    for tc in choice.message.tool_calls:
                        result.tool_calls.append({
                            "name": tc.function.name,
                            "arguments": json.loads(tc.function.arguments) if tc.function.arguments else {},
                            "id": tc.id
                        })
            
            return result
            
        except Exception as e:
            error_str = str(e).lower()
            if self.current_slot:
                if "rate" in error_str or "quota" in error_str or "429" in error_str:
                    self.current_slot.mark_rate_limited()
                else:
                    self.current_slot.mark_error(str(e))
            return LLMResponse(error=str(e), provider=self.config.name,
                             latency_ms=int((time.time() - start_time) * 1000))


class MistralProvider(OpenAICompatibleProvider):
    """Mistral AI provider - 1 slot with 4 models"""
    
    def __init__(self):
        slots = [
            APIKeySlot(
                slot_id="mistral_main",
                api_key_env="MISTRAL_API_KEY",
                models=["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest", "pixtral-12b-2409", "open-mistral-nemo-2407"]
            )
        ]
        config = ProviderConfig(
            name="mistral",
            slots=slots,
            priority=60,
            supports_function_calling=True
        )
        super().__init__(config, "https://api.mistral.ai/v1")


class CohereProvider(BaseLLMProvider):
    """Cohere provider - 1 slot with 3 models (updated for Sept 2025 deprecations)"""
    
    def __init__(self):
        slots = [
            APIKeySlot(
                slot_id="cohere_main",
                api_key_env="COHERE_API_KEY",
                models=["command-a-03-2025", "command-r-plus-08-2024", "command-r-08-2024"]
            )
        ]
        config = ProviderConfig(
            name="cohere",
            slots=slots,
            priority=55,
            supports_function_calling=True
        )
        super().__init__(config)
        self._clients: Dict[str, Any] = {}
    
    def _get_client(self, api_key: str):
        if api_key not in self._clients:
            import cohere  # type: ignore
            self._clients[api_key] = cohere.ClientV2(api_key)
        return self._clients[api_key]
    
    async def generate_with_tools(self, messages: List[Dict], tools: List[Dict],
                                   system_prompt: str) -> LLMResponse:
        start_time = time.time()
        
        if not self.current_slot:
            return LLMResponse(error="No slot selected", provider=self.config.name)
        
        api_key = self.current_slot.get_api_key()
        if not api_key:
            return LLMResponse(error="No API key", provider=self.config.name)
        
        model_name = self.current_model
        if not model_name:
            return LLMResponse(error="No model selected", provider=self.config.name)
        
        try:
            client = self._get_client(api_key)
            logger.info(f"Cohere using model: {model_name}")
            
            cohere_messages = [{"role": "system", "content": system_prompt}]
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role in ["user", "assistant"]:
                    cohere_messages.append({"role": role, "content": content})
                elif role == "tool":
                    cohere_messages.append({"role": "user", "content": f"Tool results:\n{content}"})
            
            cohere_tools = []
            for tool in tools:
                cohere_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"]
                    }
                })
            
            response = client.chat(
                model=model_name,
                messages=cohere_messages,
                tools=cohere_tools if cohere_tools else None
            )
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            result = LLMResponse(
                model=model_name,
                provider=self.config.name,
                tokens_used=getattr(response, 'meta', {}).get('billed_units', {}).get('input_tokens', 0) + 
                           getattr(response, 'meta', {}).get('billed_units', {}).get('output_tokens', 0),
                latency_ms=latency_ms
            )
            
            if hasattr(response, 'message') and response.message:
                if hasattr(response.message, 'content') and response.message.content:
                    for block in response.message.content:
                        if hasattr(block, 'text'):
                            result.text = block.text
                            break
                
                if hasattr(response.message, 'tool_calls') and response.message.tool_calls:
                    result.tool_calls = []
                    for tc in response.message.tool_calls:
                        result.tool_calls.append({
                            "name": tc.function.name,
                            "arguments": json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments,
                            "id": tc.id
                        })
            
            return result
            
        except Exception as e:
            error_str = str(e).lower()
            if self.current_slot:
                if "rate" in error_str or "quota" in error_str or "429" in error_str:
                    self.current_slot.mark_rate_limited()
                else:
                    self.current_slot.mark_error(str(e))
            return LLMResponse(error=str(e), provider=self.config.name,
                             latency_ms=int((time.time() - start_time) * 1000))


class HuggingFaceProvider(OpenAICompatibleProvider):
    """HuggingFace Inference provider - 1 slot with Qwen2.5, DeepSeek, and Kimi K2 models"""
    
    def __init__(self):
        slots = [
            APIKeySlot(
                slot_id="huggingface_main",
                api_key_env="HUGGINGFACE_API_KEY",
                models=[
                    "Qwen/Qwen2.5-72B-Instruct",
                    "deepseek-ai/DeepSeek-V3-0324",
                    "moonshotai/Kimi-K2-Instruct",
                    "mistral/mistral-7b"
                ]
            )
        ]
        config = ProviderConfig(
            name="huggingface",
            slots=slots,
            priority=50,
            supports_function_calling=True
        )
        super().__init__(config, "https://router.huggingface.co/v1")


class GroqProvider(OpenAICompatibleProvider):
    """Groq provider - OpenAI-compatible endpoint at api.groq.com"""

    def __init__(self):
        # Default model list if we cannot fetch models from Groq API
        default_models = ["groq/groq1"]

        slot = APIKeySlot(
            slot_id="groq_main",
            api_key_env="GROQ_API_KEY",
            models=default_models.copy()
        )

        # Try to populate models dynamically if GROQ_API_KEY is available
        try:
            api_key = slot.get_api_key()
            if api_key:
                import requests
                url = "https://api.groq.com/openai/v1/models"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                resp = requests.get(url, headers=headers, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    # Expecting a list of model objects or ids
                    models = []
                    if isinstance(data, dict) and data.get("data"):
                        entries = data.get("data")
                    elif isinstance(data, list):
                        entries = data
                    else:
                        entries = []

                    for e in entries:
                        if isinstance(e, dict):
                            mid = e.get("id") or e.get("model") or e.get("name")
                            if mid:
                                models.append(mid)
                        elif isinstance(e, str):
                            models.append(e)

                    if models:
                        # Replace slot models with discovered list (keep unique)
                        slot.models = list(dict.fromkeys(models))
        except Exception:
            # On any failure, keep defaults silently
            pass

        config = ProviderConfig(
            name="groq",
            slots=[slot],
            priority=65,
            supports_function_calling=True
        )
        super().__init__(config, "https://api.groq.com/openai/v1")


class NvidiaProvider(BaseLLMProvider):
    """
    Nvidia NIM provider - 5 separate slots, each with 1 model and its own API key.
    Unlike other providers, Nvidia requires a unique API key per model.
    """
    
    def __init__(self):
        slots = [
            APIKeySlot(
                slot_id="nvidia_llama4_maverick",
                api_key_env="NVIDIA_LLAMA4_128e",
                models=["meta/llama-4-maverick-17b-128e-instruct"]
            ),
            # NOTE: The 'scout' model (meta/llama-4-scout-17b-16e-instruct) has
            # been intentionally excluded from the default scoring/judge pool.
            # Rationale: repeated validation runs showed this model failing to
            # perform MCP/tool-calling reliably (no tool calls observed for
            # many vague/agent flows). Operators should add it explicitly to the
            # `scoring_whitelist` in `backend/config/BrandonBot.ini` if they
            # wish to re-enable it for experimentation.
            APIKeySlot(
                slot_id="nvidia_deepseek_r1",
                api_key_env="NVIDIA_DEEPSEEK_r1_API_KEY",
                models=["deepseek-ai/deepseek-r1"]
            ),
            APIKeySlot(
                slot_id="nvidia_llama33",
                api_key_env="NVIDIA_LLAMA33_API_KEY",
                models=["meta/llama-3.3-70b-instruct"]
            ),
            APIKeySlot(
                slot_id="nvidia_qwen25",
                api_key_env="NVIDIA_QWEN25_API_KEY",
                models=["qwen/qwen2.5-coder-32b-instruct"]
            ),
        ]
        
        config = ProviderConfig(
            name="nvidia",
            slots=slots,
            priority=90,
            supports_function_calling=True
        )
        super().__init__(config)
    
    async def generate_with_tools(self, messages: List[Dict], tools: List[Dict],
                                   system_prompt: str) -> LLMResponse:
        start_time = time.time()
        
        if not self.current_slot:
            return LLMResponse(error="No slot selected", provider=self.config.name)
        
        api_key = self.current_slot.get_api_key()
        if not api_key:
            return LLMResponse(error="No API key", provider=self.config.name)
        
        model_name = self.current_model
        if not model_name:
            return LLMResponse(error="No model selected", provider=self.config.name)
        
        try:
            logger.info(f"Nvidia using model: {model_name} (slot: {self.current_slot.slot_id})")
            
            openai_messages = [{"role": "system", "content": system_prompt}]
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role in ["user", "assistant", "system"]:
                    openai_messages.append({"role": role, "content": content})
                elif role == "tool":
                    openai_messages.append({"role": "user", "content": f"Tool results:\n{content}"})
            
            openai_tools = []
            for tool in tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"]
                    }
                })
            
            payload = {
                "model": model_name,
                "messages": openai_messages,
                "max_tokens": 2048,
                "temperature": 0.7
            }
            if openai_tools:
                payload["tools"] = openai_tools
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://integrate.api.nvidia.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            result = LLMResponse(
                model=model_name,
                provider=self.config.name,
                tokens_used=data.get("usage", {}).get("total_tokens", 0),
                latency_ms=latency_ms
            )
            
            choices = data.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                if message.get("content"):
                    result.text = message["content"]
                if message.get("tool_calls"):
                    result.tool_calls = []
                    for tc in message["tool_calls"]:
                        result.tool_calls.append({
                            "name": tc["function"]["name"],
                            "arguments": json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str) else tc["function"]["arguments"],
                            "id": tc.get("id", f"{tc['function']['name']}_0")
                        })
            
            return result
            
        except Exception as e:
            error_str = str(e).lower()
            if self.current_slot:
                if "rate" in error_str or "429" in error_str:
                    self.current_slot.mark_rate_limited()
                else:
                    self.current_slot.mark_error(str(e))
            return LLMResponse(error=str(e), provider=self.config.name,
                             latency_ms=int((time.time() - start_time) * 1000))


class LLMProviderManager:
    """
    Manages multiple LLM providers with slot-based round-robin selection.
    
    Design:
    - 9 API key slots across 5 providers, managing 17 unique models
    - Round-robin slot selection (random start, sequential advance)
    - Within each slot, rotate through models on subsequent calls
    - Session-sticky: same slot+model for entire conversation
    - Automatic failover to next slot on rate limit/error
    """
    
    def __init__(self):
        self.providers: Dict[str, BaseLLMProvider] = {}
        self._init_providers()
        
        self.all_slots: List[Tuple[str, APIKeySlot]] = []
        self._build_slot_pool()
        # Load BrandonBot INI config (non-secrets)
        try:
            if load_config:
                self.bb_config = load_config()
            else:
                self.bb_config = None
        except Exception:
            self.bb_config = None
        
        self.next_slot_idx = random.randint(0, max(1, len(self.all_slots)) - 1)
        
        self.session_assignments: Dict[str, Tuple[str, str, str]] = {}
    
    def _init_providers(self):
        """Initialize all available providers.

        Behavior changes:
        - If `LLM_MOCK=true` is set in the environment, a lightweight `MockProvider`
          will be used for deterministic responses (safe for functional tests).
        - Active provider list is controlled by `LLM_ACTIVE_PROVIDERS` env var
          (comma-separated provider names). If unset, we default to a conservative
          whitelist of higher-quality providers to avoid low-quality adapters.
        """
        # Map provider name -> class for flexible selection
        available_provider_map = {
            "ollama": OllamaProvider,
            "nvidia": NvidiaProvider,
            "gemini": GeminiProvider,
            "tinyllama": TinyLlamaProvider,
            "glaive": GlaiveProvider,
            "mistral": MistralProvider,
            "cohere": CohereProvider,
            "huggingface": HuggingFaceProvider,
            "groq": GroqProvider,
        }

        
        # If mock mode requested, register only the MockProvider
        if os.getenv("LLM_MOCK", "false").lower() in ("1", "true", "yes"):
            # Define a simple mock provider class that returns deterministic responses
            class MockProvider(BaseLLMProvider):
                def __init__(self):
                    slots = [APIKeySlot(slot_id="mock_slot", api_key_env="MOCK_KEY", models=["mock-model"])]
                    # ensure mock slot reports an API key regardless of env
                    slots[0].get_api_key = lambda: "mock-key"
                    config = ProviderConfig(name="mock", slots=slots, priority=100, supports_function_calling=True)
                    super().__init__(config)

                async def generate_with_tools(self, messages: List[Dict], tools: List[Dict], system_prompt: str) -> LLMResponse:
                    # Intelligent mock: analyze the user's last message to determine appropriate tools
                    user_message = ""
                    for msg in reversed(messages):
                        if msg.get("role") == "user":
                            user_message = msg.get("content", "").lower()
                            break
                    
                    # Response determination logic based on user message
                    text = "Thanks — I've recorded your interest as a volunteer. We'll follow up via email."
                    tool_calls = []
                    
                    # Detect callback triggers in the user message
                    callback_triggers = [
                        "give me a call", "call me", "can you call me",
                        "speak to someone", "talk to someone", "talk to a person",
                        "schedule a call", "set up a call",
                        "have someone reach out", "reach out to me", "contact me",
                        "i'm confused", "i'm frustrated", "i'm uncertain"
                    ]
                    
                    # Detect policy/position questions
                    policy_triggers = ["stance", "position", "policy", "taxes", "tax", "healthcare", "abortion", "gun", "immigration", "water"]
                    
                    # Detect comparison requests
                    comparison_triggers = ["compare", "differ", "different", "versus", "vs", "aligned with", "similar to"]
                    
                    # Detect explicit volunteer/donation requests
                    volunteer_triggers = ["volunteer", "sign up", "help", "get involved"]
                    donation_triggers = ["donate", "contribute", "give to", "support"]
                    
                    # Detect FEC compliance questions
                    fec_triggers = ["donate", "donation", "money", "cash", "fec", "anonymous", "limit"]
                    
                    has_callback_trigger = any(trigger in user_message for trigger in callback_triggers)
                    has_policy_trigger = any(trigger in user_message for trigger in policy_triggers)
                    has_comparison_trigger = any(trigger in user_message for trigger in comparison_triggers)
                    has_volunteer_trigger = any(trigger in user_message for trigger in volunteer_triggers)
                    has_donation_trigger = any(trigger in user_message for trigger in donation_triggers)
                    has_fec_trigger = any(trigger in user_message for trigger in fec_triggers)
                    
                    # Determine which tools to simulate
                    if tools:
                        for t in tools:
                            tool_name = t.get("name", "").lower()
                            
                            # Callback tool logic
                            if "callback" in tool_name:
                                if has_callback_trigger:
                                    tool_calls.append({"name": t.get("name"), "arguments": {}, "id": f"mock_callback"})
                                    text = "I'd be happy to have someone from Brandon's team call you! To set that up, could you share your name and the best phone number to reach you?"
                            
                            # Volunteer tool logic
                            elif "volunteer" in tool_name:
                                if has_volunteer_trigger and not has_callback_trigger:
                                    tool_calls.append({"name": t.get("name"), "arguments": {"name": "Anonymous", "email": "test@example.com"}, "id": f"mock_volunteer"})
                                    text = "Thanks — I've recorded your interest as a volunteer. We'll follow up via email."
                            
                            # Party comparison tool logic
                            elif "party" in tool_name or "comparison" in tool_name:
                                if has_comparison_trigger and not has_callback_trigger:
                                    tool_calls.append({"name": t.get("name"), "arguments": {"entity": "RNC"}, "id": f"mock_comparison"})
                                    text = "Brandon's positions differ from the RNC in several key ways..."
                            
                            # FEC rules tool logic
                            elif "fec" in tool_name:
                                if has_fec_trigger and "cash" in user_message and not has_callback_trigger:
                                    tool_calls.append({"name": t.get("name"), "arguments": {"question": user_message[:50]}, "id": f"mock_fec"})
                                    text = "Federal law limits contributions. Cash donations of $50,000 would violate FEC rules."
                            
                            # Donation tool logic
                            elif "donation" in tool_name or "donate" in tool_name:
                                if has_donation_trigger and not has_fec_trigger and not has_callback_trigger:
                                    tool_calls.append({"name": t.get("name"), "arguments": {"amount": 0}, "id": f"mock_donation"})
                                    text = "Thank you for your interest in supporting Brandon's campaign!"
                            
                            # Web search tool logic - use for current events and prices
                            elif "web_search" in tool_name or ("search" in tool_name and "web" in tool_name):
                                if "price" in user_message or "bitcoin" in user_message or "gas" in user_message or ("current" in user_message and not has_callback_trigger):
                                    tool_calls.append({"name": t.get("name"), "arguments": {"query": user_message[:50]}, "id": f"mock_web_search"})
                                    text = f"Based on current data: {user_message[:30]}..."
                            
                            # Search tool logic - use for policy questions (search_brandon_positions)
                            elif "search" in tool_name and "position" in tool_name:
                                # search_brandon_positions: Use for policy questions unless it's a callback request
                                if not has_callback_trigger and has_policy_trigger:
                                    tool_calls.append({"name": t.get("name"), "arguments": {"query": user_message[:50]}, "id": f"mock_search"})
                                    text = f"Based on Brandon's positions: {user_message[:30]}..."

                    return LLMResponse(text=text, tool_calls=tool_calls, model="mock-model", provider="mock", tokens_used=5, latency_ms=5)

            try:
                provider = MockProvider()
                self.providers[provider.config.name] = provider
                logger.info("LLM_MOCK=true: registered MockProvider for safe testing")
            except Exception as e:
                logger.error(f"Failed to initialize MockProvider: {e}")
            return

        # Determine active providers (allow env override); default to conservative whitelist
        env_list = os.getenv("LLM_ACTIVE_PROVIDERS")
        if env_list:
            whitelist = [p.strip().lower() for p in env_list.split(",") if p.strip()]
        else:
            # Conservative default: prefer higher-quality providers that typically give better outputs
            # Include `groq` by default as an adaptive high-quality provider when available
            # Include `ollama` by default for local self-hosted Llama models when available
            whitelist = ["nvidia", "gemini", "mistral", "groq", "ollama", "tinyllama", "glaive"]

        for name in whitelist:
            cls = available_provider_map.get(name)
            if not cls:
                logger.warning(f"Requested provider '{name}' is unknown; skipping")
                continue

            try:
                provider = cls()
                # No env-based blacklist: operator-level model/slot tuning should be
                # done via `backend/config/BrandonBot.ini` (scoring whitelist or roles).
                # We do not rely on ephemeral environment variables for whitelisting.
                self.providers[provider.config.name] = provider
                available_slots = sum(1 for s in provider.config.slots if s.is_available())
                total_models = sum(len(s.models) for s in provider.config.slots if s.is_available())
                logger.info(f"Provider {provider.config.name}: {available_slots} slots, {total_models} models")
            except Exception as e:
                logger.error(f"Failed to initialize provider {cls.__name__}: {e}")
    
    def _build_slot_pool(self):
        """Build global pool of all slots across providers.
        
        All slots are added regardless of current API key availability.
        Availability is checked dynamically at request time via is_available().
        This supports lazy-loading of environment variables.
        """
        self.all_slots = []
        for provider_name, provider in self.providers.items():
            for slot in provider.config.slots:
                self.all_slots.append((provider_name, slot))  # Add all slots
        
        random.shuffle(self.all_slots)
        total_slots = len(self.all_slots)
        available = len(self.get_available_slots())
        logger.info(f"Global slot pool: {total_slots} total slots, {available} currently available with {self._count_total_models()} unique models")
    
    def _count_total_models(self) -> int:
        """Count unique models across all slots"""
        models = set()
        for _, slot in self.all_slots:
            models.update(slot.models)
        return len(models)
    
    def get_available_slots(self) -> List[Tuple[str, APIKeySlot]]:
        """Get list of currently available slots"""
        return [(name, slot) for name, slot in self.all_slots if slot.is_available()]
    
    def _select_next_slot(self) -> Optional[Tuple[str, APIKeySlot, str]]:
        """
        Select next available slot using round-robin, then pick next model within slot.
        Returns (provider_name, slot, model_name) or None.
        """
        available = self.get_available_slots()
        if not available:
            return None
        
        start_idx = self.next_slot_idx % len(available)
        
        for i in range(len(available)):
            idx = (start_idx + i) % len(available)
            provider_name, slot = available[idx]
            model = slot.get_next_model()
            
            if model:
                self.next_slot_idx = (idx + 1) % len(available)
                return (provider_name, slot, model)
        
        return None
    
    def select_for_session(self, session_id: str, 
                           force_new: bool = False) -> Optional[Tuple[str, str, str]]:
        """
        Select a slot and model for a session.
        Returns (provider_name, slot_id, model_name) or None.
        """
        if not force_new and session_id in self.session_assignments:
            provider_name, slot_id, model_name = self.session_assignments[session_id]
            
            for pname, slot in self.all_slots:
                if pname == provider_name and slot.slot_id == slot_id and slot.is_available():
                    return (provider_name, slot_id, model_name)
        
        selection = self._select_next_slot()
        if not selection:
            logger.error("No available slots!")
            return None
        
        provider_name, slot, model_name = selection
        self.session_assignments[session_id] = (provider_name, slot.slot_id, model_name)
        
        logger.info(f"Session {session_id[:8]}... -> {provider_name}/{slot.slot_id}/{model_name}")
        return (provider_name, slot.slot_id, model_name)

    def get_candidate_slots_for_scoring(self) -> List[Tuple[str, APIKeySlot, str]]:
        """Return list of (provider_name, slot, model) filtered by scoring_whitelist in config.

        If no whitelist configured, returns all available slots+models.
        """
        candidates: List[Tuple[str, APIKeySlot, str]] = []
        whitelist = []
        if getattr(self, 'bb_config', None) and self.bb_config.scoring and self.bb_config.scoring.scoring_whitelist:
            whitelist = [(p.lower(), m) for p, m in self.bb_config.scoring.scoring_whitelist]

        for provider_name, slot in self.all_slots:
            for model in slot.models:
                if whitelist:
                    # match provider+model (model may be empty to indicate any model)
                    match_any = False
                    for wp, wm in whitelist:
                        if wp and wp != provider_name:
                            continue
                        if wm and wm != model:
                            continue
                        match_any = True
                        break
                    if not match_any:
                        continue
                candidates.append((provider_name, slot, model))
        return candidates

    def select_for_role(self, session_id: str, role: str, force_new: bool = False) -> Optional[Tuple[str, str, str]]:
        """Select a slot/model for a specific role (e.g., 'Judge' or 'User').

        - For role 'Judge': prefer provider:model pairs from `BrandonBot.ini` `roles.Judge`.
          If `require_llama_for_judge` is true and no Llama slot available, return None.
        - For role 'User': use round-robin unless `User_allow_round_robin` is false and INI lists a user preference.
        """
        role_lc = (role or "").lower()
        # If already assigned and still available and not forcing, consider returning it.
        # For the 'judge' role we prefer to honor the pinned judge from config, so
        # only return an existing assignment early if it matches the configured judge
        # or no judge preference is configured.
        if not force_new and session_id in self.session_assignments:
            provider_name, slot_id, model_name = self.session_assignments[session_id]
            # Verify slot still exists and available
            for pname, slot in self.all_slots:
                if pname == provider_name and slot.slot_id == slot_id and slot.is_available():
                    # If role is 'judge', ensure this assignment aligns with preferred judges
                    if role_lc == 'judge' and getattr(self, 'bb_config', None) and self.bb_config.roles and self.bb_config.roles.judge:
                        preferred_providers = [p.lower() for p, _ in self.bb_config.roles.judge]
                        if provider_name.lower() in preferred_providers:
                            return (provider_name, slot_id, model_name)
                        # otherwise fall through to attempt selecting a preferred judge
                    else:
                        return (provider_name, slot_id, model_name)

        # Judge selection path
        if role_lc == "judge":
            # Gather preferred judge pairs from config
            # Ensure config is loaded (reload if manager was initialized before config file fixed)
            if getattr(self, 'bb_config', None) is None and load_config:
                try:
                    self.bb_config = load_config()
                except Exception:
                    self.bb_config = None

            preferred = []
            require_llama = True
            if getattr(self, 'bb_config', None) and self.bb_config.roles:
                preferred = [(p.lower(), m) for p, m in self.bb_config.roles.judge]
            if getattr(self, 'bb_config', None) and self.bb_config.scoring:
                require_llama = bool(self.bb_config.scoring.require_llama_for_judge)

            logger.info(f"select_for_role(judge): preferred={preferred}, require_llama={require_llama}")
            logger.info("select_for_role: all_slots=" + ",".join([f"{p}:{s.slot_id}:{s.models}:{s.is_available()}" for p,s in self.all_slots]))

            # Try preferred list in order
            for prov, model in preferred:
                for pname, slot in self.all_slots:
                    if pname.lower() != prov.lower():
                        continue
                    if model:
                        # case-insensitive model match against slot models
                        model_match = any(model.lower() == sm.lower() for sm in slot.models)
                        if not model_match:
                            continue
                    if slot.is_available():
                        # assign this slot and model (use peek to avoid rotating index)
                        # we set current_model to the explicit model string
                        chosen_model = model if model else slot.get_next_model()
                        self.session_assignments[session_id] = (pname, slot.slot_id, chosen_model)
                        logger.info(f"Judge session {session_id[:8]} pinned to {pname}/{slot.slot_id}/{chosen_model}")
                        return (pname, slot.slot_id, chosen_model)

            # If require_llama_for_judge, fail closed
            if require_llama:
                logger.error("No configured Llama judge slots available and 'require_llama_for_judge' is true")
                return None
            # Otherwise fallback to normal round-robin
            return self.select_for_session(session_id, force_new=force_new)

        # User selection path: default to existing behavior
        if role_lc == "user":
            # Allow round-robin unless config specifically disables it
            allow_rr = True
            if getattr(self, 'bb_config', None) and self.bb_config.roles:
                allow_rr = bool(self.bb_config.roles.user_allow_round_robin)
            if allow_rr:
                return self.select_for_session(session_id, force_new=force_new)
            else:
                return self.select_for_session(session_id, force_new=force_new)

        # Unrecognized role: default behavior
        return self.select_for_session(session_id, force_new=force_new)
    
    def get_session_provider(self, session_id: str) -> Optional[BaseLLMProvider]:
        """Get the provider assigned to a session"""
        if session_id in self.session_assignments:
            provider_name, _, _ = self.session_assignments[session_id]
            return self.providers.get(provider_name)
        return None
    
    async def generate_with_tools(self, session_id: str, messages: List[Dict], 
                                   tools: List[Dict], system_prompt: str) -> LLMResponse:
        """Generate response with tools, handling failover."""
        selection = self.select_for_session(session_id)
        if not selection:
            return LLMResponse(
                text="I'm having trouble connecting to our AI services. Would you like someone from the team to call you back?",
                error="No slots available"
            )
        
        provider_name, slot_id, model_name = selection
        provider = self.providers[provider_name]
        
        for slot in provider.config.slots:
            if slot.slot_id == slot_id:
                provider.current_slot = slot
                break
        provider.current_model = model_name
        
        result = await provider.generate_with_tools(messages, tools, system_prompt)
        
        if result.error:
            logger.warning(f"Slot {slot_id} failed: {result.error}")
            
            del self.session_assignments[session_id]
            
            tried_slots = {slot_id}
            for _ in range(min(5, len(self.all_slots))):
                fallback = self._select_next_slot()
                if not fallback:
                    break
                    
                fb_provider_name, fb_slot, fb_model = fallback
                if fb_slot.slot_id in tried_slots:
                    continue
                tried_slots.add(fb_slot.slot_id)
                
                logger.info(f"Failover to {fb_provider_name}/{fb_slot.slot_id}/{fb_model}")
                
                fb_provider = self.providers[fb_provider_name]
                fb_provider.current_slot = fb_slot
                fb_provider.current_model = fb_model
                
                result = await fb_provider.generate_with_tools(messages, tools, system_prompt)
                if not result.error:
                    self.session_assignments[session_id] = (fb_provider_name, fb_slot.slot_id, fb_model)
                    return result
            
            return LLMResponse(
                text="I'm having trouble connecting to our AI services. Would you like someone from the team to call you back?",
                error="All slots failed"
            )
        
        return result
    
    def get_session_model_info(self, session_id: str) -> Dict[str, str]:
        """Get provider, slot, and model info for a session"""
        if session_id in self.session_assignments:
            provider_name, slot_id, model_name = self.session_assignments[session_id]
            return {"provider": provider_name, "slot": slot_id, "model": model_name}
        return {"provider": "unknown", "slot": "unknown", "model": "unknown"}
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Get statistics about providers and slots"""
        stats = {
            "total_slots": len(self.all_slots),
            "available_slots": len(self.get_available_slots()),
            "total_models": self._count_total_models(),
            "next_slot_idx": self.next_slot_idx,
            "providers": {}
        }
        
        for name, provider in self.providers.items():
            slot_info = []
            for slot in provider.config.slots:
                slot_info.append({
                    "slot_id": slot.slot_id,
                    "models": slot.models,
                    "status": slot.status.value,
                    "has_key": slot.get_api_key() is not None,
                    "last_model_idx": slot.last_model_idx,
                    "error_count": slot.error_count
                })

            stats["providers"][name] = {
                "priority": provider.config.priority,
                "slots": slot_info,
                "supports_function_calling": provider.config.supports_function_calling
            }

        return stats


class TinyLlamaProvider(BaseLLMProvider):
    """Local TinyLlama provider - expects models to be downloaded into `models/tinyllama`"""

    def __init__(self):
        slots = [
            APIKeySlot(
                slot_id="tinyllama_local",
                api_key_env="",
                models=["tinyllama-local"]
            )
        ]
        config = ProviderConfig(name="tinyllama", slots=slots, priority=40)
        super().__init__(config)

    def is_available(self) -> bool:
        # Consider the provider available if a local model directory exists
        possible_paths = [
            os.path.join(os.getcwd(), "models", "tinyllama"),
            os.path.expanduser("~/.cache/huggingface/tinyllama-local"),
        ]
        return any(os.path.isdir(p) for p in possible_paths)

    async def generate_with_tools(self, messages: List[Dict], tools: List[Dict], system_prompt: str) -> LLMResponse:
        # Placeholder: actual local model invocation should be implemented by operator
        return LLMResponse(text="[tinyllama placeholder response]", model="tinyllama-local", provider="tinyllama")


class GlaiveProvider(BaseLLMProvider):
    """Local/On-prem Glaive provider for function-calling capable models.
    Operator must place model into `models/glaive-function-calling-v1` or configure HF auth.
    """

    def __init__(self):
        slots = [
            APIKeySlot(
                slot_id="glaive_local",
                api_key_env="",
                models=["glaive-function-calling-v1"]
            )
        ]
        config = ProviderConfig(name="glaive", slots=slots, priority=45, supports_function_calling=True)
        super().__init__(config)

    def is_available(self) -> bool:
        possible_paths = [
            os.path.join(os.getcwd(), "models", "glaive-function-calling-v1"),
            os.path.expanduser("~/.cache/huggingface/glaive-function-calling-v1"),
        ]
        return any(os.path.isdir(p) for p in possible_paths)

    async def generate_with_tools(self, messages: List[Dict], tools: List[Dict], system_prompt: str) -> LLMResponse:
        return LLMResponse(text="[glaive placeholder response]", model="glaive-function-calling-v1", provider="glaive")
    
    def get_slot_rotation_summary(self) -> str:
        """Get human-readable summary of slot rotation status"""
        available = self.get_available_slots()
        lines = [f"Slot Pool: {len(available)}/{len(self.all_slots)} available, next_idx={self.next_slot_idx}"]
        
        for i, (provider_name, slot) in enumerate(available):
            marker = " -> " if i == (self.next_slot_idx % len(available)) else "    "
            model_idx = slot.last_model_idx % len(slot.models)
            lines.append(f"{marker}[{i}] {provider_name}/{slot.slot_id}: {slot.models[model_idx]} (idx {model_idx}/{len(slot.models)})")
        
        return "\n".join(lines)


llm_manager = LLMProviderManager()
