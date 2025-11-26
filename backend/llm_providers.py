"""
Unified LLM Provider Manager

Handles multiple LLM providers with failover, model rotation, and tracking.
Design: One model per conversation, switch only on rate limit/failure.
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

logger = logging.getLogger(__name__)

class ProviderStatus(Enum):
    AVAILABLE = "available"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    NO_API_KEY = "no_api_key"

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
    """Configuration for a provider"""
    name: str
    models: List[str]
    api_key_env: str
    priority: int = 50
    supports_function_calling: bool = True

class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers"""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.api_key = os.getenv(config.api_key_env)
        self.status = ProviderStatus.AVAILABLE if self.api_key else ProviderStatus.NO_API_KEY
        self.last_error_time: Optional[float] = None
        self.error_count = 0
        self.current_model: Optional[str] = None
    
    def select_model(self) -> str:
        """Randomly select a model from available models"""
        self.current_model = random.choice(self.config.models)
        return self.current_model
    
    def mark_rate_limited(self):
        """Mark provider as rate limited"""
        self.status = ProviderStatus.RATE_LIMITED
        self.last_error_time = time.time()
        self.error_count += 1
        logger.warning(f"Provider {self.config.name} marked as rate limited")
    
    def mark_error(self, error: str):
        """Mark provider as having an error"""
        self.status = ProviderStatus.ERROR
        self.last_error_time = time.time()
        self.error_count += 1
        logger.error(f"Provider {self.config.name} error: {error}")
    
    def reset_if_recovered(self, cooldown_seconds: int = 60):
        """Reset status if cooldown has passed"""
        if self.last_error_time and time.time() - self.last_error_time > cooldown_seconds:
            self.status = ProviderStatus.AVAILABLE
            logger.info(f"Provider {self.config.name} recovered after cooldown")
    
    def is_available(self) -> bool:
        """Check if provider is available"""
        self.reset_if_recovered()
        return self.status == ProviderStatus.AVAILABLE
    
    @abstractmethod
    async def generate_with_tools(self, messages: List[Dict], tools: List[Dict],
                                   system_prompt: str) -> LLMResponse:
        """Generate response with function calling support"""
        pass


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider"""
    
    def __init__(self):
        config = ProviderConfig(
            name="gemini",
            models=["gemini-2.0-flash", "gemini-2.5-flash"],
            api_key_env="GOOGLE_API_KEY",
            priority=80,
            supports_function_calling=True
        )
        super().__init__(config)
        self._configured = False
    
    async def generate_with_tools(self, messages: List[Dict], tools: List[Dict],
                                   system_prompt: str) -> LLMResponse:
        start_time = time.time()
        
        if not self.api_key:
            return LLMResponse(error="No API key", provider=self.config.name)
        
        try:
            import google.generativeai as genai
            from google.generativeai.types import FunctionDeclaration, Tool
            
            if not self._configured:
                genai.configure(api_key=self.api_key)
                self._configured = True
            
            model_name = self.current_model or self.select_model()
            logger.info(f"Gemini using model: {model_name}")
            
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
                tools=gemini_tools,
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
            if "rate" in error_str or "quota" in error_str or "429" in error_str:
                self.mark_rate_limited()
            else:
                self.mark_error(str(e))
            return LLMResponse(error=str(e), provider=self.config.name, 
                             latency_ms=int((time.time() - start_time) * 1000))


class OpenAICompatibleProvider(BaseLLMProvider):
    """Base class for OpenAI-compatible API providers"""
    
    def __init__(self, config: ProviderConfig, base_url: str):
        super().__init__(config)
        self.base_url = base_url
        self.client = None
    
    def _get_client(self):
        if not self.client:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        return self.client
    
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
        
        if not self.api_key:
            return LLMResponse(error="No API key", provider=self.config.name)
        
        try:
            client = self._get_client()
            model_name = self.current_model or self.select_model()
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
            if "rate" in error_str or "quota" in error_str or "429" in error_str:
                self.mark_rate_limited()
            else:
                self.mark_error(str(e))
            return LLMResponse(error=str(e), provider=self.config.name,
                             latency_ms=int((time.time() - start_time) * 1000))


class MistralProvider(OpenAICompatibleProvider):
    """Mistral AI provider"""
    
    def __init__(self):
        config = ProviderConfig(
            name="mistral",
            models=["mistral-small-latest", "mistral-large-latest", "pixtral-12b-latest"],
            api_key_env="MISTRAL_API_KEY",
            priority=60,
            supports_function_calling=True
        )
        super().__init__(config, "https://api.mistral.ai/v1")


class CohereProvider(BaseLLMProvider):
    """Cohere provider with native API"""
    
    def __init__(self):
        config = ProviderConfig(
            name="cohere",
            models=["command-r-plus", "command-r", "command-a-03-2025"],
            api_key_env="COHERE_API_KEY",
            priority=55,
            supports_function_calling=True
        )
        super().__init__(config)
        self.client = None
    
    def _get_client(self):
        if not self.client:
            import cohere
            self.client = cohere.ClientV2(self.api_key)
        return self.client
    
    async def generate_with_tools(self, messages: List[Dict], tools: List[Dict],
                                   system_prompt: str) -> LLMResponse:
        start_time = time.time()
        
        if not self.api_key:
            return LLMResponse(error="No API key", provider=self.config.name)
        
        try:
            client = self._get_client()
            model_name = self.current_model or self.select_model()
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
            if "rate" in error_str or "quota" in error_str or "429" in error_str:
                self.mark_rate_limited()
            else:
                self.mark_error(str(e))
            return LLMResponse(error=str(e), provider=self.config.name,
                             latency_ms=int((time.time() - start_time) * 1000))


class HuggingFaceProvider(OpenAICompatibleProvider):
    """HuggingFace Inference provider"""
    
    def __init__(self):
        config = ProviderConfig(
            name="huggingface",
            models=["meta-llama/Llama-3.3-70B-Instruct", "deepseek-ai/DeepSeek-V3-0324"],
            api_key_env="HUGGINGFACE_API_KEY",
            priority=50,
            supports_function_calling=True
        )
        super().__init__(config, "https://router.huggingface.co/v1")


class ReplicateProvider(BaseLLMProvider):
    """Replicate provider for KimiK2"""
    
    def __init__(self):
        config = ProviderConfig(
            name="replicate",
            models=["moonshotai/kimi-k2-instruct"],
            api_key_env="REPLICATE_API_TOKEN",
            priority=40,
            supports_function_calling=False
        )
        super().__init__(config)
    
    async def generate_with_tools(self, messages: List[Dict], tools: List[Dict],
                                   system_prompt: str) -> LLMResponse:
        start_time = time.time()
        
        if not self.api_key:
            return LLMResponse(error="No API key", provider=self.config.name)
        
        try:
            import replicate
            
            model_name = self.current_model or self.select_model()
            logger.info(f"Replicate using model: {model_name}")
            
            prompt_parts = [f"System: {system_prompt}"]
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt_parts.append(f"{role.capitalize()}: {content}")
            
            full_prompt = "\n\n".join(prompt_parts) + "\n\nAssistant:"
            
            output = replicate.run(
                model_name,
                input={
                    "prompt": full_prompt,
                    "max_tokens": 2048,
                    "temperature": 0.7
                }
            )
            
            response_text = "".join(output) if hasattr(output, '__iter__') else str(output)
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            return LLMResponse(
                text=response_text.strip(),
                model=model_name,
                provider=self.config.name,
                latency_ms=latency_ms
            )
            
        except Exception as e:
            error_str = str(e).lower()
            if "rate" in error_str or "limit" in error_str:
                self.mark_rate_limited()
            else:
                self.mark_error(str(e))
            return LLMResponse(error=str(e), provider=self.config.name,
                             latency_ms=int((time.time() - start_time) * 1000))


class ZaiProvider(OpenAICompatibleProvider):
    """Z.ai (Zhipu) provider"""
    
    def __init__(self):
        config = ProviderConfig(
            name="zai",
            models=["glm-4.5", "glm-4.5-air", "glm-4.6"],
            api_key_env="Z_API_KEY",
            priority=85,
            supports_function_calling=True
        )
        super().__init__(config, "https://api.z.ai/api/paas/v4")


class NvidiaProvider(BaseLLMProvider):
    """Nvidia NIM provider - handles multiple API keys for different models"""
    
    def __init__(self):
        config = ProviderConfig(
            name="nvidia",
            models=[
                "meta/llama-4-maverick-17b-128e-instruct",
                "meta/llama-4-scout-17b-16e-instruct",
                "deepseek-ai/deepseek-r1",
                "meta/llama-3.3-70b-instruct",
                "qwen/qwen2.5-coder-32b-instruct"
            ],
            api_key_env="NVIDIA_LLAMA4_128e",
            priority=90,
            supports_function_calling=True
        )
        super().__init__(config)
        
        self.model_to_api_key = {
            "meta/llama-4-maverick-17b-128e-instruct": os.getenv("NVIDIA_LLAMA4_128e"),
            "meta/llama-4-scout-17b-16e-instruct": os.getenv("NVIDIA_LLAMA4_16e"),
            "deepseek-ai/deepseek-r1": os.getenv("NVIDIA_DEEPSEEK_r1_API_KEY"),
            "meta/llama-3.3-70b-instruct": os.getenv("NVIDIA_LLAMA33_API_KEY"),
            "qwen/qwen2.5-coder-32b-instruct": os.getenv("NVIDIA_QWEN25_API_KEY")
        }
        
        available_models = [m for m, k in self.model_to_api_key.items() if k]
        if available_models:
            self.config.models = available_models
            self.status = ProviderStatus.AVAILABLE
        else:
            self.status = ProviderStatus.NO_API_KEY
    
    async def generate_with_tools(self, messages: List[Dict], tools: List[Dict],
                                   system_prompt: str) -> LLMResponse:
        start_time = time.time()
        
        model_name = self.current_model or self.select_model()
        api_key = self.model_to_api_key.get(model_name)
        
        if not api_key:
            return LLMResponse(error=f"No API key for model {model_name}", provider=self.config.name)
        
        try:
            logger.info(f"Nvidia using model: {model_name}")
            
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
            if "rate" in error_str or "429" in error_str:
                self.mark_rate_limited()
            else:
                self.mark_error(str(e))
            return LLMResponse(error=str(e), provider=self.config.name,
                             latency_ms=int((time.time() - start_time) * 1000))


class LLMProviderManager:
    """
    Manages multiple LLM providers with failover logic.
    
    Design:
    - Pick one provider+model at conversation start
    - Stick with it for the entire conversation
    - Switch only on rate limit/failure
    - Track which model was used for evaluation
    """
    
    def __init__(self):
        self.providers: Dict[str, BaseLLMProvider] = {}
        self._init_providers()
        self.session_providers: Dict[str, Tuple[str, str]] = {}
    
    def _init_providers(self):
        """Initialize all available providers"""
        provider_classes = [
            NvidiaProvider,
            ZaiProvider,
            GeminiProvider,
            MistralProvider,
            CohereProvider,
            HuggingFaceProvider,
            ReplicateProvider,
        ]
        
        for cls in provider_classes:
            try:
                provider = cls()
                self.providers[provider.config.name] = provider
                status = "available" if provider.is_available() else "unavailable"
                logger.info(f"Provider {provider.config.name}: {status} (models: {len(provider.config.models)})")
            except Exception as e:
                logger.error(f"Failed to initialize provider {cls.__name__}: {e}")
    
    def get_available_providers(self) -> List[BaseLLMProvider]:
        """Get list of available providers sorted by priority"""
        available = [p for p in self.providers.values() if p.is_available()]
        return sorted(available, key=lambda p: p.config.priority, reverse=True)
    
    def select_provider_for_session(self, session_id: str, 
                                     force_new: bool = False) -> Optional[Tuple[str, str]]:
        """
        Select a provider and model for a session.
        Returns (provider_name, model_name) or None if no providers available.
        """
        if not force_new and session_id in self.session_providers:
            provider_name, model_name = self.session_providers[session_id]
            provider = self.providers.get(provider_name)
            if provider and provider.is_available():
                return (provider_name, model_name)
        
        available = self.get_available_providers()
        if not available:
            logger.error("No available LLM providers!")
            return None
        
        weights = [p.config.priority for p in available]
        total = sum(weights)
        r = random.uniform(0, total)
        cumulative = 0
        selected_provider = available[0]
        
        for provider, weight in zip(available, weights):
            cumulative += weight
            if r <= cumulative:
                selected_provider = provider
                break
        
        model_name = selected_provider.select_model()
        self.session_providers[session_id] = (selected_provider.config.name, model_name)
        
        logger.info(f"Session {session_id[:8]}... assigned to {selected_provider.config.name}/{model_name}")
        return (selected_provider.config.name, model_name)
    
    def get_session_provider(self, session_id: str) -> Optional[BaseLLMProvider]:
        """Get the provider assigned to a session"""
        if session_id in self.session_providers:
            provider_name, _ = self.session_providers[session_id]
            return self.providers.get(provider_name)
        return None
    
    async def generate_with_tools(self, session_id: str, messages: List[Dict], 
                                   tools: List[Dict], system_prompt: str) -> LLMResponse:
        """
        Generate response with tools, handling failover.
        """
        selection = self.select_provider_for_session(session_id)
        if not selection:
            return LLMResponse(
                text="I'm having trouble connecting to our AI services. Would you like someone from the team to call you back?",
                error="No providers available"
            )
        
        provider_name, model_name = selection
        provider = self.providers[provider_name]
        provider.current_model = model_name
        
        result = await provider.generate_with_tools(messages, tools, system_prompt)
        
        if result.error:
            logger.warning(f"Provider {provider_name} failed: {result.error}")
            
            del self.session_providers[session_id]
            
            for fallback_provider in self.get_available_providers():
                if fallback_provider.config.name != provider_name:
                    logger.info(f"Failing over to {fallback_provider.config.name}")
                    fallback_provider.select_model()
                    self.session_providers[session_id] = (
                        fallback_provider.config.name, 
                        fallback_provider.current_model
                    )
                    
                    result = await fallback_provider.generate_with_tools(messages, tools, system_prompt)
                    if not result.error:
                        return result
            
            return LLMResponse(
                text="I'm having trouble connecting to our AI services. Would you like someone from the team to call you back?",
                error="All providers failed"
            )
        
        return result
    
    def get_session_model_info(self, session_id: str) -> Dict[str, str]:
        """Get provider and model info for a session"""
        if session_id in self.session_providers:
            provider_name, model_name = self.session_providers[session_id]
            return {"provider": provider_name, "model": model_name}
        return {"provider": "unknown", "model": "unknown"}
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Get statistics about providers"""
        stats = {}
        for name, provider in self.providers.items():
            stats[name] = {
                "status": provider.status.value,
                "models": provider.config.models,
                "priority": provider.config.priority,
                "error_count": provider.error_count,
                "supports_function_calling": provider.config.supports_function_calling
            }
        return stats


llm_manager = LLMProviderManager()
