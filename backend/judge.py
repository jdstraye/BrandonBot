"""
Unified Judge Module for BrandonBot Validation

Automatically selects the appropriate judge based on deployment mode:
1. SELF_HOSTED mode (32GB+ RAM): Uses local Ollama with Llama 3.1 8B
2. REPLIT mode (8GB RAM limit): Uses fixed Nvidia model (no rotation)

Environment detection:
- SELF_HOSTED=true environment variable
- Or: Available memory > 16GB
- Default: Replit mode (Nvidia)

This module provides a consistent interface regardless of which backend is used.
"""

import os
import logging
from typing import Optional, Dict, Any, List, Union

logger = logging.getLogger(__name__)

SELF_HOSTED = os.environ.get("SELF_HOSTED", "").lower() in ("true", "1", "yes")


def _check_memory_available() -> int:
    """Check available memory in MB."""
    try:
        with open("/sys/fs/cgroup/memory.max", "r") as f:
            limit = f.read().strip()
            if limit == "max":
                return 32 * 1024  # Assume unlimited means 32GB
            return int(limit) // (1024 * 1024)  # Convert to MB
    except Exception:
        pass
    
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb // 1024  # Convert to MB
    except Exception:
        pass
    
    return 8 * 1024  # Default to 8GB


def detect_deployment_mode() -> str:
    """
    Detect whether we're in self-hosted or Replit mode.
    
    Returns:
        "self_hosted" or "replit"
    """
    if SELF_HOSTED:
        logger.info("Deployment mode: SELF_HOSTED (env var)")
        return "self_hosted"
    
    memory_mb = _check_memory_available()
    if memory_mb >= 16 * 1024:  # 16GB or more
        logger.info(f"Deployment mode: SELF_HOSTED (memory: {memory_mb}MB)")
        return "self_hosted"
    
    logger.info(f"Deployment mode: REPLIT (memory: {memory_mb}MB)")
    return "replit"


class UnifiedJudge:
    """
    Unified judge that automatically selects the appropriate backend.
    
    In self-hosted mode: Uses OllamaJudge with Llama 3.1 8B
    In Replit mode: Uses NvidiaJudge with fixed model (no rotation)
    
    Provides the same interface regardless of backend.
    """
    
    def __init__(self, force_mode: Optional[str] = None):
        """
        Initialize the unified judge.
        
        Args:
            force_mode: Override automatic detection. Options: "self_hosted", "replit", "nvidia", "ollama"
        """
        self._mode = force_mode or detect_deployment_mode()
        self._judge = None
        self._judge_type = None
        self._initialized = False
    
    async def _initialize(self):
        """Lazy initialization of the appropriate judge."""
        if self._initialized:
            return
        
        if self._mode in ("self_hosted", "ollama"):
            try:
                from ollama_judge import OllamaJudge
                self._judge = OllamaJudge()
                if await self._judge.check_availability():
                    self._judge_type = "ollama"
                    self._initialized = True
                    logger.info("UnifiedJudge: Using OllamaJudge (local Llama 3.1 8B)")
                    return
                else:
                    logger.warning("Ollama not available, falling back to Nvidia")
            except Exception as e:
                logger.warning(f"Failed to initialize OllamaJudge: {e}")
        
        try:
            from nvidia_judge import NvidiaJudge
            self._judge = NvidiaJudge()
            if await self._judge.check_availability():
                self._judge_type = "nvidia"
                self._initialized = True
                logger.info(f"UnifiedJudge: Using NvidiaJudge (fixed model: {self._judge.model})")
                return
            else:
                logger.error("NvidiaJudge not available")
        except Exception as e:
            logger.error(f"Failed to initialize NvidiaJudge: {e}")
        
        raise RuntimeError("No judge backend available. Check Ollama or Nvidia API key.")
    
    async def check_availability(self) -> bool:
        """Check if any judge backend is available."""
        try:
            await self._initialize()
            return self._initialized and self._judge is not None
        except Exception as e:
            logger.error(f"Judge availability check failed: {e}")
            return False
    
    @property
    def judge_type(self) -> Optional[str]:
        """Return the type of judge being used."""
        return self._judge_type
    
    @property
    def model(self) -> str:
        """Return the model being used."""
        if self._judge:
            return getattr(self._judge, 'model', 'unknown')
        return 'not_initialized'
    
    async def score_response(
        self,
        user_query: str,
        bot_response: str,
        tool_called: Optional[str] = None,
        expected_tool: Optional[str] = None,
        context: Optional[str] = None
    ):
        """Score a BrandonBot response."""
        await self._initialize()
        if self._judge is None:
            raise RuntimeError("No judge available")
        return await self._judge.score_response(
            user_query=user_query,
            bot_response=bot_response,
            tool_called=tool_called,
            expected_tool=expected_tool,
            context=context
        )
    
    async def generate_user_response(
        self,
        bot_response: str,
        conversation_history: List[Dict[str, str]],
        persona,
        style,
        clarification_count: int = 0
    ):
        """Generate a user response for multi-turn testing."""
        await self._initialize()
        if self._judge is None:
            raise RuntimeError("No judge available")
        return await self._judge.generate_user_response(
            bot_response=bot_response,
            conversation_history=conversation_history,
            persona=persona,
            style=style,
            clarification_count=clarification_count
        )
    
    async def evaluate_vague_loop(
        self,
        initial_prompt: str,
        bot_responses: List[str],
        user_clarifications: List[str]
    ) -> Dict[str, Any]:
        """Evaluate a multi-turn vague loop interaction."""
        await self._initialize()
        if self._judge is None:
            raise RuntimeError("No judge available")
        return await self._judge.evaluate_vague_loop(
            initial_prompt=initial_prompt,
            bot_responses=bot_responses,
            user_clarifications=user_clarifications
        )


unified_judge = UnifiedJudge()


async def get_judge(force_mode: Optional[str] = None):
    """
    Get an initialized judge instance.
    
    Args:
        force_mode: Override automatic detection. Options: "self_hosted", "replit", "nvidia", "ollama"
    
    Returns:
        An initialized judge (OllamaJudge or NvidiaJudge)
    """
    if force_mode:
        judge = UnifiedJudge(force_mode=force_mode)
    else:
        judge = unified_judge
    
    await judge._initialize()
    return judge
