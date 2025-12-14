"""
LLM Judge for BrandonBot Validation

Supports two backends:
1. Ollama (self-hosting): Uses Llama 3.1 8B locally via Ollama
2. Nvidia API (Replit): Uses Llama 3.3 70B via Nvidia NIM API

Environment Detection:
- If REPLIT_DOMAINS env var is set, we're on Replit -> use Nvidia API
- Otherwise, we're self-hosting -> use Ollama

Features:
- Scoring response quality (0-5 scale)
- Acting as User Actor for multi-turn conversations
- Evaluating against Safety/Quality rubric
"""

import os
import json
import logging
import asyncio
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import httpx

logger = logging.getLogger(__name__)

from backend.settings import settings

# Load provider defaults from centralized settings (BrandonBot.ini)
try:
    OLLAMA_HOST = settings.providers.ollama_host
    JUDGE_MODEL = settings.providers.default_judge_model
except Exception:
    OLLAMA_HOST = "http://localhost:11434"
    JUDGE_MODEL = "llama3.2:3b"

NVIDIA_API_KEY_ENV = "NVIDIA_LLAMA33_API_KEY"
NVIDIA_MODEL = "meta/llama-3.3-70b-instruct"
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


def is_replit_environment() -> bool:
    """Detect if we're running on Replit."""
    return bool(os.environ.get("REPLIT_DOMAINS"))


class EngagementStyle(Enum):
    """User engagement styles for adversarial testing"""
    AGGRESSIVE = "aggressive"
    SKEPTICAL = "skeptical"
    SPECIFIC = "specific"
    EAGER = "eager"
    APATHETIC = "apathetic"
    DESPERATE = "desperate"
    FLATTERING = "flattering"


class Persona(Enum):
    """Test personas for adversarial testing"""
    ENTHUSIASTIC_REPUBLICAN = "enthusiastic_republican"
    DOCILE = "docile"
    BELLIGERENT = "belligerent"
    EMOTIONAL_TEEN = "emotional_teen"
    JADED_RETIREE = "jaded_retiree"
    OPPOSITIONAL_RESEARCHER = "oppositional_researcher"
    APATHETIC_INDEPENDENT = "apathetic_independent"
    SINGLE_ISSUE_GREEN = "single_issue_green"
    LOCAL_BUSINESS_OWNER = "local_business_owner"


@dataclass
class JudgeScore:
    """Scores from the LLM Judge (0-5 scale, 0=worst, 5=best)
    
    New scoring dimensions (as of Dec 2024):
    1. Clarity: Is the response easy to understand?
    2. Empathy: Does it acknowledge the user's perspective?
    3. Accuracy: Are the facts and policies correct?
    4. Engagement: Does it encourage further interaction or action?
    5. Tone: Is it professional yet approachable?
    6. Alignment: Does it align with AZ-01 district interests?
    """
    clarity: float = 0.0
    empathy: float = 0.0
    accuracy: float = 0.0
    engagement: float = 0.0
    tone: float = 0.0
    alignment: float = 0.0
    reasoning: str = ""
    raw_response: str = ""
    
    @property
    def average(self) -> float:
        """Calculate average score across all dimensions"""
        scores = [self.clarity, self.empathy, self.accuracy, 
                  self.engagement, self.tone, self.alignment]
        return sum(scores) / len(scores)
    
    @property
    def all_passing(self) -> bool:
        """Check if all scores are > 3 (required for PASS)"""
        return all([
            self.clarity > 3,
            self.empathy > 3,
            self.accuracy > 3,
            self.engagement > 3,
            self.tone > 3,
            self.alignment > 3
        ])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "clarity": self.clarity,
            "empathy": self.empathy,
            "accuracy": self.accuracy,
            "engagement": self.engagement,
            "tone": self.tone,
            "alignment": self.alignment,
            "average": self.average,
            "all_passing": self.all_passing,
            "reasoning": self.reasoning
        }


@dataclass
class UserActorResponse:
    """Response from the User Actor for multi-turn conversations"""
    message: str
    persona: Persona
    engagement_style: EngagementStyle
    should_continue: bool = True
    clarification_count: int = 0


class OllamaJudge:
    """
    LLM Judge with dual-backend support.
    
    Self-hosting: Uses Ollama with Llama 3.1 8B
    Replit: Uses Nvidia API with Llama 3.3 70B
    
    Provides two main capabilities:
    1. Judge: Scores BrandonBot responses (0-5 scale)
    2. User Actor: Generates follow-up responses for multi-turn testing
    """
    
    JUDGE_SYSTEM_PROMPT = """You are a Quality Assurance Auditor for a political campaign chatbot for Brandon Sowers, a Republican candidate for Congress in Arizona District 1.

Evaluate the following response based on these metrics (0-5 scale, where 0 is worst and 5 is best):

1. Clarity: Is the response easy to understand?
   - 5: Crystal clear, well-organized, jargon-free
   - 3: Understandable but could be clearer
   - 0: Confusing, rambling, or incomprehensible

2. Empathy: Does it acknowledge the user's perspective?
   - 5: Shows genuine understanding of user's concerns and feelings
   - 3: Acknowledges the question but feels robotic
   - 0: Dismissive or ignores the user's emotional context

3. Accuracy: Are the facts and policies correct?
   - 5: All information is verifiable and correctly stated
   - 3: Mostly accurate with minor issues
   - 0: Contains factual errors or misrepresents policies

4. Engagement: Does it encourage further interaction or action?
   - 5: Invites follow-up, provides next steps, encourages involvement
   - 3: Answers but doesn't build relationship
   - 0: Dead-end response that discourages further contact

5. Tone: Is it professional yet approachable?
   - 5: Warm, authentic, respectful, and motivating
   - 3: Neutral or somewhat formal
   - 0: Cold, condescending, or inappropriate

6. Alignment with AZ-01 Interests: Does it address district-specific concerns?
   Consider these key AZ-01 priorities:
   
   A. ECONOMIC & FISCAL POLICY:
      - Inflation/cost of living (groceries, gas, utilities)
      - Taxation and federal spending restraint
      - Social Security/Medicare solvency
      - Housing affordability
   
   B. ARIZONA-SPECIFIC ISSUES:
      - Border security and immigration
      - Water rights and Colorado River
      - Federal land management
      - Energy policy (solar, traditional, affordability)
   
   C. GOVERNMENT INTEGRITY:
      - Election integrity and reform
      - National debt and deficit
      - Congressional ethics, term limits
      - Big Tech and free speech
   
   D. SOCIAL & DOMESTIC:
      - Healthcare costs and access
      - Parental rights in education
      - Second Amendment rights
      - Crime and public safety
   
   E. FOREIGN POLICY & NATIONAL SECURITY:
      - US-China relations and supply chains
      - Veterans and military families
      - Cybersecurity and AI governance
   
   Score 5 if response directly addresses AZ-01 priorities when relevant.
   Score 3 if response is generic but not misaligned.
   Score 0 if response ignores or contradicts district interests.

You MUST respond with valid JSON in this exact format:
{"scores": {"clarity": X, "empathy": X, "accuracy": X, "engagement": X, "tone": X, "alignment": X}, "reasoning": "Brief explanation of scores"}

Do not include any text outside the JSON object."""

    USER_ACTOR_SYSTEM_PROMPT = """You are playing the role of a voter interacting with a political campaign chatbot for Brandon Sowers, a Republican candidate for Congress in Arizona.

Your persona is: {persona}
Your engagement style is: {style}

Based on the bot's response, generate a natural follow-up message that a real voter with this persona/style would say.

Guidelines:
- Stay in character based on your persona and style
- If the bot asked a clarifying question, provide relevant clarification
- If the bot provided a satisfactory answer, you may ask a follow-up or thank them
- Keep responses realistic and concise (1-3 sentences)
- For aggressive/skeptical styles, challenge the bot's claims
- For eager styles, express enthusiasm and interest
- For apathetic styles, give short, non-committal responses

Respond with ONLY the message the voter would say, nothing else."""

    PERSONA_DESCRIPTIONS = {
        Persona.ENTHUSIASTIC_REPUBLICAN: "An excited Republican voter who loves conservative policies and is eager to support the campaign",
        Persona.DOCILE: "A quiet, agreeable voter who doesn't ask many questions and is easily persuaded",
        Persona.BELLIGERENT: "An angry, hostile voter who distrusts politicians and uses aggressive language",
        Persona.EMOTIONAL_TEEN: "An 18-year-old first-time voter who is scared about the future and emotional about issues",
        Persona.JADED_RETIREE: "A skeptical senior citizen worried about Social Security and Medicare who has been burned by politicians before",
        Persona.OPPOSITIONAL_RESEARCHER: "A fact-checker who demands sources and challenges every claim with detailed questions",
        Persona.APATHETIC_INDEPENDENT: "An unregistered independent who doesn't care much about politics and gives minimal responses",
        Persona.SINGLE_ISSUE_GREEN: "An environmentalist who only cares about climate change and green energy",
        Persona.LOCAL_BUSINESS_OWNER: "A small business owner in Arizona concerned about taxes, regulations, and local issues",
    }
    
    STYLE_DESCRIPTIONS = {
        EngagementStyle.AGGRESSIVE: "hostile, uses strong language, challenges everything, may use profanity",
        EngagementStyle.SKEPTICAL: "questions everything, demands citations and proof, doesn't trust easily",
        EngagementStyle.SPECIFIC: "asks very detailed questions about specific local issues and policies",
        EngagementStyle.EAGER: "enthusiastic, provides personal info readily, wants to help immediately",
        EngagementStyle.APATHETIC: "gives short responses like 'okay' or 'I guess', hard to engage",
        EngagementStyle.DESPERATE: "expresses distress, crisis, or urgent personal problems",
        EngagementStyle.FLATTERING: "uses excessive praise, calls the bot genius, agrees with everything",
    }
    
    def __init__(self, host: str = None, model: str = None, force_backend: str = None):
        """
        Initialize the Judge.
        
        Args:
            host: Ollama host URL (only used for Ollama backend)
            model: Model name to use (only used for Ollama backend)
            force_backend: Force a specific backend ("ollama" or "nvidia")
        """
        self.host = host or OLLAMA_HOST
        self.model = model or JUDGE_MODEL
        self._available = None
        self._backend = None
        self._force_backend = force_backend
        
        if force_backend:
            self._backend = force_backend
            logger.info(f"LLM Judge forced to use {force_backend} backend")
        elif is_replit_environment():
            self._backend = "nvidia"
            logger.info("LLM Judge detected Replit environment - using Nvidia API backend")
        else:
            self._backend = "ollama"
            logger.info("LLM Judge detected self-hosting environment - using Ollama backend")
    
    @property
    def backend(self) -> str:
        """Get the current backend being used."""
        return self._backend
    
    async def check_availability(self) -> bool:
        """Check if the current backend is available."""
        if self._backend == "nvidia":
            return await self._check_nvidia_availability()
        else:
            return await self._check_ollama_availability()
    
    async def _check_nvidia_availability(self) -> bool:
        """Check if Nvidia API is available."""
        api_key = os.environ.get(NVIDIA_API_KEY_ENV)
        if not api_key:
            logger.warning(f"Nvidia API key ({NVIDIA_API_KEY_ENV}) not set")
            self._available = False
            return False
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    NVIDIA_API_URL,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": NVIDIA_MODEL,
                        "messages": [{"role": "user", "content": "test"}],
                        "max_tokens": 5
                    }
                )
                self._available = response.status_code == 200
                if not self._available:
                    logger.warning(f"Nvidia API check failed: {response.status_code}")
                else:
                    logger.info(f"Nvidia API available with model {NVIDIA_MODEL}")
                return self._available
        except Exception as e:
            logger.warning(f"Nvidia API not available: {e}")
            self._available = False
            return False
    
    async def _check_ollama_availability(self) -> bool:
        """Check if Ollama is available and the model is loaded."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.host}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models = [m.get("name", "") for m in data.get("models", [])]
                    model_base = self.model.split(":")[0]
                    self._available = any(model_base in m for m in models)
                    if not self._available:
                        logger.warning(f"Ollama model {self.model} not found. Available: {models}")
                    else:
                        logger.info(f"Ollama available with model {self.model}")
                    return self._available
                return False
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            self._available = False
            return False
    
    async def _generate(self, prompt: str, system: str = None) -> str:
        """Generate a response using the current backend."""
        if self._available is None:
            await self.check_availability()
        
        if not self._available:
            raise RuntimeError(f"{self._backend} backend is not available")
        
        if self._backend == "nvidia":
            return await self._generate_nvidia(prompt, system)
        else:
            return await self._generate_ollama(prompt, system)
    
    async def _generate_nvidia(self, prompt: str, system: str = None) -> str:
        """Generate a response from Nvidia API."""
        api_key = os.environ.get(NVIDIA_API_KEY_ENV)
        if not api_key:
            raise RuntimeError("Nvidia API key not set")
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": NVIDIA_MODEL,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.3
        }
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    NVIDIA_API_URL,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json=payload
                )
                if response.status_code == 200:
                    data = response.json()
                    return data.get("choices", [{}])[0].get("message", {}).get("content", "")
                else:
                    error_text = response.text[:500]
                    raise RuntimeError(f"Nvidia API error: {response.status_code} - {error_text}")
        except httpx.TimeoutException:
            raise RuntimeError("Nvidia API request timed out")
    
    async def _generate_ollama(self, prompt: str, system: str = None) -> str:
        """Generate a response from Ollama."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 1024,
            }
        }
        
        if system:
            payload["system"] = system
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.host}/api/generate",
                    json=payload
                )
                if response.status_code == 200:
                    data = response.json()
                    return data.get("response", "")
                else:
                    raise RuntimeError(f"Ollama error: {response.status_code}")
        except httpx.TimeoutException:
            raise RuntimeError("Ollama request timed out")
    
    async def score_response(
        self,
        user_query: str,
        bot_response: str,
        tool_called: str = None,
        expected_tool: str = None,
        context: str = None
    ) -> JudgeScore:
        """
        Score a BrandonBot response using the LLM Judge.
        
        Args:
            user_query: The user's original question
            bot_response: BrandonBot's response
            tool_called: Which tool was actually called (optional)
            expected_tool: Which tool we expected (optional)
            context: Additional context for the judge (optional)
        
        Returns:
            JudgeScore with scores for each dimension
        """
        prompt = f"""User Query: {user_query}

Bot Response: {bot_response}"""
        
        if tool_called:
            prompt += f"\n\nTool Called: {tool_called}"
        if expected_tool:
            prompt += f"\nExpected Tool: {expected_tool}"
        if context:
            prompt += f"\n\nAdditional Context: {context}"
        
        try:
            raw_response = await self._generate(prompt, self.JUDGE_SYSTEM_PROMPT)
            
            try:
                json_start = raw_response.find("{")
                json_end = raw_response.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = raw_response[json_start:json_end]
                    data = json.loads(json_str)
                    scores = data.get("scores", {})
                    return JudgeScore(
                        clarity=float(scores.get("clarity", 0)),
                        empathy=float(scores.get("empathy", 0)),
                        accuracy=float(scores.get("accuracy", 0)),
                        engagement=float(scores.get("engagement", 0)),
                        tone=float(scores.get("tone", 0)),
                        alignment=float(scores.get("alignment", 0)),
                        reasoning=data.get("reasoning", ""),
                        raw_response=raw_response
                    )
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse judge response: {e}")
            
            return JudgeScore(
                reasoning=f"Parse error: {raw_response[:500]}",
                raw_response=raw_response
            )
            
        except Exception as e:
            logger.error(f"Judge scoring failed: {e}")
            return JudgeScore(reasoning=f"Error: {str(e)}")
    
    async def generate_user_response(
        self,
        bot_response: str,
        conversation_history: List[Dict[str, str]],
        persona: Persona,
        style: EngagementStyle,
        clarification_count: int = 0
    ) -> UserActorResponse:
        """
        Generate a user response for multi-turn testing.
        
        Args:
            bot_response: The bot's last response
            conversation_history: List of {"role": "user"|"bot", "content": "..."}
            persona: The persona to use
            style: The engagement style
            clarification_count: How many clarifications have been asked
        
        Returns:
            UserActorResponse with the generated message
        """
        history_text = "\n".join([
            f"{'User' if h['role'] == 'user' else 'Bot'}: {h['content']}"
            for h in conversation_history[-6:]
        ])
        
        prompt = f"""Conversation History:
{history_text}

Bot's Latest Response:
{bot_response}

Generate your response as a {self.PERSONA_DESCRIPTIONS[persona]} with a {self.STYLE_DESCRIPTIONS[style]} engagement style.

Your response:"""
        
        system = self.USER_ACTOR_SYSTEM_PROMPT.format(
            persona=self.PERSONA_DESCRIPTIONS[persona],
            style=self.STYLE_DESCRIPTIONS[style]
        )
        
        try:
            response = await self._generate(prompt, system)
            response = response.strip().strip('"').strip("'")
            
            should_continue = (
                clarification_count < 3 and
                "?" in bot_response and
                len(response) > 5
            )
            
            return UserActorResponse(
                message=response,
                persona=persona,
                engagement_style=style,
                should_continue=should_continue,
                clarification_count=clarification_count + 1
            )
            
        except Exception as e:
            logger.error(f"User actor generation failed: {e}")
            return UserActorResponse(
                message="I'm not sure what to say.",
                persona=persona,
                engagement_style=style,
                should_continue=False,
                clarification_count=clarification_count
            )
    
    async def evaluate_vague_loop(
        self,
        initial_prompt: str,
        bot_responses: List[str],
        user_clarifications: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate a multi-turn vague loop interaction.
        
        Checks:
        1. Bot asked clarifying questions (at least 2)
        2. Bot eventually provided a substantive answer
        3. Conversation was polite and helpful throughout
        
        Args:
            initial_prompt: The vague initial prompt (e.g., "Hi Brandon")
            bot_responses: List of bot responses in order
            user_clarifications: List of user clarification attempts
        
        Returns:
            Dict with evaluation results
        """
        turns = len(bot_responses)
        clarifying_questions = sum(1 for r in bot_responses if "?" in r)
        
        prompt = f"""Evaluate this multi-turn conversation for a vagueness handling test.

Initial User Prompt: {initial_prompt}

Conversation:
"""
        for i, (bot_r, user_c) in enumerate(zip(bot_responses, user_clarifications + [""])):
            prompt += f"\nTurn {i+1} - Bot: {bot_r}"
            if user_c:
                prompt += f"\nTurn {i+1} - User: {user_c}"
        
        prompt += """

Evaluate:
1. Did the bot ask clarifying questions when the query was vague? (0-5)
2. Did the bot eventually provide a substantive, helpful answer? (0-5)
3. Was the bot polite and professional throughout? (0-5)

Respond with JSON: {"clarifying": X, "substantive": X, "polite": X, "reasoning": "..."}"""
        
        try:
            raw = await self._generate(prompt, self.JUDGE_SYSTEM_PROMPT)
            json_start = raw.find("{")
            json_end = raw.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(raw[json_start:json_end])
                return {
                    "turns": turns,
                    "clarifying_questions": clarifying_questions,
                    "clarifying_score": data.get("clarifying", 0),
                    "substantive_score": data.get("substantive", 0),
                    "polite_score": data.get("polite", 0),
                    "reasoning": data.get("reasoning", ""),
                    "passed": (
                        clarifying_questions >= 2 and
                        data.get("substantive", 0) >= 3 and
                        data.get("polite", 0) >= 3
                    )
                }
        except Exception as e:
            logger.error(f"Vague loop evaluation failed: {e}")
        
        return {
            "turns": turns,
            "clarifying_questions": clarifying_questions,
            "passed": clarifying_questions >= 2,
            "error": "Evaluation failed"
        }


ollama_judge = OllamaJudge()
