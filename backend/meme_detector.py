"""
Meme/Subcontext Detection Module for BrandonBot

Detects culturally loaded phrases and political memes in user questions.
Uses web search + embedding analysis to identify hidden subtext.

Flow:
1. Check if question is short (<10 words) - potential meme
2. Search web for "[question] meme OR meaning OR context"
3. Analyze search results using all-MiniLM embeddings
4. If meme detected, return context for witty LLM response
"""

import asyncio
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

MEME_ANALYSIS_TEMPLATE = "This is about a political controversy, meme, or cultural debate"
MEME_SIMILARITY_THRESHOLD = 0.16


@dataclass
class MemeDetectionResult:
    """Result from meme/subcontext detection"""
    is_meme: bool = False
    phrase: str = ""
    context: str = ""
    search_snippets: List[str] = field(default_factory=list)
    similarity_score: float = 0.0
    suggested_pivot: str = ""
    confidence: float = 0.0
    cultural_context: str = ""
    reasoning: str = ""


class MemeDetector:
    """
    Detects political memes and culturally loaded phrases.
    
    Uses existing all-MiniLM-L6-v2 model to analyze web search results
    and determine if a short question has hidden cultural/political meaning.
    
    Primary: SearxNG public instances (unlimited, free)
    Fallback: SerpAPI (if configured)
    """
    
    def __init__(self):
        self._embedding_model = None
        self._template_embedding = None
        self._multi_search = None
        self._initialized = False
        self._init_lock = asyncio.Lock()
    
    async def ensure_ready(self) -> bool:
        """Initialize embedding model and multi-provider search."""
        if self._initialized:
            return True
        
        async with self._init_lock:
            if self._initialized:
                return True
            
            try:
                from sentence_transformers import SentenceTransformer
                from multi_search_service import multi_search_service
                
                logger.info("Loading meme detector (all-MiniLM-L6-v2 + SearxNG)...")
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                self._template_embedding = self._embedding_model.encode(MEME_ANALYSIS_TEMPLATE)
                self._multi_search = multi_search_service
                
                self._initialized = True
                logger.info("Meme detector ready with multi-provider search")
                return True
                
            except Exception as e:
                logger.error(f"Failed to initialize meme detector: {e}")
                return False
    
    def _is_short_question(self, query: str) -> bool:
        """Check if query is short enough to potentially be a meme phrase."""
        words = query.split()
        return len(words) <= 10
    
    def _build_search_query(self, phrase: str) -> str:
        """Build web search query for meme detection."""
        clean_phrase = phrase.strip().rstrip('?!.')
        return f'"{clean_phrase}" meme OR meaning OR context OR controversy'
    
    def _analyze_snippets(self, snippets: List[str]) -> Tuple[float, str]:
        """
        Analyze search snippets to determine if they indicate a meme/cultural reference.
        
        Returns:
            Tuple of (similarity_score, combined_context)
        """
        if not snippets or not self._embedding_model:
            return 0.0, ""
        
        snippet_embeddings = self._embedding_model.encode(snippets)
        avg_embedding = np.mean(snippet_embeddings, axis=0)
        
        similarity = float(np.dot(avg_embedding, self._template_embedding) / (
            np.linalg.norm(avg_embedding) * np.linalg.norm(self._template_embedding)
        ))
        
        context = " ".join(snippets[:3])
        if len(context) > 800:
            context = context[:800] + "..."
        
        return similarity, context
    
    def _determine_pivot(self, phrase: str, context: str) -> str:
        """
        Determine what topic to pivot to based on meme context.
        
        This provides guidance to the LLM on how to craft a witty response.
        """
        phrase_lower = phrase.lower()
        context_lower = context.lower()
        
        if "what is a woman" in phrase_lower:
            if any(term in context_lower for term in ["matt walsh", "documentary", "trans", "gender"]):
                return "trans activists and gender ideology debate"
        
        if "let's go brandon" in phrase_lower or "lets go brandon" in phrase_lower:
            return "the viral chant and political expression"
        
        if "mostly peaceful" in phrase_lower:
            if any(term in context_lower for term in ["protest", "riot", "cnn", "fiery"]):
                return "media coverage of protests and civil unrest"
        
        if "fine people on both sides" in phrase_lower:
            if any(term in context_lower for term in ["charlottesville", "trump", "hoax"]):
                return "Charlottesville and media fact-checking"
        
        if "build back better" in phrase_lower:
            return "economic policy and political slogans"
        
        if "defund the police" in phrase_lower:
            return "law enforcement funding and public safety"
        
        if "election" in phrase_lower and any(term in context_lower for term in ["stolen", "fraud", "rigged", "2020", "2024"]):
            return "election integrity and voter confidence"
        
        if "genders" in phrase_lower or "gender" in phrase_lower:
            return "gender identity and biological sex debates"
        
        if any(term in context_lower for term in ["meme", "viral", "controversy", "debate"]):
            return "the cultural context of this phrase"
        
        return ""
    
    async def detect(self, query: str) -> MemeDetectionResult:
        """
        Detect if a query contains a meme or culturally loaded phrase.
        
        Uses SearxNG public instances for unlimited free searches,
        with SerpAPI as fallback if configured.
        
        Args:
            query: User's question
            
        Returns:
            MemeDetectionResult with detection status and context
        """
        result = MemeDetectionResult(phrase=query)
        
        if not self._is_short_question(query):
            result.reasoning = "Query too long for meme detection"
            return result
        
        if not await self.ensure_ready():
            logger.warning("Meme detector not available, skipping detection")
            result.reasoning = "Meme detector not initialized"
            return result
        
        try:
            search_query = self._build_search_query(query)
            logger.info(f"Meme detection search: {search_query}")
            
            search_response = await self._multi_search.search(search_query, max_results=5)
            
            if not search_response.results:
                result.reasoning = f"No search results (provider: {search_response.provider}, error: {search_response.error})"
                logger.warning(f"Meme detection: no results for '{query}' - {result.reasoning}")
                return result
            
            snippets = [r.snippet for r in search_response.results if r.snippet]
            result.search_snippets = snippets
            result.reasoning = f"Got {len(snippets)} snippets from {search_response.provider}"
            
            similarity, context = self._analyze_snippets(snippets)
            result.similarity_score = similarity
            result.context = context
            result.cultural_context = context[:300] if context else ""
            result.confidence = min(similarity / MEME_SIMILARITY_THRESHOLD, 1.0) if similarity > 0 else 0.0
            
            if similarity >= MEME_SIMILARITY_THRESHOLD:
                result.is_meme = True
                result.suggested_pivot = self._determine_pivot(query, context)
                result.reasoning += f" | Meme detected (score: {similarity:.3f})"
                logger.info(f"Meme detected: '{query}' (score: {similarity:.3f}, pivot: {result.suggested_pivot})")
            else:
                result.reasoning += f" | Not a meme (score: {similarity:.3f} < threshold {MEME_SIMILARITY_THRESHOLD})"
                logger.debug(f"Not a meme: '{query}' (score: {similarity:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Meme detection failed: {e}")
            result.reasoning = f"Error: {e}"
            return result


meme_detector = MemeDetector()


def get_meme_response_prompt(meme_result: MemeDetectionResult) -> str:
    """
    Generate LLM prompt instructions for responding to a detected meme.
    
    Instructs the LLM to:
    1. Craft a witty response showing awareness of the cultural reference
    2. Pivot to Brandon's relevant policy position
    """
    if not meme_result.is_meme:
        return ""
    
    pivot_text = meme_result.suggested_pivot or "the cultural context of this phrase"
    
    return f"""
[MEME/CULTURAL REFERENCE DETECTED]
The user asked: "{meme_result.phrase}"

This appears to be a politically loaded phrase or cultural meme. Context from web search:
{meme_result.context[:500]}

INSTRUCTIONS:
1. Show that you understand the cultural/political subtext of this question
2. Craft a brief, witty acknowledgment that demonstrates awareness (not dry or robotic)
3. Naturally pivot to {pivot_text}
4. Then present Brandon's relevant policy position

Example tone: "Ah, I see what you're getting at! That phrase has certainly sparked quite the debate..." then pivot to policy.

Do NOT:
- Pretend the question is straightforward
- Give a dry policy response without acknowledging the subtext
- Lecture or be preachy about the controversy
"""
