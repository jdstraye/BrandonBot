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
import re
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

MEME_ANALYSIS_TEMPLATE = "This is about a political controversy, meme, or cultural debate"
MEME_SIMILARITY_THRESHOLD = 0.28
# A complementary template that represents neutral/factual content. We use
# the difference between meme and non-meme similarities to reduce false
# positives on ambiguous queries (e.g., dictionary definitions).
NON_MEME_ANALYSIS_TEMPLATE = "This is a neutral factual definition or general informational content"
MEME_SIMILARITY_MARGIN = 0.05  # require meme_similarity to exceed non-meme similarity by this margin

# Keywords that strongly indicate a meme/viral context when present in snippets
MEME_INDICATOR_KEYWORDS = [
    "meme", "viral", "chant", "slogan", "tweet", "twitter", "hashtag",
    "reddit", "parody", "viral tweet", "video", "tiktok", "instagram",
    "viral tweet", "viral video", "chant", "slogan", "opinion piece"
]

# Keywords that more specifically indicate a political context (used for
# political meme detection rather than generic/pop-culture memes)
POLITICAL_INDICATOR_KEYWORDS = [
    "politic", "political", "slogan", "chant", "policy", "protest",
    "election", "president", "trump", "biden", "conservative", "liberal",
    "senate", "congress", "immigration", "abortion", "border", "vaccine",
    "climate", "gender", "trans", "campaign", "rally"
]

GREETING_WORDS = {"hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", "howdy", "yo"}

MEME_TRIGGER_PHRASES = {
    "let's go", "lets go", "build the wall", "covfefe", "dark brandon",
    "okay groomer", "ok groomer", "this is fine", "what is a woman",
    "mostly peaceful", "fine people", "build back", "defund",
    "stolen election", "deep state", "fake news", "great replacement"
}

# Some phrases are known to be repurposed for political/cultural commentary
# even if traditional political keywords are not nearby in snippets.
POLITICAL_MEME_OVERRIDES = {
    "this is fine", "okay groomer", "ok groomer", "lets go brandon",
    "lets go", "build the wall", "what is a woman", "mostly peaceful",
    "fine people"
}

# Known pop-culture memes that are NOT political in our detection
NON_POLITICAL_MEMES = {
    "okay boomer", "this is sparta", "what is a man", "build the team"
}


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
                from slm_manager import SLMManager

                logger.info("Loading meme detector (all-MiniLM-L6-v2 + SearxNG + local SLM)...")
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                self._template_embedding = self._embedding_model.encode(MEME_ANALYSIS_TEMPLATE)
                self._multi_search = multi_search_service
                # SLMManager will lazy-load the cross-encoder on first use
                self._slm = SLMManager()

                self._initialized = True
                logger.info("Meme detector ready with multi-provider search and SLM")
                return True
                
            except Exception as e:
                logger.error(f"Failed to initialize meme detector: {e}")
                return False
    
    def _is_short_question(self, query: str) -> bool:
        """Check if query is short enough to potentially be a meme phrase."""
        words = query.split()
        return len(words) <= 10
    
    def _is_greeting(self, query: str) -> bool:
        """
        Check if query is a simple greeting that should skip meme detection.
        
        Simple greetings like "Hi Brandon" should not trigger meme detection
        even though web search might return "Let's Go Brandon" results.
        
        Uses heuristic: starts with greeting word + no meme trigger phrases
        """
        query_lower = query.lower().strip()
        
        if self._contains_meme_trigger(query_lower):
            return False
        
        for greeting in GREETING_WORDS:
            if query_lower.startswith(greeting):
                remaining = query_lower[len(greeting):].strip()
                remaining = remaining.lstrip(',').lstrip('-').lstrip('–').strip()
                
                if not remaining:
                    return True
                
                if remaining.startswith("brandon"):
                    after_brandon = remaining[7:].strip()
                    if not after_brandon:
                        return True
                    
                    greeting_continuations = [
                        "i'm", "im", "i am", "my name is", "this is",
                        "how are you", "how's it going", "what's up",
                        "nice to meet you", "pleased to meet you",
                        ".", "!", "?", ","
                    ]
                    
                    for cont in greeting_continuations:
                        if after_brandon.lstrip(',').lstrip('.').lstrip('!').strip().startswith(cont):
                            return True
                    
                    if after_brandon.startswith(",") or after_brandon.startswith("."):
                        return True
                    
                    words = after_brandon.split()
                    if len(words) <= 6 and not self._contains_meme_trigger(after_brandon):
                        is_intro = any(w in after_brandon for w in ["i'm", "im", "name", "call me", "this is"])
                        is_casual = any(w in after_brandon for w in ["how", "what", "nice", "great", "thanks"])
                        if is_intro or is_casual:
                            return True
        
        return False
    
    def _contains_meme_trigger(self, text: str) -> bool:
        """Check if text contains known meme trigger phrases."""
        text_lower = text.lower()
        for trigger in MEME_TRIGGER_PHRASES:
            if trigger in text_lower:
                return True
        return False
    
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

        meme_similarity = float(np.dot(avg_embedding, self._template_embedding) / (
            np.linalg.norm(avg_embedding) * np.linalg.norm(self._template_embedding)
        ))

        # Compute a non-meme similarity as a negative control to detect cases
        # where snippets are neutral/definitional despite matching the meme
        # template by accident.
        try:
            non_meme_embedding = self._embedding_model.encode([NON_MEME_ANALYSIS_TEMPLATE])[0]
            non_meme_similarity = float(np.dot(avg_embedding, non_meme_embedding) / (
                np.linalg.norm(avg_embedding) * np.linalg.norm(non_meme_embedding)
            ))
        except Exception:
            non_meme_similarity = 0.0

        similarity = meme_similarity

        context = " ".join(snippets[:3])
        if len(context) > 800:
            context = context[:800] + "..."

        return similarity, context, non_meme_similarity
    
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
    
    async def detect(self, query: str, session_id: Optional[str] = None, test_id: Optional[str] = None, request_id: Optional[str] = None) -> MemeDetectionResult:
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
        
        if self._is_greeting(query):
            result.reasoning = "Simple greeting - skipping meme detection"
            logger.debug(f"Skipping meme detection for greeting: '{query}'")
            return result
        
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

            # Precompute snippet-level indicators so we can make context-aware
            # decisions even when the SLM asserts 'not_meme' with high confidence.
            import re
            snippet_text = " ".join(snippets).lower()
            snippet_text_clean = re.sub(r"[^\w\s]", " ", snippet_text)
            # Normalize common abbreviated forms (e.g., 'ok' -> 'okay') to
            # help phrase matching for variants like 'OK Groomer' vs 'Okay Groomer'.
            snippet_text_clean = re.sub(r"\bok\b", "okay", snippet_text_clean)
            indicator_count = 0
            for kw in MEME_INDICATOR_KEYWORDS:
                indicator_count += snippet_text_clean.count(kw)

            political_count = 0
            for kw in POLITICAL_INDICATOR_KEYWORDS:
                political_count += snippet_text_clean.count(kw)

            phrase_norm = re.sub(r"[^\w\s]", "", query.lower()).strip()
            # If this phrase is a known non-political meme, skip detection
            if phrase_norm in NON_POLITICAL_MEMES:
                result.reasoning += " | Phrase is a known non-political meme - skipping political detection"
                logger.debug(f"Skipping political meme detection for known non-political meme: '{query}'")
                return result

            # Ask the local SLM (cross-encoder) to classify snippets for meme signal.
            try:
                # Pass the original query/phrase so SLM can check phrase presence
                slm_resp = await self._slm.classify_meme(snippets, phrase=query, test_id=test_id, session_id=session_id, request_id=request_id)
                result.reasoning += f" | SLM: {slm_resp.explanation}"

                # Strong SLM meme decisions take precedence, but require either
                # explicit political indicators or membership in overrides to
                # avoid SLM labeling pop-culture-only memes as political.
                if slm_resp.decision == "meme" and slm_resp.confidence >= 0.5 and (phrase_norm in POLITICAL_MEME_OVERRIDES or political_count >= 1):
                    result.is_meme = True
                    result.similarity_score = slm_resp.confidence
                    result.suggested_pivot = self._determine_pivot(query, result.reasoning)
                    result.reasoning += f" | Meme detected (SLM confidence: {slm_resp.confidence:.3f})"
                    logger.info(f"Meme detected by SLM: '{query}' (confidence: {slm_resp.confidence:.3f})")
                    return result

                # If SLM is strongly confident this is NOT a meme, we still allow
                # heuristic overrides (e.g., explicit political indicators or a
                # curated override phrase). This avoids SLM overconfidence blocking
                # obvious political memes when snippets are noisy.
                if slm_resp.decision == "not_meme" and slm_resp.confidence >= 0.6:
                    if not (phrase_norm in POLITICAL_MEME_OVERRIDES or (" " + phrase_norm + " " in " " + snippet_text_clean + " " and (political_count >= 1 or indicator_count >= 2))):
                        result.is_meme = False
                        result.similarity_score = 0.0
                        result.reasoning += f" | Classified non-meme by SLM (confidence: {slm_resp.confidence:.3f})"
                        logger.info(f"SLM classified non-meme: '{query}' (confidence: {slm_resp.confidence:.3f})")
                        return result
                    else:
                        # Ignore SLM's not_meme and fall through to heuristics
                        logger.info(f"Ignoring SLM non-meme for '{query}' due to contextual indicators (political={political_count}, indicators={indicator_count})")
            except Exception as e:
                logger.warning(f"SLM meme classification failed: {e}")

            similarity, context, non_meme_similarity = self._analyze_snippets(snippets)
            result.similarity_score = similarity
            result._non_meme_similarity = non_meme_similarity  # debug aid
            result.context = context
            result.cultural_context = context[:300] if context else ""
            result.confidence = min(similarity / MEME_SIMILARITY_THRESHOLD, 1.0) if similarity > 0 else 0.0
            # Check for explicit indicators in snippets (strong signal).
            # Use a cleaned snippet text (no punctuation) for reliable matching.
            import re
            snippet_text = " ".join(snippets).lower()
            snippet_text_clean = re.sub(r"[^\w\s]", " ", snippet_text)
            indicator_count = 0
            for kw in MEME_INDICATOR_KEYWORDS:
                indicator_count += snippet_text_clean.count(kw)

            # Final decision uses a combination of signals:
            # 1) similarity threshold
            # 2) meme vs non-meme similarity margin
            # 3) presence of explicit indicator keywords
            is_meme_by_similarity = similarity >= MEME_SIMILARITY_THRESHOLD
            is_meme_by_margin = (similarity - non_meme_similarity) >= MEME_SIMILARITY_MARGIN
            # Phrase presence check: indicator keywords should relate to the phrase
            import re
            phrase_clean = re.sub(r"[^\w\s]", "", query.lower()).strip()
            phrase_present = phrase_clean in snippet_text_clean

            # Special-case guard: for very short generic phrases (<=2 words) where
            # the snippets predominantly reference a longer political phrase
            # (e.g., 'lets go brandon'), do not label the short phrase as a meme.
            phrase_word_count = len(phrase_clean.split()) if phrase_clean else 0
            if phrase_word_count <= 2 and 'brandon' in snippet_text_clean and 'brandon' not in phrase_clean:
                result.reasoning += " | Short generic phrase appears only as part of a longer meme (e.g., 'lets go brandon') - not labeling as meme"
                logger.info(f"Short generic phrase '{query}' appears in political snippets but without explicit phrase match - skipping as meme")
                return result

            # Count political-specific indicators separately; we treat a phrase as a political
            # meme even if there is only a single political indicator present in snippets
            political_count = 0
            for kw in POLITICAL_INDICATOR_KEYWORDS:
                political_count += snippet_text_clean.count(kw)

            # Short-circuit for pop-culture/political memes: if the phrase is present
            # and there are explicit meme/viral indicators, treat as a meme. In addition,
            # treat curated override phrases (e.g., 'okay groomer') as political memes
            # if the snippet indicators show meme/viral context, even when the phrase
            # itself isn't present exactly in the snippets (helps with 'Okay, Groomer').
            phrase_norm = re.sub(r"[^\w\s]", "", query.lower()).strip()
            if (phrase_present and indicator_count >= 2 and (political_count >= 1 or phrase_norm in POLITICAL_MEME_OVERRIDES) and (phrase_norm in POLITICAL_MEME_OVERRIDES or (not phrase_norm.startswith('what is') and not phrase_norm.startswith('who is')))) or (phrase_norm in POLITICAL_MEME_OVERRIDES and indicator_count >= 2):
                result.is_meme = True
                result.suggested_pivot = self._determine_pivot(query, context)
                result.reasoning += f" | Meme detected (indicators: {indicator_count})"
                logger.info(f"Meme detected by indicators: '{query}' (indicators: {indicator_count}, pivot: {result.suggested_pivot})")

            # Decision rules:
            # - If political indicators are present and the phrase occurs in snippets, treat as political meme
            # - Otherwise require similarity >= threshold AND a positive margin vs non-meme
            elif phrase_present:
                # For phrases in our overrides, a single political indicator is enough
                if phrase_norm in POLITICAL_MEME_OVERRIDES and political_count >= 1:
                    result.is_meme = True
                    result.suggested_pivot = self._determine_pivot(query, context)
                    result.reasoning += f" | Meme detected (override + political_indicators: {political_count})"
                    logger.info(f"Meme detected by override+political indicators: '{query}' (political_indicators: {political_count}, pivot: {result.suggested_pivot})")
                # Otherwise require a stronger political signal to avoid false positives
                elif political_count >= 3 and (indicator_count >= 2 or (is_meme_by_similarity and is_meme_by_margin)):
                    result.is_meme = True
                    result.suggested_pivot = self._determine_pivot(query, context)
                    result.reasoning += f" | Meme detected (political_indicators: {political_count})"
                    logger.info(f"Meme detected by political indicators: '{query}' (political_indicators: {political_count}, pivot: {result.suggested_pivot})")
            # Final similarity-based rule: require stronger political signal for
            # question-like phrases (e.g., "What is a tree?") to avoid false positives.
            if is_meme_by_similarity and is_meme_by_margin:
                if phrase_norm.startswith('what is') or phrase_norm.startswith('who is'):
                    # require >=3 political indicators for question-like phrases
                    if political_count >= 3:
                        result.is_meme = True
                        result.suggested_pivot = self._determine_pivot(query, context)
                        result.reasoning += f" | Meme detected (score: {similarity:.3f}, indicators: {indicator_count})"
                        logger.info(f"Meme detected by similarity (question case): '{query}' (score: {similarity:.3f}, pivot: {result.suggested_pivot}, indicators: {indicator_count})")
                else:
                    # non-question phrases need only a single political indicator (or overrides handled earlier)
                    if political_count >= 1:
                        result.is_meme = True
                        result.suggested_pivot = self._determine_pivot(query, context)
                        result.reasoning += f" | Meme detected (score: {similarity:.3f}, indicators: {indicator_count})"
                        logger.info(f"Meme detected by similarity: '{query}' (score: {similarity:.3f}, pivot: {result.suggested_pivot}, indicators: {indicator_count})")
            else:
                result.reasoning += f" | Not a meme (score: {similarity:.3f} < threshold {MEME_SIMILARITY_THRESHOLD} or margin fail (non_meme={non_meme_similarity:.3f}, indicators={indicator_count}))"
                logger.debug(f"Not a meme: '{query}' (score: {similarity:.3f}, non_meme={non_meme_similarity:.3f}, indicators={indicator_count})")
            
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
[MEME/CULTURAL REFERENCE DETECTED - RESPOND DIRECTLY, NO TOOLS NEEDED]
The user asked: "{meme_result.phrase}"

This is a politically loaded meme/cultural reference. Context from web search:
{meme_result.context[:500]}

CRITICAL: RESPOND DIRECTLY - DO NOT call search_brandon_positions or any other tools.
The context above is sufficient for a witty pivot response.

INSTRUCTIONS:
1. Craft a brief, witty acknowledgment that shows you get the cultural/political subtext
2. Naturally pivot to {pivot_text}
3. Share Brandon's relevant perspective on the underlying issue

Example opening: "Ha! I know exactly what you're getting at - that phrase has become quite the cultural flashpoint..." then pivot.

AVOID:
- Using any tools or searches (you have the context already)
- Pretending the question is straightforward
- Giving a dry policy response without acknowledging the subtext
- Being preachy or lecturing about the controversy
"""
