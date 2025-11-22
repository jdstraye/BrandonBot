"""
Retrieval Orchestrator - Coordinates multi-tier RAG retrieval
Handles tiered queries, dual-source enforcement, and confidence calculation
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
import asyncio

from analysis_pipeline import QuestionAnalysis


logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Single retrieval result with confidence scoring"""
    text: str
    collection: str
    similarity: float
    trust_multiplier: float
    confidence: float  # similarity × trust_multiplier
    metadata: Dict
    
    def __post_init__(self):
        """Calculate confidence on initialization"""
        self.confidence = self.similarity * self.trust_multiplier


@dataclass
class RetrievalContext:
    """Complete retrieval context for prompt generation"""
    content_results: List[RetrievalResult]  # Factual evidence
    marketguru_results: List[RetrievalResult]  # Style guidance
    bible_results: List[RetrievalResult]  # Biblical references
    web_results: List[RetrievalResult]  # External search results
    best_confidence: float
    has_dual_sources: bool  # For comparisons: do we have Brandon AND opponent?


class RetrievalOrchestrator:
    """
    Orchestrates multi-tier retrieval with proper confidence scoring
    and dual-source enforcement for comparison questions
    """
    
    # Trust multipliers for each source type
    TRUST_MULTIPLIERS = {
        "BrandonPlatform": 1.0,      # Most trustworthy
        "PreviousQA": 1.0,           # Very trustworthy
        "brandonsowers_web": 0.8,    # Official website (web search)
        "PartyPlatform": 0.6,        # Moderately trustworthy
        "external_web": 0.3,         # General internet (low trust)
        "MarketGurus": 1.0,          # Full trust for style (not content)
        "Bible": 1.0                 # Full trust for spiritual guidance
    }
    
    # Confidence thresholds for each tier (applied AFTER trust multiplier)
    CONFIDENCE_THRESHOLDS = {
        "BrandonPlatform": 0.45,
        "PreviousQA": 0.45,
        "brandonsowers_web": 0.40,
        "PartyPlatform": 0.35,
        "external_web": 0.25
    }
    
    def __init__(self, weaviate_manager, web_search_service=None):
        """
        Initialize orchestrator
        
        Args:
            weaviate_manager: WeaviateManager instance
            web_search_service: Optional web search service
        """
        self.weaviate = weaviate_manager
        self.web_search = web_search_service
    
    async def retrieve(
        self,
        question: str,
        analysis: QuestionAnalysis,
        limit_per_collection: int = 5
    ) -> RetrievalContext:
        """
        Orchestrate retrieval across all tiers
        
        Args:
            question: User's question
            analysis: QuestionAnalysis from QuestionAnalyzer
            limit_per_collection: Max results per collection
            
        Returns:
            RetrievalContext with all retrieved content
        """
        # Tier 1: Content retrieval (sequential search with confidence thresholds)
        content_results = await self._retrieve_content(
            question,
            analysis,
            limit_per_collection
        )
        
        # Calculate best confidence from content
        best_confidence = max([r.confidence for r in content_results], default=0.0)
        
        # Tier 2: MarketGuru retrieval (INVERSE confidence: more when uncertain)
        marketguru_results = await self._retrieve_marketgurus(
            question,
            analysis,
            best_confidence,
            limit_per_collection
        )
        
        # Tier 3: Bible verses (if needed)
        bible_results = await self._retrieve_bible_verses(
            analysis,
            limit_per_collection
        ) if analysis.needs_bible_verse else []
        
        # Tier 4: Web search (if needed)
        web_results = await self._retrieve_web_search(
            analysis,
            limit_per_collection
        ) if analysis.needs_external_search else []
        
        # Verify dual-source requirement for comparisons
        has_dual_sources = self._verify_dual_sources(content_results, analysis)
        
        return RetrievalContext(
            content_results=content_results,
            marketguru_results=marketguru_results,
            bible_results=bible_results,
            web_results=web_results,
            best_confidence=best_confidence,
            has_dual_sources=has_dual_sources
        )
    
    async def _retrieve_content(
        self,
        question: str,
        analysis: QuestionAnalysis,
        limit: int
    ) -> List[RetrievalResult]:
        """
        Parallel tiered content retrieval
        
        Order: BrandonPlatform → PreviousQA → brandonsowers.com → PartyPlatform
        Each tier filtered by confidence threshold
        """
        # Query all collections in parallel for efficiency
        brandon_task = self._query_collection(
            "BrandonPlatform",
            question,
            limit,
            self.CONFIDENCE_THRESHOLDS["BrandonPlatform"]
        )
        qa_task = self._query_collection(
            "PreviousQA",
            question,
            limit,
            self.CONFIDENCE_THRESHOLDS["PreviousQA"]
        )
        party_task = self._query_collection(
            "PartyPlatform",
            question,
            limit,
            self.CONFIDENCE_THRESHOLDS["PartyPlatform"]
        )
        
        # Execute queries in parallel
        brandon_results, qa_results, party_results = await asyncio.gather(
            brandon_task, qa_task, party_task
        )
        
        all_results = []
        all_results.extend(brandon_results)
        all_results.extend(qa_results)
        
        # Tier 3: brandonsowers.com web search (before PartyPlatform)
        if self.web_search and analysis.search_queries:
            web_results = await self._retrieve_web_search_for_content(
                question,
                analysis,
                limit
            )
            # Filter to only brandonsowers.com results for content tier
            brandonsowers_results = [r for r in web_results if r.collection == "brandonsowers_web"]
            all_results.extend(brandonsowers_results)
            logger.info(f"brandonsowers.com web search: Added {len(brandonsowers_results)} results to content tier")
        
        all_results.extend(party_results)
        
        # Sort by confidence (highest first)
        all_results.sort(key=lambda r: r.confidence, reverse=True)
        
        # For comparison questions, enforce dual sources
        if analysis.needs_dual_sources:
            all_results = await self._enforce_dual_sources(all_results, analysis)
        
        return all_results
    
    async def _query_collection(
        self,
        collection_name: str,
        query: str,
        limit: int,
        confidence_threshold: float
    ) -> List[RetrievalResult]:
        """
        Query a single Weaviate collection and convert to RetrievalResults
        
        Args:
            collection_name: Name of Weaviate collection
            query: Search query
            limit: Max results
            confidence_threshold: Minimum confidence to include
            
        Returns:
            List of RetrievalResults that pass threshold
        """
        try:
            # Query Weaviate (FIX: parameter is 'query' not 'query_text', and it's async)
            raw_results = await self.weaviate.search(
                collection_name=collection_name,
                query=query,
                limit=limit
            )
            
            # Convert to RetrievalResults with confidence calculation
            results = []
            trust_multiplier = self.TRUST_MULTIPLIERS.get(collection_name, 0.5)
            
            for item in raw_results:
                # The WeaviateManager returns 'content' not 'text', and 'confidence' directly
                content = item.get("content", "")
                # WeaviateManager already calculates confidence as 1 - distance
                confidence_from_search = item.get("confidence", 0.5)
                
                # Apply trust multiplier to the similarity score
                final_confidence = confidence_from_search * trust_multiplier
                
                # Filter by threshold
                if final_confidence >= confidence_threshold:
                    result = RetrievalResult(
                        text=content,
                        collection=collection_name,
                        similarity=confidence_from_search,
                        trust_multiplier=trust_multiplier,
                        confidence=final_confidence,
                        metadata=item
                    )
                    results.append(result)
            
            logger.info(f"{collection_name}: {len(results)}/{len(raw_results)} passed threshold {confidence_threshold}")
            return results
            
        except Exception as e:
            logger.error(f"Error querying {collection_name}: {e}")
            return []
    
    async def _retrieve_marketgurus(
        self,
        question: str,
        analysis: QuestionAnalysis,
        best_content_confidence: float,
        limit: int
    ) -> List[RetrievalResult]:
        """
        Retrieve MarketGuru guidance (INVERSE confidence: more when uncertain)
        
        - confidence < 0.5: Use full semantic search (need guidance)
        - 0.5 ≤ confidence < 0.8: Moderate guidance
        - confidence ≥ 0.8: Minimal/optional guidance
        """
        # Determine how much guidance we need
        if best_content_confidence >= 0.8:
            # High confidence: minimal guidance
            limit = min(limit, 2)
        elif best_content_confidence >= 0.5:
            # Moderate confidence: normal guidance
            pass  # use default limit
        else:
            # Low confidence: maximum guidance
            limit = limit * 2
        
        # Build search query from analysis keywords
        search_query = " ".join(analysis.marketgurus_keywords[:5])  # Top 5 keywords
        
        try:
            # Full semantic search (not templates!)
            raw_results = await self.weaviate.search(
                collection_name="MarketGurus",
                query=search_query,
                limit=limit
            )
            
            # Convert to RetrievalResults
            results = []
            for item in raw_results:
                content = item.get("content", "")
                confidence_from_search = item.get("confidence", 0.5)
                
                result = RetrievalResult(
                    text=content,
                    collection="MarketGurus",
                    similarity=confidence_from_search,
                    trust_multiplier=self.TRUST_MULTIPLIERS["MarketGurus"],
                    confidence=confidence_from_search * self.TRUST_MULTIPLIERS["MarketGurus"],
                    metadata=item
                )
                results.append(result)
            
            logger.info(f"MarketGurus: Retrieved {len(results)} guidance snippets (confidence={best_content_confidence:.2f})")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving MarketGurus: {e}")
            return []
    
    async def _retrieve_bible_verses(
        self,
        analysis: QuestionAnalysis,
        limit: int
    ) -> List[RetrievalResult]:
        """Retrieve relevant Bible verses based on topics"""
        if not analysis.bible_topics:
            return []
        
        all_results = []
        
        for topic in analysis.bible_topics[:3]:  # Top 3 topics
            try:
                raw_results = await self.weaviate.search(
                    collection_name=f"Bible_{topic.capitalize()}",
                    query=f"{topic} scripture biblical",
                    limit=2  # 2 verses per topic
                )
                
                for item in raw_results:
                    content = item.get("content", "")
                    result = RetrievalResult(
                        text=content,
                        collection=f"Bible_{topic}",
                        similarity=1.0,  # Topical match
                        trust_multiplier=self.TRUST_MULTIPLIERS["Bible"],
                        confidence=1.0,
                        metadata=item
                    )
                    all_results.append(result)
                    
            except Exception as e:
                logger.warning(f"Bible collection Bible_{topic} not found: {e}")
        
        return all_results
    
    async def _retrieve_web_search(
        self,
        analysis: QuestionAnalysis,
        limit: int
    ) -> List[RetrievalResult]:
        """Retrieve web search results"""
        if not self.web_search or not analysis.search_queries:
            return []
        
        all_results = []
        
        for query in analysis.search_queries[:2]:  # Top 2 queries
            try:
                search_response = await self.web_search.search(query, max_results=limit)
                search_results = search_response.results if search_response else []
                
                for sr in search_results:
                    # Determine trust multiplier based on domain
                    if "brandonsowers.com" in sr.url.lower():
                        trust = self.TRUST_MULTIPLIERS["brandonsowers_web"]
                        collection = "brandonsowers_web"
                    else:
                        trust = self.TRUST_MULTIPLIERS["external_web"]
                        collection = "external_web"
                    
                    result = RetrievalResult(
                        text=sr.snippet,
                        collection=collection,
                        similarity=0.6,  # Assumed relevance (search results don't have scores in stub)
                        trust_multiplier=trust,
                        confidence=0.6 * trust,
                        metadata={
                            "url": sr.url,
                            "title": sr.source_name,
                            "source": sr.source_name
                        }
                    )
                    all_results.append(result)
                    
            except Exception as e:
                logger.error(f"Web search error for '{query}': {e}")
        
        return all_results
    
    async def _retrieve_web_search_for_content(
        self,
        question: str,
        analysis: QuestionAnalysis,
        limit: int
    ) -> List[RetrievalResult]:
        """
        Retrieve web search results specifically for content tier integration
        This is a wrapper around _retrieve_web_search for use in _retrieve_content
        
        Args:
            question: User's question
            analysis: Question analysis
            limit: Max results per query
            
        Returns:
            List of web search results (both brandonsowers.com and external)
        """
        return await self._retrieve_web_search(analysis, limit)
    
    async def _enforce_dual_sources(
        self,
        results: List[RetrievalResult],
        analysis: QuestionAnalysis
    ) -> List[RetrievalResult]:
        """
        For comparison questions: ensure at least one Brandon and one opponent source
        Uses web search for generic comparisons when opponent sources unavailable
        
        Args:
            results: Current results
            analysis: Question analysis
            
        Returns:
            Results with dual-source enforcement
        """
        if not analysis.needs_dual_sources:
            return results
        
        # Separate Brandon sources vs opponent sources (including web search external sources)
        brandon_sources = [r for r in results if r.collection in ["BrandonPlatform", "PreviousQA", "brandonsowers_web"]]
        opponent_sources = [r for r in results if r.collection in ["PartyPlatform", "external_web"]]
        
        # Check if we have at least one from each side
        has_brandon = len(brandon_sources) > 0
        has_opponent = len(opponent_sources) > 0
        
        if has_brandon and has_opponent:
            # Perfect: we have both sides
            logger.info(f"Dual sources satisfied: {len(brandon_sources)} Brandon + {len(opponent_sources)} opponent")
            return results
        
        # Missing Brandon: try BrandonPlatform + brandonsowers.com web search
        if not has_brandon:
            logger.warning("Missing Brandon sources for comparison - fetching BrandonPlatform and brandonsowers.com")
            # Try BrandonPlatform with lower threshold
            additional_brandon = await self._query_collection(
                "BrandonPlatform",
                " ".join(analysis.comparison_targets) if analysis.comparison_targets else analysis.question,
                limit=3,
                confidence_threshold=0.25
            )
            results.extend(additional_brandon)
            
            # Also try brandonsowers.com web search
            if not additional_brandon:
                logger.info("No BrandonPlatform results - trying brandonsowers.com web search")
                try:
                    brandonsowers_query = f"Brandon Sowers position on {' '.join(analysis.comparison_targets)}" if analysis.comparison_targets else analysis.question
                    web_results = await self.web_search.search(brandonsowers_query, max_results=3)
                    for web_result in web_results:
                        if "brandonsowers.com" in web_result.url.lower():
                            results.append(RetrievalResult(
                                text=f"{web_result.title}\n{web_result.snippet}",
                                collection="brandonsowers_web",
                                similarity=0.7,  # Assumed relevance
                                trust_multiplier=self.TRUST_MULTIPLIERS["brandonsowers_web"],
                                confidence=0.7 * self.TRUST_MULTIPLIERS["brandonsowers_web"],
                                metadata={"url": web_result.url, "title": web_result.title}
                            ))
                except Exception as e:
                    logger.error(f"Web search for Brandon failed: {e}")
        
        # Missing opponent: try PartyPlatform first, then web search for generic comparisons
        if not has_opponent:
            logger.warning("Missing opponent sources for comparison - trying PartyPlatform and web search")
            # Try PartyPlatform with lower threshold
            additional_opponent = await self._query_collection(
                "PartyPlatform",
                " ".join(analysis.comparison_targets) if analysis.comparison_targets else analysis.question,
                limit=3,
                confidence_threshold=0.20
            )
            results.extend(additional_opponent)
            
            # If still no opponent sources and we have comparison targets, use web search
            if not additional_opponent and analysis.comparison_targets:
                logger.info(f"No PartyPlatform results - using web search for opponent positions: {analysis.comparison_targets}")
                try:
                    # Build queries for each comparison target (e.g., "Democrat position on immigration")
                    for target in analysis.comparison_targets[:2]:  # Limit to 2 targets to avoid excessive searches
                        opponent_query = f"{target} position on {' '.join([t for t in analysis.comparison_targets if t != target])}"
                        web_results = await self.web_search.search(opponent_query, max_results=2)
                        for web_result in web_results:
                            # External web sources get lower trust (0.3)
                            results.append(RetrievalResult(
                                text=f"{web_result.title}\n{web_result.snippet}",
                                collection="external_web",
                                similarity=0.6,  # Assumed relevance
                                trust_multiplier=self.TRUST_MULTIPLIERS["external_web"],
                                confidence=0.6 * self.TRUST_MULTIPLIERS["external_web"],
                                metadata={"url": web_result.url, "title": web_result.title, "comparison_target": target}
                            ))
                        
                        # Break after first successful search if we got results
                        if web_results:
                            logger.info(f"Found {len(web_results)} web results for {target}")
                            break
                except Exception as e:
                    logger.error(f"Web search for opponent failed: {e}")
        
        # Re-sort by confidence
        results.sort(key=lambda r: r.confidence, reverse=True)
        
        final_brandon = sum(1 for r in results if r.collection in ["BrandonPlatform", "PreviousQA", "brandonsowers_web"])
        final_opponent = sum(1 for r in results if r.collection in ["PartyPlatform", "external_web"])
        logger.info(f"After enforcement: {final_brandon} Brandon sources, {final_opponent} opponent sources")
        
        return results
    
    def _verify_dual_sources(
        self,
        content_results: List[RetrievalResult],
        analysis: QuestionAnalysis
    ) -> bool:
        """Verify if comparison has both Brandon and opponent sources"""
        if not analysis.needs_dual_sources:
            return True  # Not a comparison, no requirement
        
        has_brandon = any(r.collection in ["BrandonPlatform", "PreviousQA", "brandonsowers_web"] for r in content_results)
        has_opponent = any(r.collection in ["PartyPlatform", "external_web"] for r in content_results)
        
        return has_brandon and has_opponent
