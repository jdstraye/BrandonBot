"""
Web Search Service with Citation Management
Handles external searches and formats citations with full URLs
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import re
import asyncio
from datetime import datetime


@dataclass
class SearchResult:
    """A single search result with citation"""
    content: str
    source_name: str
    url: str
    snippet: str
    date: Optional[str] = None


@dataclass
class SearchResponse:
    """Complete search response with formatted citations"""
    summary: str
    results: List[SearchResult]
    citations: List[str]  # Formatted footnote-style citations


class WebSearchService:
    """
    Performs web searches and manages citations
    NOTE: In this implementation, we're creating a stub that will be enhanced
    when we integrate with actual web search APIs
    """
    
    def __init__(self):
        """Initialize web search service"""
        self.search_available = False  # Will be True when web search is integrated
    
    async def search(self, query: str, max_results: int = 3) -> SearchResponse:
        """
        Perform web search and return formatted results with citations
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return (default 3)
            
        Returns:
            SearchResponse with summary, results, and formatted citations
        """
        # Placeholder implementation - will be replaced with actual web search
        # For now, return empty results to indicate search is needed
        return SearchResponse(
            summary="External search required but not yet available in this version.",
            results=[],
            citations=[]
        )
    
    def format_citations(self, results: List[SearchResult]) -> List[str]:
        """
        Format search results as footnote-style citations
        
        Args:
            results: List of SearchResult objects
            
        Returns:
            List of formatted citation strings
        """
        citations = []
        for i, result in enumerate(results, 1):
            # Format: [1] Source Name (Date if available). URL
            date_str = f" ({result.date})" if result.date else ""
            citation = f"[{i}] {result.source_name}{date_str}. {result.url}"
            citations.append(citation)
        return citations
    
    def extract_url_with_anchor(self, url: str, heading_text: Optional[str] = None) -> str:
        """
        Try to create URL with anchor link to specific section
        
        Args:
            url: Base URL
            heading_text: Text of heading/section (if available)
            
        Returns:
            URL with anchor if possible, otherwise base URL
        """
        if not heading_text:
            return url
        
        # Create anchor from heading text
        # Remove special characters, replace spaces with hyphens, lowercase
        anchor = re.sub(r'[^\w\s-]', '', heading_text)
        anchor = re.sub(r'\s+', '-', anchor).lower()
        
        return f"{url}#{anchor}"


class ExternalSearchIntegrator:
    """
    Integrates external search results into BrandonBot responses
    Handles different framing based on question type
    """
    
    @staticmethod
    def frame_statistics_response(search_summary: str, citations: List[str]) -> str:
        """
        Frame external search results for statistics questions
        User stays in character as Brandon
        
        Args:
            search_summary: Summary of search findings
            citations: List of formatted citations
            
        Returns:
            Framed response text
        """
        response = f"I had to use the internet to get the latest data and statistics, but it seems that {search_summary}\n\n"
        
        if citations:
            response += "Sources:\n"
            response += "\n".join(citations)
        
        return response
    
    @staticmethod
    def frame_comparison_response(
        search_summary: str,
        citations: List[str],
        brandon_position: str
    ) -> str:
        """
        Frame external search results for comparison questions
        Breaks character to be honest about limitations
        
        Args:
            search_summary: Summary of opponent's position from search
            citations: List of formatted citations
            brandon_position: Brandon's position from RAG
            
        Returns:
            Framed response text
        """
        response = "[Breaking character for a moment] This is a challenging topic for which I don't have enough training data to respond with complete confidence. "
        response += "Let me get your contact information so a staffer can follow up with a thorough, accurate comparison.\n\n"
        response += f"Based on what I know:\n\n"
        response += f"My position: {brandon_position}\n\n"
        response += f"Based on internet search: {search_summary}\n\n"
        
        if citations:
            response += "Sources:\n"
            response += "\n".join(citations)
            response += "\n\n"
        
        response += "What are your thoughts on this? I'd love to discuss it further with you directly."
        
        return response
    
    @staticmethod
    def frame_low_confidence_response(search_summary: str, citations: List[str]) -> str:
        """
        Frame external search results for low-confidence scenarios
        Breaks character to be honest
        
        Args:
            search_summary: Summary of search findings
            citations: List of formatted citations
            
        Returns:
            Framed response text
        """
        response = "[Breaking character] I don't have specific information on that topic in my platform materials. "
        response += "Let me get your contact information so a staffer can provide you with a thorough answer.\n\n"
        response += f"Based on an internet search: {search_summary}\n\n"
        
        if citations:
            response += "Sources:\n"
            response += "\n".join(citations)
            response += "\n\n"
        
        response += "This is clearly an important question to you. I'd love to give you the best possible answer - "
        response += "can I have my team reach out to discuss this further?"
        
        return response
    
    @staticmethod
    def frame_recent_event_response(search_summary: str, citations: List[str]) -> str:
        """
        Frame external search results for recent events
        Breaks character for complex/developing situations
        
        Args:
            search_summary: Summary of recent event from search
            citations: List of formatted citations
            
        Returns:
            Framed response text
        """
        response = "[Breaking character] This is a developing situation and I want to make sure I give you accurate information. "
        response += "Let me get your contact information so a staffer can follow up with a detailed response.\n\n"
        response += f"Based on recent news: {search_summary}\n\n"
        
        if citations:
            response += "Sources:\n"
            response += "\n".join(citations)
            response += "\n\n"
        
        response += "What's your take on this? I'd be happy to discuss it with you personally."
        
        return response


# Mock implementation for testing - will be replaced with actual search integration
class MockWebSearchService(WebSearchService):
    """Mock search service for testing without actual web API"""
    
    def __init__(self):
        """Initialize mock search service"""
        super().__init__()
        self.search_available = True  # Mock is always available for testing
    
    async def search(self, query: str, max_results: int = 3) -> SearchResponse:
        """Return mock search results for testing"""
        query_lower = query.lower()
        
        # Provide contextual mock results based on query type
        if any(word in query_lower for word in ["compare", "comparison", "differ", "vs", "versus"]):
            # Comparison query
            mock_results = [
                SearchResult(
                    content="Policy comparison showing differences in approach to the issue.",
                    source_name="Political Policy Database",
                    url="https://example.com/policy-comparison#key-differences",
                    snippet="Detailed comparison of approaches to this policy area",
                    date="2024-11-15"
                ),
                SearchResult(
                    content="Analysis of different positions on this topic.",
                    source_name="Policy Analysis Institute",
                    url="https://example.com/analysis#positions",
                    snippet="Comprehensive breakdown of political positions",
                    date="2024-11-10"
                )
            ]
            summary = "Based on available policy documents, the main differences center on approach and priorities in this area."
        
        elif any(word in query_lower for word in ["statistics", "data", "how many", "percentage", "rate"]):
            # Statistics query
            mock_results = [
                SearchResult(
                    content="Latest statistical data on this topic.",
                    source_name="National Statistics Bureau",
                    url="https://example.com/statistics#latest-data",
                    snippet="Current numbers show 42% increase over last year",
                    date="2024-11-18"
                )
            ]
            summary = "the latest data shows significant trends, with a 42% increase compared to last year according to recent reports."
        
        elif any(word in query_lower for word in ["recent", "latest", "new", "current"]):
            # Recent event query
            mock_results = [
                SearchResult(
                    content="Latest news on this developing topic.",
                    source_name="News Network",
                    url="https://example.com/news#breaking",
                    snippet="Breaking developments from this week",
                    date="2024-11-19"
                )
            ]
            summary = "Recent developments show ongoing changes in this area. The situation is still evolving."
        
        else:
            # Generic query
            mock_results = [
                SearchResult(
                    content="General information about this topic.",
                    source_name="Reference Database",
                    url="https://example.com/reference#topic",
                    snippet="Comprehensive overview of the subject",
                    date="2024-11-01"
                )
            ]
            summary = "Available information suggests multiple perspectives on this topic. More research may be needed for a complete answer."
        
        citations = self.format_citations(mock_results[:max_results])
        
        return SearchResponse(
            summary=summary,
            results=mock_results[:max_results],
            citations=citations
        )


# Real DuckDuckGo implementation
class DuckDuckGoSearchService(WebSearchService):
    """Real web search using DuckDuckGo (free, no API key required)"""
    
    def __init__(self):
        """Initialize DuckDuckGo search service"""
        super().__init__()
        self.search_available = True
        # Domain trust scoring - boost results from official website
        self.trusted_domains = {
            "brandonsowers.com": 2.0,  # 2x boost for official website
            "brandonforarizona.com": 2.0,  # Alternative domain
        }
        try:
            from duckduckgo_search import DDGS
            self.ddgs = DDGS()
            self.search_available = True
        except ImportError:
            import logging
            logging.warning("duckduckgo-search not installed. Web search disabled.")
            self.search_available = False
            self.ddgs = None
    
    async def search(self, query: str, max_results: int = 3) -> SearchResponse:
        """
        Perform real web search using DuckDuckGo
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return (default 3)
            
        Returns:
            SearchResponse with summary, results, and formatted citations
        """
        if not self.search_available or not self.ddgs:
            return SearchResponse(
                summary="External search not available",
                results=[],
                citations=[]
            )
        
        try:
            # Run DDG search in executor to avoid blocking
            loop = asyncio.get_event_loop()
            search_results = await loop.run_in_executor(
                None,
                lambda: list(self.ddgs.text(query, max_results=max_results))
            )
            
            # Convert DDG results to SearchResult objects with trust scoring
            results = []
            for idx, result in enumerate(search_results[:max_results * 2]):  # Get more results for reranking
                url = result.get('href', '')
                # Calculate trust score based on domain
                trust_score = self._get_domain_trust(url)
                
                results.append({
                    'result': SearchResult(
                        content=result.get('body', ''),
                        source_name=result.get('title', 'Unknown Source'),
                        url=url,
                        snippet=result.get('body', ''),
                        date=None  # DDG doesn't always provide dates
                    ),
                    'trust_score': trust_score,
                    'original_rank': idx
                })
            
            # Rerank results based on trust score (prioritize trusted domains)
            results.sort(key=lambda x: (-x['trust_score'], x['original_rank']))
            
            # Extract top results after reranking
            final_results = [r['result'] for r in results[:max_results]]
            
            # Generate summary from top results
            if final_results:
                summary = self._generate_summary(query, final_results)
            else:
                summary = "No search results found for this query."
            
            citations = self.format_citations(final_results)
            
            return SearchResponse(
                summary=summary,
                results=final_results,
                citations=citations
            )
            
        except Exception as e:
            import logging
            logging.error(f"DuckDuckGo search failed: {str(e)}")
            return SearchResponse(
                summary=f"Search error: {str(e)}",
                results=[],
                citations=[]
            )
    
    def _generate_summary(self, query: str, results: List[SearchResult]) -> str:
        """
        Generate a summary from search results
        
        Args:
            query: Original search query
            results: List of SearchResult objects
            
        Returns:
            Summary string
        """
        if not results:
            return "No relevant information found."
        
        # Take snippets from top 2-3 results
        snippets = [r.snippet[:200] for r in results[:min(3, len(results))]]
        summary = " ".join(snippets)
        
        # Clean up summary
        summary = summary.replace('\n', ' ').strip()
        if len(summary) > 500:
            summary = summary[:497] + "..."
        
        return summary
    
    def _get_domain_trust(self, url: str) -> float:
        """
        Calculate trust score for a URL based on domain
        
        Args:
            url: URL to check
            
        Returns:
            Trust score (higher = more trustworthy, 1.0 = neutral)
        """
        if not url:
            return 1.0
        
        # Extract domain from URL
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.lower()
            # Remove www. prefix
            domain = domain.replace('www.', '')
            
            # Check if domain is in trusted list
            for trusted_domain, trust_score in self.trusted_domains.items():
                if trusted_domain in domain:
                    return trust_score
            
            return 1.0  # Neutral trust for unknown domains
        except:
            return 1.0
