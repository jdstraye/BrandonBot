"""
Multi-Provider Search Service for BrandonBot

Three-tier failover for reliable web search:
1. SearxNG public instances (primary) - unlimited, free
2. DuckDuckGo (first fallback) - free, no API key
3. SerpAPI (final fallback) - uses API key when others fail

Flow:
1. Try SearxNG public instances in round-robin order
2. On failure/timeout, try next SearxNG instance  
3. If all SearxNG fail, try DuckDuckGo
4. If DDG fails, fall back to SerpAPI (if configured)
5. If all fail, return empty results
"""

import asyncio
import logging
import os
import httpx
import concurrent.futures
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

SEARXNG_PUBLIC_INSTANCES = [
    "https://search.bus-hit.me",
    "https://searx.tiekoetter.com", 
    "https://search.ononoki.org",
    "https://searx.be",
    "https://paulgo.io",
    "https://search.sapti.me",
    "https://searx.fmac.xyz",
    "https://searx.prvcy.eu",
]

SEARXNG_TIMEOUT = 8.0
DDG_TIMEOUT = 10.0
SERPAPI_TIMEOUT = 10.0
INSTANCE_COOLDOWN = 60


@dataclass
class SearchResult:
    """A single search result"""
    title: str
    url: str
    snippet: str
    source: str = ""
    date: Optional[str] = None


@dataclass
class SearchResponse:
    """Complete search response"""
    results: List[SearchResult]
    provider: str = ""
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        return len(self.results) > 0 and self.error is None


@dataclass
class InstanceHealth:
    """Tracks health of a SearxNG instance"""
    url: str
    failures: int = 0
    last_failure: Optional[datetime] = None
    last_success: Optional[datetime] = None
    
    def is_healthy(self) -> bool:
        if self.failures == 0:
            return True
        if self.last_failure is None:
            return True
        cooldown = timedelta(seconds=INSTANCE_COOLDOWN * min(self.failures, 5))
        return datetime.now() > self.last_failure + cooldown
    
    def record_success(self):
        self.failures = 0
        self.last_success = datetime.now()
    
    def record_failure(self):
        self.failures += 1
        self.last_failure = datetime.now()


class MultiSearchService:
    """
    Multi-provider search service with SearxNG primary and SerpAPI fallback.
    
    SearxNG instances are tried in round-robin order with automatic failover.
    Unhealthy instances are temporarily skipped based on failure count.
    """
    
    def __init__(self):
        self._instance_health: Dict[str, InstanceHealth] = {
            url: InstanceHealth(url=url) for url in SEARXNG_PUBLIC_INSTANCES
        }
        self._current_instance_idx = 0
        self._serpapi_key = os.environ.get("SERPAPI_KEY") or os.environ.get("SERPAPI_API_KEY")
        self._lock = asyncio.Lock()
        
        if self._serpapi_key:
            logger.info("MultiSearchService initialized with SearxNG + SerpAPI fallback")
        else:
            logger.info("MultiSearchService initialized with SearxNG only (no SerpAPI key)")
    
    def _get_next_healthy_instance(self) -> Optional[str]:
        """Get next healthy SearxNG instance in round-robin order"""
        start_idx = self._current_instance_idx
        checked = 0
        
        while checked < len(SEARXNG_PUBLIC_INSTANCES):
            url = SEARXNG_PUBLIC_INSTANCES[self._current_instance_idx]
            self._current_instance_idx = (self._current_instance_idx + 1) % len(SEARXNG_PUBLIC_INSTANCES)
            checked += 1
            
            health = self._instance_health[url]
            if health.is_healthy():
                return url
        
        return SEARXNG_PUBLIC_INSTANCES[start_idx]
    
    async def _search_searxng(self, instance_url: str, query: str, max_results: int) -> SearchResponse:
        """Search using a specific SearxNG instance"""
        try:
            params = {
                "q": query,
                "format": "json",
                "categories": "general",
            }
            
            async with httpx.AsyncClient(timeout=SEARXNG_TIMEOUT) as client:
                response = await client.get(
                    f"{instance_url}/search",
                    params=params,
                    headers={"Accept": "application/json"}
                )
                response.raise_for_status()
                data = response.json()
            
            results = []
            for item in data.get("results", [])[:max_results]:
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", "") or item.get("snippet", ""),
                    source=item.get("engine", "searxng"),
                    date=item.get("publishedDate")
                ))
            
            if results:
                self._instance_health[instance_url].record_success()
                logger.debug(f"SearxNG {instance_url} returned {len(results)} results")
            
            return SearchResponse(results=results, provider=f"searxng:{instance_url}")
            
        except httpx.TimeoutException:
            self._instance_health[instance_url].record_failure()
            logger.warning(f"SearxNG {instance_url} timeout")
            return SearchResponse(results=[], error="timeout")
            
        except httpx.HTTPStatusError as e:
            self._instance_health[instance_url].record_failure()
            logger.warning(f"SearxNG {instance_url} HTTP error: {e.response.status_code}")
            return SearchResponse(results=[], error=f"http_{e.response.status_code}")
            
        except Exception as e:
            self._instance_health[instance_url].record_failure()
            logger.warning(f"SearxNG {instance_url} error: {e}")
            return SearchResponse(results=[], error=str(e))
    
    def _ddg_sync_search(self, query: str, max_results: int) -> list:
        """Synchronous DDG search for executor"""
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        ddgs = DDGS()
        return list(ddgs.text(query, max_results=max_results))
    
    async def _search_duckduckgo(self, query: str, max_results: int) -> SearchResponse:
        """Search using DuckDuckGo (first fallback after SearxNG)"""
        try:
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                search_results = await asyncio.wait_for(
                    loop.run_in_executor(
                        executor,
                        lambda: self._ddg_sync_search(query, max_results)
                    ),
                    timeout=DDG_TIMEOUT
                )
            
            results = []
            for item in search_results[:max_results]:
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("href", ""),
                    snippet=item.get("body", ""),
                    source="duckduckgo",
                    date=None
                ))
            
            if results:
                logger.info(f"DuckDuckGo returned {len(results)} results")
                return SearchResponse(results=results, provider="duckduckgo")
            else:
                return SearchResponse(results=[], error="no_results")
            
        except asyncio.TimeoutError:
            logger.warning("DuckDuckGo timeout")
            return SearchResponse(results=[], error="timeout")
        except Exception as e:
            logger.warning(f"DuckDuckGo error: {e}")
            return SearchResponse(results=[], error=str(e))
    
    async def _search_serpapi(self, query: str, max_results: int) -> SearchResponse:
        """Search using SerpAPI (final fallback)"""
        if not self._serpapi_key:
            return SearchResponse(results=[], error="no_api_key")
        
        try:
            params = {
                "q": query,
                "api_key": self._serpapi_key,
                "engine": "google",
                "num": max_results,
            }
            
            async with httpx.AsyncClient(timeout=SERPAPI_TIMEOUT) as client:
                response = await client.get(
                    "https://serpapi.com/search",
                    params=params
                )
                response.raise_for_status()
                data = response.json()
            
            results = []
            for item in data.get("organic_results", [])[:max_results]:
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    source="google",
                    date=item.get("date")
                ))
            
            logger.info(f"SerpAPI returned {len(results)} results")
            return SearchResponse(results=results, provider="serpapi")
            
        except Exception as e:
            logger.error(f"SerpAPI error: {e}")
            return SearchResponse(results=[], error=str(e))
    
    async def search(self, query: str, max_results: int = 5) -> SearchResponse:
        """
        Perform search with three-tier provider failover.
        
        Order: SearxNG -> DuckDuckGo -> SerpAPI
        
        Args:
            query: Search query string
            max_results: Maximum results to return
            
        Returns:
            SearchResponse with results from first successful provider
        """
        async with self._lock:
            # Tier 1: Try SearxNG public instances
            attempts = 0
            max_attempts = min(len(SEARXNG_PUBLIC_INSTANCES), 3)
            
            while attempts < max_attempts:
                instance_url = self._get_next_healthy_instance()
                if not instance_url:
                    break
                
                response = await self._search_searxng(instance_url, query, max_results)
                if response.success:
                    return response
                
                attempts += 1
            
            # Tier 2: Try DuckDuckGo
            logger.info("SearxNG instances failed, trying DuckDuckGo fallback")
            response = await self._search_duckduckgo(query, max_results)
            if response.success:
                return response
            
            # Tier 3: Try SerpAPI (if configured)
            if self._serpapi_key:
                logger.info("DuckDuckGo failed, trying SerpAPI fallback")
                response = await self._search_serpapi(query, max_results)
                if response.success:
                    return response
            
            logger.warning("All search providers failed")
            return SearchResponse(
                results=[],
                error="all_providers_failed"
            )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all search providers"""
        searxng_status = {}
        for url, health in self._instance_health.items():
            searxng_status[url] = {
                "healthy": health.is_healthy(),
                "failures": health.failures,
                "last_success": health.last_success.isoformat() if health.last_success else None,
                "last_failure": health.last_failure.isoformat() if health.last_failure else None,
            }
        
        return {
            "searxng": searxng_status,
            "serpapi_configured": bool(self._serpapi_key),
            "current_instance_idx": self._current_instance_idx,
        }


multi_search_service = MultiSearchService()
