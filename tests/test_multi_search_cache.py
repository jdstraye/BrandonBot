import asyncio
import os
import sys
from datetime import datetime, timedelta

import pytest

# Ensure repo root is on sys.path for test imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.multi_search_service import MultiSearchService, SearchResponse, SearchResult, CACHE_TTL


def test_search_caches_results(monkeypatch):
    # Use a single fake instance and create the service after patching instances
    from types import SimpleNamespace
    monkeypatch.setattr('backend.multi_search_service.load_config',
                        lambda: SimpleNamespace(providers=SimpleNamespace(searxng_url="https://fake.searx")))
    svc = MultiSearchService()

    call_count = {"n": 0}

    async def fake_search(instance_url, query, max_results):
        call_count["n"] += 1
        return SearchResponse(results=[SearchResult(title="t", url="u", snippet="s")], provider=f"searxng:{instance_url}")

    # Patch the private search implementation
    monkeypatch.setattr(svc, "_search_searxng", fake_search)

    # First call - should call underlying searx search
    r1 = asyncio.run(svc.search("foo bar", max_results=3))
    assert r1.success
    assert call_count["n"] == 1

    # Second call - should return cached response and not increment call count
    r2 = asyncio.run(svc.search("foo bar", max_results=3))
    assert r2.success
    assert call_count["n"] == 1

