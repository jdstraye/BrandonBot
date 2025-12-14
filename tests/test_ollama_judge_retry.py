import asyncio
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from backend.ollama_judge import OllamaJudge


def test_ollama_retry(monkeypatch):
    j = OllamaJudge()

    # Ensure judge config requests 1 retry and small backoff
    monkeypatch.setattr('backend.ollama_judge.settings.cfg', SimpleNamespace(judge={"timeout_seconds": 0.1, "retries": 1, "retry_backoff_ms": 1}))

    # Avoid real network calls for availability check
    async def _avail():
        return True

    monkeypatch.setattr(j, 'check_availability', _avail)
    # Mark judge as available (check_availability usually sets this)
    j._available = True

    calls = {"n": 0}

    class DummyResponse:
        def __init__(self, data):
            self._data = data
            self.status_code = 200

        def raise_for_status(self):
            if self.status_code != 200:
                raise httpx.HTTPStatusError('status', request=None, response=None)

        def json(self):
            return self._data

    class DummyClient:
        def __init__(self, timeout=None):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json):
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.TimeoutException('simulated timeout')
            return DummyResponse({"response": '{"scores": {"clarity": 4, "empathy": 4, "accuracy": 4, "engagement": 4, "tone": 4, "alignment": 4}, "reasoning": "ok"}'})

    import httpx
    monkeypatch.setattr('backend.ollama_judge.httpx.AsyncClient', DummyClient)

    score = asyncio.run(j.score_response(user_query="q", bot_response="r"))
    assert score.average == 4.0
    assert calls["n"] == 2
