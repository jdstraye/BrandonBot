import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.multi_search_service import MultiSearchService, SEARXNG_PUBLIC_INSTANCES


def test_ini_precedence(monkeypatch):
    # INI value should take precedence over env var
    monkeypatch.setenv("SEARXNG_URL", "https://env.example")
    monkeypatch.setattr('backend.multi_search_service.load_config', lambda: SimpleNamespace(providers=SimpleNamespace(searxng_url="https://ini.example")))
    svc = MultiSearchService()
    assert svc._instances == ["https://ini.example"]


def test_env_fallback(monkeypatch):
    # When INI missing, env var used
    monkeypatch.delenv("SEARXNG_URL", raising=False)
    monkeypatch.setenv("SEARXNG_URL", "https://env.example")
    monkeypatch.setattr('backend.multi_search_service.load_config', lambda: SimpleNamespace(providers=SimpleNamespace(searxng_url="")))
    svc = MultiSearchService()
    assert svc._instances == ["https://env.example"]


def test_defaults_fallback(monkeypatch):
    # When neither INI nor env present, use defaults
    monkeypatch.delenv("SEARXNG_URL", raising=False)
    monkeypatch.setattr('backend.multi_search_service.load_config', lambda: SimpleNamespace(providers=SimpleNamespace(searxng_url="")))
    svc = MultiSearchService()
    assert svc._instances == SEARXNG_PUBLIC_INSTANCES
