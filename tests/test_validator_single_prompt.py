import asyncio
import os
import sys
import pytest

# Ensure repo root is on sys.path for test imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.validation.validator import BrandonBotValidator


class FakeAgent:
    async def process_message(self, user_message, session_id=None):
        return (f"Echo: {user_message}", {"model": "fake-model"})


def test_run_single_prompt_by_index(monkeypatch):
    # Make the agent available for the validator
    monkeypatch.setattr('backend.validation.validator.AGENT_AVAILABLE', True)

    validator = BrandonBotValidator(use_judge=False, use_agent=True, require_slm=False)
    # Ensure _ensure_agent_ready returns True and wire a fake agent
    async def _ensure_agent_ready():
        return True

    validator._ensure_agent_ready = _ensure_agent_ready
    validator.agent = FakeAgent()

    # Set test data with two categories so global indexing spans them
    validator.test_data = {
        "categories": {
            "A": {"prompts": ["p0", "p1"]},
            "B": {"prompts": ["p2", "p3"]}
        }
    }

    results = asyncio.run(validator.run_full_validation(max_prompts=None, target_prompt_index=2))
    assert len(results) == 1
    assert results[0].test_id == "B-002"


def test_run_single_prompt_by_id(monkeypatch):
    monkeypatch.setattr('backend.validation.validator.AGENT_AVAILABLE', True)

    validator = BrandonBotValidator(use_judge=False, use_agent=True, require_slm=False)
    async def _ensure_agent_ready():
        return True

    validator._ensure_agent_ready = _ensure_agent_ready
    validator.agent = FakeAgent()

    validator.test_data = {
        "categories": {
            "A": {"prompts": ["p0", "p1"]},
            "B": {"prompts": ["p2", "p3"]}
        }
    }

    results = asyncio.run(validator.run_full_validation(max_prompts=None, target_prompt_id="A-001"))
    assert len(results) == 1
    assert results[0].test_id == "A-001"
