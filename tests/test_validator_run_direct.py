import asyncio
import os
import sys

import pytest

# Ensure repo root is on sys.path for test imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.validation.validator import BrandonBotValidator, TestPhase


class DummyAgent:
    async def process_message(self, user_message, session_id=None):
        return ("Dummy response", {"model": "dummy-model"})


def test_run_validation_phase_full_by_index(monkeypatch):
    # ensure Agent import path considered available
    monkeypatch.setattr('backend.validation.validator.AGENT_AVAILABLE', True)

    validator = BrandonBotValidator(use_judge=False, use_agent=True, require_slm=False)

    async def _ensure_agent_ready():
        return True

    validator._ensure_agent_ready = _ensure_agent_ready
    validator.agent = DummyAgent()

    validator.test_data = {
        "categories": {
            "A": {"prompts": ["p0", "p1"]},
            "B": {"prompts": ["p2", "p3"]}
        }
    }

    # Set max_prompts so the validator does not run the additional vague-loop and callback edge tests
    # Use a value greater than the target_prompt_index so we can reach index 2 but still avoid running the extra tests
    session = asyncio.run(validator.run_validation(TestPhase.FULL, max_prompts=3, target_prompt_index=2))
    # The validator returns a ValidationSession; ensure exactly one result was executed
    assert session.results is not None
    assert len(session.results) == 1
    assert session.results[0].test_id == "B-002"
