import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
import pytest
from types import SimpleNamespace

from backend.validation.validator import BrandonBotValidator, TestPhase
from backend.validation_debug import get_debug_db


class FakeJudge:
    async def check_availability(self):
        return True

    async def ensure_ready(self):
        return True

    async def generate_user_response(self, bot_response, conversation_history, persona=None, style=None, clarification_count=0):
        # Simulate a user actor who replies succinctly after a clarifying question
        if bot_response.strip().endswith('?'):
            return SimpleNamespace(message="I'd like to know about your views on water policy.")
        return SimpleNamespace(message="Thanks, that's helpful.")

    async def score_response(self, user_query, bot_response, context=None):
        # Return a passing score object
        return SimpleNamespace(
            clarity=4.0,
            empathy=4.0,
            accuracy=4.0,
            engagement=4.0,
            tone=4.0,
            alignment=4.0,
            reasoning="Fake judge: passing",
            all_passing=True
        )


class DeterministicClarifyAgent:
    def __init__(self):
        self._asked = False

    async def ensure_ready(self):
        return True

    async def process_message(self, user_message, session_id=None):
        txt = (user_message or '').lower()
        if not self._asked and any(w in txt for w in ('hi', 'hello', 'brandon')):
            self._asked = True
            return ("Hi — can you tell me what you'd like to discuss (policy, volunteering, or something else)?", {"model": "deterministic/1"})
        return ("Thanks for clarifying. Here's a concise answer on water policy: Brandon supports targeted investments in water infrastructure focusing on low-income households.", {"model": "deterministic/1"})


def test_real_prompt_stable_run(monkeypatch):
    """Run prompt 0 in a stable (fake judge + deterministic agent) environment and assert it completes without looping."""
    monkeypatch.setenv('TESTING_MODE', 'true')
    monkeypatch.setattr('backend.validation.validator.AGENT_AVAILABLE', True)

    validator = BrandonBotValidator(use_judge=True, use_agent=True, require_slm=True)

    # Swap in fake judge and deterministic agent
    validator.judge = FakeJudge()
    agent = DeterministicClarifyAgent()
    validator.agent = agent
    async def _ensure_agent_ready():
        return True
    validator._ensure_agent_ready = _ensure_agent_ready

    # Run the validation (prompt index 0) with a short timeout to ensure determinism
    res = asyncio.run(asyncio.wait_for(validator.run_validation(TestPhase.FULL, max_prompts=None, target_prompt_index=0), timeout=30.0))
    assert res is not None
    assert len(res.results) == 1
    r = res.results[0]
    # OV should pass or be resolved via leniency; final should not be in a regeneration loop
    assert r.ov_passed or r.pass_fail == 'PASS'

    # Check that regen prompts (if any) contained the reminder text when intent_checking failed
    dbg = get_debug_db()
    reqs = dbg.get_llm_requests_by_test(r.test_id)
    regen_reqs = [rq for rq in reqs if rq.get('extra') and rq.get('extra').get('phase') == 'regeneration']
    for rq in regen_reqs:
        sp = rq.get('system_prompt') or ''
        assert "REMINDER: The user's original question" in sp
