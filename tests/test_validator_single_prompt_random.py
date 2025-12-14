import os
import sys
import json
import random
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from backend.validation.validator import BrandonBotValidator, TestPhase
from backend.validation_debug import get_debug_db
import sqlite3
from backend.ollama_judge import JudgeScore


def load_prompts():
    path = os.path.join(os.path.dirname(__file__), '..', 'backend', 'validation', 'test_prompts.json')
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    prompts = []
    for cat, obj in data.get('categories', {}).items():
        for p in obj.get('prompts', []):
            prompts.append((cat, p))
    return prompts


def test_random_prompt_pass_with_tuned_agent(monkeypatch):
    """Pick a random prompt and verify it can pass using a tuned FakeAgent + FakeJudge.

    This test purposefully stubs the agent and judge so we can iterate quickly
    and demonstrate a single prompt can reach PASS (scores>3 and turnaround<3s).
    """
    prompts = load_prompts()
    assert len(prompts) > 0

    # Pick a pseudo-random prompt but allow reproducibility via env
    seed = int(os.environ.get('TEST_PROMPT_SEED', '0'))
    random.seed(seed)
    idx = random.randrange(len(prompts))
    cat, prompt_text = prompts[idx]

    # Tuned agent: returns safe, empathetic responses that include AZ-01 alignment hints
    class TunedAgent:
        async def process_message(self, user_message, session_id=None):
            text = (user_message or "").lower()
            reply = "Thanks for asking. "
            if any(k in text for k in ["water", "river"]):
                reply += "Water and Colorado River stewardship are priorities; Brandon supports practical conservation and local solutions. "
            elif any(k in text for k in ["border", "immigr", "security"]):
                reply += "Border security and humanitarian processes are important; Brandon supports secure borders balanced with lawful pathways. "
            elif any(k in text for k in ["tax", "donat", "money"]):
                reply += "We follow FEC rules strictly; on taxes, Brandon favors fiscal responsibility and targeted relief. "
            elif any(k in text for k in ["health", "medic", "care"]):
                reply += "Healthcare affordability matters; Brandon supports measures to lower costs while protecting care access. "
            else:
                reply += "I'm happy to provide resources or specific policy steps for AZ-01 residents; what would you like to know? "

            reply += "(If this is urgent, we can arrange for the team to follow up.)"
            return (reply, {"model": "tuned-fake"})

    class FakeJudge:
        async def check_availability(self):
            return True

        async def score_response(self, *args, **kwargs):
            return JudgeScore(clarity=4.5, empathy=4.5, accuracy=4.5, engagement=4.2, tone=4.0, alignment=4.0, reasoning="ok")

    monkeypatch.setenv("TESTING_MODE", "true")
    monkeypatch.setattr('backend.validation.validator.AGENT_AVAILABLE', True)

    validator = BrandonBotValidator(use_judge=True, use_agent=True, require_slm=True)

    async def _ensure_agent_ready():
        return True

    validator._ensure_agent_ready = _ensure_agent_ready
    validator.agent = TunedAgent()

    validator.judge = FakeJudge()

    async def _create_session_judge(session_id):
        return validator.judge

    validator._create_session_judge = _create_session_judge

    # When targeting a specific prompt index, do not limit max_prompts (None)
    # — otherwise the internal prompt_count >= max_prompts check will break
    # before reaching high-index prompts.
    res = asyncio.run(validator.run_validation(TestPhase.FULL, max_prompts=None, target_prompt_index=idx))
    assert res is not None and res.results
    r = res.results[0]

    # Expect OV passed and core scores > 3
    assert r.ov_passed is True, f"OV failed for prompt idx {idx}: {r.ov_issues}"
    assert r.score_clarity > 3
    assert r.score_empathy > 3
    assert r.score_accuracy > 3

    final_turn = r.turns[-1]
    assert final_turn.duration_ms < 3000, f"Final response too slow: {final_turn.duration_ms}ms"

    # Ensure LLM request payloads were logged for this test run
    vdb = get_debug_db()
    conn = sqlite3.connect(vdb.db_path)
    cur = conn.cursor()
    cur.execute('SELECT id, system_prompt, messages FROM llm_requests WHERE test_id = ? ORDER BY id', (r.test_id,))
    rows = cur.fetchall()
    conn.close()

    assert len(rows) > 0, "No llm_requests were logged for this run"
    # Quick sanity check: messages JSON should contain the user's prompt fragment
    found = False
    for _id, sys_prompt, msgs in rows:
        if prompt_text[:20].lower() in (msgs or "").lower():
            found = True
            break
    assert found, "User prompt not found in logged llm request messages"
