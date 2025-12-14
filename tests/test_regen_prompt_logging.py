import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.ov_utils import build_regen_prompt
from backend.output_validator import OVResult, OVSafeguard, OVValidationResult
from backend.validation_debug import get_debug_db


def test_regen_prompt_logged_contains_reminder():
    # Create a fake validation result with intent_checking hard failure
    res_map = {
        OVSafeguard.INTENT_CHECKING: OVResult(
            safeguard=OVSafeguard.INTENT_CHECKING,
            score=4,
            confidence=0.9,
            explanation="Mostly unrelated",
            method="ms_marco"
        )
    }
    ovv = OVValidationResult(query="Hi Brandon", response="I apologize...", pq_confidence=0.5, results=res_map)

    prompt = build_regen_prompt(ovv, "Hi Brandon")

    # Log into debug DB as if we were about to regenerate
    debug_db = get_debug_db()
    test_id = "REGEN-TEST-001"
    debug_db.log_llm_request(system_prompt=prompt, messages=[{"role":"system","content":prompt}], test_id=test_id, extra={"phase":"regeneration"})

    # Retrieve regen log for our test id
    reqs = debug_db.get_llm_requests_by_test(test_id)
    found = [r for r in reqs if r.get('extra') and r.get('extra').get('phase') == 'regeneration']
    assert found, "No regeneration LLM request logged"
    assert any("REMINDER: The user's original question" in (r.get('system_prompt') or '') for r in found)