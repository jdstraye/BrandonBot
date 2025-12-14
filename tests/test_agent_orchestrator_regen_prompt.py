import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.ov_utils import build_regen_prompt
from backend.output_validator import OVValidationResult, OVResult, OVSafeguard


def test_regen_prompt_includes_original_query_on_intent_failure():
    # Build a validation_result that simulates an intent_checking hard-fail
    res_map = {
        OVSafeguard.INTENT_CHECKING: OVResult(
            safeguard=OVSafeguard.INTENT_CHECKING,
            score=4,
            confidence=0.9,
            explanation="Mostly unrelated",
            method="ms_marco"
        )
    }
    ovv = OVValidationResult(query="Hi Brandon", response="I apologize, ...", pq_confidence=0.5, results=res_map)

    # Import the helper directly from module
    prompt = build_regen_prompt(ovv, "Hi Brandon")

    assert "The user's original question was" in prompt
    assert "Hi Brandon" in prompt
