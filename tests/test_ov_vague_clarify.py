import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
import pytest

from backend.output_validator import OutputValidatorSLM, OVResult, OVSafeguard


def test_clarifying_question_bypasses_intent():
    """A short clarifying question in response to a short/vague query should bypass intent checking."""
    ov = OutputValidatorSLM(require_slm=True)

    query = "Hi Brandon"
    response = "Hi — can you tell me what you'd like to discuss (policy, volunteering, or something else)?"

    # Call the internal check directly; should short-circuit and return score 0
    res = asyncio.run(ov._check_intent(query, response, is_callback_flow=False))

    assert isinstance(res, OVResult)
    assert res.safeguard == OVSafeguard.INTENT_CHECKING
    assert res.score == 0
    assert res.method == "vague_clarify_bypass"


def test_fallback_message_bypasses_intent():
    ov = OutputValidatorSLM(require_slm=True)
    query = "Hi Brandon"
    variants = [
        "I apologize, but I'm having trouble completing this request. Would you like someone from the team to call you back to discuss this?",
        "I'm having trouble completing this request; would you like someone to call you back?",
        "Would you like someone from Brandon's team to call you back to discuss this?",
        "I want to make sure I give you accurate information. Would you like someone from the team to call you back?",
    ]
    for response in variants:
        res = asyncio.run(ov._check_intent(query, response, is_callback_flow=False))
        assert res.score == 0, f"Fallback variant should bypass intent: {response}"
        assert res.method in ("fallback_bypass", "vague_clarify_bypass")
