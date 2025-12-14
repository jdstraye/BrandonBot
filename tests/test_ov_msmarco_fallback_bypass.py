import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
from types import SimpleNamespace

from backend.output_validator import OutputValidatorSLM, OVResult, OVSafeguard


class FakeMSMarco:
    def __init__(self, primary_score, secondary_score):
        self.primary_score = primary_score
        self.secondary_score = secondary_score
        self._initialized = True

    async def ensure_ready(self):
        return True

    async def check_intent(self, query, response):
        # If query looks like the fallback template (we'll use substring), return secondary score
        q = (query or "").lower()
        if any(s in q for s in ("call you back", "having trouble completing")):
            return SimpleNamespace(score=self.secondary_score, confidence=0.9, explanation="matched fallback")
        return SimpleNamespace(score=self.primary_score, confidence=0.9, explanation="primary check")


def test_msmarco_fallback_bypass():
    ov = OutputValidatorSLM(require_slm=True)
    # Inject fake ms_marco
    ov._msmarco = FakeMSMarco(primary_score=4, secondary_score=1)

    query = "Hi Brandon"
    # Use a fallback-style response that is not a short clarifying question
    response = "I apologize, but I'm having trouble completing this request. We can have someone from the team call you back to discuss this."

    res = asyncio.run(ov._check_intent(query, response, is_callback_flow=False))

    assert isinstance(res, OVResult)
    assert res.safeguard == OVSafeguard.INTENT_CHECKING
    assert res.score == 0
    assert res.method in ("fallback_bypass", "ms_marco_fallback_bypass")
