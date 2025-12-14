import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
from types import SimpleNamespace

from backend.output_validator import OutputValidatorSLM, OVResult, OVSafeguard


class FakeMSMarco2:
    def __init__(self, score=4, explanation='Mostly unrelated (heuristic: refusal with alternative)'):
        self.score = score
        self.explanation = explanation
        self.confidence = 0.9
        self._initialized = True

    async def ensure_ready(self):
        return True

    async def check_intent(self, query, response):
        return SimpleNamespace(score=self.score, confidence=self.confidence, explanation=self.explanation)


def test_msmarco_leniency_refusal_with_alternative():
    ov = OutputValidatorSLM(require_slm=True)
    ov._msmarco = FakeMSMarco2(score=4, explanation='Mostly unrelated (heuristic: refusal with alternative)')

    res = asyncio.run(ov._check_intent('Hi Brandon', "I apologize, but I'm having trouble completing this request. Would you like someone to call you back?", is_callback_flow=False, is_vague_query=True))

    assert isinstance(res, OVResult)
    assert res.safeguard == OVSafeguard.INTENT_CHECKING
    # The response could be short-circuited as a clarifying/fallback bypass
    # (score 0) or be downgraded to a lenient score (2). Accept both.
    assert res.score in (0, 2)
    assert res.method in ('ms_marco_lenient', 'vague_clarify_bypass', 'fallback_bypass')


def test_msmarco_leniency_topic_overlap():
    ov = OutputValidatorSLM(require_slm=True)
    ov._msmarco = FakeMSMarco2(score=4, explanation='Mostly unrelated (heuristic: topic overlap (5 keywords))')

    res = asyncio.run(ov._check_intent('Hi Brandon', "Here's some unrelated content", is_callback_flow=False, is_vague_query=True))

    assert res.score == 2
    assert res.method == 'ms_marco_lenient'


def test_msmarco_leniency_on_fallback_phrase_with_confidence():
    """Ensure that known fallback phrases are treated leniently when the
    query is classified as vague with high confidence (numeric vagueness
    confidence supplied). This protects against MS-MARCO non-determinism."""
    ov = OutputValidatorSLM(require_slm=True)
    # Simulate MS-MARCO that returns a hard failure (4)
    ov._msmarco = FakeMSMarco2(score=4, explanation='Mostly unrelated (heuristic: refusal with alternative)')

    response = "I apologize, but I'm having trouble completing this request. Would you like someone from the team to call you back to discuss this?"
    res = asyncio.run(ov._check_intent('Hi Brandon', response, is_callback_flow=False, is_vague_query=True, vagueness_confidence=0.95))

    assert isinstance(res, OVResult)
    assert res.safeguard == OVSafeguard.INTENT_CHECKING
    # With high vagueness confidence, MS-MARCO hard-fail (4) should be
    # downgraded by multiplicative leniency or bypassed as a fallback.
    assert res.score in (0, 2)
    assert res.method in ('ms_marco_lenient_vagueness', 'vague_clarify_bypass', 'fallback_bypass')
