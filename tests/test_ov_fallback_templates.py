import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio

from backend.output_validator import OutputValidatorSLM, OVResult, OVSafeguard


def _run_check(response_text: str):
    ov = OutputValidatorSLM(require_slm=True)
    # Use a short, vague query trigger
    query = "Hi Brandon"
    return asyncio.run(ov._check_intent(query, response_text, is_callback_flow=False, is_vague_query=True))


def test_fallback_variants_are_bypassed():
    samples = [
        "I apologize, but I'm having trouble completing this request. Would you like someone from the team to call you back to discuss this?",
        "I want to make sure I give you accurate information. Would you like someone from Brandon's team to call you back to discuss this personally?",
        "I can see you're really upset and want straight answers from us. If you'd like, I can arrange for a member of the team to call you back to discuss this.",
        "I'm sorry — I'm having trouble with that right now. We can arrange for someone from our team to call you back and follow up.",
    ]

    for s in samples:
        res = _run_check(s)
        assert isinstance(res, OVResult)
        assert res.safeguard == OVSafeguard.INTENT_CHECKING
        # We expect either an early substring bypass or the ms_marco fallback bypass
        assert res.method in ("fallback_bypass", "ms_marco_fallback_bypass", "ms_marco_lenient", "vague_clarify_bypass"), f"Unexpected method {res.method} for sample {s}"


def test_long_fallback_variants_are_deterministically_bypassed():
    """Some long fallback variants should be deterministically matched by
    the substring checks (not just by MS-MARCO secondary checks) to avoid
    intermittent rejections due to SLM non-determinism."""
    long_msg = (
        "I can see you're really upset and want straight answers from us. "
        "I apologize if our automated responses have come across as evasive or unclear - that's not the intention. "
        "We're working to improve our conversations and provide more personalized interactions with our team members. "
        "However, I don't have a direct source to share regarding our policy on calling back with accurate information, "
        "but rest assured we're committed to transparency and would love to discuss this further with you personally."
    )

    res = _run_check(long_msg)
    assert isinstance(res, OVResult)
    assert res.safeguard == OVSafeguard.INTENT_CHECKING
    # This should be deterministically recognized as a fallback/callback
    assert res.method == 'fallback_bypass' or res.score == 0, f"Expected deterministic fallback bypass, got {res.method} score={res.score}"


def test_fallback_with_unicode_and_em_dash_variants():
    """Ensure punctuation variants (curly apostrophes, em-dashes) are
    matched by substring checks."""
    samples = [
        "I apologize, but I’m having trouble completing this request. Would you like someone from the team to call you back to discuss this?",
        "I apologize — I'm having trouble completing this request. Would you like someone from the team to call you back?",
        "I apologize - I'm having trouble completing this request. Would you like someone from the team to call you back?",
    ]

    for s in samples:
        res = _run_check(s)
        assert isinstance(res, OVResult)
        assert res.safeguard == OVSafeguard.INTENT_CHECKING
        assert res.method == 'fallback_bypass' or res.score == 0, f"Punctuation variant not bypassed: method={res.method} score={res.score}"


def test_replay_sequence_does_not_loop_or_reject():
    # A sequence of messages observed in diagnostics that previously caused
    # intermittent rejections. Ensure each is treated as fallback/allowed.
    seq = [
        "I apologize, but I'm having trouble completing this request. Would you like someone from the team to call you back to discuss this?",
        "Thanks — I've recorded your interest as a volunteer. We'll follow up via email.",
        "I want to make sure I give you accurate information. Would you like someone from Brandon's team to call you back to discuss this personally?",
        "I can see you're really upset and want straight answers from us. I apologize if our automated responses have come across as evasive or unclear - that's not the intention. We're working to improve our conversations and provide more personalized interactions with our team members. However, I don't have a direct source to share regarding our policy on calling back with accurate information, but rest assured we're committed to transparency and would love to discuss this further with you personally."
    ]

    class FakeMSMarco:
        def __init__(self):
            self._initialized = True

        async def ensure_ready(self):
            return True

        async def check_intent(self, query, response):
            # For replay sequence tests, assume MS-MARCO will be permissive
            # (low scores) so that non-fallback messages can also pass.
            from types import SimpleNamespace
            return SimpleNamespace(score=1, confidence=0.9, explanation="permissive")

    ov = OutputValidatorSLM(require_slm=True)
    ov._msmarco = FakeMSMarco()
    query = "Hi Brandon"
    for s in seq:
        res = asyncio.run(ov._check_intent(query, s, is_callback_flow=False, is_vague_query=True))
        assert isinstance(res, OVResult)
        assert res.safeguard == OVSafeguard.INTENT_CHECKING
        assert res.score <= 2, f"Unexpected high score {res.score} for message: {s}"


def test_callback_leniency_when_msmarco_hardfails():
    # If MS-MARCO hard-fails (score=4) but the response contains callback-like
    # phrasing, OV should downgrade to a soft-fail (score=2) for vague queries.
    class FakeMSMarcoHardFail:
        def __init__(self):
            self._initialized = True

        async def ensure_ready(self):
            return True

        async def check_intent(self, q, r):
            from types import SimpleNamespace
            return SimpleNamespace(score=4, confidence=0.5, explanation="mostly unrelated")

    ov = OutputValidatorSLM(require_slm=True)
    ov._msmarco = FakeMSMarcoHardFail()

    msg = "If you'd like to have someone from Brandon's team call you back, please let me know and I can arrange that."
    res = asyncio.run(ov._check_intent("Hi Brandon", msg, is_callback_flow=False, is_vague_query=True, vagueness_confidence=0.95))
    assert isinstance(res, OVResult)
    assert res.safeguard == OVSafeguard.INTENT_CHECKING
    assert res.score in (0, 2)
    assert (res.method == "fallback_bypass") or res.method.startswith("ms_marco_lenient")
