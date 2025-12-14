import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
import pytest

from backend.output_validator import OutputValidatorSLM, SLMNotAvailableError


def test_repetition_fallback_to_pattern_when_embeddings_unavailable():
    ov = OutputValidatorSLM(require_slm=True)

    # Ensure weaviate manager is not set and require_slm=True
    ov._weaviate_manager = None

    # Previous responses to compare against
    prev = ["I apologize, but I'm having trouble completing this request. Would you like someone from the team to call you back to discuss this?",]

    # When embeddings are missing, check_repetition should raise SLMNotAvailableError
    with pytest.raises(SLMNotAvailableError):
        asyncio.run(ov.check_repetition(response="Some response", previous_responses=prev))

    # But the pattern fallback should return an OVResult (no exception)
    fallback = ov._check_repetition_pattern_fallback("Some response", prev)
    assert fallback is not None
    assert hasattr(fallback, 'score')
