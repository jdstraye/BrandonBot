import asyncio
import pytest

from backend.weaviate_manager import WeaviateManager
from backend.output_validator import OutputValidatorSLM


def skip_if_no_weaviate():
    # Simple probe: try to contact local weaviate readiness endpoint
    import requests
    try:
        r = requests.get('http://127.0.0.1:8079/v1/.well-known/ready', timeout=2)
        return r.status_code != 200
    except Exception:
        return True


@pytest.mark.skipif(skip_if_no_weaviate(), reason="Local Weaviate not available")
def test_fec_rag_seeded_and_operational():
    async def run():
        wm = WeaviateManager()
        await wm.initialize()
        count = await wm.get_collection_count('FECProhibited')
        assert count > 0, "FECProhibited must be seeded for compliance checks"

        ov = OutputValidatorSLM(require_slm=True)
        ov.set_fec_rag(wm)
        res = await ov._check_fec("Please enter your credit card number to donate now.")
        assert res.score >= 4
        assert res.method in ("rag_pattern", "pattern", "hybrid")

    asyncio.run(run())
