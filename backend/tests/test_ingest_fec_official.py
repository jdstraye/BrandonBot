import asyncio
import pytest

from backend.weaviate_manager import WeaviateManager


def skip_if_no_weaviate():
    import requests
    try:
        r = requests.get('http://127.0.0.1:8079/v1/.well-known/ready', timeout=2)
        return r.status_code != 200
    except Exception:
        return True


@pytest.mark.skipif(skip_if_no_weaviate(), reason="Local Weaviate not available")
def test_ingest_fec_official_runs():
    async def run():
        wm = WeaviateManager()
        await wm.initialize()
        # Run the script logic inline: fetch a known FEC snippet and add
        ok = await wm.add_document(
            collection_name='FECProhibited',
            content='TEST OFFICIAL: Federal law limits contributions to candidates. Source: https://www.fec.gov',
            source='https://www.fec.gov',
            category='official_fec'
        )
        assert ok

    asyncio.run(run())
