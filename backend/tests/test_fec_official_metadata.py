import asyncio
import pytest
import json

from backend.weaviate_manager import WeaviateManager


def skip_if_no_weaviate():
    import requests
    try:
        r = requests.get('http://127.0.0.1:8079/v1/.well-known/ready', timeout=2)
        return r.status_code != 200
    except Exception:
        return True


@pytest.mark.skipif(skip_if_no_weaviate(), reason="Local Weaviate not available")
def test_official_fec_metadata_present():
    async def run():
        wm = WeaviateManager()
        await wm.initialize()
        # Run the official ingest script inline
        from scripts import ingest_fec_official as inf
        await inf.main()

        # Search for terms likely present in FEC official pages (e.g., 'help' or 'press')
        # Use REST /v1/objects to inspect all FECProhibited objects and check metadata
        import requests
        r = requests.get('http://127.0.0.1:8079/v1/objects', params={'class': 'FECProhibited', 'limit': 100})
        r.raise_for_status()
        data = r.json()
        found = False
        for obj in data.get('objects', []):
            meta = obj.get('properties', {}).get('metadata', '')
            if not meta:
                continue
            try:
                d = json.loads(meta)
                if d.get('origin') == 'official_fec' and d.get('fetched_at'):
                    found = True
                    break
            except Exception:
                continue
        assert found, 'No official_fec document with fetched_at metadata found in collection'

    asyncio.run(run())
