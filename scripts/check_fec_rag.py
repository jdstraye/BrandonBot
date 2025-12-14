#!/usr/bin/env python3
"""Check FEC RAG readiness and sample queries.

Usage: python scripts/check_fec_rag.py
"""
import asyncio
import sys
import os
# Add repo root to sys.path for direct script execution
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.weaviate_manager import WeaviateManager


async def main():
    try:
        wm = WeaviateManager()
        await wm.initialize()
        count = await wm.get_collection_count("FECProhibited")
        print(f"FECProhibited count: {count}")

        if count == 0:
            print("FECProhibited is empty — run ingest_all.ingest_fec_prohibited or scripts/seed_weaviate_local.sh --full")
            return 2

        # Run a sample search
        results = await wm.search("FECProhibited", "enter your credit card", limit=3)
        print("Sample search results:")
        for r in results:
            print(f" - [{r.get('confidence',0):.2f}] {r.get('content','')[:120]}")

        return 0
    except Exception as e:
        print(f"Check failed: {e}")
        return 1


if __name__ == '__main__':
    code = asyncio.run(main())
    sys.exit(code)
