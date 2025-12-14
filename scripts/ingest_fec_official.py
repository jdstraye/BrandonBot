#!/usr/bin/env python3
"""Fetch official FEC pages and ingest them into the FECProhibited collection.

This script downloads a small set of authoritative pages (FEC, eCFR) and
adds them to the `FECProhibited` collection via the `WeaviateManager`.

Usage: python scripts/ingest_fec_official.py
"""
import asyncio
import sys
import os
import logging
from textwrap import shorten

import httpx

# Ensure repo root is on sys.path when run as a script
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.weaviate_manager import WeaviateManager
from backend.ingest_documents import extract_text_from_pdf
from datetime import datetime, timezone

logger = logging.getLogger("ingest_fec_official")
logging.basicConfig(level=logging.INFO)

# List of authoritative pages to ingest. Each entry is (url, short_title)
FEC_OFFICIAL_PAGES = [
    ("https://www.fec.gov/updates/adjusting-contribution-limits/", "FEC: Adjusting contribution limits"),
    ("https://www.fec.gov/resources/cms-content/documents/11-CFR-110-11.pdf", "11 CFR 110.11 (tax deductibility)"),
    ("https://www.fec.gov/help-candidates-and-committees/", "FEC: Help for candidates and committees"),
    ("https://www.fec.gov/press/", "FEC: Press and announcements"),
    ("https://www.ecfr.gov/current/title-11", "eCFR Title 11 - Federal Elections"),
    ("https://www.fec.gov/updates/fec-contribution-limits/", "FEC: Contribution limits (updates)"),
    ("https://www.fec.gov/answers/", "FEC: Answers (FAQs)"),
]


def extract_text(html: str) -> str:
    """Minimal HTML -> text extraction; avoid heavy deps like BeautifulSoup.

    We remove script/style tags and collapse whitespace. This is sufficient for
    ingesting FEC reference material for RAG.
    """
    import re

    # Remove script/style
    cleaned = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", " ", html)
    # Remove HTML tags
    cleaned = re.sub(r"(?s)<.*?>", " ", cleaned)
    # Collapse whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


async def fetch_page(client: httpx.AsyncClient, url: str) -> httpx.Response | None:
    try:
        # allow redirects explicitly; some sites redirect via 302
        r = await client.get(url, timeout=30.0, follow_redirects=True)
        r.raise_for_status()
        return r
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None


async def main():
    wm = WeaviateManager()
    await wm.initialize()

    async with httpx.AsyncClient() as client:
        added = 0
        for url, title in FEC_OFFICIAL_PAGES:
            logger.info(f"Fetching {url} ...")
            resp = await fetch_page(client, url)
            if resp is None:
                logger.warning(f"Skipping {url}: empty content")
                continue

            # Determine if this is PDF content
            ctype = resp.headers.get('content-type', '').lower()
            if 'pdf' in ctype or url.lower().endswith('.pdf'):
                # extract text from PDF binary
                try:
                    import tempfile
                    tf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
                    tf.write(resp.content)
                    tf.flush()
                    tf.close()
                    text = extract_text_from_pdf(tf.name)
                except Exception as e:
                    logger.warning(f"Failed to extract PDF content from {url}: {e}")
                    text = ''
            else:
                text = extract_text(resp.text)

            # Limit document size and include fetch metadata
            snippet = text[:20000]
            fetched_at = datetime.now(timezone.utc).isoformat()
            metadata = {"source_url": url, "fetched_at": fetched_at, "origin": "official_fec"}
            content = f"{title}\nSource: {url}\nFetched: {fetched_at}\n\n{snippet}"

            ok = await wm.add_document(collection_name="FECProhibited", content=content, source=url, category="official_fec", metadata=metadata)
            if ok:
                logger.info(f"Ingested: {shorten(title, width=80)}")
                added += 1
            else:
                logger.warning(f"Failed to ingest: {url}")

        logger.info(f"Ingestion complete: {added} pages added")
        return 0


if __name__ == '__main__':
    try:
        code = asyncio.run(main())
        sys.exit(code)
    except Exception as e:
        logger.error(f"Ingest failed: {e}")
        sys.exit(1)
