# FEC RAG Maintenance

This document describes how to keep the `FECProhibited` RAG collection up-to-date with authoritative FEC/eCFR sources.

Recommended approach:

- Add `scripts/refresh_fec_rag.sh` to a cron job (weekly or monthly) on your deployment host. Example crontab entry:

```
# Run every Sunday at 03:00 UTC
0 3 * * 0 /path/to/repo/scripts/refresh_fec_rag.sh
```

- The script writes logs to `logs/refresh_fec_rag_<timestamp>.log` for auditing; keep these logs for compliance reviews.

- The ingestion process tags each document with `metadata.source_url` and `metadata.fetched_at` so you can reconstruct the exact authoritative content used at any point in time.

- If a page is a PDF, the ingestion attempts to download and extract text; if extraction fails, the ingest will fall back to HTML extraction where possible.

- Monitor logs for HTTP 403/302/404 responses; these indicate the authoritative page may have moved or is behind access controls. You should investigate these manually and add canonical endpoints where necessary.

- For security / compliance, consider storing a compressed archive of ingested docs (or their hashes) to support future audits.

Contact: ops@yourorg.example (replace with your team contact) for operational support or to add additional authoritative sources.
