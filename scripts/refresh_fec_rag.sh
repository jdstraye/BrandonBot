#!/usr/bin/env bash
set -euo pipefail

# Refresh FEC RAG by ingesting official pages and appending to a log
ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "Refreshing FEC RAG at $TIMESTAMP..." | tee "$LOG_DIR/refresh_fec_rag_$TIMESTAMP.log"
python3 "$ROOT_DIR/scripts/ingest_fec_official.py" 2>&1 | tee -a "$LOG_DIR/refresh_fec_rag_$TIMESTAMP.log"
echo "Done. Log: $LOG_DIR/refresh_fec_rag_$TIMESTAMP.log"
