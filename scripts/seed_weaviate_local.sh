#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/seed_weaviate_local.sh
# Starts a local weaviate via docker-compose.test.yml, waits for health, and runs the seed script.

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
# Default to repo-root compose file if present, otherwise fallback to script-local name
COMPOSE_FILE="$REPO_ROOT/docker-compose.test.yml"

FULL_INGEST=false
EXTRA_COMPOSE_ARG=""

# Parse args: --full and optional --compose-file <path>
while [ "$#" -gt 0 ]; do
  case "$1" in
    --full)
      FULL_INGEST=true
      shift
      ;;
    --compose-file)
      if [ -n "${2:-}" ]; then
        COMPOSE_FILE="$2"
        shift 2
      else
        echo "--compose-file requires a path" >&2
        exit 2
      fi
      ;;
    --*)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
    *)
      # ignore positional
      shift
      ;;
  esac
done

if [ ! -f "$COMPOSE_FILE" ]; then
  echo "Warning: compose file '$COMPOSE_FILE' not found in repo root. Falling back to script-local docker-compose.test.yml"
  COMPOSE_FILE="$(cd "$(dirname "$0")" && pwd)/docker-compose.test.yml"
fi

echo "Starting Weaviate (docker-compose: $COMPOSE_FILE)..."
docker compose -f "$COMPOSE_FILE" up -d --remove-orphans

echo "Waiting for Weaviate to become ready (http://localhost:8079)..."
for i in {1..30}; do
  if curl -sS http://127.0.0.1:8079/v1/.well-known/ready >/dev/null 2>&1; then
    echo "Weaviate is ready"
    break
  fi
  echo "  waiting... ($i)"
  sleep 2
done

if ! curl -sS http://127.0.0.1:8079/v1/.well-known/ready >/dev/null 2>&1; then
  echo "Weaviate did not become ready in time" >&2
  exit 1
fi

if [ "$FULL_INGEST" = "true" ]; then
  echo "Running full ingestion via backend/ingest_all.py (this may take a while)..."
  (cd "$REPO_ROOT" && python3 backend/ingest_all.py ./documents) || {
    echo "Full ingestion failed" >&2
    exit 1
  }
else
  echo "Seeding Weaviate collections (this may take a moment)..."
  (cd "$REPO_ROOT" && python3 backend/seed_weaviate.py)
fi

echo "Seed complete. You can now run: pytest -q backend/tests"