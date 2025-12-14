# Developer Guide

## Getting started (local dev)
1. Create and activate a Python virtualenv and install deps:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Start local services used by tests (Weaviate, local judge, etc.)

```bash
docker compose -f docker-compose.test.yml up -d
```

3. Seed the FEC RAG collection:

```bash
scripts/seed_weaviate_local.sh --full
```

4. Run unit tests and validation:

```bash
pytest -q
scripts/run_full_validation.py --output-dir validation_runs
```

## Crons & maintenance
- Install weekly FEC RAG refresh: `scripts/install_refresh_cron.sh --install`
- Install nightly validation cron (00:00 Pacific / midnight): `scripts/install_nightly_validation_cron.sh` or `scripts/install_refresh_cron.sh --install-all`

## Troubleshooting checklist
- If judge calls are timing out: check `backend/ollama_judge.py` settings and local model readiness.
- If OV is repeatedly hard-failing: inspect `raw_llm_responses` and `ov_attempts` rows to find canonical fallback strings; use `scripts/generate_validation_report.py` to collect artifacts.

## Contacts & Owners
- Primary owner: @jdstraye
- Validation infra: ops/validation@example.com

## TODOs
- Flesh out environment matrix for CI and local (which judge model(s) we allow for scoring).
- Add a runbook for judge cold-start performance and retries.
