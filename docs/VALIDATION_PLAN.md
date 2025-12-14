# Validation Plan

## Objective
Define how to run, collect, and act on validation runs for BrandonBot, including nightly runs, telemetry capture, and triage procedures.

## Components
- `scripts/run_full_validation.py` — runs pytest and `TestPhase.ALL` validator sessions and stores JSON artifacts under `validation_runs/`
- `scripts/generate_validation_report.py` — consolidates run artifacts and generates an HTML report plus a plot of average judge score over time
- Cron jobs: weekly FEC refresh and nightly validation (03:00 America/Los_Angeles)

## Runbook (local)
1. Ensure local environment: judge available, agent slots healthy, Weaviate seeded:

```bash
# example quick checks
curl -s http://127.0.0.1:5000/health | python3 -m json.tool
scripts/seed_weaviate_local.sh --full
```

2. Run a single full validation (writes artifacts to `validation_runs/<timestamp>`):

```bash
scripts/run_full_validation.py --output-dir validation_runs
```

3. Generate local report for runs directory:

```bash
scripts/generate_validation_report.py --runs-dir validation_runs --out-dir validation_report
```

## Nightly job
- Installed via `scripts/install_nightly_validation_cron.sh` or `scripts/install_refresh_cron.sh --install-all`
- Runs at 00:00 (midnight) America/Los_Angeles and writes artifacts to `validation_runs/` and generated HTML into `validation_report/`.

## Triage
- Check `llm_requests`, `raw_llm_responses`, `ov_attempts`, `spiral_events` telemetry tables when a failure occurs.
- Use `scripts/generate_validation_report.py` plots to review judge score trends.

## TODOs
- Add an integration test enforcing pass criteria (judge score >=3 and per-turn latency < 3s).
- Document triage playbook with example queries and SQL snippets to fetch telemetry.
