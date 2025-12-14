# Validation Results & Nightly Reporting

This document describes the nightly validation job and reporting for BrandonBot.

Overview
- Nightly job runs the full validation (or a subset) and stores a JSON artifact under `validation_runs/validation_run_<timestamp>.json`.
- A report generator collates artifacts and produces `validation_report/index.html` and `avg_judge_score.png`.

Scripts
- `scripts/run_full_validation.py` — Run a single prompt in FULL mode (index 0) and persist results.
- `scripts/generate_validation_report.py` — Collate runs and generate HTML + plot.
- `scripts/install_nightly_validation_cron.sh` — Helper to install a nightly cron (02:00 UTC) that runs validation and generates report.

Cron / CI
- To install: `scripts/install_nightly_validation_cron.sh` (or run with `--print` to preview).
- The cron runs at 00:00 (midnight) Pacific Time nightly by default (uses `CRON_TZ=America/Los_Angeles`). Adjust schedule in `scripts/install_nightly_validation_cron.sh` as needed.
Content of the report
- Table of runs with timestamp, average judge score, and artifact path.
- PNG plot with average judge score (y-axis) and timestamp (x-axis).
- The report includes the latest pytest failures if the validation run captures any.

Retention & Auditing
- Store run artifacts and report logs; rotate or archive periodically (e.g., keep last 90 days).
- Keep a backup of `validation_runs` and `validation_report` in your archival storage for audits.
