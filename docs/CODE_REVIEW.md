# Code Review / Design Notes

## Goals for reviews
- Ensure validation runs are deterministic and testable.
- Confirm telemetry is comprehensive for triage (llm_requests, raw_llm_responses, ov_attempts, spiral_events).
- Check for idempotent ingestion and robust REST fallbacks for Weaviate.

## Checklist
- [ ] Unit tests added for new behaviors
- [ ] Integration tests added for critical flows (FEC RAG ingestion, nightly reporting)
- [ ] Scripts that modify crontab expose `--print` and are idempotent
- [ ] Documentation updated (`docs/DEVELOPER_GUIDE.md`, `docs/VALIDATION_PLAN.md`)

## Common review concerns
- Adding new cron entries should never silently duplicate entries.
- Avoid non-deterministic reliance on external APIs in unit tests (use fixtures/mocks).

## TODOs
- Add explicit instructions for reviewer to test `scripts/install_refresh_cron.sh --install-all` in a disposable environment.
