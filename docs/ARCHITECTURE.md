# Architecture Overview

## Components
- Agent orchestration: `backend/validation/validator.py`, `backend/agent_orchestrator.py`
- Output Validator: `backend/output_validator.py` (SLM checks, MS‑MARCO, deterministic fallbacks)
- Judge: `backend/ollama_judge.py` (scoring; pluggable model parameters)
- RAG: `backend/weaviate_manager.py`, ingestion scripts under `scripts/`
- Telemetry: DB tables `llm_requests`, `raw_llm_responses`, `ov_attempts`, `ov_rejections`, `spiral_events`, `perf_metrics`

## Data flows
1. PQ (prompt qualification) -> Agent is invoked -> raw response stored in `raw_llm_responses`.
2. Output Validator (OV) evaluates the response (MS‑MARCO + deterministic checks); OV decisions and `ov_attempts` are recorded.
3. If enabled, Judge scores the response out-of-band; judge output is stored and linked to the validation run.
4. Nightly runs call the full validation and generate HTML reports for trend analysis.

## Key design choices
- Per-agent-turn timeouts (`AGENT_TURN_TIMEOUT`) to prevent endless loops.
- Deterministic fallback generation and death-spiral detection (persisted to `spiral_events`) to allow replays for forensic analysis.
- FEC RAG is authoritative source for disallowed phrases and is refreshed weekly.

## Scalability & reliability
- Nightly runs are batched and write artifacts to disk so historical plots are available even if database state changes.
- Judge scoring is tunable; possible future addition: lightweight scoring worker to reduce latency.

## Open design items
- Finalize judge model and config for consistent latencies in CI.
- Add optional warm-up or persistent judge process to avoid cold-start latencies.
