# Software Requirements Specification (SRS)

## Purpose
This SRS captures the high-level requirements for the BrandonBot validation system: ensuring deterministic validation of agent responses (PQ -> Agent -> Output Validator), reliable judge scoring, and authoritative FEC RAG coverage.

## Scope
- Orchestrate full validation runs (PQ -> Agent -> OV -> Judge) with telemetry and forensic artifacts.
- Nightly validation runs with reporting and time-series of judge scores.
- Maintain authoritative FEC RAG dataset and ingestion pipeline for official sources.

## Stakeholders
- Owners: @jdstraye (primary), validation & infra team
- Consumers: CI, nightly monitoring dashboards, developers triaging validation failures

## High-level requirements
1. Full validation must be reproducible and deterministic when using the same seeded environment and models.
2. The Output Validator (OV) must avoid causing death-spiral regenerations; deterministic fallback and substring normalization must be enforced.
3. The FEC RAG must be regularly refreshed (weekly) and include official sources with metadata.
4. Nightly validation must run automatically and publish an HTML report including average judge score plot and pytest results.

## Acceptance criteria
- A single real prompt can be run in FULL mode and reach PASS (judge score >= 3) with agent PQ->Agent->OV latency < 3s, when environment prerequisites are met.

## Traceability matrix
TBD

## Open items / TODOs
- Define strict numeric thresholds for MS-MARCO leniency and vagueness.
- Finalize judge configuration for CI (num_predict/timeout tuning).
- Add integration test enforcing PASS criteria (optional integration test in CI).
