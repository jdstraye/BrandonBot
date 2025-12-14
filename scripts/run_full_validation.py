#!/usr/bin/env python3
"""Run the full validation suite in non-interactive mode and persist results.

Produces a JSON artifact in `validation_runs/` containing:
- pytest failures (if any)
- validation session results (from BrandonBotValidator)
- timestamp and run metadata

Usage: python scripts/run_full_validation.py --output-dir validation_runs
"""
import argparse
import asyncio
import json
import os
from datetime import datetime, timezone

from backend.validation.validator import BrandonBotValidator, TestPhase


async def run_validation(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    tstamp = datetime.now(timezone.utc).isoformat()
    out_path = os.path.join(output_dir, f"validation_run_{tstamp}.json")

    # Run pytest first and capture failures (include stdout/stderr)
    import subprocess
    pytest_proc = subprocess.run(["pytest", "-q"], capture_output=True, text=True)

    # Run full validation (ALL phases) to exercise everything
    validator = BrandonBotValidator(use_judge=True, use_agent=True, require_slm=True)
    # Ensure dependencies wired
    try:
        await validator._ensure_agent_ready()
    except Exception:
        pass

    session = await validator.run_validation(TestPhase.ALL)

    # Gather basic summary and serialize
    serializable = {
        "timestamp": tstamp,
        "pytest": {
            "returncode": pytest_proc.returncode,
            "stdout": pytest_proc.stdout,
            "stderr": pytest_proc.stderr,
        },
        "validation_session": {
            "session_id": session.session_id if session else None,
            "results": [r.__dict__ for r in (session.results if session else [])]
        }
    }

    with open(out_path, 'w') as f:
        json.dump(serializable, f, default=str, indent=2)

    print(f"Wrote validation run to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='validation_runs')
    args = parser.parse_args()
    asyncio.run(run_validation(args.output_dir))


if __name__ == '__main__':
    main()
