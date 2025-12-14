#!/usr/bin/env python3
"""Run lightweight diagnostics against configured LLM providers.

Saves results to `/tmp/llm_providers_status.json` for later analysis.
"""
import asyncio
import json
import time
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure repo root and `backend` are importable when running this script directly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.llm_providers import llm_manager
from output_validator import output_validator, SLMNotAvailableError


async def test_provider(provider_name: str, provider) -> Dict[str, Any]:
    results = {
        "provider": provider_name,
        "slots": []
    }

    for slot in provider.config.slots:
        slot_info = {
            "slot_id": slot.slot_id,
            "has_key": bool(slot.get_api_key()),
            "status": slot.status.value,
            "last_model_idx": slot.last_model_idx,
            "error_count": slot.error_count,
            "models": slot.models,
            "test": None
        }

        # Set provider to use this slot and model
        try:
            provider.current_slot = slot
            provider.current_model = slot.peek_next_model()

            messages = [{"role": "user", "content": "I want to volunteer. My email is diag@example.test"}]
            start = time.time()
            try:
                coro = provider.generate_with_tools(messages, [], "System: diagnostics")
                res = await asyncio.wait_for(coro, timeout=30)
            except asyncio.TimeoutError:
                slot_info["test"] = {"error": "timeout"}
                results["slots"].append(slot_info)
                continue

            elapsed = time.time() - start

            test_result = {
                "error": res.error,
                "text_snippet": (res.text or "")[:200],
                "tokens_used": res.tokens_used,
                "latency_ms": res.latency_ms,
                "measured_sec": elapsed,
                "tool_calls": res.tool_calls or []
            }

            # Attempt OV validation if available
            try:
                ov = await asyncio.wait_for(output_validator.validate(messages[0]["content"], res.text or "", pq_confidence=0.9), timeout=20)
                test_result["ov_passed"] = ov.passed
                test_result["ov_max_violation"] = ov.max_violation
                test_result["ov_rejection_reason"] = ov.rejection_reason
            except SLMNotAvailableError:
                test_result["ov_available"] = False
            except Exception as e:
                test_result["ov_error"] = str(e)

            slot_info["test"] = test_result

        except Exception as e:
            slot_info["test"] = {"error": str(e)}

        results["slots"].append(slot_info)

    return results


async def main():
    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "providers": {}
    }

    for name, provider in llm_manager.providers.items():
        try:
            print(f"Testing provider: {name}")
            res = await test_provider(name, provider)
            out["providers"][name] = res
        except Exception as e:
            out["providers"][name] = {"error": str(e)}

    with open("/tmp/llm_providers_status.json", "w") as f:
        json.dump(out, f, indent=2)

    print("Wrote /tmp/llm_providers_status.json")


if __name__ == "__main__":
    asyncio.run(main())
