#!/usr/bin/env python3
"""
LLM Validation Script for BrandonBot
Tests all 20 models across 11 API key slots to verify they work correctly.

Usage:
    python scripts/validate_llms.py [--quick] [--slot SLOT_ID] [--model MODEL_NAME]

Options:
    --quick     Only test the first model from each slot (11 tests instead of 20)
    --slot      Test only a specific slot by ID
    --model     Test only a specific model by name
"""

import sys
import os
import asyncio
import argparse
import time
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from llm_providers import (
    LLMProviderManager, 
    APIKeySlot, 
    SlotStatus,
    GeminiProvider,
    MistralProvider, 
    CohereProvider,
    HuggingFaceProvider,
    ReplicateProvider,
    ZaiProvider,
    NvidiaProvider
)


@dataclass
class TestResult:
    slot_id: str
    model: str
    provider: str
    success: bool
    latency_ms: int
    error: str = ""
    response_preview: str = ""


SIMPLE_PROMPT = "What is 2+2? Answer with just the number."
TOOL_TEST_PROMPT = "What is the capital of France? Use the available tools if needed."

TEST_TOOLS = [
    {
        "name": "get_capital",
        "description": "Gets the capital of a country",
        "parameters": {
            "type": "object",
            "properties": {
                "country": {"type": "string", "description": "The country name"}
            },
            "required": ["country"]
        }
    }
]


async def test_single_model(provider, slot: APIKeySlot, model: str) -> TestResult:
    """Test a single model with a simple prompt"""
    start = time.time()
    
    provider.current_slot = slot
    provider.current_model = model
    
    try:
        messages = [{"role": "user", "content": SIMPLE_PROMPT}]
        result = await provider.generate_with_tools(
            messages=messages,
            tools=[],
            system_prompt="You are a helpful assistant. Be concise."
        )
        
        latency = int((time.time() - start) * 1000)
        
        if result.error:
            return TestResult(
                slot_id=slot.slot_id,
                model=model,
                provider=provider.config.name,
                success=False,
                latency_ms=latency,
                error=result.error[:200]
            )
        
        response_text = result.text or ""
        return TestResult(
            slot_id=slot.slot_id,
            model=model,
            provider=provider.config.name,
            success=True,
            latency_ms=latency,
            response_preview=response_text[:100]
        )
        
    except Exception as e:
        latency = int((time.time() - start) * 1000)
        return TestResult(
            slot_id=slot.slot_id,
            model=model,
            provider=provider.config.name,
            success=False,
            latency_ms=latency,
            error=str(e)[:200]
        )


async def test_function_calling(provider, slot: APIKeySlot, model: str) -> TestResult:
    """Test function calling capability"""
    start = time.time()
    
    provider.current_slot = slot
    provider.current_model = model
    
    try:
        messages = [{"role": "user", "content": TOOL_TEST_PROMPT}]
        result = await provider.generate_with_tools(
            messages=messages,
            tools=TEST_TOOLS,
            system_prompt="You are a helpful assistant. Use tools when appropriate."
        )
        
        latency = int((time.time() - start) * 1000)
        
        if result.error:
            return TestResult(
                slot_id=slot.slot_id,
                model=model,
                provider=provider.config.name,
                success=False,
                latency_ms=latency,
                error=f"FC: {result.error[:180]}"
            )
        
        has_tool_call = bool(result.tool_calls)
        has_text = bool(result.text)
        
        return TestResult(
            slot_id=slot.slot_id,
            model=model,
            provider=provider.config.name,
            success=True,
            latency_ms=latency,
            response_preview=f"FC={'Yes' if has_tool_call else 'No'}, Text={'Yes' if has_text else 'No'}"
        )
        
    except Exception as e:
        latency = int((time.time() - start) * 1000)
        return TestResult(
            slot_id=slot.slot_id,
            model=model,
            provider=provider.config.name,
            success=False,
            latency_ms=latency,
            error=f"FC Exception: {str(e)[:150]}"
        )


def create_provider_instances() -> Dict[str, Any]:
    """Create fresh provider instances for testing"""
    return {
        "nvidia": NvidiaProvider(),
        "zai": ZaiProvider(),
        "gemini": GeminiProvider(),
        "mistral": MistralProvider(),
        "cohere": CohereProvider(),
        "huggingface": HuggingFaceProvider(),
        "replicate": ReplicateProvider(),
    }


async def run_validation(quick: bool = False, target_slot: str = None, target_model: str = None):
    """Run validation tests on all models"""
    print("=" * 70)
    print(f"BrandonBot LLM Validation - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    providers = create_provider_instances()
    
    all_tests: List[Tuple[Any, APIKeySlot, str]] = []
    
    for provider_name, provider in providers.items():
        for slot in provider.config.slots:
            if not slot.is_available():
                print(f"  SKIP {slot.slot_id} - No API key")
                continue
                
            if target_slot and slot.slot_id != target_slot:
                continue
            
            models_to_test = slot.models[:1] if quick else slot.models
            
            for model in models_to_test:
                if target_model and model != target_model:
                    continue
                all_tests.append((provider, slot, model))
    
    print(f"\nTesting {len(all_tests)} model configurations...")
    print("-" * 70)
    
    results: List[TestResult] = []
    
    for i, (provider, slot, model) in enumerate(all_tests, 1):
        print(f"\n[{i}/{len(all_tests)}] {provider.config.name}/{slot.slot_id}")
        print(f"    Model: {model}")
        
        result = await test_single_model(provider, slot, model)
        results.append(result)
        
        status = "OK" if result.success else "FAIL"
        print(f"    Simple: {status} ({result.latency_ms}ms)")
        
        if result.success:
            print(f"    Response: {result.response_preview[:60]}...")
        else:
            print(f"    Error: {result.error[:60]}...")
        
        if provider.config.supports_function_calling and result.success:
            fc_result = await test_function_calling(provider, slot, model)
            print(f"    FuncCall: {'OK' if fc_result.success else 'FAIL'} ({fc_result.latency_ms}ms) - {fc_result.response_preview}")
        
        await asyncio.sleep(0.5)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)
    
    print(f"\nTotal: {len(results)} | Passed: {passed} | Failed: {failed}")
    
    if failed > 0:
        print("\nFailed Models:")
        for r in results:
            if not r.success:
                print(f"  - {r.provider}/{r.slot_id}/{r.model}")
                print(f"    Error: {r.error[:80]}")
    
    by_provider = {}
    for r in results:
        if r.provider not in by_provider:
            by_provider[r.provider] = {"passed": 0, "failed": 0}
        if r.success:
            by_provider[r.provider]["passed"] += 1
        else:
            by_provider[r.provider]["failed"] += 1
    
    print("\nBy Provider:")
    for provider, stats in by_provider.items():
        total = stats["passed"] + stats["failed"]
        print(f"  {provider}: {stats['passed']}/{total} passed")
    
    avg_latency = sum(r.latency_ms for r in results if r.success) / max(1, passed)
    print(f"\nAverage Latency (successful): {avg_latency:.0f}ms")
    
    print("\n" + "=" * 70)
    
    return passed, failed, results


def main():
    parser = argparse.ArgumentParser(description="Validate BrandonBot LLM providers")
    parser.add_argument("--quick", action="store_true", help="Quick test (1 model per slot)")
    parser.add_argument("--slot", type=str, help="Test specific slot ID")
    parser.add_argument("--model", type=str, help="Test specific model name")
    
    args = parser.parse_args()
    
    passed, failed, results = asyncio.run(
        run_validation(
            quick=args.quick,
            target_slot=args.slot,
            target_model=args.model
        )
    )
    
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
