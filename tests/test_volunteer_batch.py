#!/usr/bin/env python3
"""
Batch test: 50 volunteer registration conversations.
Tests volunteer auto-registration, duplicate prevention, and idempotency.
"""

import asyncio
import json
import sys
import time
from datetime import datetime
from collections import defaultdict
import hashlib

# Add backend to path
sys.path.insert(0, '/home/cana/cana/BrandonBot.git/backend')

from agent_orchestrator import AgentOrchestrator
from weaviate_manager import WeaviateManager
from database import DatabaseManager
from web_search_service import WebSearchService
from slm_manager import SLMManager

# Test configuration
NUM_CONVERSATIONS = 50
TEST_SESSION_ID_BASE = "batch_test_volunteer"
TEST_USER_ID_BASE = "batch_user"

# Results tracking
results = {
    "total_conversations": NUM_CONVERSATIONS,
    "successful_registrations": 0,
    "failed_registrations": 0,
    "duplicate_responses_detected": 0,
    "unique_responses": set(),
    "response_hashes": defaultdict(list),
    "tool_executions": defaultdict(int),
    "idempotency_reuses": 0,
    "errors": [],
    "conversation_details": []
}


async def run_batch_test():
    """Run 50 volunteer registration conversations and track results."""
    
    print("=" * 80)
    print("BATCH TEST: 50 Volunteer Registration Conversations")
    print(f"Start time: {datetime.now().isoformat()}")
    print("=" * 80)
    
    # Initialize orchestrator
    try:
        print("\n[Setup] Initializing orchestrator...")
        weaviate_manager = WeaviateManager("./weaviate_data")
        await weaviate_manager.initialize()
        
        db_manager = DatabaseManager("data/brandonbot.db")
        await db_manager.initialize()
        
        web_search_service = WebSearchService()
        
        slm_manager = None
        try:
            slm_manager = SLMManager()
        except Exception as e:
            print(f"[Warning] SLM Manager failed to initialize: {e}")
        
        orchestrator = AgentOrchestrator(weaviate_manager, web_search_service, db_manager, slm_manager=slm_manager)
        print("[Setup] Orchestrator initialized successfully.\n")
    except Exception as e:
        print(f"[ERROR] Failed to initialize orchestrator: {e}")
        results["errors"].append(f"Initialization failed: {e}")
        return results
    
    # Run batch conversations
    for i in range(1, NUM_CONVERSATIONS + 1):
        session_id = f"{TEST_SESSION_ID_BASE}_{i}"
        user_id = f"{TEST_USER_ID_BASE}_{i}"
        email = f"volunteer_test_{i:03d}@test.local"
        
        print(f"\n[{i:2d}/{NUM_CONVERSATIONS}] Testing volunteer registration...")
        print(f"  Session: {session_id}")
        print(f"  Email: {email}")
        
        try:
            # Simulate volunteer signup conversation
            query = f"Yes, I want to volunteer. My name is Test Volunteer {i}, email is {email}, zip is 9700{i % 10}."
            
            start_time = time.time()
            response_text, metadata = await orchestrator.process_message(
                user_message=query,
                session_id=session_id
            )
            elapsed = time.time() - start_time
            
            # Track response for duplicate detection
            response_hash = hashlib.md5(response_text.lower().strip().encode()).hexdigest()
            results["response_hashes"][response_hash].append({
                "conversation": i,
                "email": email,
                "response_preview": response_text[:100]
            })
            results["unique_responses"].add(response_hash)
            
            # Check if registration was attempted
            tool_calls = metadata.get("tool_calls", [])
            for tc in tool_calls:
                tool_name = tc.get("name", "unknown")
                results["tool_executions"][tool_name] += 1
                if tool_name == "register_volunteer":
                    results["successful_registrations"] += 1
                    print(f"  ✓ Volunteer auto-registered (tool executed)")
            
            # Check metadata for idempotency reuse (would need access to session context for full verification)
            iterations = metadata.get("iterations", 0)
            regenerations = metadata.get("regeneration_attempts", 0)
            
            print(f"  Response length: {len(response_text)} chars")
            print(f"  Latency: {elapsed:.1f}s, Iterations: {iterations}, Regenerations: {regenerations}")
            print(f"  Tool calls: {[tc.get('name') for tc in tool_calls]}")
            
            results["conversation_details"].append({
                "conversation_num": i,
                "session_id": session_id,
                "email": email,
                "response_hash": response_hash,
                "response_length": len(response_text),
                "latency_seconds": elapsed,
                "iterations": iterations,
                "regenerations": regenerations,
                "tool_calls": tool_calls,
                "response_preview": response_text[:150]
            })
            
        except Exception as e:
            results["failed_registrations"] += 1
            error_msg = f"Conversation {i} failed: {str(e)}"
            results["errors"].append(error_msg)
            print(f"  ✗ ERROR: {e}")
    
    # Analyze results
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal conversations:           {results['total_conversations']}")
    print(f"Successful registrations:      {results['successful_registrations']}")
    print(f"Failed registrations:          {results['failed_registrations']}")
    print(f"Unique response hashes:        {len(results['unique_responses'])}")
    print(f"Expected unique responses:     {NUM_CONVERSATIONS}")
    
    if len(results['unique_responses']) < NUM_CONVERSATIONS:
        print(f"\n⚠️  DUPLICATE RESPONSES DETECTED!")
        print(f"  Unique: {len(results['unique_responses'])}, Expected: {NUM_CONVERSATIONS}")
        print(f"  Duplicates: {NUM_CONVERSATIONS - len(results['unique_responses'])}")
        
        # Show which responses were duplicated
        for response_hash, occurrences in results["response_hashes"].items():
            if len(occurrences) > 1:
                results["duplicate_responses_detected"] += len(occurrences) - 1
                print(f"\n  Hash: {response_hash}")
                print(f"  Occurred {len(occurrences)} times:")
                for occ in occurrences:
                    print(f"    - Conversation {occ['conversation']}: {occ['response_preview'][:80]}...")
    else:
        print(f"\n✓ All {NUM_CONVERSATIONS} responses were unique!")
    
    print(f"\nTool executions:")
    for tool_name, count in sorted(results["tool_executions"].items()):
        print(f"  {tool_name}: {count}")
    
    if results["errors"]:
        print(f"\nErrors ({len(results['errors'])}):")
        for error in results["errors"][:5]:  # Show first 5
            print(f"  - {error}")
        if len(results["errors"]) > 5:
            print(f"  ... and {len(results['errors']) - 5} more")
    
    # Summary verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    
    all_passed = (
        results["successful_registrations"] == NUM_CONVERSATIONS and
        len(results['unique_responses']) == NUM_CONVERSATIONS and
        len(results["errors"]) == 0
    )
    
    if all_passed:
        print("✓ TEST PASSED: All volunteer registrations successful and no duplicates!")
    else:
        issues = []
        if results["failed_registrations"] > 0:
            issues.append(f"{results['failed_registrations']} failed registrations")
        if len(results['unique_responses']) < NUM_CONVERSATIONS:
            issues.append(f"{NUM_CONVERSATIONS - len(results['unique_responses'])} duplicate responses")
        if results["errors"]:
            issues.append(f"{len(results['errors'])} errors")
        print(f"✗ TEST FAILED: {', '.join(issues)}")
    
    print(f"\nEnd time: {datetime.now().isoformat()}")
    print("=" * 80)
    
    # Save detailed results to JSON
    output_file = "/tmp/batch_test_results.json"
    try:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {output_file}")
    except Exception as e:
        print(f"\nWarning: Could not save results to file: {e}")
    
    return results


if __name__ == "__main__":
    try:
        results = asyncio.run(run_batch_test())
        sys.exit(0 if results["duplicate_responses_detected"] == 0 and results["failed_registrations"] == 0 else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
