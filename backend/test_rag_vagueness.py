#!/usr/bin/env python3
"""
Test RAG+Qwen Vagueness Detection

Tests the new RAG-informed vagueness classification:
1. Query Weaviate for relevant content
2. Pass query + RAG results + similarity scores to Qwen
3. Qwen classifies CLEAR/VAGUE with full context

Measures latency for each component and overall.
"""

import asyncio
import time
import logging
import weaviate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEST_QUERIES = {
    "clear_policy": [
        "What is Brandon's position on healthcare?",
        "How does Brandon plan to address inflation?",
        "What are Brandon's views on education?",
        "Tell me about Brandon's tax policy",
        "What is Brandon's stance on immigration?",
    ],
    "clear_specific": [
        "Does Brandon support the second amendment?",
        "What is Brandon's plan for job creation?",
        "How will Brandon reduce government spending?",
        "What does Brandon think about border security?",
        "Where does Brandon stand on energy independence?",
    ],
    "vague_greeting": [
        "hi",
        "hello",
        "hey there",
        "good morning",
        "what's up",
    ],
    "vague_short": [
        "taxes",
        "healthcare",
        "help",
        "question",
        "info",
    ],
    "vague_ambiguous": [
        "What about that thing?",
        "Tell me more",
        "What do you think?",
        "Can you explain?",
        "I have a question",
    ],
    "edge_cases": [
        "What about healthcare?",
        "Tell me about the economy",
        "What's Brandon's opinion?",
        "I want to know more about Brandon",
        "What are the policies?",
    ],
}


async def test_rag_vagueness():
    """Test RAG+Qwen vagueness detection with latency measurement."""
    
    print("=" * 60)
    print("RAG + Qwen Vagueness Detection Test")
    print("=" * 60)
    
    print("\n[1/3] Connecting to Weaviate...")
    try:
        client = weaviate.connect_to_local(port=8079, grpc_port=50050)
        if not client.is_ready():
            print("ERROR: Weaviate not ready")
            return
        print("Weaviate connected successfully")
    except Exception as e:
        print(f"ERROR: Could not connect to Weaviate: {e}")
        print("Make sure Weaviate is running on ports 8079/50050")
        return
    
    print("\n[2/3] Loading SLM (Cross-Encoder + Sentiment)...")
    load_start = time.time()
    
    from slm_manager import SLMManager
    from weaviate_manager import WeaviateManager
    from prequalifier import Prequalifier, RAGResult
    
    slm = SLMManager()
    await slm._ensure_cross_encoder_loaded()
    load_time = time.time() - load_start
    print(f"Cross-encoder loaded in {load_time:.2f}s")
    
    weaviate_mgr = WeaviateManager()
    weaviate_mgr.client = client
    
    pq = Prequalifier()
    pq.set_slm_provider(slm)
    pq.set_weaviate_manager(weaviate_mgr)
    
    print("\n[3/3] Running vagueness classification tests...")
    print("-" * 60)
    
    results = {
        "clear_policy": {"correct": 0, "total": 0, "latencies": []},
        "clear_specific": {"correct": 0, "total": 0, "latencies": []},
        "vague_greeting": {"correct": 0, "total": 0, "latencies": []},
        "vague_short": {"correct": 0, "total": 0, "latencies": []},
        "vague_ambiguous": {"correct": 0, "total": 0, "latencies": []},
        "edge_cases": {"correct": 0, "total": 0, "latencies": []},
    }
    
    for category, queries in TEST_QUERIES.items():
        expected = "CLEAR" if category.startswith("clear") else "VAGUE"
        print(f"\n### {category.upper()} (expected: {expected}) ###")
        
        for query in queries:
            start_time = time.time()
            
            rag_start = time.time()
            rag_results, avg_confidence = await pq._retrieve_rag_context(query)
            rag_time = time.time() - rag_start
            
            slm_start = time.time()
            rag_dicts = [r.to_dict() for r in rag_results] if rag_results else []
            response = await slm.classify_vagueness_with_rag(
                message=query,
                rag_results=rag_dicts,
                avg_confidence=avg_confidence
            )
            slm_time = time.time() - slm_start
            
            total_time = time.time() - start_time
            
            decision = response.decision
            is_correct = decision == expected
            
            results[category]["total"] += 1
            results[category]["latencies"].append(total_time)
            if is_correct:
                results[category]["correct"] += 1
            
            status = "PASS" if is_correct else "FAIL"
            print(f"  [{status}] \"{query}\"")
            print(f"        -> {decision} (RAG: {rag_time*1000:.0f}ms, SLM: {slm_time*1000:.0f}ms, Total: {total_time*1000:.0f}ms)")
            print(f"        RAG: {len(rag_results)} results, avg_score={avg_confidence:.2f}")
            if rag_results:
                top_result = rag_results[0]
                print(f"        Top result: [{top_result.collection}] {top_result.content[:60]}...")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_correct = 0
    total_tests = 0
    all_latencies = []
    
    for category, data in results.items():
        accuracy = data["correct"] / data["total"] * 100 if data["total"] > 0 else 0
        avg_latency = sum(data["latencies"]) / len(data["latencies"]) * 1000 if data["latencies"] else 0
        
        print(f"{category:20s}: {data['correct']}/{data['total']} ({accuracy:.0f}%) - avg latency: {avg_latency:.0f}ms")
        
        total_correct += data["correct"]
        total_tests += data["total"]
        all_latencies.extend(data["latencies"])
    
    print("-" * 60)
    overall_accuracy = total_correct / total_tests * 100 if total_tests > 0 else 0
    avg_total_latency = sum(all_latencies) / len(all_latencies) * 1000 if all_latencies else 0
    
    print(f"{'OVERALL':20s}: {total_correct}/{total_tests} ({overall_accuracy:.0f}%)")
    print(f"{'AVG LATENCY':20s}: {avg_total_latency:.0f}ms per query")
    print(f"{'SLM LOAD TIME':20s}: {load_time:.2f}s (one-time)")
    
    client.close()
    
    print("\n" + "=" * 60)
    if overall_accuracy >= 80:
        print("RESULT: RAG+Qwen vagueness detection is VIABLE")
    elif overall_accuracy >= 60:
        print("RESULT: RAG+Qwen vagueness detection needs TUNING")
    else:
        print("RESULT: RAG+Qwen vagueness detection needs MAJOR WORK")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_rag_vagueness())
