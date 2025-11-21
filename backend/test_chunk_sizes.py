#!/usr/bin/env python3
"""
Comprehensive chunk size testing script
Tests multiple chunk sizes with and without query expansion
"""
import asyncio
import sys
import os
import json
import urllib.request
from typing import Dict, List
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test configuration
CHUNK_SIZES = [256, 384, 512, 640, 768, 1024]
TEST_QUERIES = ["education", "healthcare", "immigration", "economy"]
OVERLAP_RATIO = 0.2  # 20% overlap
DOCUMENTS_DIR = "../documents"

# Baseline results (1000-char chunks, 200 overlap)
BASELINE = {
    "education": 35.21,
    "healthcare": 53.44,
    "immigration": 47.71,
    "economy": 43.76
}

def reingest_documents(chunk_size: int, overlap: int) -> bool:
    """Re-ingest documents with specified chunk size"""
    import subprocess
    
    print(f"\n{'='*80}")
    print(f"Re-ingesting with chunk_size={chunk_size}, overlap={overlap}")
    print(f"{'='*80}")
    
    # Modify ingest_documents.py temporarily
    with open('ingest_documents.py', 'r') as f:
        original_content = f.read()
    
    # Replace default chunk_size and overlap
    modified_content = original_content.replace(
        'def chunk_text(text: str, chunk_size: int = 512, overlap: int = 100)',
        f'def chunk_text(text: str, chunk_size: int = {chunk_size}, overlap: int = {overlap})'
    )
    
    with open('ingest_documents.py', 'w') as f:
        f.write(modified_content)
    
    try:
        # Stop server to avoid conflicts
        subprocess.run(['pkill', '-f', 'uvicorn main:app'], capture_output=True)
        time.sleep(3)
        
        # Clear old data
        subprocess.run(['rm', '-rf', 'weaviate_data'], capture_output=True)
        time.sleep(2)
        
        # Start server (Weaviate will auto-start)
        server_process = subprocess.Popen(
            ['python3', '-m', 'uvicorn', 'main:app', '--host', '0.0.0.0', '--port', '5000'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Wait for Weaviate to be ready
        print("Waiting for Weaviate to start...")
        time.sleep(30)
        
        # Run ingestion
        result = subprocess.run(
            ['python3', 'ingest_documents.py', DOCUMENTS_DIR],
            capture_output=True,
            text=True,
            timeout=180
        )
        
        # Check if ingestion succeeded
        if "Ingestion Complete" in result.stdout:
            print("✅ Ingestion successful")
            # Extract chunk counts
            for line in result.stdout.split('\n'):
                if 'BrandonPlatform:' in line or 'chunks created:' in line:
                    print(f"  {line.strip()}")
            return True
        else:
            print("❌ Ingestion failed")
            print(result.stderr[-500:] if result.stderr else "")
            return False
            
    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
        return False
    finally:
        # Restore original ingest_documents.py
        with open('ingest_documents.py', 'w') as f:
            f.write(original_content)

def test_query(query: str, expand: bool) -> float:
    """Test a single query and return confidence score"""
    try:
        url = f'http://localhost:5000/api/test_rag?query={query}&expand={"true" if expand else "false"}'
        response = urllib.request.urlopen(url, timeout=30).read()
        data = json.loads(response)
        confidence = data.get('best_confidence', 0) * 100
        return round(confidence, 2)
    except Exception as e:
        print(f"  ⚠️  Error testing {query}: {e}")
        return 0.0

def test_chunk_size(chunk_size: int) -> Dict:
    """Test a specific chunk size with and without expansion"""
    overlap = int(chunk_size * OVERLAP_RATIO)
    
    print(f"\n{'='*80}")
    print(f"TESTING CHUNK SIZE: {chunk_size} chars (overlap: {overlap})")
    print(f"{'='*80}")
    
    # Re-ingest documents
    if not reingest_documents(chunk_size, overlap):
        return None
    
    # Wait for server to stabilize
    time.sleep(5)
    
    results = {
        "chunk_size": chunk_size,
        "overlap": overlap,
        "without_expansion": {},
        "with_expansion": {}
    }
    
    # Test without expansion
    print(f"\n🔍 Testing WITHOUT query expansion...")
    for query in TEST_QUERIES:
        confidence = test_query(query, expand=False)
        results["without_expansion"][query] = confidence
        baseline = BASELINE[query]
        improvement = confidence - baseline
        print(f"  {query:12s}: {confidence:5.2f}% (baseline: {baseline:5.2f}%, Δ: {improvement:+5.2f}%)")
    
    # Test with expansion
    print(f"\n🔍 Testing WITH query expansion...")
    for query in TEST_QUERIES:
        confidence = test_query(query, expand=True)
        results["with_expansion"][query] = confidence
        baseline = BASELINE[query]
        improvement = confidence - baseline
        print(f"  {query:12s}: {confidence:5.2f}% (baseline: {baseline:5.2f}%, Δ: {improvement:+5.2f}%)")
    
    return results

def main():
    """Run comprehensive chunk size testing"""
    print("="*80)
    print("COMPREHENSIVE CHUNK SIZE TESTING")
    print("="*80)
    print(f"\nTesting chunk sizes: {CHUNK_SIZES}")
    print(f"Test queries: {TEST_QUERIES}")
    print(f"Overlap ratio: {OVERLAP_RATIO} (20%)")
    print(f"\nBaseline (1000-char chunks, 200 overlap):")
    for query, conf in BASELINE.items():
        print(f"  {query:12s}: {conf:.2f}%")
    
    all_results = []
    
    for chunk_size in CHUNK_SIZES:
        result = test_chunk_size(chunk_size)
        if result:
            all_results.append(result)
            
            # Save intermediate results
            with open('/tmp/chunk_test_results.json', 'w') as f:
                json.dump(all_results, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"Progress: {len(all_results)}/{len(CHUNK_SIZES)} chunk sizes tested")
        print(f"{'='*80}")
    
    # Generate final report
    print("\n\n")
    print("="*80)
    print("FINAL RESULTS MATRIX")
    print("="*80)
    
    # Create results table
    print(f"\n{'Chunk Size':<12} {'Query':<12} {'No Expansion':<15} {'With Expansion':<15} {'Expansion Δ':<12}")
    print("-"*80)
    
    for result in all_results:
        chunk_size = result['chunk_size']
        for query in TEST_QUERIES:
            no_exp = result['without_expansion'][query]
            with_exp = result['with_expansion'][query]
            delta = with_exp - no_exp
            
            print(f"{chunk_size:<12} {query:<12} {no_exp:>6.2f}%         {with_exp:>6.2f}%         {delta:+5.2f}%")
    
    # Find best configuration
    print("\n" + "="*80)
    print("BEST CONFIGURATIONS")
    print("="*80)
    
    best_overall = None
    best_overall_score = 0
    
    for result in all_results:
        # Calculate average improvement over baseline
        avg_no_exp = sum(result['without_expansion'].values()) / len(TEST_QUERIES)
        avg_with_exp = sum(result['with_expansion'].values()) / len(TEST_QUERIES)
        avg_baseline = sum(BASELINE.values()) / len(TEST_QUERIES)
        
        improvement_no_exp = avg_no_exp - avg_baseline
        improvement_with_exp = avg_with_exp - avg_baseline
        
        print(f"\nChunk size: {result['chunk_size']}")
        print(f"  Without expansion: {avg_no_exp:.2f}% (Δ: {improvement_no_exp:+.2f}%)")
        print(f"  With expansion:    {avg_with_exp:.2f}% (Δ: {improvement_with_exp:+.2f}%)")
        
        if avg_with_exp > best_overall_score:
            best_overall_score = avg_with_exp
            best_overall = result
    
    if best_overall:
        print("\n" + "="*80)
        print("🏆 OPTIMAL CONFIGURATION")
        print("="*80)
        print(f"Chunk size: {best_overall['chunk_size']} chars")
        print(f"Overlap: {best_overall['overlap']} chars")
        print(f"Query expansion: {'YES' if any(best_overall['with_expansion'][q] > best_overall['without_expansion'][q] for q in TEST_QUERIES) else 'NO'}")
        avg_score = sum(best_overall['with_expansion'].values()) / len(TEST_QUERIES)
        avg_baseline = sum(BASELINE.values()) / len(TEST_QUERIES)
        print(f"Average confidence: {avg_score:.2f}% (baseline: {avg_baseline:.2f}%, improvement: {avg_score - avg_baseline:+.2f}%)")
    
    # Save final results
    with open('/tmp/chunk_test_results_final.json', 'w') as f:
        json.dump({
            'results': all_results,
            'baseline': BASELINE,
            'best_config': best_overall
        }, f, indent=2)
    
    print(f"\n✅ Results saved to /tmp/chunk_test_results_final.json")

if __name__ == "__main__":
    main()
