#!/usr/bin/env python3
"""Test a single chunk size configuration"""
import sys
import os
import subprocess
import time
import json
import urllib.request

def modify_chunk_size(chunk_size: int, overlap: int):
    """Modify ingest_documents.py to use specified chunk size"""
    with open('ingest_documents.py', 'r') as f:
        content = f.read()
    
    # Replace the default parameters
    content = content.replace(
        'def chunk_text(text: str, chunk_size: int = 512, overlap: int = 100)',
        f'def chunk_text(text: str, chunk_size: int = {chunk_size}, overlap: int = {overlap})'
    )
    
    with open('ingest_documents.py', 'w') as f:
        f.write(content)
    
    print(f"✅ Modified ingest_documents.py: chunk_size={chunk_size}, overlap={overlap}")

def restart_and_ingest(chunk_size: int):
    """Restart server and reingest documents"""
    print("\n" + "="*80)
    print(f"RESTARTING SERVER AND INGESTING WITH CHUNK SIZE: {chunk_size}")
    print("="*80)
    
    # Kill server
    subprocess.run(['pkill', '-f', 'uvicorn'], capture_output=True)
    time.sleep(3)
    
    # Clear weaviate data
    subprocess.run(['rm', '-rf', 'weaviate_data'], capture_output=True)
    time.sleep(1)
    
    # Start server in background
    with open('/tmp/server.log', 'w') as log:
        subprocess.Popen(
            ['python3', '-m', 'uvicorn', 'main:app', '--host', '0.0.0.0', '--port', '5000'],
            stdout=log,
            stderr=log
        )
    
    print("Waiting for server to start (30 seconds)...")
    time.sleep(30)
    
    # Run ingestion
    result = subprocess.run(
        ['python3', 'ingest_documents.py', '../documents'],
        capture_output=True,
        text=True,
        timeout=180
    )
    
    if "Ingestion Complete" in result.stdout:
        # Extract chunk count
        for line in result.stdout.split('\n'):
            if 'BrandonPlatform:' in line or 'Total chunks' in line:
                print(f"  {line.strip()}")
        return True
    else:
        print("❌ Ingestion failed")
        return False

def test_query(query: str, expand: bool) -> float:
    """Test a single query"""
    try:
        url = f'http://localhost:5000/api/test_rag?query={query}&expand={"true" if expand else "false"}'
        response = urllib.request.urlopen(url, timeout=30).read()
        data = json.loads(response)
        return round(data.get('best_confidence', 0) * 100, 2)
    except Exception as e:
        print(f"  ⚠️  Error: {e}")
        return 0.0

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 test_single_chunk_size.py <chunk_size>")
        sys.exit(1)
    
    chunk_size = int(sys.argv[1])
    overlap = int(chunk_size * 0.2)  # 20% overlap
    
    queries = ["education", "healthcare", "immigration", "economy"]
    
    # Modify ingestion script
    modify_chunk_size(chunk_size, overlap)
    
    # Restart and ingest
    if not restart_and_ingest(chunk_size):
        print("❌ Failed to ingest documents")
        sys.exit(1)
    
    # Wait for server to stabilize
    time.sleep(5)
    
    results = {
        "chunk_size": chunk_size,
        "overlap": overlap,
        "without_expansion": {},
        "with_expansion": {}
    }
    
    # Test without expansion
    print("\n🔍 Testing WITHOUT query expansion:")
    for query in queries:
        conf = test_query(query, False)
        results["without_expansion"][query] = conf
        print(f"  {query:12s}: {conf:.2f}%")
    
    # Test with expansion
    print("\n🔍 Testing WITH query expansion:")
    for query in queries:
        conf = test_query(query, True)
        results["with_expansion"][query] = conf
        print(f"  {query:12s}: {conf:.2f}%")
    
    # Save results
    output_file = f'/tmp/chunk_{chunk_size}_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_file}")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
