#!/usr/bin/env python3
"""
Unified Document Ingestion Script for BrandonBot

This script consolidates all ingestion:
- Brandon Platform documents
- Party Platform documents  
- Previous Q&A
- Marketing Guru content (with smart parsing)

Usage:
    python ingest_all.py [documents_directory]
    python ingest_all.py --chunk-size 256 --overlap 100
"""
import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingest_documents import (
    ingest_file, chunk_text, extract_text_from_pdf, 
    extract_text_from_docx, extract_text_from_txt, COLLECTION_MAP
)
from marketguru_ingester import MarketGuruIngester
from weaviate_manager import WeaviateManager
import weaviate as weaviate_client

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


async def connect_or_start_weaviate(data_path: str) -> WeaviateManager:
    """Connect to existing Weaviate or start embedded mode"""
    weaviate = WeaviateManager(data_path)
    
    try:
        logger.info("Connecting to existing Weaviate instance...")
        weaviate.client = await asyncio.to_thread(
            lambda: weaviate_client.connect_to_local(host="localhost", port=8079, grpc_port=50050)
        )
        logger.info("Connected to existing Weaviate instance")
    except Exception as e:
        logger.info(f"No existing instance found, starting embedded mode: {e}")
        await weaviate.initialize()
    
    return weaviate


async def ingest_standard_collections(weaviate: WeaviateManager, data_dir: str,
                                       chunk_size: int, overlap: int) -> dict:
    """Ingest Brandon Platform, Party Platforms, and Previous Q&A"""
    stats = {"docs": 0, "chunks": 0}
    
    collection_dirs = {
        "brandon_platform": ("brandon", "Brandon's Platform"),
        "party_platforms": ("party", "Party Platforms"),
        "previous_qa": ("qa", "Previous Q&A")
    }
    
    for dir_name, (collection_key, display_name) in collection_dirs.items():
        dir_path = os.path.join(data_dir, dir_name)
        
        if not os.path.exists(dir_path):
            logger.info(f"  Skipping {display_name}: Directory not found")
            continue
        
        files = [f for f in os.listdir(dir_path) 
                if f.endswith(('.pdf', '.docx', '.txt'))]
        
        if not files:
            logger.info(f"  Skipping {display_name}: No documents found")
            continue
        
        logger.info(f"\n  {display_name} ({len(files)} files)")
        
        for file_name in files:
            file_path = os.path.join(dir_path, file_name)
            chunks = await ingest_file(weaviate, file_path, collection_key, display_name,
                                       chunk_size=chunk_size, overlap=overlap)
            if chunks > 0:
                stats["docs"] += 1
                stats["chunks"] += chunks
                logger.info(f"    + {file_name}: {chunks} chunks")
    
    return stats


async def ingest_market_gurus(weaviate: WeaviateManager, data_dir: str) -> dict:
    """Ingest marketing guru content with smart parsing"""
    stats = {"files": 0, "chunks": 0}
    
    market_dir = os.path.join(data_dir, "market_gurus")
    
    if not os.path.exists(market_dir):
        logger.info("  Skipping MarketGurus: Directory not found")
        return stats
    
    logger.info(f"\n  Marketing Gurus (smart parsing)")
    
    ingester = MarketGuruIngester(weaviate)
    guru_stats = await ingester.ingest_directory(market_dir)
    
    stats["files"] = guru_stats.get("files", 0)
    stats["chunks"] = guru_stats.get("chunks", 0)
    
    return stats


async def run_unified_ingestion(data_dir: str, chunk_size: int = 1000, overlap: int = 200):
    """Run complete ingestion pipeline"""
    logger.info("=" * 70)
    logger.info("BrandonBot Unified Document Ingestion")
    logger.info("=" * 70)
    logger.info(f"Source: {data_dir}")
    logger.info(f"Chunk size: {chunk_size} | Overlap: {overlap}")
    logger.info("")
    
    weaviate = await connect_or_start_weaviate("./weaviate_data")
    
    try:
        logger.info("PHASE 1: Standard Collections")
        logger.info("-" * 40)
        std_stats = await ingest_standard_collections(weaviate, data_dir, chunk_size, overlap)
        
        logger.info("\nPHASE 2: Marketing Gurus (Smart Parsing)")
        logger.info("-" * 40)
        guru_stats = await ingest_market_gurus(weaviate, data_dir)
        
        logger.info("\n" + "=" * 70)
        logger.info("INGESTION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Standard docs: {std_stats['docs']} files, {std_stats['chunks']} chunks")
        logger.info(f"MarketGurus:   {guru_stats['files']} files, {guru_stats['chunks']} chunks")
        logger.info(f"TOTAL:         {std_stats['docs'] + guru_stats['files']} files, "
                   f"{std_stats['chunks'] + guru_stats['chunks']} chunks")
        logger.info("")
        
        logger.info("Collection Summary:")
        for collection_key in ["brandon", "party", "qa", "market"]:
            collection_name = COLLECTION_MAP[collection_key]
            count = await weaviate.get_collection_count(collection_name)
            logger.info(f"  {collection_name}: {count} chunks")
        
    finally:
        if hasattr(weaviate, 'client') and weaviate.client:
            weaviate.client.close()


def main():
    parser = argparse.ArgumentParser(
        description="Unified document ingestion for BrandonBot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest_all.py documents/
  python ingest_all.py --chunk-size 256 --overlap 100 documents/
        """
    )
    
    parser.add_argument(
        "data_directory",
        nargs="?",
        default="./documents",
        help="Path to documents directory (default: ./documents)"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum chunk size in characters (default: 1000)"
    )
    
    parser.add_argument(
        "--overlap",
        type=int,
        default=200,
        help="Overlap between chunks in characters (default: 200)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_directory):
        logger.error(f"Data directory not found: {args.data_directory}")
        logger.info("\nExpected directory structure:")
        logger.info("  documents/")
        logger.info("    brandon_platform/")
        logger.info("    party_platforms/")
        logger.info("    market_gurus/")
        logger.info("    previous_qa/")
        sys.exit(1)
    
    asyncio.run(run_unified_ingestion(
        args.data_directory,
        chunk_size=args.chunk_size,
        overlap=args.overlap
    ))


if __name__ == "__main__":
    main()
