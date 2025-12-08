#!/usr/bin/env python3
"""
Unified Document Ingestion Script for BrandonBot

This script consolidates all ingestion and setup:
- Brandon Platform documents
- Party Platform documents  
- Previous Q&A
- Marketing Guru content (with smart parsing)
- FEC Prohibited phrases (MANDATORY for compliance)
- SQLite database initialization

After running this script, the system should be fully operational.

Usage:
    python ingest_all.py [documents_directory]
    python ingest_all.py --chunk-size 256 --overlap 100
    python ingest_all.py --skip-db  # Skip database initialization
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
from database import DatabaseManager
import weaviate as weaviate_client

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

FEC_PROHIBITED_DATA = [
    {
        "content": "PROHIBITED: Making tax deductibility claims for political contributions. Phrases like 'your donation is tax deductible' or 'tax write-off for contributions' are strictly prohibited as political contributions are NOT tax deductible. Violation: 11 CFR 110.11",
        "source": "FEC Regulations - 11 CFR 110.11",
        "category": "tax_advice",
    },
    {
        "content": "PROHIBITED: Soliciting or processing financial transactions. The bot must never request credit card numbers, bank account information, or process donations directly. All donations must go through the official, FEC-compliant donation portal. Phrases like 'enter your credit card' or 'provide payment information' are prohibited.",
        "source": "FEC Regulations - Financial Solicitation",
        "category": "financial_solicitation",
    },
    {
        "content": "PROHIBITED: Making defamatory statements about opponents. Statements like 'is a criminal', 'committed fraud', 'stole money', 'is corrupt', or 'took bribes' without verified factual basis constitute defamation. Focus on policy differences, not personal attacks.",
        "source": "FEC Regulations - Defamation Guidelines",
        "category": "defamation",
    },
    {
        "content": "PROHIBITED: Claiming to be the candidate or a human. The AI assistant must never claim 'I am Brandon', 'I am the candidate', 'I am a human', or 'speaking as the candidate'. It must always identify as an AI assistant for the campaign.",
        "source": "FEC Regulations - Identity Disclosure",
        "category": "false_identity",
    },
    {
        "content": "PROHIBITED: Making unverified endorsement claims or guarantees. Statements like 'endorsed by [organization]' without verification, 'guaranteed to win', 'will definitely happen', or '100% certain' are prohibited. Campaign promises must be aspirational, not absolute.",
        "source": "FEC Regulations - False Claims",
        "category": "false_claims",
    },
    {
        "content": "PROHIBITED: Coercive language regarding support or donations. Phrases like 'vote for us or else', 'if you don't support us', 'you must donate', or 'failure to support' are prohibited. All supporter engagement must be voluntary and respectful.",
        "source": "FEC Regulations - Anti-Coercion",
        "category": "coercion",
    },
    {
        "content": "PROHIBITED: Providing medical advice or treatment recommendations. The campaign bot must never suggest medications, treatments, or make health claims. Phrases like 'you should take', 'this treatment will cure', or 'I recommend this medication' are strictly prohibited.",
        "source": "FEC Regulations - Medical Advice",
        "category": "medical_advice",
    },
]


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


async def ingest_fec_prohibited(weaviate: WeaviateManager) -> dict:
    """
    Ingest FEC Prohibited phrases - MANDATORY for compliance.
    
    This collection is REQUIRED for the system to operate.
    """
    stats = {"docs": 0}
    
    logger.info("\n  FEC Prohibited Phrases (MANDATORY)")
    
    count = await weaviate.get_collection_count("FECProhibited")
    if count > 0:
        logger.info(f"    FECProhibited already has {count} documents")
        stats["docs"] = count
        return stats
    
    added = 0
    for item in FEC_PROHIBITED_DATA:
        success = await weaviate.add_document(
            collection_name="FECProhibited",
            content=item["content"],
            source=item["source"],
            category=item.get("category", ""),
        )
        if success:
            added += 1
    
    logger.info(f"    Added {added} FEC prohibited phrases")
    stats["docs"] = added
    return stats


async def ingest_fec_from_files(weaviate: WeaviateManager, data_dir: str, 
                                 chunk_size: int, overlap: int) -> dict:
    """Ingest additional FEC data from files if available"""
    stats = {"docs": 0, "chunks": 0}
    
    fec_dir = os.path.join(data_dir, "fec_prohibited")
    
    if not os.path.exists(fec_dir):
        return stats
    
    files = [f for f in os.listdir(fec_dir) 
            if f.endswith(('.pdf', '.docx', '.txt'))]
    
    if not files:
        return stats
    
    logger.info(f"    Found {len(files)} additional FEC files")
    
    for file_name in files:
        file_path = os.path.join(fec_dir, file_name)
        try:
            if file_path.endswith('.pdf'):
                text = extract_text_from_pdf(file_path)
            elif file_path.endswith('.docx'):
                text = extract_text_from_docx(file_path)
            else:
                text = extract_text_from_txt(file_path)
            
            chunks = chunk_text(text, chunk_size, overlap)
            
            for i, chunk in enumerate(chunks):
                success = await weaviate.add_document(
                    collection_name="FECProhibited",
                    content=chunk,
                    source=f"{file_name} (chunk {i+1})",
                    category="fec_regulation",
                )
                if success:
                    stats["chunks"] += 1
            
            stats["docs"] += 1
            logger.info(f"      + {file_name}: {len(chunks)} chunks")
            
        except Exception as e:
            logger.warning(f"      Failed to process {file_name}: {e}")
    
    return stats


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


async def initialize_database(db_path: str) -> bool:
    """Initialize SQLite database with all required tables."""
    logger.info(f"\n  Initializing SQLite database: {db_path}")
    
    try:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        db = DatabaseManager(db_path)
        await db.initialize()
        
        logger.info("    Database initialized successfully")
        logger.info("    Tables created:")
        logger.info("      - user_consent")
        logger.info("      - interactions")
        logger.info("      - callback_requests")
        logger.info("      - new_questions")
        logger.info("      - conversation_history")
        logger.info("      - request_logs")
        logger.info("      - model_performance")
        logger.info("      - volunteers")
        logger.info("      - donation_interests")
        logger.info("      - compliance_log")
        
        return True
        
    except Exception as e:
        logger.error(f"    Failed to initialize database: {e}")
        return False


async def run_unified_ingestion(data_dir: str, chunk_size: int = 1000, 
                                 overlap: int = 200, skip_db: bool = False,
                                 db_path: str = "data/brandonbot.db"):
    """Run complete ingestion pipeline"""
    logger.info("=" * 70)
    logger.info("BrandonBot Unified Setup & Ingestion")
    logger.info("=" * 70)
    logger.info(f"Source: {data_dir}")
    logger.info(f"Database: {db_path}")
    logger.info(f"Chunk size: {chunk_size} | Overlap: {overlap}")
    logger.info("")
    
    if not skip_db:
        logger.info("PHASE 0: Database Initialization")
        logger.info("-" * 40)
        db_ok = await initialize_database(db_path)
        if not db_ok:
            logger.error("Database initialization failed!")
            return
    
    weaviate = await connect_or_start_weaviate("./weaviate_data")
    
    try:
        logger.info("\nPHASE 1: FEC Compliance Data (MANDATORY)")
        logger.info("-" * 40)
        fec_stats = await ingest_fec_prohibited(weaviate)
        fec_file_stats = await ingest_fec_from_files(weaviate, data_dir, chunk_size, overlap)
        
        fec_count = await weaviate.get_collection_count("FECProhibited")
        if fec_count == 0:
            logger.error("CRITICAL: FECProhibited collection is empty!")
            logger.error("System cannot operate without FEC compliance data.")
            return
        
        logger.info(f"    FECProhibited: {fec_count} total documents")
        
        logger.info("\nPHASE 2: Standard Collections")
        logger.info("-" * 40)
        std_stats = await ingest_standard_collections(weaviate, data_dir, chunk_size, overlap)
        
        logger.info("\nPHASE 3: Marketing Gurus (Smart Parsing)")
        logger.info("-" * 40)
        guru_stats = await ingest_market_gurus(weaviate, data_dir)
        
        logger.info("\n" + "=" * 70)
        logger.info("SETUP COMPLETE")
        logger.info("=" * 70)
        
        logger.info("\nDatabase:")
        if not skip_db:
            logger.info(f"  SQLite: {db_path} (initialized)")
        else:
            logger.info("  SQLite: skipped (--skip-db)")
        
        logger.info("\nWeaviate Collections:")
        all_collections = ["FECProhibited", "BrandonPlatform", "PartyPlatform", 
                          "PreviousQA", "MarketGurus"]
        for collection in all_collections:
            count = await weaviate.get_collection_count(collection)
            status = "OK" if count > 0 else "EMPTY"
            mandatory = " (MANDATORY)" if collection == "FECProhibited" else ""
            logger.info(f"  {collection}: {count} chunks [{status}]{mandatory}")
        
        logger.info("\nIngestion Summary:")
        logger.info(f"  FEC Prohibited: {fec_stats['docs']} built-in + {fec_file_stats.get('chunks', 0)} from files")
        logger.info(f"  Standard docs: {std_stats['docs']} files, {std_stats['chunks']} chunks")
        logger.info(f"  MarketGurus:   {guru_stats['files']} files, {guru_stats['chunks']} chunks")
        
        logger.info("\nSystem Status:")
        logger.info("  [OK] SQLite database ready" if not skip_db else "  [--] SQLite database skipped")
        logger.info(f"  [OK] FEC RAG ready ({fec_count} documents)")
        logger.info("  [OK] Ready for operations!")
        
    finally:
        if hasattr(weaviate, 'client') and weaviate.client:
            weaviate.client.close()


def main():
    parser = argparse.ArgumentParser(
        description="Unified document ingestion and setup for BrandonBot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script sets up BrandonBot for operation:
1. Initializes SQLite database (data/brandonbot.db)
2. Loads FEC prohibited phrases (MANDATORY)
3. Ingests document collections into Weaviate

Examples:
  python ingest_all.py                           # Default setup
  python ingest_all.py documents/                # Custom documents directory
  python ingest_all.py --chunk-size 256          # Custom chunk size
  python ingest_all.py --skip-db                 # Skip database initialization

Expected directory structure:
  documents/
    brandon_platform/    # Brandon's policy documents
    party_platforms/     # Republican/Independent platform docs
    previous_qa/         # Verified Q&A pairs
    market_gurus/        # Marketing/copywriting guidance
    fec_prohibited/      # Additional FEC compliance docs (optional)
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
    
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="Skip SQLite database initialization"
    )
    
    parser.add_argument(
        "--db-path",
        default="data/brandonbot.db",
        help="Path to SQLite database (default: data/brandonbot.db)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_directory):
        logger.warning(f"Documents directory not found: {args.data_directory}")
        logger.info("Will create FEC data and initialize database only.")
        os.makedirs(args.data_directory, exist_ok=True)
    
    asyncio.run(run_unified_ingestion(
        args.data_directory,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        skip_db=args.skip_db,
        db_path=args.db_path
    ))


if __name__ == "__main__":
    main()
