#!/usr/bin/env python3
"""
Document ingestion script for BrandonBot knowledge base
Processes PDFs, DOCX, and TXT files into the 4-tier collection system
"""
import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path
from typing import List, Dict
import pypdf
import docx
import weaviate as weaviate_client
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weaviate_manager import WeaviateManager

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

COLLECTION_MAP = {
    "brandon": "BrandonPlatform",
    "party": "PartyPlatform",
    "market": "MarketGurus",
    "qa": "PreviousQA"
}

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
    except Exception as e:
        logger.error(f"Error reading PDF {file_path}: {e}")
        return ""

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text.strip()
    except Exception as e:
        logger.error(f"Error reading DOCX {file_path}: {e}")
        return ""

def extract_text_from_txt(file_path: str) -> str:
    """Extract text from TXT file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except Exception as e:
        logger.error(f"Error reading TXT {file_path}: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks with smart boundary detection.
    Respects boundaries in priority order: sections > paragraphs > sentences > characters.
    
    Args:
        text: Text to chunk
        chunk_size: Target maximum chunk size in characters (must be > 0)
        overlap: Number of characters to overlap between chunks (must be >= 0 and < chunk_size)
        
    Returns:
        List of text chunks
        
    Raises:
        ValueError: If chunk_size or overlap parameters are invalid
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be < chunk_size ({chunk_size})")
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        
        if end >= len(text):
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break
        
        chunk_text_raw = text[start:end]
        best_break = None
        
        search_window = chunk_text_raw[int(chunk_size * 0.6):]
        
        section_pattern = r'\n\n+'
        section_matches = list(re.finditer(section_pattern, search_window))
        if section_matches:
            best_match = section_matches[0]
            best_break = int(chunk_size * 0.6) + best_match.end()
        
        if not best_break:
            para_pattern = r'\n'
            para_matches = list(re.finditer(para_pattern, search_window))
            if para_matches:
                best_match = para_matches[0]
                best_break = int(chunk_size * 0.6) + best_match.end()
        
        if not best_break:
            sentence_pattern = r'[.!?][\s\n]'
            sentence_matches = list(re.finditer(sentence_pattern, search_window))
            if sentence_matches:
                best_match = sentence_matches[0]
                best_break = int(chunk_size * 0.6) + best_match.end()
        
        if best_break:
            end = start + best_break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
    
    return chunks

async def ingest_file(weaviate: WeaviateManager, file_path: str, collection_key: str, 
                     category: str = "", chunk_size: int = 1000, overlap: int = 200) -> int:
    """Ingest a single file into specified collection"""
    file_ext = Path(file_path).suffix.lower()
    file_name = Path(file_path).name
    
    if file_ext == '.pdf':
        text = extract_text_from_pdf(file_path)
    elif file_ext == '.docx':
        text = extract_text_from_docx(file_path)
    elif file_ext == '.txt':
        text = extract_text_from_txt(file_path)
    else:
        logger.warning(f"Unsupported file type: {file_ext}")
        return 0
    
    if not text:
        logger.warning(f"No text extracted from {file_name}")
        return 0
    
    collection_name = COLLECTION_MAP.get(collection_key)
    if not collection_name:
        logger.error(f"Invalid collection key: {collection_key}")
        return 0
    
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    logger.info(f"  Processing {file_name}: {len(chunks)} chunks")
    
    success_count = 0
    for i, chunk in enumerate(chunks):
        success = await weaviate.add_document(
            collection_name=collection_name,
            content=chunk,
            source=file_name,
            category=category,
            metadata={"chunk_index": i, "total_chunks": len(chunks)}
        )
        if success:
            success_count += 1
    
    return success_count

async def ingest_directory(data_dir: str, chunk_size: int = 1000, overlap: int = 200):
    """Ingest all documents from structured directory"""
    logger.info("=" * 70)
    logger.info("BrandonBot Document Ingestion")
    logger.info("=" * 70)
    logger.info(f"Chunk size: {chunk_size} chars | Overlap: {overlap} chars")
    logger.info("")
    
    weaviate = WeaviateManager("./weaviate_data")
    try:
        logger.info("Connecting to existing Weaviate instance...")
        weaviate.client = await asyncio.to_thread(
            lambda: weaviate_client.connect_to_local(host="localhost", port=8079, grpc_port=50050)
        )
        logger.info("Connected to existing Weaviate instance")
    except Exception as e:
        logger.info(f"No existing instance found, starting embedded mode: {e}")
        await weaviate.initialize()
    
    expected_structure = {
        "brandon_platform": ("brandon", "Brandon's Platform"),
        "party_platforms": ("party", "Party Platforms"),
        "market_gurus": ("market", "Marketing Books"),
        "previous_qa": ("qa", "Previous Q&A")
    }
    
    total_docs = 0
    total_chunks = 0
    
    for dir_name, (collection_key, display_name) in expected_structure.items():
        dir_path = os.path.join(data_dir, dir_name)
        
        if not os.path.exists(dir_path):
            logger.info(f"⊘ {display_name}: Directory not found ({dir_name}/)")
            continue
        
        files = [f for f in os.listdir(dir_path) 
                if f.endswith(('.pdf', '.docx', '.txt'))]
        
        if not files:
            logger.info(f"⊘ {display_name}: No documents found")
            continue
        
        logger.info(f"📁 {display_name} ({len(files)} files)")
        
        for file_name in files:
            file_path = os.path.join(dir_path, file_name)
            chunks = await ingest_file(weaviate, file_path, collection_key, display_name, 
                                      chunk_size=chunk_size, overlap=overlap)
            if chunks > 0:
                total_docs += 1
                total_chunks += chunks
        
        logger.info("")
    
    logger.info("=" * 70)
    logger.info(f"✓ Ingestion Complete!")
    logger.info(f"  Documents processed: {total_docs}")
    logger.info(f"  Total chunks created: {total_chunks}")
    logger.info("=" * 70)
    logger.info("")
    
    for collection_key, display_name in expected_structure.values():
        collection_name = COLLECTION_MAP[collection_key]
        count = await weaviate.get_collection_count(collection_name)
        logger.info(f"  {collection_name}: {count} chunks")
    
    if hasattr(weaviate, 'client') and weaviate.client:
        weaviate.client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest documents into BrandonBot knowledge base with smart chunking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default chunking (1000 chars, 200 overlap)
  python ingest_documents.py documents/
  
  # Optimized small chunks (128 chars, 51 overlap = 40%)
  python ingest_documents.py documents/ --chunk-size 128 --overlap 51
  
  # Custom configuration
  python ingest_documents.py documents/ --chunk-size 256 --overlap 100
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
        logger.info("")
        logger.info("Expected directory structure:")
        logger.info("  documents/")
        logger.info("    brandon_platform/     (Brandon's own statements & platform)")
        logger.info("    party_platforms/       (Republican, RNC platforms)")
        logger.info("    market_gurus/          (Marketing/copywriting books)")
        logger.info("    previous_qa/           (Historical Q&A)")
        sys.exit(1)
    
    asyncio.run(ingest_directory(args.data_directory, chunk_size=args.chunk_size, overlap=args.overlap))
