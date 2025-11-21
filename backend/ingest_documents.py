#!/usr/bin/env python3
"""
Document ingestion script for BrandonBot knowledge base
Processes PDFs, DOCX, and TXT files into the 4-tier collection system
"""
import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import List, Dict
import pypdf
import docx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weaviate_manager import WeaviateManager

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

COLLECTION_MAP = {
    "brandon": "BrandonPlatform",
    "party": "PartyPlatform",
    "market": "MarketGurus",
    "internet": "InternetSources",
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
    """Split text into overlapping chunks"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            cutoff = max(last_period, last_newline)
            if cutoff > chunk_size * 0.7:
                end = start + cutoff + 1
                chunk = text[start:end]
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return chunks

async def ingest_file(weaviate: WeaviateManager, file_path: str, collection_key: str, 
                     category: str = "") -> int:
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
    
    chunks = chunk_text(text)
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

async def ingest_directory(data_dir: str):
    """Ingest all documents from structured directory"""
    logger.info("=" * 70)
    logger.info("BrandonBot Document Ingestion")
    logger.info("=" * 70)
    logger.info("")
    
    weaviate = WeaviateManager("./weaviate_data")
    await weaviate.initialize()
    
    expected_structure = {
        "brandon_platform": ("brandon", "Brandon's Platform"),
        "party_platforms": ("party", "Party Platforms"),
        "market_gurus": ("market", "Marketing Books"),
        "internet_sources": ("internet", "Internet Research"),
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
            chunks = await ingest_file(weaviate, file_path, collection_key, display_name)
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
    data_directory = sys.argv[1] if len(sys.argv) > 1 else "./documents"
    
    if not os.path.exists(data_directory):
        logger.error(f"Data directory not found: {data_directory}")
        logger.info("")
        logger.info("Expected directory structure:")
        logger.info("  documents/")
        logger.info("    brandon_platform/     (Brandon's own statements & platform)")
        logger.info("    party_platforms/       (Republican, RNC platforms)")
        logger.info("    market_gurus/          (Marketing/copywriting books)")
        logger.info("    internet_sources/      (Research articles)")
        logger.info("    previous_qa/           (Historical Q&A)")
        sys.exit(1)
    
    asyncio.run(ingest_directory(data_directory))
