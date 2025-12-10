#!/usr/bin/env python3
"""
BrandonBot Model Downloader

Downloads and sets up all models required for self-hosted BrandonBot:

1. Ollama + Llama 3.2 (LLM Judge) - Required for validation
2. HuggingFace SLM Models (Safeguards) - Required for operation

After running this script, all models should be cached and ready.

Usage:
    python download_models.py                    # Download all models
    python download_models.py --ollama-only      # Just Ollama/Llama setup
    python download_models.py --slm-only         # Just SLM safeguard models
    python download_models.py --verify-only      # Check what's installed
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path

OLLAMA_MODEL = "llama3.2:3b"

SLM_MODELS = {
    "embeddings": {
        "name": "all-MiniLM-L6-v2 (Embeddings)",
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "type": "sentence-transformers-embeddings",
        "size_mb": 90,
        "description": "Text embeddings for semantic search and RAG",
        "required": True,
    },
    "cross_encoder": {
        "name": "MS-MARCO Cross-Encoder (Intent/Vagueness)",
        "model_id": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "type": "cross-encoder",
        "size_mb": 120,
        "description": "Query-response semantic alignment scoring",
        "required": True,
    },
    "emotion": {
        "name": "Emotion Classifier (Frustration)",
        "model_id": "j-hartmann/emotion-english-distilroberta-base",
        "type": "pipeline",
        "size_mb": 320,
        "description": "7-emotion classification for frustration detection",
        "required": True,
    },
    "ethics": {
        "name": "ME2-BERT (Ethics)",
        "model_id": "lorenzozan/ME2-BERT",
        "type": "transformers",
        "size_mb": 420,
        "description": "Moral foundations classification for ethics checking",
        "required": True,
    },
    "pii": {
        "name": "DeBERTa-PII (PII Detection)",
        "model_id": "lakshyakh93/deberta_finetuned_pii",
        "type": "ner",
        "size_mb": 550,
        "description": "Named entity recognition for PII detection",
        "required": True,
    },
    "confidence": {
        "name": "BERT-tiny (Confidence)",
        "model_id": "prajjwal1/bert-tiny",
        "type": "transformers",
        "size_mb": 15,
        "description": "Lightweight model for confidence verification",
        "required": True,
    },
}


def check_ollama_installed():
    """Check if Ollama is installed and accessible."""
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            version = result.stdout.strip() or result.stderr.strip()
            print(f"  Ollama: {version}")
            return True
    except FileNotFoundError:
        pass
    except subprocess.TimeoutExpired:
        print("  Ollama: timeout checking version")
    except Exception as e:
        print(f"  Ollama: error - {e}")
    return False


def check_ollama_running():
    """Check if Ollama server is running."""
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def pull_ollama_model(model_name):
    """Pull an Ollama model."""
    print(f"    Pulling {model_name}...")
    try:
        result = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=False,
            timeout=1800
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"    Timeout pulling {model_name}")
        return False
    except Exception as e:
        print(f"    Error: {e}")
        return False


def check_ollama_model_exists(model_name):
    """Check if a specific Ollama model is already downloaded."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return model_name.split(":")[0] in result.stdout
    except Exception:
        pass
    return False


def verify_ollama_model(model_name):
    """Verify an Ollama model works by running a quick test."""
    try:
        result = subprocess.run(
            ["ollama", "run", model_name, "Say 'OK'"],
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.returncode == 0 and len(result.stdout.strip()) > 0
    except Exception:
        return False


def setup_ollama():
    """Set up Ollama and pull the LLM Judge model."""
    print("\n" + "=" * 60)
    print("Ollama Setup (LLM Judge for Validation)")
    print("=" * 60)
    
    if not check_ollama_installed():
        print("\n  Ollama is not installed.")
        print("\n  Installation instructions:")
        print("    Linux:   curl -fsSL https://ollama.com/install.sh | sh")
        print("    macOS:   brew install ollama")
        print("    Windows: Download from https://ollama.com/download")
        print("\n  After installing, run: ollama serve")
        return False
    
    if not check_ollama_running():
        print("\n  Ollama server is not running.")
        print("  Start it with: ollama serve")
        print("  (or: systemctl start ollama)")
        return False
    
    print(f"\n  Target model: {OLLAMA_MODEL}")
    
    if check_ollama_model_exists(OLLAMA_MODEL):
        print("  Model already downloaded.")
    else:
        print("  Model not found, downloading...")
        if not pull_ollama_model(OLLAMA_MODEL):
            print("  Failed to pull model.")
            return False
    
    print("  Verifying model...")
    if verify_ollama_model(OLLAMA_MODEL):
        print("  Model verified OK!")
        return True
    else:
        print("  Model verification failed.")
        return False


def check_slm_dependencies():
    """Check that required packages for SLM models are installed."""
    missing = []
    
    try:
        import torch
        print(f"  torch: {torch.__version__}")
    except ImportError:
        missing.append("torch")
    
    try:
        import transformers
        print(f"  transformers: {transformers.__version__}")
    except ImportError:
        missing.append("transformers")
    
    try:
        import sentence_transformers
        print(f"  sentence-transformers: {sentence_transformers.__version__}")
    except ImportError:
        missing.append("sentence-transformers")
    
    try:
        from huggingface_hub import snapshot_download
        print(f"  huggingface-hub: installed")
    except ImportError:
        missing.append("huggingface-hub")
    
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True


def download_embeddings_model(model_id):
    """Download a sentence-transformers embeddings model."""
    from sentence_transformers import SentenceTransformer
    
    print(f"    Downloading embeddings model...")
    SentenceTransformer(model_id)
    return True


def download_cross_encoder(model_id):
    """Download a cross-encoder model."""
    from sentence_transformers import CrossEncoder
    
    print(f"    Downloading cross-encoder...")
    CrossEncoder(model_id)
    return True


def download_pipeline_model(model_id):
    """Download a transformers pipeline model."""
    from transformers import pipeline
    
    print(f"    Downloading pipeline model...")
    pipeline("text-classification", model=model_id)
    return True


def download_transformers_model(model_id):
    """Download a transformers model to default cache."""
    from transformers import AutoTokenizer, AutoModel
    
    print(f"    Downloading tokenizer...")
    AutoTokenizer.from_pretrained(model_id)
    
    print(f"    Downloading model weights...")
    AutoModel.from_pretrained(model_id)
    
    return True


def download_ner_model(model_id):
    """Download a NER (token classification) model."""
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    
    print(f"    Downloading tokenizer...")
    AutoTokenizer.from_pretrained(model_id)
    
    print(f"    Downloading NER model weights...")
    AutoModelForTokenClassification.from_pretrained(model_id)
    
    return True


def verify_slm_model(model_key, model_info):
    """Verify an SLM model is properly cached and loadable."""
    try:
        model_type = model_info["type"]
        model_id = model_info["model_id"]
        
        if model_type == "sentence-transformers-embeddings":
            from sentence_transformers import SentenceTransformer
            SentenceTransformer(model_id)
            return True
        elif model_type == "cross-encoder":
            from sentence_transformers import CrossEncoder
            CrossEncoder(model_id)
            return True
        elif model_type == "pipeline":
            from transformers import pipeline
            pipeline("text-classification", model=model_id)
            return True
        elif model_type == "transformers":
            from transformers import AutoTokenizer, AutoModel
            AutoTokenizer.from_pretrained(model_id)
            AutoModel.from_pretrained(model_id)
            return True
        elif model_type == "ner":
            from transformers import AutoTokenizer, AutoModelForTokenClassification
            AutoTokenizer.from_pretrained(model_id)
            AutoModelForTokenClassification.from_pretrained(model_id)
            return True
            
    except Exception as e:
        print(f"    Verification failed: {e}")
        return False
    
    return False


def setup_slm_models(verify_only=False, model_filter=None):
    """Set up SLM safeguard models."""
    print("\n" + "=" * 60)
    print("SLM Safeguard Models")
    print("=" * 60)
    
    print("\nChecking dependencies...")
    if not check_slm_dependencies():
        return False
    
    models_to_process = {model_filter: SLM_MODELS[model_filter]} if model_filter else SLM_MODELS
    
    total_size = sum(m["size_mb"] for m in models_to_process.values())
    required_count = sum(1 for m in models_to_process.values() if m.get("required", False))
    print(f"\nModels to {'verify' if verify_only else 'download'}: {len(models_to_process)} ({required_count} required)")
    print(f"Estimated total size: ~{total_size}MB")
    print("-" * 60)
    
    results = {}
    
    for key, info in models_to_process.items():
        required = info.get("required", False)
        required_str = " [REQUIRED]" if required else ""
        
        print(f"\n[{key.upper()}] {info['name']}{required_str}")
        print(f"  Model ID: {info['model_id']}")
        print(f"  Type: {info['type']}")
        print(f"  Size: ~{info['size_mb']}MB")
        
        if verify_only:
            print("  Verifying...")
            success = verify_slm_model(key, info)
        else:
            print("  Downloading...")
            try:
                model_type = info["type"]
                if model_type == "sentence-transformers-embeddings":
                    success = download_embeddings_model(info["model_id"])
                elif model_type == "cross-encoder":
                    success = download_cross_encoder(info["model_id"])
                elif model_type == "pipeline":
                    success = download_pipeline_model(info["model_id"])
                elif model_type == "transformers":
                    success = download_transformers_model(info["model_id"])
                elif model_type == "ner":
                    success = download_ner_model(info["model_id"])
                else:
                    print(f"  Unknown type: {model_type}")
                    success = False
                
                if success:
                    print("  Verifying...")
                    success = verify_slm_model(key, info)
            except Exception as e:
                print(f"  ERROR: {e}")
                success = False
        
        results[key] = success
        status = "OK" if success else "FAILED"
        print(f"  Status: {status}")
    
    required_ok = all(
        results.get(k, False) 
        for k, v in SLM_MODELS.items() 
        if v.get("required", False) and (model_filter is None or k == model_filter)
    )
    
    return required_ok


def main():
    parser = argparse.ArgumentParser(
        description="Download BrandonBot models for self-hosted deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script downloads all models required for BrandonBot:

SLM Models (Safeguards):
  - all-MiniLM-L6-v2: Text embeddings for RAG
  - ms-marco-MiniLM: Cross-encoder for intent/vagueness
  - emotion-distilroberta: Frustration detection
  - ME2-BERT: Ethics classification
  - deberta-pii: PII detection
  - bert-tiny: Confidence verification

Ollama (LLM Judge):
  - llama3.2:3b: Local LLM for validation scoring

After running this script:
  1. Run ingest_all.py to set up Weaviate and SQLite
  2. System is ready for operations
        """
    )
    parser.add_argument(
        "--ollama-only",
        action="store_true",
        help="Only set up Ollama and Llama model"
    )
    parser.add_argument(
        "--slm-only",
        action="store_true",
        help="Only download SLM safeguard models"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing models, don't download"
    )
    parser.add_argument(
        "--model",
        choices=list(SLM_MODELS.keys()),
        help="Download specific SLM model only"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("BrandonBot Model Downloader")
    print("=" * 60)
    
    results = {}
    
    if not args.slm_only:
        results["ollama"] = setup_ollama()
    
    if not args.ollama_only:
        results["slm"] = setup_slm_models(
            verify_only=args.verify_only,
            model_filter=args.model
        )
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    if "ollama" in results:
        status = "Ready" if results["ollama"] else "FAILED (optional for Replit)"
        print(f"  Ollama ({OLLAMA_MODEL}): {status}")
    
    if "slm" in results:
        status = "Ready" if results["slm"] else "FAILED"
        print(f"  SLM Safeguard Models: {status}")
    
    slm_ok = results.get("slm", True)
    
    if slm_ok:
        print("\nAll required models ready!")
        print("\nNext steps:")
        print("  1. Run: python ingest_all.py")
        print("  2. System will be ready for operations")
        
        if "ollama" in results and results["ollama"]:
            print("\nTo run validation with local LLM judge:")
            print("  export USE_LOCAL_JUDGE=true")
            print("  cd backend/validation && python validator.py")
        return 0
    else:
        print("\nSome required components failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
