#!/usr/bin/env python3
"""
BrandonBot SLM Model Downloader

Downloads and caches all 4 SLM models required for the 6-safeguard system:
1. ME2-BERT (ethics) - ~420MB
2. MS-MARCO cross-encoder (intent) - ~120MB  
3. DeBERTa-PII (PII detection) - ~550MB
4. BERT-tiny (confidence) - ~15MB

Total: ~1.1GB disk space, ~2-3GB RAM when running

Usage:
    python download_models.py [--cache-dir PATH] [--verify-only]
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

MODELS = {
    "ethics": {
        "name": "ME2-BERT (Ethics)",
        "model_id": "bert-base-uncased",
        "type": "transformers",
        "size_mb": 420,
        "description": "Moral foundations classification for ethics checking"
    },
    "intent": {
        "name": "MS-MARCO Cross-Encoder (Intent)",
        "model_id": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "type": "sentence-transformers",
        "size_mb": 120,
        "description": "Query-response semantic alignment scoring"
    },
    "pii": {
        "name": "DeBERTa-PII (PII Detection)",
        "model_id": "lakshyakh93/deberta_finetuned_pii",
        "type": "transformers",
        "size_mb": 550,
        "description": "Named entity recognition for PII detection"
    },
    "confidence": {
        "name": "BERT-tiny (Confidence)",
        "model_id": "prajjwal1/bert-tiny",
        "type": "transformers",
        "size_mb": 15,
        "description": "Lightweight model for confidence verification"
    }
}

def check_dependencies():
    """Check that required packages are installed."""
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


def get_cache_dir(custom_dir=None):
    """
    Get the cache directory for models.
    
    Priority (matches ov_slm_models.get_model_cache_dir):
    1. Custom directory (CLI argument)
    2. MODEL_CACHE_DIR env var (project-specific)
    3. HF_HOME env var (HuggingFace standard, maps to <hf_home>/hub)
    4. TRANSFORMERS_CACHE env var (transformers standard)
    5. Default: ~/.cache/huggingface
    """
    if custom_dir:
        cache_dir = Path(custom_dir)
    elif os.environ.get("MODEL_CACHE_DIR"):
        cache_dir = Path(os.environ["MODEL_CACHE_DIR"])
    elif os.environ.get("HF_HOME"):
        cache_dir = Path(os.environ["HF_HOME"]) / "hub"
    elif os.environ.get("TRANSFORMERS_CACHE"):
        cache_dir = Path(os.environ["TRANSFORMERS_CACHE"])
    else:
        cache_dir = Path.home() / ".cache" / "huggingface"
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_transformers_model(model_id, cache_dir):
    """Download a transformers model."""
    from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification
    
    print(f"    Downloading tokenizer...")
    AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    
    print(f"    Downloading model weights...")
    try:
        AutoModelForTokenClassification.from_pretrained(model_id, cache_dir=cache_dir)
    except:
        AutoModel.from_pretrained(model_id, cache_dir=cache_dir)
    
    return True


def download_sentence_transformer(model_id, cache_dir):
    """Download a sentence-transformers model."""
    from sentence_transformers import CrossEncoder
    import sentence_transformers
    
    print(f"    Downloading cross-encoder...")
    
    version = tuple(int(x) for x in sentence_transformers.__version__.split('.')[:2])
    if version >= (2, 7):
        CrossEncoder(model_id, cache_folder=str(cache_dir))
    else:
        old_cache = os.environ.get("SENTENCE_TRANSFORMERS_HOME")
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(cache_dir)
        try:
            CrossEncoder(model_id)
        finally:
            if old_cache:
                os.environ["SENTENCE_TRANSFORMERS_HOME"] = old_cache
            else:
                os.environ.pop("SENTENCE_TRANSFORMERS_HOME", None)
    
    return True


def verify_model(model_key, model_info, cache_dir):
    """Verify a model is properly cached and loadable."""
    try:
        if model_info["type"] == "transformers":
            from transformers import AutoTokenizer, AutoModel
            AutoTokenizer.from_pretrained(model_info["model_id"], cache_dir=cache_dir, local_files_only=True)
            return True
        elif model_info["type"] == "sentence-transformers":
            from sentence_transformers import CrossEncoder
            import sentence_transformers
            
            version = tuple(int(x) for x in sentence_transformers.__version__.split('.')[:2])
            if version >= (2, 7):
                CrossEncoder(model_info["model_id"], cache_folder=str(cache_dir), local_files_only=True)
            else:
                old_cache = os.environ.get("SENTENCE_TRANSFORMERS_HOME")
                os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(cache_dir)
                try:
                    CrossEncoder(model_info["model_id"])
                finally:
                    if old_cache:
                        os.environ["SENTENCE_TRANSFORMERS_HOME"] = old_cache
                    else:
                        os.environ.pop("SENTENCE_TRANSFORMERS_HOME", None)
            return True
    except Exception as e:
        print(f"    Verification failed: {e}")
        return False
    return False


def get_cache_size(cache_dir):
    """Get total size of cache directory in MB."""
    total = 0
    cache_path = Path(cache_dir)
    if cache_path.exists():
        for f in cache_path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    return total / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(description="Download BrandonBot SLM models")
    parser.add_argument("--cache-dir", help="Custom cache directory for models")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing models")
    parser.add_argument("--model", choices=list(MODELS.keys()), help="Download specific model only")
    args = parser.parse_args()
    
    print("=" * 60)
    print("BrandonBot SLM Model Downloader")
    print("=" * 60)
    
    print("\nChecking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    cache_dir = get_cache_dir(args.cache_dir)
    print(f"\nCache directory: {cache_dir}")
    
    models_to_process = {args.model: MODELS[args.model]} if args.model else MODELS
    
    total_size = sum(m["size_mb"] for m in models_to_process.values())
    print(f"\nModels to {'verify' if args.verify_only else 'download'}: {len(models_to_process)}")
    print(f"Estimated total size: ~{total_size}MB")
    print("-" * 60)
    
    results = {}
    
    for key, info in models_to_process.items():
        print(f"\n[{key.upper()}] {info['name']}")
        print(f"  Model ID: {info['model_id']}")
        print(f"  Type: {info['type']}")
        print(f"  Size: ~{info['size_mb']}MB")
        
        if args.verify_only:
            print("  Verifying...")
            success = verify_model(key, info, cache_dir)
        else:
            print("  Downloading...")
            try:
                if info["type"] == "transformers":
                    success = download_transformers_model(info["model_id"], cache_dir)
                else:
                    success = download_sentence_transformer(info["model_id"], cache_dir)
                
                if success:
                    print("  Verifying...")
                    success = verify_model(key, info, cache_dir)
            except Exception as e:
                print(f"  ERROR: {e}")
                success = False
        
        results[key] = success
        status = "OK" if success else "FAILED"
        print(f"  Status: {status}")
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    success_count = sum(1 for v in results.values() if v)
    print(f"\nModels ready: {success_count}/{len(results)}")
    
    actual_size = get_cache_size(cache_dir)
    print(f"Cache size: {actual_size:.1f}MB")
    
    for key, success in results.items():
        status = "Ready" if success else "MISSING"
        print(f"  {MODELS[key]['name']}: {status}")
    
    if success_count == len(results):
        print("\nAll models ready! You can now run the full test suite:")
        print("  cd backend && python -m pytest tests/test_ov_*.py tests/test_pq.py -v")
        return 0
    else:
        print("\nSome models failed to download. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
