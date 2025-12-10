#!/usr/bin/env python3
"""
Download models for BrandonBot Output Validator

Models to download:
1. Llama-Guard-3-1B (ONNX) - Safety/ethics classification (1B params)
2. Phi-3 Mini ONNX - General validation tasks
3. DistilBERT (optional) - Fast intent classification

Uses huggingface_hub Python API for reliable downloads.
"""
import os
import sys
import shutil
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

MODELS = {
    "llama-guard-1b": {
        "repo_id": "onnx-community/Llama-Guard-3-1B",
        "local_dir": "./backend/llama_guard_model",
        "description": "Llama Guard 3 1B ONNX - Content safety classification",
        "size": "~2GB",
        "files": [
            "onnx/model.onnx",
            "onnx/model.onnx_data",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "config.json",
        ]
    },
    "phi3": {
        "repo_id": "microsoft/Phi-3-mini-4k-instruct-onnx",
        "local_dir": "./backend/phi3_model",
        "description": "Phi-3 Mini ONNX INT4 - General validation",
        "size": "~2GB",
        "subdir": "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4",
        "files": [
            "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/phi3-mini-4k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx",
            "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/phi3-mini-4k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx.data",
            "genai_config.json",
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "config.json",
            "configuration_phi3.py",
        ]
    }
}


def check_model_exists(model_key: str) -> bool:
    """Check if model is already downloaded."""
    config = MODELS[model_key]
    local_dir = config["local_dir"]
    
    if not os.path.exists(local_dir):
        return False
    
    if model_key == "phi3":
        required = ["genai_config.json", "phi3-mini-4k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx"]
    elif model_key == "llama-guard-1b":
        required = ["config.json"]
        onnx_dir = os.path.join(local_dir, "onnx")
        if os.path.exists(onnx_dir):
            if os.path.exists(os.path.join(onnx_dir, "model.onnx")):
                required = ["config.json"]
        else:
            if os.path.exists(os.path.join(local_dir, "model.onnx")):
                required = ["config.json"]
    else:
        required = ["config.json"]
    
    for req in required:
        if not os.path.exists(os.path.join(local_dir, req)):
            return False
    
    return True


def download_model(model_key: str, force: bool = False) -> bool:
    """Download a specific model using huggingface_hub."""
    if model_key not in MODELS:
        logger.error(f"Unknown model: {model_key}")
        logger.info(f"Available models: {list(MODELS.keys())}")
        return False
    
    config = MODELS[model_key]
    
    if not force and check_model_exists(model_key):
        logger.info(f"✓ {model_key} already downloaded at {config['local_dir']}")
        return True
    
    logger.info("=" * 70)
    logger.info(f"Downloading: {config['description']}")
    logger.info("=" * 70)
    logger.info(f"Repository: {config['repo_id']}")
    logger.info(f"Size: {config['size']}")
    logger.info(f"Target: {config['local_dir']}")
    logger.info("")
    
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        logger.info("Installing huggingface_hub...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=True)
        from huggingface_hub import hf_hub_download, snapshot_download
    
    os.makedirs(config["local_dir"], exist_ok=True)
    
    try:
        for file_path in config.get("files", []):
            logger.info(f"Downloading: {file_path}")
            hf_hub_download(
                repo_id=config["repo_id"],
                filename=file_path,
                local_dir=config["local_dir"],
                local_dir_use_symlinks=False
            )
        
        if "subdir" in config:
            source_path = os.path.join(config["local_dir"], config["subdir"])
            if os.path.exists(source_path):
                logger.info("")
                logger.info("Moving nested files to root directory...")
                for item in os.listdir(source_path):
                    src = os.path.join(source_path, item)
                    dst = os.path.join(config["local_dir"], item)
                    if not os.path.exists(dst):
                        shutil.move(src, dst)
                        logger.info(f"  Moved: {item}")
        
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"✓ {model_key} downloaded successfully!")
        logger.info("=" * 70)
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to download {model_key}: {e}")
        return False


def list_models():
    """List all available models and their status."""
    logger.info("=" * 70)
    logger.info("Available Models for BrandonBot Output Validator")
    logger.info("=" * 70)
    
    for key, config in MODELS.items():
        status = "✓ Downloaded" if check_model_exists(key) else "○ Not downloaded"
        logger.info(f"\n{key}:")
        logger.info(f"  Description: {config['description']}")
        logger.info(f"  Size: {config['size']}")
        logger.info(f"  Status: {status}")


def main():
    parser = argparse.ArgumentParser(description="Download models for BrandonBot")
    parser.add_argument("command", choices=["download", "list", "all"], 
                       help="Command: download <model>, list, or all")
    parser.add_argument("model", nargs="?", help="Model key to download")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_models()
        return 0
    
    if args.command == "all":
        success = True
        for model_key in MODELS.keys():
            if not download_model(model_key, args.force):
                success = False
        return 0 if success else 1
    
    if args.command == "download":
        if not args.model:
            logger.error("Please specify a model to download")
            logger.info(f"Available models: {list(MODELS.keys())}")
            return 1
        return 0 if download_model(args.model, args.force) else 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
