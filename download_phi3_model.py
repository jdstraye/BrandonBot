#!/usr/bin/env python3
"""
Download Phi-3 Mini ONNX model (CPU-optimized) from Hugging Face
This is a one-time setup script.

Uses hf_hub_download Python API instead of deprecated huggingface-cli.
"""
import os
import sys
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

PHI3_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct-onnx"
MODEL_PATH = "./backend/phi3_model"
CPU_INT4_SUBDIR = "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"

REQUIRED_FILES = [
    f"{CPU_INT4_SUBDIR}/phi3-mini-4k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx",
    f"{CPU_INT4_SUBDIR}/phi3-mini-4k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx.data",
    "genai_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "config.json",
    "configuration_phi3.py",
]

def download_model():
    """Download the Phi-3 ONNX model optimized for CPU using Python API"""
    logger.info("=" * 70)
    logger.info("Downloading Phi-3 Mini ONNX Model (CPU-Optimized, INT4 Quantized)")
    logger.info("=" * 70)
    logger.info(f"Model: {PHI3_MODEL_ID}")
    logger.info(f"Size: ~2GB (INT4 quantized)")
    logger.info(f"Target: {MODEL_PATH}")
    logger.info("")
    
    final_onnx = os.path.join(MODEL_PATH, "phi3-mini-4k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx")
    final_data = os.path.join(MODEL_PATH, "phi3-mini-4k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx.data")
    final_config = os.path.join(MODEL_PATH, "genai_config.json")
    
    if os.path.exists(final_onnx) and os.path.exists(final_data) and os.path.exists(final_config):
        logger.info("✓ Model already downloaded!")
        return True
    
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        logger.info("Installing huggingface_hub...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=True)
        from huggingface_hub import hf_hub_download, snapshot_download
    
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    logger.info("Downloading model files (this may take 5-10 minutes)...")
    logger.info("")
    
    try:
        for file_path in REQUIRED_FILES:
            logger.info(f"Downloading: {file_path}")
            hf_hub_download(
                repo_id=PHI3_MODEL_ID,
                filename=file_path,
                local_dir=MODEL_PATH,
                local_dir_use_symlinks=False
            )
        
        source_path = os.path.join(MODEL_PATH, CPU_INT4_SUBDIR)
        if os.path.exists(source_path):
            logger.info("")
            logger.info("Moving model files to root directory...")
            for item in os.listdir(source_path):
                src = os.path.join(source_path, item)
                dst = os.path.join(MODEL_PATH, item)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
                    logger.info(f"  Moved: {item}")
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("✓ Phi-3 model downloaded successfully!")
        logger.info("=" * 70)
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to download model: {e}")
        return False

if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)
