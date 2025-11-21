#!/usr/bin/env python3
"""
Download Phi-3 Mini ONNX model (CPU-optimized) from Hugging Face
This is a one-time setup script.
"""
import os
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

PHI3_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct-onnx"
MODEL_PATH = "./backend/phi3_model"
CPU_INT4_SUBDIR = "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"

def download_model():
    """Download the Phi-3 ONNX model optimized for CPU"""
    logger.info("=" * 70)
    logger.info("Downloading Phi-3 Mini ONNX Model (CPU-Optimized, INT4 Quantized)")
    logger.info("=" * 70)
    logger.info(f"Model: {PHI3_MODEL_ID}")
    logger.info(f"Size: ~2GB (INT4 quantized)")
    logger.info(f"Target: {MODEL_PATH}")
    logger.info("")
    
    if os.path.exists(os.path.join(MODEL_PATH, "genai_config.json")):
        logger.info("✓ Model already downloaded!")
        return True
    
    logger.info("Installing huggingface-cli if needed...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub[cli]"], check=True)
    
    logger.info("Downloading model files (this may take 5-10 minutes)...")
    logger.info("")
    
    cmd = [
        "huggingface-cli", "download",
        PHI3_MODEL_ID,
        "--include", f"{CPU_INT4_SUBDIR}/*",
        "--local-dir", MODEL_PATH
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        
        source_path = os.path.join(MODEL_PATH, CPU_INT4_SUBDIR)
        if os.path.exists(source_path):
            logger.info("Moving model files to root directory...")
            for item in os.listdir(source_path):
                src = os.path.join(source_path, item)
                dst = os.path.join(MODEL_PATH, item)
                if not os.path.exists(dst):
                    os.rename(src, dst)
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("✓ Phi-3 model downloaded successfully!")
        logger.info("=" * 70)
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed to download model: {e}")
        return False

if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)
