#!/usr/bin/env python3
import os
import sys
import uvicorn
from pathlib import Path

def main():
    # Set environment variables
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    
    # Check if model files exist
    config_path = "configs/viton512.yaml"
    ckpt_path = "checkpoint/mvg.ckpt"
    
    if not Path(config_path).exists():
        print(f"⚠️  Config file not found: {config_path}")
        print("Please update the path in main.py startup_event()")
    
    if not Path(ckpt_path).exists():
        print(f"⚠️  Checkpoint file not found: {ckpt_path}")
        print("Please update the path in main.py startup_event()")
    
    # Create required directories
    Path("generated_images").mkdir(exist_ok=True)
    Path("temp_uploads").mkdir(exist_ok=True)
    
    print("🚀 Starting Virtual Try-On API server...")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        access_log=True
    )

if __name__ == "__main__":
    main()
