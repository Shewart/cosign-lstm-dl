#!/usr/bin/env python3
"""
🔥 QUICK START SCRIPT 🔥
========================
One-click setup and execution for sign language recognition
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse


def run_command(command, description):
    """Run a command with nice output"""
    print(f"\n🚀 {description}")
    print("="*60)
    print(f"Command: {command}")
    print("="*60)
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
    else:
        print("❌ Failed!")
        if result.stderr:
            print(result.stderr)
        return False
    
    return True


def activate_environment():
    """Activate virtual environment"""
    if sys.platform == "win32":
        return "ml_env\\Scripts\\activate"
    else:
        return "source ml_env/bin/activate"


def main():
    parser = argparse.ArgumentParser(description='Quick Start for Sign Language Recognition')
    parser.add_argument('--step', type=str, 
                       choices=['preprocess', 'train', 'evaluate', 'inference', 'all'],
                       default='all', help='Which step to run')
    parser.add_argument('--data_dir', type=str, default='data/raw',
                       help='Directory containing raw video data')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    
    args = parser.parse_args()
    
    print("🔥" * 60)
    print("🚀 SIGN LANGUAGE RECOGNITION - QUICK START")
    print("🔥" * 60)
    print("📋 Pipeline Steps:")
    print("   1️⃣  Preprocess raw video data")
    print("   2️⃣  Train Advanced CNN-LSTM model")
    print("   3️⃣  Evaluate model performance")
    print("   4️⃣  Run real-time inference")
    print("🔥" * 60)
    
    # Activate environment
    activate_cmd = activate_environment()
    
    if args.step in ['preprocess', 'all']:
        # Step 1: Preprocess data
        cmd = f"{activate_cmd} && python scripts/01_preprocess_data.py --input_dir {args.data_dir} --output_dir data/processed"
        if not run_command(cmd, "STEP 1: Preprocessing Data"):
            print("❌ Preprocessing failed. Please check your data directory.")
            return 1
    
    if args.step in ['train', 'all']:
        # Step 2: Train model
        cmd = f"{activate_cmd} && python scripts/02_train_model.py --data_dir data/processed --epochs {args.epochs}"
        if not run_command(cmd, "STEP 2: Training Model"):
            print("❌ Training failed. Please check the preprocessed data.")
            return 1
    
    if args.step in ['evaluate', 'all']:
        # Step 3: Evaluate model
        model_path = "logs/training/advanced_cnn_lstm_best.keras"
        metadata_path = "data/processed/metadata.json"
        cmd = f"{activate_cmd} && python scripts/03_evaluate_model.py --data_dir data/processed --model_path {model_path}"
        
        if Path(model_path).exists():
            if not run_command(cmd, "STEP 3: Evaluating Model"):
                print("❌ Evaluation failed.")
                return 1
        else:
            print(f"⚠️  Model not found at {model_path}. Skipping evaluation.")
    
    if args.step in ['inference', 'all']:
        # Step 4: Setup inference
        model_path = "logs/training/advanced_cnn_lstm_best.keras"
        metadata_path = "data/processed/metadata.json"
        
        if Path(model_path).exists() and Path(metadata_path).exists():
            print(f"\n🎉 SETUP COMPLETE!")
            print(f"✅ Your model is ready for inference!")
            print(f"\n🚀 To run real-time inference:")
            print(f"   Webcam: python scripts/04_inference.py --model_path {model_path} --metadata_path {metadata_path} --mode webcam --show_video")
            print(f"   Video:  python scripts/04_inference.py --model_path {model_path} --metadata_path {metadata_path} --mode video --video_path your_video.mp4")
        else:
            print(f"⚠️  Model or metadata not found. Please run training first.")
    
    print("\n🎉 Quick start complete!")
    return 0


if __name__ == "__main__":
    exit(main()) 