#!/usr/bin/env python3
"""
🔥 STEP 3: MODEL EVALUATION 🔥
==============================
Evaluate trained model performance
"""

import sys
import os
from pathlib import Path
import json

# Add core modules to path
sys.path.append(str(Path(__file__).parent.parent))

from core.training.advanced_trainer import AdvancedModelTrainer
from core.preprocessing.advanced_preprocessing import load_processed_dataset
import tensorflow as tf
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Advanced CNN-LSTM model')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing processed data')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model file (.keras)')
    parser.add_argument('--output_dir', type=str, default='results/evaluation',
                       help='Directory for evaluation outputs')
    
    args = parser.parse_args()
    
    print("🔥" * 50)
    print("🚀 ADVANCED CNN-LSTM MODEL EVALUATION")
    print("🔥" * 50)
    print(f"📁 Data Directory: {args.data_dir}")
    print(f"🤖 Model Path: {args.model_path}")
    print(f"📁 Output Directory: {args.output_dir}")
    print("🔥" * 50)
    
    try:
        # Load processed dataset
        print("📥 Loading processed dataset...")
        data, metadata = load_processed_dataset(args.data_dir)
        
        if 'test' not in data:
            logger.error("❌ No test data found")
            return 1
        
        print(f"✅ Dataset loaded successfully!")
        print(f"   📊 Test samples: {data['test']['sequences'].shape}")
        
        # Load trained model
        print("🤖 Loading trained model...")
        model = tf.keras.models.load_model(args.model_path)
        print(f"✅ Model loaded from: {args.model_path}")
        
        # Initialize trainer for evaluation
        trainer = AdvancedModelTrainer(model, args.output_dir, "advanced_cnn_lstm")
        
        # Prepare test data
        test_data = (data['test']['sequences'], data['test']['labels'])
        class_names = [metadata['class_names'][str(i)] for i in range(metadata['num_classes'])]
        
        # Run evaluation
        print("🧪 Running comprehensive evaluation...")
        results = trainer.evaluate_model(test_data, class_names)
        
        # Print summary
        print("\n🎉 EVALUATION COMPLETE!")
        print(f"✅ Test Accuracy: {results['test_accuracy']*100:.2f}%")
        print(f"✅ Test Loss: {results['test_loss']:.4f}")
        print(f"📁 Detailed results saved to: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main()) 