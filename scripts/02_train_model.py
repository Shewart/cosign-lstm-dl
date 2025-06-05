#!/usr/bin/env python3
"""
🔥 STEP 2: MODEL TRAINING 🔥
============================
Train the Advanced CNN-LSTM Hybrid model
"""

import sys
import os
from pathlib import Path
import json

# Add core modules to path
sys.path.append(str(Path(__file__).parent.parent))

from core.models.advanced_cnn_lstm import create_advanced_model
from core.training.advanced_trainer import create_training_session
from core.preprocessing.advanced_preprocessing import load_processed_dataset
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Train Advanced CNN-LSTM model')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing processed data')
    parser.add_argument('--log_dir', type=str, default='logs/training',
                       help='Directory for training logs and outputs')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Initial learning rate')
    
    args = parser.parse_args()
    
    print("🔥" * 50)
    print("🚀 ADVANCED CNN-LSTM MODEL TRAINING")
    print("🔥" * 50)
    print(f"📁 Data Directory: {args.data_dir}")
    print(f"📁 Log Directory: {args.log_dir}")
    print(f"🔄 Epochs: {args.epochs}")
    print(f"📦 Batch Size: {args.batch_size}")
    print(f"📈 Learning Rate: {args.learning_rate}")
    print("🔥" * 50)
    
    try:
        # Load processed dataset
        print("📥 Loading processed dataset...")
        data, metadata = load_processed_dataset(args.data_dir)
        
        if 'train' not in data or 'validation' not in data:
            logger.error("❌ Missing train or validation data")
            return 1
        
        print(f"✅ Dataset loaded successfully!")
        print(f"   📊 Classes: {metadata['num_classes']}")
        print(f"   📊 Feature Dim: {metadata['feature_dim']}")
        print(f"   📊 Train: {data['train']['sequences'].shape}")
        print(f"   📊 Val: {data['validation']['sequences'].shape}")
        
        # Create model
        print("🏗️  Building Advanced CNN-LSTM model...")
        model, model_builder = create_advanced_model(
            num_classes=metadata['num_classes'],
            sequence_length=metadata['sequence_length']
        )
        
        # Prepare data
        train_data = (data['train']['sequences'], data['train']['labels'])
        val_data = (data['validation']['sequences'], data['validation']['labels'])
        test_data = None
        if 'test' in data:
            test_data = (data['test']['sequences'], data['test']['labels'])
        
        # Get class names
        class_names = [metadata['class_names'][str(i)] for i in range(metadata['num_classes'])]
        
        # Start training session
        print("🚀 Starting advanced training session...")
        results = create_training_session(
            model=model,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            class_names=class_names,
            log_dir=args.log_dir,
            model_name="advanced_cnn_lstm",
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # Print final results
        training_results = results['training_results']
        best_metrics = training_results['best_metrics']
        
        print("\n🎉 TRAINING COMPLETE!")
        print(f"✅ Best Validation Accuracy: {best_metrics['best_val_accuracy']*100:.2f}%")
        print(f"✅ Best Epoch: {best_metrics['best_epoch'] + 1}")
        print(f"📁 Results saved to: {args.log_dir}")
        
        if results['evaluation_results']:
            eval_results = results['evaluation_results']
            print(f"🧪 Test Accuracy: {eval_results['test_accuracy']*100:.2f}%")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main()) 