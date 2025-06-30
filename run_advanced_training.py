#!/usr/bin/env python3
"""
🚀 ADVANCED CNN-LSTM ASL RECOGNITION TRAINING 🚀
===============================================
Professional training pipeline with:
- Automated data preprocessing
- Batch-optimized training
- Real-time metrics & visualization  
- Comprehensive logging
- Dissertation-quality results
"""

import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add core modules to path
sys.path.append(str(Path(__file__).parent / "core"))
sys.path.append(str(Path(__file__).parent))

from core.models.advanced_cnn_lstm import create_advanced_model, get_model_summary
from core.training.advanced_trainer import AdvancedTrainer
from core.preprocessing.data_loader import DataLoader
from core.evaluation.metrics import ModelEvaluator

class TrainingOrchestrator:
    """Orchestrates the complete training pipeline with professional logging."""
    
    def __init__(self, config_path=None):
        self.setup_directories()
        self.setup_logging()
        self.config = self.load_config(config_path)
        self.start_time = time.time()
        
        # GPU optimization
        self.setup_gpu()
        
        print("🔥 ADVANCED CNN-LSTM ASL TRAINING INITIALIZED 🔥")
        print("=" * 60)
    
    def setup_directories(self):
        """Create all necessary directories."""
        dirs = [
            'results/training_logs',
            'results/models',
            'results/plots',
            'results/metrics',
            'results/checkpoints',
            'data/processed',
            'data/splits'
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self):
        """Setup comprehensive logging system."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(f"results/training_logs/training_{timestamp}")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / 'training.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # TensorBoard logging
        self.tensorboard_dir = self.log_dir / 'tensorboard'
        self.tensorboard_dir.mkdir(exist_ok=True)
    
    def setup_gpu(self):
        """Optimize GPU settings for training."""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                self.logger.info(f"✅ GPU acceleration enabled: {len(gpus)} GPU(s) found")
            except RuntimeError as e:
                self.logger.warning(f"⚠️  GPU setup error: {e}")
        else:
            self.logger.info("ℹ️  Running on CPU")
    
    def load_config(self, config_path):
        """Load training configuration."""
        default_config = {
            "model": {
                "sequence_length": 30,
                "num_classes": 26,
                "input_shape": [30, 1629]
            },
            "training": {
                "batch_size": 16,  # Optimized for PC memory
                "epochs": 100,
                "validation_split": 0.2,
                "early_stopping_patience": 15,
                "reduce_lr_patience": 7,
                "initial_lr": 0.001,
                "min_lr": 1e-6
            },
            "data": {
                "train_path": "data/train",
                "validation_path": "data/validation",
                "test_path": "data/test",
                "augmentation": True,
                "normalization": True
            },
            "logging": {
                "save_best_only": True,
                "monitor": "val_accuracy",
                "mode": "max",
                "verbose": 1
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                # Merge configs
                for key, value in user_config.items():
                    if key in default_config:
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
        
        return default_config
    
    def print_training_header(self):
        """Print beautiful training header with system info."""
        print("\n" + "🔥" * 60)
        print("    ADVANCED CNN-LSTM ASL RECOGNITION TRAINING")
        print("🔥" * 60)
        print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💾 Logs: {self.log_dir}")
        print(f"🎯 Target Accuracy: 70%+")
        print(f"🏗️  Architecture: Body-part-aware CNN-LSTM with Attention")
        print("=" * 60)
        
        # System info
        print("🖥️  SYSTEM INFORMATION:")
        print(f"   • TensorFlow: {tf.__version__}")
        print(f"   • Python: {sys.version.split()[0]}")
        print(f"   • GPUs Available: {len(tf.config.experimental.list_physical_devices('GPU'))}")
        print(f"   • Batch Size: {self.config['training']['batch_size']}")
        print(f"   • Max Epochs: {self.config['training']['epochs']}")
        print("=" * 60)
    
    def load_and_preprocess_data(self):
        """Load and preprocess training data with progress tracking."""
        print("\n📊 LOADING AND PREPROCESSING DATA...")
        
        # Initialize data loader
        data_loader = DataLoader(
            train_path=self.config['data']['train_path'],
            validation_path=self.config['data']['validation_path'],
            test_path=self.config['data']['test_path'],
            sequence_length=self.config['model']['sequence_length'],
            batch_size=self.config['training']['batch_size']
        )
        
        # Load data with progress tracking
        with tqdm(total=4, desc="Data Loading", bar_format="{desc}: {percentage:3.0f}%|{bar}|") as pbar:
            
            pbar.set_description("Loading training data")
            train_data = data_loader.load_training_data()
            pbar.update(1)
            
            pbar.set_description("Loading validation data")
            val_data = data_loader.load_validation_data()
            pbar.update(1)
            
            pbar.set_description("Preprocessing features")
            train_data, val_data = data_loader.preprocess_data(train_data, val_data)
            pbar.update(1)
            
            pbar.set_description("Creating batches")
            train_dataset = data_loader.create_batched_dataset(train_data)
            val_dataset = data_loader.create_batched_dataset(val_data)
            pbar.update(1)
        
        # Log data statistics
        self.logger.info(f"✅ Data loaded successfully")
        self.logger.info(f"   • Training samples: {len(train_data[0])}")
        self.logger.info(f"   • Validation samples: {len(val_data[0])}")
        self.logger.info(f"   • Features per sample: {train_data[0].shape[-1]}")
        self.logger.info(f"   • Sequence length: {train_data[0].shape[1]}")
        
        return train_dataset, val_dataset, data_loader
    
    def create_model(self):
        """Create and display the advanced model."""
        print("\n🏗️  BUILDING ADVANCED CNN-LSTM MODEL...")
        
        # Create model
        model = create_advanced_model(
            num_classes=self.config['model']['num_classes'],
            sequence_length=self.config['model']['sequence_length'],
            input_shape=tuple(self.config['model']['input_shape'])
        )
        
        # Display model summary
        print("\n📋 MODEL ARCHITECTURE:")
        print("=" * 50)
        model.summary()
        
        # Calculate and display model stats
        total_params = model.count_params()
        trainable_params = sum([tf.keras.utils.count_params(layer.weights) for layer in model.layers])
        memory_mb = (total_params * 4) / (1024 * 1024)
        
        print(f"\n📊 MODEL STATISTICS:")
        print(f"   • Total Parameters: {total_params:,}")
        print(f"   • Trainable Parameters: {trainable_params:,}")
        print(f"   • Model Size: {memory_mb:.1f} MB")
        print(f"   • Architecture: Advanced CNN-LSTM Hybrid")
        print(f"   • Features: Body-part aware + Attention")
        
        self.logger.info(f"Model created with {total_params:,} parameters")
        return model
    
    def setup_callbacks(self, model):
        """Setup comprehensive training callbacks."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        callbacks = [
            # Model checkpointing
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"results/checkpoints/best_model_{timestamp}.h5",
                monitor=self.config['logging']['monitor'],
                mode=self.config['logging']['mode'],
                save_best_only=self.config['logging']['save_best_only'],
                save_weights_only=False,
                verbose=1
            ),
            
            # Early stopping
            tf.keras.callbacks.EarlyStopping(
                monitor=self.config['logging']['monitor'],
                patience=self.config['training']['early_stopping_patience'],
                mode=self.config['logging']['mode'],
                restore_best_weights=True,
                verbose=1
            ),
            
            # Learning rate scheduling
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=self.config['logging']['monitor'],
                factor=0.5,
                patience=self.config['training']['reduce_lr_patience'],
                min_lr=self.config['training']['min_lr'],
                verbose=1
            ),
            
            # TensorBoard logging
            tf.keras.callbacks.TensorBoard(
                log_dir=str(self.tensorboard_dir),
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                update_freq='epoch'
            ),
            
            # CSV logging
            tf.keras.callbacks.CSVLogger(
                filename=str(self.log_dir / 'training_metrics.csv'),
                separator=',',
                append=False
            ),
            
            # Custom progress callback
            self.AdvancedProgressCallback(self.logger)
        ]
        
        return callbacks
    
    class AdvancedProgressCallback(tf.keras.callbacks.Callback):
        """Custom callback for detailed progress logging."""
        
        def __init__(self, logger):
            super().__init__()
            self.logger = logger
            self.epoch_start_time = None
            self.best_accuracy = 0.0
        
        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start_time = time.time()
            self.logger.info(f"🚀 Starting Epoch {epoch + 1}")
        
        def on_epoch_end(self, epoch, logs=None):
            epoch_time = time.time() - self.epoch_start_time
            
            # Extract metrics
            train_loss = logs.get('loss', 0)
            train_acc = logs.get('accuracy', 0) * 100
            val_loss = logs.get('val_loss', 0)
            val_acc = logs.get('val_accuracy', 0) * 100
            lr = logs.get('lr', 0)
            
            # Track best accuracy
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                improvement = "🎯 NEW BEST!"
            else:
                improvement = ""
            
            # Beautiful epoch summary
            print("\n" + "="*60)
            print(f"📊 EPOCH {epoch + 1} COMPLETE {improvement}")
            print("="*60)
            print(f"⏱️  Time: {epoch_time:.1f}s")
            print(f"📉 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"📈 Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%") 
            print(f"🎯 Best Val Acc: {self.best_accuracy:.2f}%")
            print(f"📚 Learning Rate: {lr:.2e}")
            print("="*60)
            
            self.logger.info(f"Epoch {epoch + 1}: val_acc={val_acc:.2f}%, train_acc={train_acc:.2f}%")
    
    def create_training_plots(self, history):
        """Create comprehensive training visualization plots."""
        print("\n📊 GENERATING TRAINING PLOTS...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('🔥 Advanced CNN-LSTM Training Results 🔥', fontsize=16, fontweight='bold')
        
        # Training & Validation Loss
        axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('📉 Model Loss Over Time')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Training & Validation Accuracy
        axes[0, 1].plot([acc * 100 for acc in history.history['accuracy']], label='Training Accuracy', linewidth=2)
        axes[0, 1].plot([acc * 100 for acc in history.history['val_accuracy']], label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_title('📈 Model Accuracy Over Time')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning Rate
        if 'lr' in history.history:
            axes[1, 0].semilogy(history.history['lr'], linewidth=2, color='orange')
            axes[1, 0].set_title('📚 Learning Rate Schedule')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate (log scale)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Additional Metrics
        if 'val_precision' in history.history and 'val_recall' in history.history:
            axes[1, 1].plot([p * 100 for p in history.history['val_precision']], label='Precision', linewidth=2)
            axes[1, 1].plot([r * 100 for r in history.history['val_recall']], label='Recall', linewidth=2)
            axes[1, 1].set_title('🎯 Precision & Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Score (%)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.log_dir / 'training_results.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training plots saved to: {plot_path}")
        
        return fig
    
    def run_training(self):
        """Execute the complete training pipeline."""
        try:
            # Print header
            self.print_training_header()
            
            # Load data
            train_dataset, val_dataset, data_loader = self.load_and_preprocess_data()
            
            # Create model
            model = self.create_model()
            
            # Setup callbacks
            callbacks = self.setup_callbacks(model)
            
            # Start training
            print("\n🚀 STARTING TRAINING...")
            print("=" * 60)
            
            training_start = time.time()
            
            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=self.config['training']['epochs'],
                callbacks=callbacks,
                verbose=1
            )
            
            training_time = time.time() - training_start
            
            # Training complete
            print("\n" + "🎉" * 60)
            print("    TRAINING COMPLETED SUCCESSFULLY!")
            print("🎉" * 60)
            
            # Final results
            final_accuracy = max(history.history['val_accuracy']) * 100
            final_loss = min(history.history['val_loss'])
            
            print(f"🏆 FINAL RESULTS:")
            print(f"   • Best Validation Accuracy: {final_accuracy:.2f}%")
            print(f"   • Final Validation Loss: {final_loss:.4f}")
            print(f"   • Training Time: {training_time/60:.1f} minutes")
            print(f"   • Total Epochs: {len(history.history['loss'])}")
            
            # Save final model
            final_model_path = f"results/models/advanced_cnn_lstm_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
            model.save(final_model_path)
            print(f"💾 Model saved: {final_model_path}")
            
            # Generate plots
            self.create_training_plots(history)
            
            # Save training config and results
            results = {
                "final_accuracy": float(final_accuracy),
                "final_loss": float(final_loss),
                "training_time_minutes": training_time / 60,
                "total_epochs": len(history.history['loss']),
                "model_path": final_model_path,
                "config": self.config
            }
            
            with open(self.log_dir / 'training_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            self.logger.info(f"🎯 Training completed with {final_accuracy:.2f}% accuracy")
            
            return model, history, results
            
        except Exception as e:
            self.logger.error(f"❌ Training failed: {str(e)}")
            raise
        
        finally:
            total_time = time.time() - self.start_time
            print(f"\n⏱️  Total execution time: {total_time/60:.1f} minutes")

def main():
    """Main execution function."""
    print("🔥 INITIALIZING ADVANCED CNN-LSTM TRAINING PIPELINE 🔥")
    
    # Create orchestrator
    orchestrator = TrainingOrchestrator()
    
    # Run training
    model, history, results = orchestrator.run_training()
    
    print("\n✅ ALL OPERATIONS COMPLETED SUCCESSFULLY!")
    print(f"🎯 Achievement: {results['final_accuracy']:.2f}% accuracy")
    print("📊 Check results/ directory for detailed outputs")

if __name__ == "__main__":
    main() 