#!/usr/bin/env python3
"""
Improved training script for CNN-LSTM hybrid model for sign language recognition.
This script fixes the issues in the original hybrid approach and implements
a proper CNN-LSTM architecture for better accuracy.
"""

import os
import sys
import json
import numpy as np
import argparse
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Import our new hybrid model
sys.path.append('src')
from cnn_lstm_hybrid import create_cnn_lstm_hybrid_model, create_advanced_cnn_lstm_model, compile_hybrid_model

# Configuration
DATA_PATH = "processed_vids/wlasl-video/processed_keypoints_fixed"
JSON_PATH = "models/wlasl_label_mapping.json"
FALLBACK_JSON_PATH = "models/label_mapping.json"
EXPECTED_FEATURES = 1629
SEQUENCE_LENGTH = 50

# Training parameters
BATCH_SIZE = 16  # Start smaller for hybrid model
EPOCHS = 100
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2

# GPU setup
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_GPU_THREAD_COUNT"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Check GPU availability
has_gpu = len(tf.config.list_physical_devices('GPU')) > 0
print(f"GPU available: {has_gpu}")

# Use mixed precision for better performance if GPU available
if has_gpu:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("Using mixed precision for GPU training")
else:
    tf.keras.mixed_precision.set_global_policy('float32')
    print("Using float32 precision for CPU training")


class HybridDataGenerator(Sequence):
    """
    Optimized data generator for CNN-LSTM hybrid model training.
    """
    
    def __init__(self, file_list, label_list, data_path, batch_size, seq_length, 
                 features, num_classes, is_training=True, debug=False):
        self.file_list = file_list
        self.label_list = label_list
        self.data_path = data_path
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.features = features
        self.num_classes = num_classes
        self.is_training = is_training
        self.debug = debug
        
        # Cache for improved performance
        self.cache = {}
        self.cache_size = 200
        
        # Pre-load and validate some data
        self._validate_data()
    
    def _validate_data(self):
        """Validate that data files exist and have correct format"""
        print(f"Validating {len(self.file_list)} data files...")
        valid_files = []
        valid_labels = []
        
        for i, (file, label) in enumerate(zip(self.file_list[:10], self.label_list[:10])):
            try:
                file_path = os.path.join(self.data_path, file)
                if os.path.exists(file_path):
                    seq = np.load(file_path, allow_pickle=True)
                    if seq.shape[1] == self.features and not np.isnan(seq).all():
                        valid_files.append(file)
                        valid_labels.append(label)
                        if self.debug:
                            print(f"✅ Valid: {file} - Shape: {seq.shape}")
                    else:
                        if self.debug:
                            print(f"❌ Invalid shape/data: {file} - Shape: {seq.shape}")
                else:
                    if self.debug:
                        print(f"❌ File not found: {file_path}")
            except Exception as e:
                if self.debug:
                    print(f"❌ Error loading {file}: {e}")
        
        print(f"Validation complete: {len(valid_files)}/{len(self.file_list[:10])} files valid")
    
    def __len__(self):
        return int(np.ceil(len(self.file_list) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_files = self.file_list[idx * self.batch_size:(idx+1) * self.batch_size]
        batch_labels = self.label_list[idx * self.batch_size:(idx+1) * self.batch_size]
        
        sequences = []
        valid_labels = []
        
        for f, label in zip(batch_files, batch_labels):
            # Check cache first
            if f in self.cache:
                sequences.append(self.cache[f])
                valid_labels.append(label)
                continue
            
            # Load and process file
            try:
                file_path = os.path.join(self.data_path, f)
                seq = np.load(file_path, allow_pickle=True)
                
                # Ensure correct data type
                seq = np.asarray(seq, dtype=np.float32)
                
                # Handle NaN and infinite values
                if np.isnan(seq).any() or np.isinf(seq).any():
                    seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Normalize the sequence
                seq_normalized = self._normalize_sequence(seq)
                
                # Add to cache
                if len(self.cache) < self.cache_size:
                    self.cache[f] = seq_normalized
                
                sequences.append(seq_normalized)
                valid_labels.append(label)
                
            except Exception as e:
                if self.debug:
                    print(f"Error loading {f}: {e}")
                continue
        
        if not sequences:
            # Return empty batch if no valid sequences
            empty_x = np.zeros((0, self.seq_length, self.features), dtype=np.float32)
            empty_y = np.zeros((0, self.num_classes), dtype=np.float32)
            return empty_x, empty_y
        
        # Pad sequences to uniform length
        padded_sequences = []
        for seq in sequences:
            if len(seq) > self.seq_length:
                # Take the middle part of the sequence
                start_idx = (len(seq) - self.seq_length) // 2
                padded_seq = seq[start_idx:start_idx + self.seq_length]
            elif len(seq) < self.seq_length:
                # Pad with zeros
                padded_seq = np.zeros((self.seq_length, self.features), dtype=np.float32)
                padded_seq[:len(seq)] = seq
            else:
                padded_seq = seq
            
            padded_sequences.append(padded_seq)
        
        # Create batch
        X_batch = np.array(padded_sequences, dtype=np.float32)
        y_batch = tf.keras.utils.to_categorical(valid_labels, num_classes=self.num_classes)
        
        return X_batch, y_batch
    
    def _normalize_sequence(self, seq):
        """Robust normalization for sequence data"""
        # Global normalization across all features
        mean = np.mean(seq)
        std = np.std(seq)
        
        if std > 1e-8:
            normalized = (seq - mean) / (std + 1e-8)
        else:
            normalized = seq - mean
        
        # Clip extreme values
        normalized = np.clip(normalized, -5.0, 5.0)
        
        return normalized.astype(np.float32)
    
    def on_epoch_end(self):
        if self.is_training:
            # Shuffle data between epochs
            indices = np.arange(len(self.file_list))
            np.random.shuffle(indices)
            self.file_list = [self.file_list[i] for i in indices]
            self.label_list = [self.label_list[i] for i in indices]


class ImprovedMetricsCallback(Callback):
    """Callback for better metrics display"""
    
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        
        # Format and display metrics
        metrics_str = f"\nEpoch {epoch+1}: "
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                if 'acc' in key.lower():
                    metrics_str += f"{key}: {value:.4f} ({value*100:.2f}%)  "
                else:
                    metrics_str += f"{key}: {value:.4f}  "
            else:
                metrics_str += f"{key}: {value}  "
        
        print(metrics_str)


def load_json_mapping(json_path):
    """Load mapping between video IDs and labels from JSON file"""
    try:
        with open(json_path, "r") as json_file:
            data = json.load(json_file)
        
        # Convert to proper format
        if isinstance(list(data.values())[0], int):
            # gloss_to_label format
            gloss_to_label = data
            label_to_gloss = {str(v): k for k, v in data.items()}
        else:
            # label_to_gloss format
            label_to_gloss = data
            gloss_to_label = {v: int(k) for k, v in data.items()}
        
        print(f"✅ Loaded mapping with {len(gloss_to_label)} classes from {json_path}")
        return gloss_to_label, label_to_gloss
        
    except Exception as e:
        print(f"❌ Could not load {json_path}: {e}")
        
        # Try fallback
        try:
            with open(FALLBACK_JSON_PATH, "r") as f:
                fallback_data = json.load(f)
            print(f"✅ Using fallback mapping from {FALLBACK_JSON_PATH}")
            
            if isinstance(list(fallback_data.values())[0], int):
                gloss_to_label = fallback_data
                label_to_gloss = {str(v): k for k, v in fallback_data.items()}
            else:
                label_to_gloss = fallback_data
                gloss_to_label = {v: int(k) for k, v in fallback_data.items()}
                
            return gloss_to_label, label_to_gloss
            
        except Exception as fallback_e:
            print(f"❌ Fallback also failed: {fallback_e}")
            print("Creating dummy mapping for testing...")
            
            # Create dummy mapping
            dummy_gloss_to_label = {f"sign_{i}": i for i in range(10)}
            dummy_label_to_gloss = {str(i): f"sign_{i}" for i in range(10)}
            return dummy_gloss_to_label, dummy_label_to_gloss


def prepare_dataset(data_path, gloss_to_label):
    """Prepare dataset by scanning directory and mapping files to labels"""
    if not os.path.exists(data_path):
        print(f"❌ Data directory not found: {data_path}")
        return [], [], 0
    
    # Get all .npy files
    all_files = [f for f in os.listdir(data_path) if f.endswith('.npy')]
    print(f"Found {len(all_files)} .npy files in {data_path}")
    
    if len(all_files) == 0:
        print("❌ No .npy files found!")
        return [], [], 0
    
    # Map files to labels
    file_list = []
    label_list = []
    
    # Try direct mapping first (video_id.npy -> gloss -> label)
    wlasl_mapping = load_wlasl_mapping()
    
    for file in all_files:
        video_id = os.path.splitext(file)[0]
        
        # Try direct gloss mapping
        if video_id in gloss_to_label:
            file_list.append(file)
            label_list.append(gloss_to_label[video_id])
            continue
        
        # Try WLASL mapping
        if wlasl_mapping and video_id in wlasl_mapping:
            gloss = wlasl_mapping[video_id]
            if gloss in gloss_to_label:
                file_list.append(file)
                label_list.append(gloss_to_label[gloss])
                continue
        
        # If no mapping found, skip or use default
        if len(gloss_to_label) <= 10:  # For testing with small datasets
            file_list.append(file)
            label_list.append(hash(video_id) % len(gloss_to_label))
    
    num_classes = max(label_list) + 1 if label_list else len(gloss_to_label)
    print(f"✅ Prepared dataset: {len(file_list)} files, {num_classes} classes")
    
    return file_list, label_list, num_classes


def load_wlasl_mapping():
    """Load WLASL video ID to gloss mapping"""
    wlasl_path = "processed_vids/wlasl-video/WLASL_v0.3.json"
    
    if not os.path.exists(wlasl_path):
        return None
    
    try:
        with open(wlasl_path, "r") as f:
            wlasl_data = json.load(f)
        
        # Create mapping from video ID to gloss
        video_id_to_gloss = {}
        for entry in wlasl_data:
            gloss = entry.get("gloss", "").lower()
            instances = entry.get("instances", [])
            for instance in instances:
                video_id = str(instance.get("video_id"))
                video_id_to_gloss[video_id] = gloss
        
        print(f"✅ Loaded WLASL mapping: {len(video_id_to_gloss)} video-to-gloss mappings")
        return video_id_to_gloss
        
    except Exception as e:
        print(f"❌ Error loading WLASL mapping: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Train CNN-LSTM hybrid model for sign language recognition")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--data-path", type=str, default=DATA_PATH, help="Path to processed keypoints")
    parser.add_argument("--advanced", action="store_true", help="Use advanced body-part-aware model")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--test-run", action="store_true", help="Quick test run with few epochs")
    
    args = parser.parse_args()
    
    if args.test_run:
        args.epochs = 2
        print("🧪 Test run mode: Using only 2 epochs")
    
    print("🚀 Starting CNN-LSTM Hybrid Training")
    print(f"Data path: {args.data_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Advanced model: {args.advanced}")
    
    # Load data mapping
    print("\n📊 Loading data mapping...")
    gloss_to_label, label_to_gloss = load_json_mapping(JSON_PATH)
    
    # Prepare dataset
    print("\n📁 Preparing dataset...")
    file_list, label_list, num_classes = prepare_dataset(args.data_path, gloss_to_label)
    
    if len(file_list) == 0:
        print("❌ No valid data files found. Exiting.")
        return
    
    # Split dataset
    print(f"\n✂️ Splitting dataset ({VALIDATION_SPLIT:.0%} validation)...")
    train_files, val_files, train_labels, val_labels = train_test_split(
        file_list, label_list, test_size=VALIDATION_SPLIT, 
        random_state=42, stratify=label_list if len(set(label_list)) > 1 else None
    )
    
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    
    # Create data generators
    print("\n🔄 Creating data generators...")
    train_generator = HybridDataGenerator(
        train_files, train_labels, args.data_path, args.batch_size,
        SEQUENCE_LENGTH, EXPECTED_FEATURES, num_classes, 
        is_training=True, debug=args.debug
    )
    
    val_generator = HybridDataGenerator(
        val_files, val_labels, args.data_path, args.batch_size,
        SEQUENCE_LENGTH, EXPECTED_FEATURES, num_classes,
        is_training=False, debug=args.debug
    )
    
    # Create model
    print(f"\n🧠 Creating {'advanced' if args.advanced else 'basic'} CNN-LSTM hybrid model...")
    input_shape = (SEQUENCE_LENGTH, EXPECTED_FEATURES)
    
    if args.advanced:
        model = create_advanced_cnn_lstm_model(input_shape, num_classes)
    else:
        model = create_cnn_lstm_hybrid_model(input_shape, num_classes, use_attention=True)
    
    # Compile model
    model = compile_hybrid_model(model, learning_rate=args.learning_rate)
    
    print(f"Model parameters: {model.count_params():,}")
    print("\nModel architecture:")
    model.summary()
    
    # Setup callbacks
    print("\n⚙️ Setting up callbacks...")
    
    # Model checkpoint
    checkpoint_path = "models/cnn_lstm_hybrid_best.h5"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1,
        save_weights_only=False
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1,
        min_delta=0.001
    )
    
    # Learning rate reduction
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=1
    )
    
    # Metrics callback
    metrics_callback = ImprovedMetricsCallback()
    
    callbacks = [checkpoint, early_stopping, reduce_lr, metrics_callback]
    
    # Train model
    print(f"\n🏋️ Training model for {args.epochs} epochs...")
    
    try:
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=1,
            workers=4 if not sys.platform.startswith('win') else 0,
            use_multiprocessing=not sys.platform.startswith('win')
        )
        
        # Save final model
        final_model_path = "models/cnn_lstm_hybrid_final.h5"
        model.save(final_model_path)
        print(f"✅ Final model saved to {final_model_path}")
        
        # Save training history
        history_path = "models/cnn_lstm_hybrid_history.json"
        with open(history_path, 'w') as f:
            json.dump(history.history, f, indent=2)
        print(f"✅ Training history saved to {history_path}")
        
        # Plot training history
        if 'accuracy' in history.history:
            plt.figure(figsize=(15, 5))
            
            # Accuracy plot
            plt.subplot(1, 3, 1)
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            # Loss plot
            plt.subplot(1, 3, 2)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # Top-5 accuracy (if available)
            if 'top_5_accuracy' in history.history:
                plt.subplot(1, 3, 3)
                plt.plot(history.history['top_5_accuracy'], label='Training Top-5 Acc')
                plt.plot(history.history['val_top_5_accuracy'], label='Validation Top-5 Acc')
                plt.title('Top-5 Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Top-5 Accuracy')
                plt.legend()
            
            plt.tight_layout()
            plt.savefig('models/cnn_lstm_hybrid_training_history.png', dpi=300, bbox_inches='tight')
            print("✅ Training plots saved to models/cnn_lstm_hybrid_training_history.png")
        
        print("\n🎉 Training completed successfully!")
        print(f"Best model saved at: {checkpoint_path}")
        print(f"Final model saved at: {final_model_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        # Try to save current model
        interrupted_path = "models/cnn_lstm_hybrid_interrupted.h5"
        try:
            model.save(interrupted_path)
            print(f"✅ Interrupted model saved to {interrupted_path}")
        except Exception as e:
            print(f"❌ Could not save interrupted model: {e}")
    
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 