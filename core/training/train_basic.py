"""
train_optimized.py
-----------------
Optimized training script for the ASL recognition LSTM model
with fixed percentage display, loss normalization, and model saving
"""

import numpy as np
import os
import sys
import json
import platform
import tensorflow as tf
import time
import argparse
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from tensorflow.keras.optimizers import Adam
from lstm_model import create_lstm_model

# Disable GPU for now - add back if needed 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# ===== CONFIGURATION =====
# Paths
DATA_PATH = "processed_vids/wlasl-video/processed_keypoints_fixed"
JSON_PATH = "processed_vids/wlasl-video/WLASL_v0.3.json"
MODEL_SAVE_PATH = "models/lstm_model.h5"
CHECKPOINT_PATH = "models/lstm_model_checkpoint.h5"

# Training parameters
BATCH_SIZE = 32        # Larger batch for stable gradients
EPOCHS = 100           # Train longer with early stopping
VALIDATION_SPLIT = 0.2 # 80/20 split
LEARNING_RATE = 3e-4   # Adjusted learning rate
DROPOUT_RATE = 0.5     # Increased dropout for better generalization
L2_REGULARIZATION = 1e-5  # Weight regularization to prevent overfitting
CLIP_NORM = 1.0        # Gradient clipping to prevent exploding gradients

# Model parameters
EXPECTED_FEATURES = 1629  # Features per frame
SEQUENCE_LENGTH = 50      # Frames per sequence
LSTM_UNITS = 128          # Reduced size for CPU training
USE_ATTENTION = True      # Use attention mechanism

# ===== HELPER FUNCTIONS =====
def print_header(message):
    """Print a formatted header message"""
    print("\n" + "="*80)
    print(f" {message}")
    print("="*80)
    sys.stdout.flush()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="ASL Recognition Model Training")
    parser.add_argument("--test-only", action="store_true", 
                        help="Run in test mode without full training")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size for training (default: {BATCH_SIZE})")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Maximum number of epochs (default: {EPOCHS})")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help=f"Learning rate (default: {LEARNING_RATE})")
    return parser.parse_args()

def verify_data_paths():
    """Verify that data paths exist and contain necessary files"""
    # For test-only mode, don't exit if paths don't exist
    is_test = "--test-only" in sys.argv
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ ERROR: Data directory {DATA_PATH} not found!")
        if is_test:
            print("Test mode: Continuing despite missing data path")
            return []
        sys.exit(1)
    
    if not os.path.exists(JSON_PATH):
        print(f"❌ ERROR: JSON file {JSON_PATH} not found!")
        if is_test:
            print("Test mode: Continuing despite missing JSON file")
            return []
        sys.exit(1)
    
    # Check for .npy files
    try:
        npy_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.npy')]
        if not npy_files:
            print(f"❌ ERROR: No .npy files found in {DATA_PATH}")
            if is_test:
                print("Test mode: Continuing despite missing .npy files")
                return []
            sys.exit(1)
        
        print(f"✅ Found {len(npy_files)} .npy files in {DATA_PATH}")
        return npy_files
    except Exception as e:
        print(f"❌ ERROR accessing data directory: {e}")
        if is_test:
            print("Test mode: Continuing despite error")
            return []
        sys.exit(1)

def load_json_mapping(json_path):
    """Load the JSON mapping file and create videoid_to_gloss mapping"""
    # For test-only mode, don't exit if JSON loading fails
    is_test = "--test-only" in sys.argv
    
    try:
        with open(json_path, "r") as json_file:
            json_data = json.load(json_file)
    except Exception as e:
        print(f"❌ ERROR loading JSON file: {e}")
        if is_test:
            print("Test mode: Using dummy data")
            # Create dummy mapping for test mode
            return {"dummy": "DUMMY"}, {"DUMMY": 0}
        sys.exit(1)
    
    # Create video ID to gloss mapping
    videoid_to_gloss = {}
    for entry in json_data:
        gloss = entry.get("gloss")
        instances = entry.get("instances", [])
        for inst in instances:
            video_id = str(inst.get("video_id")).zfill(5)
            videoid_to_gloss[video_id] = gloss
    
    # Create gloss to label index mapping
    gloss_set = set(videoid_to_gloss.values())
    gloss_to_label = {gloss: idx for idx, gloss in enumerate(sorted(gloss_set))}
    
    print(f"✅ Found {len(gloss_set)} unique sign classes")
    return videoid_to_gloss, gloss_to_label

def prepare_dataset(npy_files, videoid_to_gloss, gloss_to_label):
    """Prepare dataset files and labels"""
    # For test-only mode, use minimal data
    is_test = "--test-only" in sys.argv
    
    if is_test and not npy_files:
        print("Test mode: Using dummy dataset")
        return ["dummy.npy"], [0], 1
        
    file_list = []
    label_list = []
    
    for f in sorted(npy_files):
        video_id = os.path.splitext(f)[0]
        if video_id not in videoid_to_gloss:
            continue
        
        gloss = videoid_to_gloss[video_id]
        label = gloss_to_label[gloss]
        file_list.append(f)
        label_list.append(label)
        
        # In test mode, only process a few files
        if is_test and len(file_list) >= 2:
            break
    
    # Print class distribution info
    label_counts = {}
    for label in label_list:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    # Print number of samples per class (top 5 and bottom 5)
    top_classes = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    bottom_classes = sorted(label_counts.items(), key=lambda x: x[1])[:5]
    
    print(f"✅ Dataset prepared with {len(file_list)} samples")
    print(f"Top 5 classes: {top_classes}")
    print(f"Bottom 5 classes: {bottom_classes}")
    return file_list, label_list, len(gloss_to_label)

def split_train_val(file_list, label_list, val_split=0.2):
    """Split dataset into training and validation sets"""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create indices and shuffle
    indices = np.arange(len(file_list))
    np.random.shuffle(indices)
    
    # Split into train and validation
    split_idx = int(len(indices) * (1 - val_split))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    # Create train and validation lists
    train_files = [file_list[i] for i in train_indices]
    train_labels = [label_list[i] for i in train_indices]
    val_files = [file_list[i] for i in val_indices]
    val_labels = [label_list[i] for i in val_indices]
    
    print(f"✅ Split dataset into {len(train_files)} training and {len(val_files)} validation samples")
    return train_files, train_labels, val_files, val_labels

def save_label_mapping(gloss_to_label):
    """Save label mapping to file for inference"""
    # Create models directory if it doesn't exist
    try:
        os.makedirs("models", exist_ok=True)
        label_to_gloss = {str(idx): gloss for gloss, idx in gloss_to_label.items()}
        
        with open("models/label_mapping.json", "w") as f:
            json.dump(label_to_gloss, f)
        
        print("✅ Label mapping saved to models/label_mapping.json")
    except Exception as e:
        print(f"❌ Error saving label mapping: {e}")

# ===== DATA GENERATOR =====
class ImprovedDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_list, label_list, data_path, batch_size, seq_length, 
                 features, num_classes, is_training=True, is_test_mode=False):
        self.file_list = file_list
        self.label_list = label_list
        self.data_path = data_path
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.features = features
        self.num_classes = num_classes
        self.is_training = is_training
        self.is_test_mode = is_test_mode
        
        # Cache for improved performance
        self.cache = {}
        self.cache_size = 150  # Increased for better performance
    
    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(len(self.file_list) / self.batch_size))
    
    def __getitem__(self, idx):
        """Get a batch of data"""
        # For test mode, generate dummy data
        if self.is_test_mode:
            X_batch = np.zeros((self.batch_size, self.seq_length, self.features), dtype='float32')
            y_batch = np.zeros((self.batch_size, self.num_classes), dtype='float32')
            # Set some random values for testing
            X_batch = np.random.rand(*X_batch.shape).astype('float32')
            for i in range(min(self.batch_size, len(self.label_list))):
                y_batch[i, self.label_list[i % len(self.label_list)]] = 1.0
            return X_batch, y_batch
        
        batch_files = self.file_list[idx * self.batch_size:(idx+1) * self.batch_size]
        batch_labels = self.label_list[idx * self.batch_size:(idx+1) * self.batch_size]
        
        sequences = []
        valid_labels = []
        
        # Process each file in the batch
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
                
                # Validate shape
                if seq.ndim != 2 or seq.shape[1] != self.features:
                    continue
                
                # Convert to float32 for better performance
                seq = np.asarray(seq, dtype='float32')
                
                # Normalize with robust method
                # Use robust normalization with epsilon to avoid division by zero
                seq = (seq - np.mean(seq)) / (np.std(seq) + 1e-3)
                
                # Replace any NaN or inf values (can happen with strange input data)
                seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Add to cache if not full
                if len(self.cache) < self.cache_size:
                    self.cache[f] = seq
                
                sequences.append(seq)
                valid_labels.append(label)
            except Exception as e:
                print(f"Error loading {f}: {e}")
                continue
        
        # Handle empty batch
        if not sequences:
            return (
                np.zeros((0, self.seq_length, self.features), dtype='float32'),
                np.zeros((0, self.num_classes), dtype='float32')
            )
        
        # Pad sequences to same length
        padded_sequences = []
        for seq in sequences:
            # If sequence is too long, truncate
            if len(seq) > self.seq_length:
                padded_seq = seq[:self.seq_length]
            # If sequence is too short, pad with zeros
            elif len(seq) < self.seq_length:
                padded_seq = np.zeros((self.seq_length, self.features), dtype='float32')
                padded_seq[:len(seq)] = seq
            else:
                padded_seq = seq
            
            padded_sequences.append(padded_seq)
        
        # Stack into batch
        X_batch = np.stack(padded_sequences)
        
        # Convert labels to one-hot
        y_batch = to_categorical(valid_labels, num_classes=self.num_classes)
        
        return X_batch, y_batch
    
    def on_epoch_end(self):
        """Called at the end of each epoch"""
        if self.is_training:
            # Shuffle data between epochs
            indices = np.arange(len(self.file_list))
            np.random.shuffle(indices)
            self.file_list = [self.file_list[i] for i in indices]
            self.label_list = [self.label_list[i] for i in indices]

# ===== CALLBACKS =====
class FixedMetricsCallback(Callback):
    """Custom callback for logging metrics with fixed percentage display"""
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Get metrics (with safety checks)
        loss = max(0.0, float(logs.get('loss', 0)))
        acc = max(0.0, min(1.0, float(logs.get('accuracy', 0))))
        val_loss = max(0.0, float(logs.get('val_loss', 0)))
        val_acc = max(0.0, min(1.0, float(logs.get('val_accuracy', 0))))
        
        # Safely get learning rate
        try:
            lr = float(self.model.optimizer.learning_rate.numpy())
        except:
            lr = float(self.model.optimizer.lr)
        
        # Print metrics
        print(f"\nEpoch {epoch+1}/{self.params['epochs']}:")
        print(f"  Loss: {loss:.4f}, Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"  Learning Rate: {lr:.6f}")
        sys.stdout.flush()

# ===== MAIN FUNCTION =====
def main():
    # Parse command line arguments
    args = parse_args()
    
    # Update global constants from args
    global BATCH_SIZE, EPOCHS, LEARNING_RATE
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    
    # If running in test-only mode, use minimal epochs
    if args.test_only:
        EPOCHS = 1
        print_header("TEST MODE - MINIMAL TRAINING")
    else:
        print_header("LSTM Sign Language Training")
    
    # Verify paths and load data
    npy_files = verify_data_paths()
    videoid_to_gloss, gloss_to_label = load_json_mapping(JSON_PATH)
    file_list, label_list, num_classes = prepare_dataset(npy_files, videoid_to_gloss, gloss_to_label)
    train_files, train_labels, val_files, val_labels = split_train_val(file_list, label_list, VALIDATION_SPLIT)
    save_label_mapping(gloss_to_label)
    
    # Create data generators
    print_header("Creating Data Generators")
    train_generator = ImprovedDataGenerator(
        train_files, train_labels, DATA_PATH, BATCH_SIZE, 
        SEQUENCE_LENGTH, EXPECTED_FEATURES, num_classes, 
        is_training=True, is_test_mode=args.test_only
    )
    val_generator = ImprovedDataGenerator(
        val_files, val_labels, DATA_PATH, BATCH_SIZE, 
        SEQUENCE_LENGTH, EXPECTED_FEATURES, num_classes, 
        is_training=False, is_test_mode=args.test_only
    )
    
    # Create and compile model
    print_header("Building and Compiling Model")
    input_shape = (SEQUENCE_LENGTH, EXPECTED_FEATURES)
    print(f"Input shape: {input_shape}, Output classes: {num_classes}")
    
    # Create model
    model = create_lstm_model(
        input_shape=input_shape,
        num_classes=num_classes,
        use_attention=USE_ATTENTION,
        is_cpu_only=True  # Force CPU-optimized model
    )
    
    # Use fixed optimizer with gradient clipping
    optimizer = Adam(
        learning_rate=LEARNING_RATE,
        clipnorm=CLIP_NORM,
        epsilon=1e-7
    )
    
    # Compile with float32 computation
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model Summary:")
    model.summary()
    
    # Setup callbacks
    print_header("Configuring Training")
    
    # Checkpointing
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    checkpoint = ModelCheckpoint(
        CHECKPOINT_PATH,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,  # More patience
        restore_best_weights=True,
        verbose=1
    )
    
    # Learning rate reducer
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=1
    )
    
    # Metrics logger
    metrics_logger = FixedMetricsCallback()
    
    # If in test-only mode, train for just 1 step
    if args.test_only:
        print("Test mode: Training for 1 step only")
        model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=1,
            steps_per_epoch=1,
            validation_steps=1,
            callbacks=[metrics_logger],
            verbose=0
        )
        print_header("Test Completed Successfully")
        return
    
    # Train model
    print_header("Starting Training")
    
    try:
        # Fit model
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=EPOCHS,
            callbacks=[checkpoint, early_stopping, reduce_lr, metrics_logger],
            workers=0,  # Single-process data loading
            use_multiprocessing=False,  # Avoid multiprocessing issues
            verbose=0  # Use our custom callback instead
        )
        
        # Save final model
        print_header("Saving Model")
        try:
            # First try with save_model
            model.save(MODEL_SAVE_PATH, save_format='h5')
            print(f"✅ Model saved to {MODEL_SAVE_PATH}")
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            # Alternative method
            try:
                # Try saving weights separately
                weights_path = os.path.join(os.path.dirname(MODEL_SAVE_PATH), "model_weights.h5")
                model.save_weights(weights_path)
                print(f"✅ Model weights saved to {weights_path}")
            except Exception as e2:
                print(f"❌ Could not save weights either: {e2}")
        
        # Save training history
        history_dict = {
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'accuracy': history.history['accuracy'],
            'val_accuracy': history.history['val_accuracy']
        }
        
        with open('models/training_history.json', 'w') as f:
            json.dump(history_dict, f)
        
        print("✅ Training history saved to models/training_history.json")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        # Still try to save the model
        try:
            model.save(MODEL_SAVE_PATH, save_format='h5')
            print(f"✅ Interrupted model saved to {MODEL_SAVE_PATH}")
        except:
            print("❌ Could not save interrupted model")
    
    print_header("Training Complete")

if __name__ == "__main__":
    main() 