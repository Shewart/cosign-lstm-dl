import numpy as np
import os
import sys
import json
import platform
import tensorflow as tf
import time
import pkg_resources
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback, TensorBoard
from lstm_model import create_lstm_model, compile_model  # Import the updated model creator

# Check compatibility of installed packages
def check_version_compatibility():
    print("Checking version compatibility...", flush=True)
    
    warnings = []
    
    # Check TensorFlow version
    tf_version = tf.__version__
    expected_tf_version = "2.10.0"
    if tf_version != expected_tf_version:
        warnings.append(f"⚠️ TensorFlow version mismatch: found {tf_version}, expected {expected_tf_version}")
    
    try:
        # Check OpenCV version
        import cv2
        cv_version = cv2.__version__
        expected_cv_version = "4.8.0.76"
        if cv_version != expected_cv_version:
            warnings.append(f"⚠️ OpenCV version mismatch: found {cv_version}, expected {expected_cv_version}")
    except ImportError:
        warnings.append("⚠️ OpenCV not installed")
    
    try:
        # Check MediaPipe version
        import mediapipe as mp
        mp_version = pkg_resources.get_distribution("mediapipe").version
        expected_mp_version = "0.10.8"
        if mp_version != expected_mp_version:
            warnings.append(f"⚠️ MediaPipe version mismatch: found {mp_version}, expected {expected_mp_version}")
    except ImportError:
        warnings.append("⚠️ MediaPipe not installed")
    
    # Print warnings if any
    if warnings:
        print("\nVersion compatibility issues detected:")
        for warning in warnings:
            print(warning)
        print("\nThese mismatches might cause unexpected behavior. Consider installing the expected versions.")
        # Ask user to continue
        if input("Continue anyway? (y/n): ").lower().strip() != 'y':
            print("Exiting...")
            sys.exit(1)
    else:
        print("✅ All package versions match expected values", flush=True)

# Run compatibility check
check_version_compatibility()

# Check operating system - Windows has issues with multiprocessing
IS_WINDOWS = platform.system() == 'Windows'
if IS_WINDOWS:
    print("⚠️ Windows detected - multiprocessing will be disabled for compatibility", flush=True)

# Check if GPU is available
has_gpu = len(tf.config.list_physical_devices('GPU')) > 0
print(f"GPU available: {has_gpu}", flush=True)

# Enable mixed precision only when GPU is available
if has_gpu:
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("✅ Mixed precision enabled for GPU acceleration", flush=True)
    except Exception as e:
        print(f"⚠️ Mixed precision not available: {e}", flush=True)
else:
    print("ℹ️ Running on CPU - mixed precision disabled for better compatibility", flush=True)

print("🚀 Starting model training with data generator...", flush=True)

# === Paths & Hyperparameters ===
DATA_PATH = "processed_vids/wlasl-video/processed_keypoints_fixed"
JSON_PATH = "processed_vids/wlasl-video/WLASL_v0.3.json"

EXPECTED_FEATURES = 1629        # Features per frame
TARGET_SEQUENCE_LENGTH = 50     # Frames per sequence
BATCH_SIZE = 8                  # Reduced batch size for CPU training
EPOCHS = 100                    # More epochs, with early stopping
VALIDATION_SPLIT = 0.2          # Percentage of data to use for validation

# Optimized learning rate with cosine decay
initial_learning_rate = 1e-4    # Reduced learning rate for CPU training
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate,
    decay_steps=1000,
    alpha=1e-5
)

# === List Processed Files ===
try:
    all_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".npy")]
    print(f"Found {len(all_files)} processed files.", flush=True)
    if not all_files:
        print("❌ ERROR: No training data found in", DATA_PATH, flush=True)
        sys.exit(1)
except FileNotFoundError:
    print(f"❌ ERROR: Directory not found: {DATA_PATH}", flush=True)
    print("Please make sure you have the correct path to the processed data.", flush=True)
    sys.exit(1)

# === Load JSON Mapping ===
try:
    with open(JSON_PATH, "r") as json_file:
        json_data = json.load(json_file)
except FileNotFoundError:
    print(f"❌ ERROR: JSON file not found: {JSON_PATH}", flush=True)
    print("Please make sure you have the correct path to the JSON mapping file.", flush=True)
    sys.exit(1)
except Exception as e:
    print("❌ ERROR loading JSON file:", e, flush=True)
    sys.exit(1)

videoid_to_gloss = {}
for entry in json_data:
    gloss = entry.get("gloss")
    instances = entry.get("instances", [])
    for inst in instances:
        video_id = str(inst.get("video_id")).zfill(5)
        videoid_to_gloss[video_id] = gloss

# === Build Gloss-to-Label Mapping ===
gloss_set = set(videoid_to_gloss.values())
gloss_to_label = {gloss: idx for idx, gloss in enumerate(sorted(gloss_set))}
NUM_CLASSES = len(gloss_to_label)
print(f"Found {NUM_CLASSES} unique gloss labels.", flush=True)

# === Build File and Label Lists ===
file_list = []
label_list = []
for f in sorted(all_files):
    video_id = os.path.splitext(f)[0]
    if video_id not in videoid_to_gloss:
        print(f"Skipping {f}: video_id {video_id} not found in JSON mapping.", flush=True)
        continue
    gloss = videoid_to_gloss[video_id]
    label = gloss_to_label[gloss]
    file_list.append(f)
    label_list.append(label)
    
print(f"Using {len(file_list)} files for training.", flush=True)

# === Split data into training and validation sets ===
indices = np.arange(len(file_list))
np.random.shuffle(indices)
split_idx = int(len(indices) * (1 - VALIDATION_SPLIT))
train_indices = indices[:split_idx]
val_indices = indices[split_idx:]

train_file_list = [file_list[i] for i in train_indices]
train_label_list = [label_list[i] for i in train_indices]
val_file_list = [file_list[i] for i in val_indices]
val_label_list = [label_list[i] for i in val_indices]

print(f"Training set: {len(train_file_list)} samples", flush=True)
print(f"Validation set: {len(val_file_list)} samples", flush=True)

# Save the label mapping for inference
os.makedirs("models", exist_ok=True)
label_to_gloss = {str(idx): gloss for gloss, idx in gloss_to_label.items()}
with open("models/label_mapping.json", "w") as f:
    json.dump(label_to_gloss, f)
print("✅ Label mapping saved to models/label_mapping.json", flush=True)

# === Optimized Data Generator with Prefetching and Caching ===
class DataGenerator(Sequence):
    def __init__(self, file_list, label_list, data_path, batch_size, target_len, expected_features, num_classes):
        self.file_list = file_list
        self.label_list = label_list
        self.data_path = data_path
        self.batch_size = batch_size
        self.target_len = target_len
        self.expected_features = expected_features
        self.num_classes = num_classes
        # Cache for frequently accessed files (smaller for CPU training)
        self.cache = {}
        self.cache_size = 50 if not has_gpu else 100

    def __len__(self):
        return int(np.ceil(len(self.file_list) / self.batch_size))

    def __getitem__(self, idx):
        batch_files = self.file_list[idx * self.batch_size:(idx+1) * self.batch_size]
        batch_labels = self.label_list[idx * self.batch_size:(idx+1) * self.batch_size]
        sequences = []
        valid_labels = []
        
        for f, label in zip(batch_files, batch_labels):
            # Try to get from cache first
            if f in self.cache:
                sequences.append(self.cache[f])
                valid_labels.append(label)
                continue
                
            file_path = os.path.join(self.data_path, f)
            try:
                seq = np.load(file_path, allow_pickle=True)
                if seq.ndim != 2 or seq.shape[1] != self.expected_features:
                    print(f"Skipping {f} in generator: invalid shape {seq.shape}", flush=True)
                    continue
                np_seq = np.asarray(seq, dtype='float32')
                
                # Advanced normalization: z-score standardization for better convergence
                np_seq = (np_seq - np.mean(np_seq)) / (np.std(np_seq) + 1e-7)
                
                # Add to cache if not full
                if len(self.cache) < self.cache_size:
                    self.cache[f] = np_seq.tolist()
                
                sequences.append(np_seq.tolist())
                valid_labels.append(label)
            except Exception as e:
                print(f"Error loading {f} in generator: {e}", flush=True)
                continue
                
        if not sequences:
            # Return empty batch with correct shapes if no valid sequences
            return (
                np.zeros((0, self.target_len, self.expected_features), dtype='float32'),
                np.zeros((0, self.num_classes), dtype='float32')
            )
            
            X_batch = pad_sequences(
                sequences,
                maxlen=self.target_len,
                padding='post',
                truncating='post',
                dtype='float32'
            )
        
        # Convert labels to one-hot vectors (shape: [batch_size, num_classes])
        y_batch = to_categorical(np.array(valid_labels), num_classes=self.num_classes)
        
        return X_batch, y_batch

    def on_epoch_end(self):
        # Shuffle data at the end of each epoch
        indices = np.arange(len(self.file_list))
        np.random.shuffle(indices)
        self.file_list = [self.file_list[i] for i in indices]
        self.label_list = [self.label_list[i] for i in indices]

# Create training and validation generators
train_generator = DataGenerator(
    file_list=train_file_list,
    label_list=train_label_list,
    data_path=DATA_PATH,
    batch_size=BATCH_SIZE,
    target_len=TARGET_SEQUENCE_LENGTH,
    expected_features=EXPECTED_FEATURES,
    num_classes=NUM_CLASSES
)

val_generator = DataGenerator(
    file_list=val_file_list,
    label_list=val_label_list,
    data_path=DATA_PATH,
    batch_size=BATCH_SIZE,
    target_len=TARGET_SEQUENCE_LENGTH,
    expected_features=EXPECTED_FEATURES,
    num_classes=NUM_CLASSES
)

# === Custom Callback for Detailed Metrics Logging with Percentages ===
class MetricsPercentageCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss', 0)
        acc = logs.get('accuracy', 0)
        val_loss = logs.get('val_loss', 0)
        val_acc = logs.get('val_accuracy', 0)
        
        # Fix the learning rate access
        try:
            # For TF 2.x
            if hasattr(self.model.optimizer, '_decayed_lr'):
                lr = self.model.optimizer._decayed_lr(tf.float32).numpy()
            # Fallback for older TF versions
            elif hasattr(self.model.optimizer, 'lr'):
                if callable(self.model.optimizer.lr):
                    lr = self.model.optimizer.lr(self.model.optimizer.iterations).numpy()
                else:
                    lr = float(self.model.optimizer.lr.numpy())
            else:
                lr = 0.0
        except Exception:
            # Final fallback
            lr = 0.0
        
        print(f"Epoch {epoch+1}/{self.params['epochs']}:")
        print(f"  Loss: {loss:.4f}, Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"  Learning Rate: {lr:.6f}")

# === Build the Enhanced LSTM Model ===
input_shape = (TARGET_SEQUENCE_LENGTH, EXPECTED_FEATURES)
print(f"Input shape for model: {input_shape}", flush=True)

# Create model with attention mechanism, smaller for CPU-only
model = create_lstm_model(
    input_shape, 
    NUM_CLASSES, 
    use_attention=True,
    is_cpu_only=not has_gpu  # Use smaller model for CPU
)
model = compile_model(model, learning_rate=lr_schedule, clipnorm=1.0)

print("Model summary:")
model.summary()

# === TensorBoard Setup ===
log_dir = "logs/fit/" + time.strftime("%Y%m%d-%H%M%S")
os.makedirs(log_dir, exist_ok=True)
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch='500,520')

# === Define enhanced callbacks ===
early_stopping = EarlyStopping(
    monitor="val_loss", 
    patience=10,  # More patience for better convergence
    verbose=1, 
    restore_best_weights=True,
    min_delta=0.001  # Minimum improvement to count
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", 
    factor=0.5, 
    patience=5, 
    verbose=1,
    min_delta=0.001,
    min_lr=1e-6
)

# Define a model checkpoint path with h5 format instead of keras
checkpoint_path = "models/lstm_model_best.h5"
checkpoint = ModelCheckpoint(
    checkpoint_path, 
    monitor="val_accuracy",
    save_best_only=True, 
    verbose=1,
    save_weights_only=False
)

metrics_callback = MetricsPercentageCallback()

print("🛠️ Starting model training with generator...", flush=True)

# Determine multiprocessing settings based on OS
use_multiprocessing = not IS_WINDOWS
workers = 0 if IS_WINDOWS else 4

print(f"Training with multiprocessing: {use_multiprocessing}, workers: {workers}", flush=True)

# Print hardware info
print(f"Training on CPU-optimized model: {not has_gpu}", flush=True)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr, checkpoint, metrics_callback, tensorboard_callback],
    verbose=0,  # We use our custom callback for logging
    workers=workers,  # Adjust based on OS
    use_multiprocessing=use_multiprocessing,
    max_queue_size=10
)

# Save the final model using h5 format which is more compatible
final_model_path = "models/lstm_model.h5"
try:
    print(f"Saving model to {final_model_path}...", flush=True)
    # Use tf.keras.models.save_model with h5 format
    tf.keras.models.save_model(
        model,
        final_model_path,
        save_format='h5'  # Explicitly specify h5 format
    )
    print("✅ Model saved successfully!", flush=True)
except Exception as e:
    print(f"❌ Error saving model: {e}", flush=True)
    # Try alternative saving method
    try:
        model.save(final_model_path, save_format='h5')
        print("✅ Model saved with alternative method!", flush=True)
    except Exception as e2:
        print(f"❌ Second attempt failed: {e2}", flush=True)

# Save the training history for later analysis
history_dict = {
    'loss': history.history['loss'],
    'val_loss': history.history['val_loss'],
    'accuracy': history.history['accuracy'],
    'val_accuracy': history.history['val_accuracy']
}
with open('models/training_history.json', 'w') as f:
    json.dump(history_dict, f)
print("✅ Training history saved to models/training_history.json", flush=True)
