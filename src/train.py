import numpy as np
import os
import sys
import json
import tensorflow as tf
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from lstm_model import create_lstm_model  # Your bidirectional model file

print("🚀 Starting model training with data generator...", flush=True)

# === Paths & Hyperparameters ===
DATA_PATH = "processed_vids/wlasl-video/processed_keypoints"
JSON_PATH = "processed_vids/wlasl-video/WLASL_v0.3.json"

EXPECTED_FEATURES = 1629        # Features per frame
TARGET_SEQUENCE_LENGTH = 50     # Frames per sequence
BATCH_SIZE = 8
EPOCHS = 50

# Adjust the learning rate here; experiment with values like 1e-4, 5e-4, or 1e-3
LEARNING_RATE = 1e-4  
# Use Adam with gradient clipping (clipnorm helps prevent exploding gradients)
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)

# === List Processed Files ===
all_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".npy")]
print(f"Found {len(all_files)} processed files.", flush=True)
if not all_files:
    print("❌ ERROR: No training data found in", DATA_PATH, flush=True)
    sys.exit(1)

# === Load JSON Mapping ===
try:
    with open(JSON_PATH, "r") as json_file:
        json_data = json.load(json_file)
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

# === Data Generator with Optional Normalization ===
class DataGenerator(Sequence):
    def __init__(self, file_list, label_list, data_path, batch_size, target_len, expected_features, num_classes):
        self.file_list = file_list
        self.label_list = label_list
        self.data_path = data_path
        self.batch_size = batch_size
        self.target_len = target_len
        self.expected_features = expected_features
        self.num_classes = num_classes

    def __len__(self):
        return int(np.ceil(len(self.file_list) / self.batch_size))

    def __getitem__(self, idx):
        batch_files = self.file_list[idx * self.batch_size:(idx+1) * self.batch_size]
        batch_labels = self.label_list[idx * self.batch_size:(idx+1) * self.batch_size]
        sequences = []
        for f in batch_files:
            file_path = os.path.join(self.data_path, f)
            try:
                seq = np.load(file_path, allow_pickle=True)
                if seq.ndim != 2 or seq.shape[1] != self.expected_features:
                    print(f"Skipping {f} in generator: invalid shape {seq.shape}", flush=True)
                    continue
                np_seq = np.asarray(seq, dtype='float32')
                # Optional: Normalize keypoints if needed (uncomment below if data isn't normalized)
                # np_seq = (np_seq - np.min(np_seq)) / (np.max(np_seq) - np.min(np_seq) + 1e-7)
                sequences.append(np_seq.tolist())
            except Exception as e:
                print(f"Error loading {f} in generator: {e}", flush=True)
                continue
        if sequences:
            X_batch = pad_sequences(
                sequences,
                maxlen=self.target_len,
                padding='post',
                truncating='post',
                dtype='float32'
            )
        else:
            X_batch = np.zeros((len(batch_files), self.target_len, self.expected_features), dtype='float32')
        y_batch = to_categorical(np.array(batch_labels[:X_batch.shape[0]]), num_classes=self.num_classes)
        return X_batch, y_batch

generator = DataGenerator(
    file_list=file_list,
    label_list=label_list,
    data_path=DATA_PATH,
    batch_size=BATCH_SIZE,
    target_len=TARGET_SEQUENCE_LENGTH,
    expected_features=EXPECTED_FEATURES,
    num_classes=NUM_CLASSES
)

steps_per_epoch = len(generator)
print(f"Generator will yield {steps_per_epoch} steps per epoch.", flush=True)

# === Custom Callback for Detailed Metrics Logging with Percentages ===
class MetricsPercentageCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss', 0)
        acc = logs.get('accuracy', 0)
        val_loss = logs.get('val_loss', 0)
        val_acc = logs.get('val_accuracy', 0)
        print(f"Epoch {epoch+1} complete:")
        print(f"  Loss: {loss:.4f} (raw), Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(f"  Val Loss: {val_loss:.4f} (raw), Val Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")

# === Build the Bidirectional LSTM Model ===
input_shape = (TARGET_SEQUENCE_LENGTH, EXPECTED_FEATURES)
print(f"Input shape for model: {input_shape}", flush=True)
model = create_lstm_model(input_shape, NUM_CLASSES)

print("Model summary:")
model.summary()

# Compile model with our optimizer
model.compile(optimizer=OPTIMIZER, 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor="val_loss", patience=5, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
checkpoint = ModelCheckpoint("models/lstm_model_best.h5", monitor="val_loss", save_best_only=True, verbose=1)
metrics_callback = MetricsPercentageCallback()

print("🛠️ Starting model training with generator...", flush=True)
history = model.fit(
    generator,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr, checkpoint, metrics_callback],
    verbose=0  # We use our custom callback for logging
)

os.makedirs("models", exist_ok=True)
model.save("models/lstm_model.h5")
print("✅ Model training complete. Saved to models/lstm_model.h5", flush=True)
