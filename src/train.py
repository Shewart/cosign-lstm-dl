import numpy as np
import os
import sys
import json
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from lstm_model import create_lstm_model

print("🚀 Starting model training...", flush=True)

# Paths
DATA_PATH = "I:/wlasl-video/processed_keypoints"
JSON_PATH = "I:/wlasl-video/WLASL_v0.3.json"
sequences = []
labels = []

print("📂 Loading data from", DATA_PATH, flush=True)

EXPECTED_FEATURES = 1629  # from extraction
TARGET_SEQUENCE_LENGTH = 50  # number of frames per sequence

# List all .npy files in the processed data folder
files = [f for f in os.listdir(DATA_PATH) if f.endswith(".npy")]
print(f"Found {len(files)} processed files.", flush=True)

if not files:
    print("❌ ERROR: No training data found in", DATA_PATH, flush=True)
    sys.exit(1)

# Load JSON mapping for video_id to gloss labels
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
        video_id = str(inst.get("video_id")).zfill(5)  # zero-pad if needed
        if video_id not in videoid_to_gloss:
            videoid_to_gloss[video_id] = gloss

# Build gloss to label mapping (each unique sign gets a unique integer)
gloss_set = set(videoid_to_gloss.values())
gloss_to_label = {gloss: idx for idx, gloss in enumerate(sorted(gloss_set))}
print(f"Found {len(gloss_to_label)} unique gloss labels.", flush=True)

# Build file and label lists using the JSON mapping
file_list = []
label_list = []
for f in sorted(files):
    video_id = os.path.splitext(f)[0]  # assuming filename matches video_id
    if video_id not in videoid_to_gloss:
        print(f"Skipping {f}: video_id {video_id} not found in JSON mapping.", flush=True)
        continue
    gloss = videoid_to_gloss[video_id]
    label = gloss_to_label[gloss]
    file_list.append(f)
    label_list.append(label)
    
print(f"Using {len(file_list)} files for training.", flush=True)

# For testing, limit to a subset (uncomment if needed)
file_list = file_list[:1000]
label_list = label_list[:1000]

# Load sequences from files
for idx, file in enumerate(file_list):
    file_path = os.path.join(DATA_PATH, file)
    print(f"Loading file {idx+1}/{len(file_list)}: {file}", flush=True)
    try:
        seq = np.load(file_path, allow_pickle=True)
        if seq.ndim != 2:
            print(f"Skipping {file}: Expected 2D array but got {seq.ndim}D", flush=True)
            continue
        if seq.shape[1] != EXPECTED_FEATURES:
            print(f"Skipping {file}: Expected feature dimension {EXPECTED_FEATURES} but got {seq.shape[1]}", flush=True)
            continue
        np_seq = np.asarray(seq, dtype='float32')
        sequences.append(np_seq.tolist())
    except Exception as e:
        print(f"Error processing file {file}: {e}", flush=True)
        continue

print(f"✅ Loaded {len(sequences)} valid sequences for training.", flush=True)

if sequences and sequences[0]:
    feature_dim = len(sequences[0][0])
    print(f"Determined feature dimension: {feature_dim}", flush=True)
else:
    print("❌ No valid sequence frames found.", flush=True)
    sys.exit(1)

print("Padding sequences...", flush=True)
try:
    X = pad_sequences(
        sequences,
        maxlen=TARGET_SEQUENCE_LENGTH,
        padding='post',
        truncating='post',
        dtype='float32'
    )
except MemoryError as me:
    print("Memory error during padding:", me, flush=True)
    sys.exit(1)

print("Padded sequences shape:", X.shape, flush=True)

# Convert labels to numpy array
y = np.array(label_list)
print(f"y shape before one-hot: {y.shape}", flush=True)

# One-hot encoding:
# For each label, create a binary vector of length num_classes where only the index corresponding to the label is 1.
num_classes = len(gloss_to_label)
print(f"📊 Using fixed number of classes: {num_classes}", flush=True)
y = to_categorical(y, num_classes=num_classes)
print(f"✅ y after one-hot encoding: {y.shape}", flush=True)

input_shape = (X.shape[1], X.shape[2])
print(f"Input shape for model: {input_shape}", flush=True)

print("🛠️ Building LSTM model...", flush=True)
model = create_lstm_model(input_shape, num_classes)

# Callbacks for training improvements
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
early_stopping = EarlyStopping(monitor="val_loss", patience=5, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
checkpoint = ModelCheckpoint("models/lstm_model_best.h5", monitor="val_loss", save_best_only=True, verbose=1)

print(f"📈 Training model with X shape {X.shape} and y shape {y.shape}...", flush=True)
history = model.fit(
    X, y,
    validation_split=0.2,
    epochs=50,
    batch_size=8,
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=2
)

os.makedirs("models", exist_ok=True)
model.save("models/lstm_model.h5")
print("✅ Model training complete. Saved to models/lstm_model.h5", flush=True)
