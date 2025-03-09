import numpy as np
import os
import sys
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from lstm_model import create_lstm_model

print("🚀 Starting model training...", flush=True)

# Update the data path to your processed keypoints on the external drive
DATA_PATH = "I:/wlasl-video/processed_keypoints"
sequences = []
labels = []

print("📂 Loading data from", DATA_PATH, flush=True)

# Set the expected feature dimension per frame (e.g., 1629 from our extraction)
EXPECTED_FEATURES = 1629

# List all .npy files in the processed data folder
files = [f for f in os.listdir(DATA_PATH) if f.endswith(".npy")]
print("Found processed files:", files, flush=True)

if not files:
    print("❌ ERROR: No training data found in", DATA_PATH, flush=True)
    sys.exit(1)

# Load each sequence and assign a dummy label based on file order
for idx, file in enumerate(files):
    file_path = os.path.join(DATA_PATH, file)
    print(f"Loading file {idx+1}/{len(files)}: {file}", flush=True)
    try:
        seq = np.load(file_path, allow_pickle=True)
        # Ensure the sequence is a 2D array and its feature dimension matches EXPECTED_FEATURES
        if seq.ndim != 2:
            print(f"Skipping {file}: Expected 2D array but got {seq.ndim}D", flush=True)
            continue
        if seq.shape[1] != EXPECTED_FEATURES:
            print(f"Skipping {file}: Expected feature dimension {EXPECTED_FEATURES} but got {seq.shape[1]}", flush=True)
            continue
        # Try converting the sequence to a numpy array with dtype float32
        try:
            np_seq = np.asarray(seq, dtype='float32')
        except MemoryError as me:
            print(f"Skipping {file} due to memory error: {me}", flush=True)
            continue
        sequences.append(np_seq.tolist())  # Convert back to list for pad_sequences
        labels.append(idx)  # Dummy label; update if you have actual labels
    except Exception as e:
        print(f"Error processing file {file}: {e}", flush=True)
        continue

print(f"✅ Loaded {len(sequences)} valid sequences for training.", flush=True)

# Set the target sequence length (number of frames)
TARGET_SEQUENCE_LENGTH = 50  

# Determine feature dimension from the first frame of the first sequence
if sequences and sequences[0]:
    feature_dim = len(sequences[0][0])
    print(f"Determined feature dimension: {feature_dim}", flush=True)
else:
    print("❌ No valid sequence frames found.", flush=True)
    sys.exit(1)

# Pad (or truncate) each sequence to the target length.
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

# Convert labels to a numpy array
y = np.array(labels)
print(f"y shape before one-hot: {y.shape}", flush=True)

# Determine the number of unique classes dynamically
num_classes = len(np.unique(y))
print(f"📊 Detected {num_classes} unique classes.", flush=True)

# Convert labels to one-hot encoding
y = to_categorical(y, num_classes=num_classes)
print(f"✅ y after one-hot encoding: {y.shape}", flush=True)

# The padded X has shape (num_samples, TARGET_SEQUENCE_LENGTH, feature_dim)
input_shape = (X.shape[1], X.shape[2])
print(f"Input shape for model: {input_shape}", flush=True)

# Build the LSTM model
print("🛠️ Building LSTM model...", flush=True)
model = create_lstm_model(input_shape, num_classes)

print(f"📈 Training model with X shape {X.shape} and y shape {y.shape}...", flush=True)
history = model.fit(
    X, y,
    validation_split=0.2,
    epochs=10,       
    batch_size=8,
    verbose=2
)

# Ensure the models directory exists
os.makedirs("models", exist_ok=True)

# Save the trained model (using the native Keras format)
model.save("models/lstm_model.keras")
print("✅ Model training complete. Saved to models/lstm_model.keras", flush=True)
