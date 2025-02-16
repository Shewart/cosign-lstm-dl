import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from lstm_model import create_lstm_model

print("🚀 Starting model training...")

# Update the data path to your processed keypoints on the external drive
DATA_PATH = "I:/wlasl-video/processed_keypoints"
sequences = []
labels = []

print("📂 Loading data from", DATA_PATH)

# List all .npy files in the processed data folder
files = [f for f in os.listdir(DATA_PATH) if f.endswith(".npy")]
print("Found processed files:", files)

if not files:
    print("❌ ERROR: No training data found in", DATA_PATH)
    exit()

# Load each sequence and assign a dummy label based on file order
for idx, file in enumerate(files):
    file_path = os.path.join(DATA_PATH, file)
    try:
        seq = np.load(file_path, allow_pickle=True)
        # Ensure each sequence is a 2D array: (num_frames, features)
        if seq.ndim != 2:
            print(f"Skipping {file}: Expected 2D array but got {seq.ndim}D")
            continue
        sequences.append(seq.tolist())  # Convert to list for pad_sequences
        labels.append(idx)  # Dummy label; update this if you have actual labels
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        continue

print(f"✅ Loaded {len(sequences)} valid sequences for training.")

# Set the target sequence length. You can adjust this value.
TARGET_SEQUENCE_LENGTH = 50

# Determine feature dimension from the first frame of the first sequence
if sequences and sequences[0]:
    feature_dim = len(sequences[0][0])
else:
    print("❌ No valid sequence frames found.")
    exit()

# Pad (or truncate) each sequence to the target length.
# This will convert each sequence to shape (TARGET_SEQUENCE_LENGTH, feature_dim)
X = pad_sequences(sequences, maxlen=TARGET_SEQUENCE_LENGTH, padding='post', truncating='post', dtype='float32')
print("Padded sequences shape:", X.shape)

# Convert labels to a numpy array
y = np.array(labels)
print(f"y shape before one-hot: {y.shape}")

# Determine the number of unique classes dynamically
num_classes = len(np.unique(y))
print(f"📊 Detected {num_classes} unique classes.")

# Convert labels to one-hot encoding
y = to_categorical(y, num_classes=num_classes)
print(f"✅ y after one-hot encoding: {y.shape}")

# The padded X has shape (num_samples, TARGET_SEQUENCE_LENGTH, feature_dim)
input_shape = (X.shape[1], X.shape[2])
print(f"Input shape for model: {input_shape}")

# Build the LSTM model
print("🛠️ Building LSTM model...")
model = create_lstm_model(input_shape, num_classes)

print(f"📈 Training model with X shape {X.shape} and y shape {y.shape}...")
history = model.fit(
    X, y,
    validation_split=0.2,
    epochs=10,        # Adjust epochs as needed
    batch_size=8,
    verbose=2
)

# Ensure the models directory exists
os.makedirs("models", exist_ok=True)

# Save the trained model (using the native Keras format)
model.save("models/lstm_model.keras")
print("✅ Model training complete. Saved to models/lstm_model.keras")
