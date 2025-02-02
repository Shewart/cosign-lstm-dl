import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from lstm_model import create_lstm_model

print("🚀 Starting model training...")

# Load preprocessed data
DATA_PATH = "data/"
sequences = []
labels = []

print("📂 Loading data...")

for file in os.listdir(DATA_PATH):
    if file.endswith(".npy"):
        sequence = np.load(os.path.join(DATA_PATH, file))
        sequences.append(sequence)
        labels.append(int(file.split("_")[1].split(".")[0]))  # Extract label from filename

# Check if data is available
if len(sequences) == 0:
    print("❌ ERROR: No training data found! Make sure .npy files exist in 'data/'")
    exit()

X = np.array(sequences)
y = np.array(labels)

print(f"✅ Loaded {len(X)} sequences for training.")

# Detect number of unique labels dynamically
num_classes = len(np.unique(y))
print(f"📊 Detected {num_classes} unique classes.")

# Convert labels to one-hot encoding
y = to_categorical(y, num_classes=num_classes)
print(f"✅ y after one-hot encoding: {y.shape}")

# Define input shape
input_shape = (X.shape[1], X.shape[2])  # (sequence_length, features)

# Create LSTM model
print("🛠️ Building LSTM model...")
model = create_lstm_model(input_shape, num_classes)

# Print data shape before training
print(f"📈 Training model with input shape {X.shape} and output shape {y.shape}...")

# Train the model
history = model.fit(
    X, y,
    validation_split=0.2,
    epochs=10,  # Reduce for testing, increase after debugging
    batch_size=8,
    verbose=2
)

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Save trained model
model.save("models/lstm_model.keras")
print("✅ Model training complete. Saved to models/lstm_model.h5")
