import numpy as np
import os
import sys
import json
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from lstm_model import create_lstm_model

# Paths
DATA_PATH = "I:/wlasl-video/processed_keypoints"
JSON_PATH = "I:/wlasl-video/WLASL_v0.3.json"

print("🚀 Starting training with data generator...")

# Set expected feature dimension per frame
EXPECTED_FEATURES = 1629
TARGET_SEQUENCE_LENGTH = 50  # Desired number of frames per sequence

# Load JSON mapping for video IDs to gloss labels
try:
    with open(JSON_PATH, "r") as json_file:
        json_data = json.load(json_file)
except Exception as e:
    print("❌ ERROR loading JSON file:", e)
    sys.exit(1)

# Build mapping: video_id (str) -> gloss
videoid_to_gloss = {}
for entry in json_data:
    gloss = entry.get("gloss")
    instances = entry.get("instances", [])
    for inst in instances:
        video_id = str(inst.get("video_id")).zfill(5)  # Zero-pad if needed
        if video_id not in videoid_to_gloss:
            videoid_to_gloss[video_id] = gloss

# Build mapping: gloss -> integer label
gloss_set = set(videoid_to_gloss.values())
gloss_to_label = {gloss: idx for idx, gloss in enumerate(sorted(gloss_set))}
NUM_CLASSES = len(gloss_to_label)
print(f"Using {NUM_CLASSES} unique classes.", flush=True)

# Create a list of valid files and their labels
all_files = sorted([f for f in os.listdir(DATA_PATH) if f.endswith(".npy")])
file_list = []
label_list = []
for f in all_files:
    video_id = os.path.splitext(f)[0]  # Filename without extension
    if video_id not in videoid_to_gloss:
        print(f"Skipping {f}: video_id not found in JSON mapping.", flush=True)
        continue
    gloss = videoid_to_gloss[video_id]
    label = gloss_to_label[gloss]
    file_list.append(f)
    label_list.append(label)
    
print(f"Found {len(file_list)} files with valid labels.", flush=True)

# Define a generator class using Keras Sequence
class DataGenerator(Sequence):
    def __init__(self, file_list, labels, batch_size, data_path, target_len, expected_features, num_classes):
        self.file_list = file_list
        self.labels = labels
        self.batch_size = batch_size
        self.data_path = data_path
        self.target_len = target_len
        self.expected_features = expected_features
        self.num_classes = num_classes

    def __len__(self):
        return int(np.ceil(len(self.file_list) / self.batch_size))

    def __getitem__(self, index):
        batch_files = self.file_list[index*self.batch_size:(index+1)*self.batch_size]
        batch_labels = self.labels[index*self.batch_size:(index+1)*self.batch_size]
        batch_sequences = []
        for f in batch_files:
            try:
                seq = np.load(os.path.join(self.data_path, f), allow_pickle=True)
                # Validate sequence shape
                if seq.ndim != 2 or seq.shape[1] != self.expected_features:
                    print(f"Skipping {f} in generator: Invalid shape {seq.shape}", flush=True)
                    continue
                # Convert to float32
                np_seq = np.asarray(seq, dtype='float32')
                batch_sequences.append(np_seq.tolist())
            except Exception as e:
                print(f"Error loading {f} in generator: {e}", flush=True)
                continue
        if not batch_sequences:
            # If no valid data in this batch, return zeros (or handle differently)
            X_batch = np.zeros((self.batch_size, self.target_len, self.expected_features), dtype='float32')
            y_batch = to_categorical(np.zeros(self.batch_size, dtype=int), num_classes=self.num_classes)
        else:
            X_batch = pad_sequences(
                batch_sequences,
                maxlen=self.target_len,
                padding='post',
                truncating='post',
                dtype='float32'
            )
            # Adjust y_batch length if some files were skipped
            y_batch = to_categorical(np.array(batch_labels[:X_batch.shape[0]]), num_classes=self.num_classes)
        return X_batch, y_batch

# Parameters for the generator
BATCH_SIZE = 8
generator = DataGenerator(file_list, label_list, BATCH_SIZE, DATA_PATH, TARGET_SEQUENCE_LENGTH, EXPECTED_FEATURES, NUM_CLASSES)
steps_per_epoch = len(generator)
print(f"Generator will yield {steps_per_epoch} steps per epoch.", flush=True)

# Build the model using shape from generator
input_shape = (TARGET_SEQUENCE_LENGTH, EXPECTED_FEATURES)
print(f"Input shape for model: {input_shape}", flush=True)
model = create_lstm_model(input_shape, NUM_CLASSES)

print("🛠️ Starting model training...", flush=True)
history = model.fit(
    generator,
    epochs=10,
    verbose=2
)

os.makedirs("models", exist_ok=True)
model.save("models/lstm_model.keras")
print("✅ Model training complete. Saved to models/lstm_model.keras", flush=True)
