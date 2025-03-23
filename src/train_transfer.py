import sys
import numpy as np
import os
import json
import tensorflow as tf
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras.optimizers import Adam

# --- Workaround for deserializing Kevin's pre-trained model ---
# Patch the module path so that "keras.src.models.functional" resolves correctly.
sys.modules["keras.src.models.functional"] = tf.keras.models

# Monkey-patch InputLayer.__init__ to ignore the 'batch_shape' argument.
original_inputlayer_init = InputLayer.__init__
def patched_inputlayer_init(self, *args, **kwargs):
    kwargs.pop("batch_shape", None)
    original_inputlayer_init(self, *args, **kwargs)
InputLayer.__init__ = patched_inputlayer_init

print("🚀 Starting transfer learning training with data generator...", flush=True)

# Paths
DATA_PATH = "I:/wlasl-video/processed_keypoints"
JSON_PATH = "I:/wlasl-video/WLASL_v0.3.json"
PRETRAINED_MODEL_PATH = "models/model.keras"  # Kevin's pre-trained model in native Keras format

# Hyperparameters
EXPECTED_FEATURES = 1629         # Number of features per frame
TARGET_SEQUENCE_LENGTH = 50      # Desired frames per sequence
BATCH_SIZE = 8
EPOCHS = 20

# List all .npy files in the processed data folder
all_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".npy")]
print(f"Found {len(all_files)} processed files.", flush=True)
if not all_files:
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
        video_id = str(inst.get("video_id")).zfill(5)  # Zero-pad if needed
        videoid_to_gloss[video_id] = gloss

# Build gloss-to-label mapping
gloss_set = set(videoid_to_gloss.values())
gloss_to_label = {gloss: idx for idx, gloss in enumerate(sorted(gloss_set))}
NUM_CLASSES = len(gloss_to_label)
print(f"Found {NUM_CLASSES} unique gloss labels.", flush=True)

# Build file and label lists using the JSON mapping
file_list = []
label_list = []
for f in sorted(all_files):
    video_id = os.path.splitext(f)[0]  # Assuming filename matches video_id
    if video_id not in videoid_to_gloss:
        print(f"Skipping {f}: video_id {video_id} not found in JSON mapping.", flush=True)
        continue
    gloss = videoid_to_gloss[video_id]
    label = gloss_to_label[gloss]
    file_list.append(f)
    label_list.append(label)
    
print(f"Using {len(file_list)} files for training.", flush=True)

# Optionally, limit to a subset for testing:
# file_list = file_list[:1000]
# label_list = label_list[:1000]

# Define the DataGenerator class for batch loading and padding
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

# Instantiate the generator
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

# Load the pre-trained model using our monkey-patched InputLayer and module path patch
print("Loading pre-trained model from", PRETRAINED_MODEL_PATH, flush=True)
try:
    pretrained_model = load_model(PRETRAINED_MODEL_PATH, compile=False)
except Exception as e:
    print("Error loading pre-trained model:", e, flush=True)
    sys.exit(1)

# Restore the original InputLayer.__init__
InputLayer.__init__ = original_inputlayer_init

# Freeze the pre-trained layers
for layer in pretrained_model.layers:
    layer.trainable = False

# Replace the classification head with a new Dense layer for our task.
x = pretrained_model.layers[-2].output  # Use the penultimate layer's output as features
x = Dropout(0.5)(x)
new_output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=pretrained_model.input, outputs=new_output)
print("Modified model summary:")
model.summary()

# Compile the model with a low learning rate for fine-tuning
model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for training improvements
early_stopping = EarlyStopping(monitor="val_loss", patience=5, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
checkpoint = ModelCheckpoint("models/transfer_model_best.h5", monitor="val_loss", save_best_only=True, verbose=1)

print("Training transfer learning model with generator...", flush=True)
history = model.fit(
    generator,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=2
)

os.makedirs("models", exist_ok=True)
model.save("models/transfer_model.h5")
print("✅ Transfer learning model training complete. Saved to models/transfer_model.h5", flush=True)
