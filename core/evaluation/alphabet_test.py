"""
alphabet_recognition.py
-----------------------
Specialized script for American Sign Language (ASL) alphabet recognition.
Uses MediaPipe for landmark extraction and the existing LSTM model architecture.
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp
from collections import deque
import matplotlib.pyplot as plt
import json

# Fix import path for data_preprocessing_fixed
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.data_preprocessing_fixed import extract_keypoints  # Fixed import

# Configuration
SEQUENCE_LENGTH = 20  # ASL alphabet uses static poses, so fewer frames are needed
CONFIDENCE_THRESHOLD = 0.75
HOLD_FRAMES = 15  # Number of frames to hold the same prediction for confirmation
MODEL_PATH = "models/alphabet_model.h5"  # We'll train this model separately

# Check compatibility with installed versions
def check_compatibility():
    """Verify compatibility of installed packages with this script"""
    compatibility_issues = []
    
    # Check OpenCV
    cv_version = cv2.__version__
    if cv_version.startswith('4.11'):
        compatibility_issues.append(f"OpenCV {cv_version} might have conflicts with MediaPipe 0.10.8. " 
                                   f"Consider downgrading to 4.8.0.76.")
    
    # Check MediaPipe
    mp_version = mp.__version__
    if mp_version != "0.10.8":
        compatibility_issues.append(f"This script was tested with MediaPipe 0.10.8, but {mp_version} is installed. " 
                                   f"There might be compatibility issues.")
    
    # Check TensorFlow
    tf_version = tf.__version__
    if not tf_version.startswith('2.10'):
        compatibility_issues.append(f"This script was tested with TensorFlow 2.10.0, but {tf_version} is installed.")
    
    # Report any issues
    if compatibility_issues:
        print("⚠️ Compatibility Warnings:")
        for issue in compatibility_issues:
            print(f"  - {issue}")
        print("The script may still work, but consider updating packages if you encounter errors.")
    else:
        print("✅ All package versions are compatible.")
    
    return len(compatibility_issues) == 0

class AlphabetRecognizer:
    def __init__(self, model_path=MODEL_PATH):
        self.initialize_model(model_path)
        self.initialize_mediapipe()
        self.initialize_display()
        
    def initialize_model(self, model_path):
        """Load model if exists, otherwise prepare for training mode"""
        if os.path.exists(model_path):
            try:
                self.model = tf.keras.models.load_model(model_path)
                print(f"✅ Model loaded from {model_path}")
                self.mode = "recognition"
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                self.model = None
                self.mode = "collection"
        else:
            print(f"⚠️ Model not found at {model_path}, entering data collection mode")
            self.model = None
            self.mode = "collection"
        
        # Try to load alphabet mapping from JSON
        mapping_path = os.path.join(os.path.dirname(model_path), "alphabet_mapping.json")
        if os.path.exists(mapping_path):
            try:
                with open(mapping_path, "r") as f:
                    # Convert string keys back to integers
                    mapping = json.load(f)
                    self.alphabet = {int(k): v for k, v in mapping.items()}
                print(f"✅ Loaded alphabet mapping with {len(self.alphabet)} letters")
            except Exception as e:
                print(f"⚠️ Could not load alphabet mapping: {e}")
                # Fall back to default mapping
                self.alphabet = {i: chr(65 + i) for i in range(26)}
                self.alphabet[26] = "SPACE"
        else:
            # Define default alphabet mapping (0-25 for A-Z, 26 for space)
            self.alphabet = {i: chr(65 + i) for i in range(26)}
            self.alphabet[26] = "SPACE"
            
        self.reverse_alphabet = {v: k for k, v in self.alphabet.items()}
        
    def initialize_mediapipe(self):
        """Initialize MediaPipe with optimized settings for hand landmarks"""
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        try:
            # For ASL alphabet, we need higher detection confidence since we're looking for precise hand shapes
            self.holistic = self.mp_holistic.Holistic(
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7,
                model_complexity=1,  # Use 1 for balance between accuracy and speed
                refine_face_landmarks=False  # Face details less important for alphabet
            )
            print("✅ MediaPipe initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing MediaPipe: {e}")
            print("This might be due to version incompatibility. Try using mediapipe==0.10.8")
            sys.exit(1)
        
    def initialize_display(self):
        """Prepare display and tracking variables"""
        self.sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.prediction_history = deque(maxlen=5)
        self.hold_prediction = None
        self.hold_counter = 0
        self.last_letter = None
        self.collected_sequences = {}  # For data collection mode
        self.current_collection_letter = None
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.fps = 0
        
    def open_webcam(self, device_id=0, width=640, height=480):
        """Open webcam with specific settings"""
        self.cap = cv2.VideoCapture(device_id)
        if not self.cap.isOpened():
            print("❌ ERROR: Could not open webcam")
            sys.exit(1)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        print(f"🎥 Webcam opened successfully at {width}x{height}")
        
        # Create display windows
        cv2.namedWindow('ASL Alphabet Recognition', cv2.WINDOW_NORMAL)
        if self.mode == "collection":
            cv2.namedWindow('Collection Guide', cv2.WINDOW_NORMAL)
            
    def process_frame(self, frame):
        """Process a single frame through MediaPipe"""
        # Calculate FPS
        self.fps_counter += 1
        if (time.time() - self.fps_start_time) > 1:
            self.fps = self.fps_counter / (time.time() - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = time.time()
            
        # Prepare frame
        frame = cv2.flip(frame, 1)  # Mirror for intuitive interaction
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        frame_rgb.flags.writeable = False
        results = self.holistic.process(frame_rgb)
        frame_rgb.flags.writeable = True
        
        # Extract keypoints for hand landmarks
        try:
            keypoints = extract_keypoints(results)
            if keypoints is not None:
                self.sequence_buffer.append(keypoints)
        except Exception as e:
            print(f"❌ Error extracting keypoints: {e}")
            return frame, results, None
            
        return frame, results, keypoints
        
    def draw_landmarks(self, frame, results):
        """Draw MediaPipe landmarks on the frame"""
        # Draw hand landmarks with custom style
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, 
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style()
            )
            
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, 
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style()
            )
            
        # Draw pose landmarks (simplified)
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        return frame
    
    def make_prediction(self):
        """Make prediction based on sequence buffer"""
        if len(self.sequence_buffer) < SEQUENCE_LENGTH:
            return None, 0
            
        # Prepare model input
        model_input = np.expand_dims(np.array(self.sequence_buffer), axis=0)
        
        # Normalize input (use same normalization as training)
        model_input = (model_input - np.mean(model_input)) / (np.std(model_input) + 1e-7)
        
        # Get prediction
        prediction = self.model.predict(model_input, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        # Add to prediction history
        self.prediction_history.append((predicted_class, confidence))
        
        # Stabilize prediction by checking for consistent prediction
        if confidence >= CONFIDENCE_THRESHOLD:
            if self.hold_prediction == predicted_class:
                self.hold_counter += 1
            else:
                self.hold_prediction = predicted_class
                self.hold_counter = 1
                
            # If we've held the same prediction for enough frames, return it
            if self.hold_counter >= HOLD_FRAMES:
                predicted_letter = self.alphabet.get(predicted_class, "?")
                
                # Only update if it's a new letter (reduce flickering)
                if predicted_letter != self.last_letter:
                    self.last_letter = predicted_letter
                    return predicted_letter, confidence
                    
        return self.last_letter, confidence
    
    def handle_data_collection(self, frame, keypoints):
        """Handle data collection mode"""
        # Display guide for collecting alphabet data
        guide_img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        if self.current_collection_letter is None:
            text = "Press A-Z to start collecting data for that letter"
            cv2.putText(guide_img, text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            # Show what letter is being collected
            text = f"Collecting data for letter {self.current_collection_letter}"
            samples_text = f"Samples: {len(self.collected_sequences.get(self.current_collection_letter, []))}/100"
            cv2.putText(guide_img, text, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(guide_img, samples_text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(guide_img, "Press SPACE to finish", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # If we have keypoints and enough frames, add to collection
            if keypoints is not None and len(self.sequence_buffer) == SEQUENCE_LENGTH:
                if self.current_collection_letter not in self.collected_sequences:
                    self.collected_sequences[self.current_collection_letter] = []
                    
                if len(self.collected_sequences[self.current_collection_letter]) < 100:  # Limit samples
                    self.collected_sequences[self.current_collection_letter].append(np.array(self.sequence_buffer))
                    # Clear buffer to collect new sequence
                    self.sequence_buffer.clear()
                    
        # Show collection guide window
        cv2.imshow('Collection Guide', guide_img)
        
        # Draw collection status on main frame
        if self.current_collection_letter is not None:
            count = len(self.collected_sequences.get(self.current_collection_letter, []))
            status = f"Collecting '{self.current_collection_letter}': {count}/100 samples"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
        return frame
        
    def handle_recognition(self, frame):
        """Handle recognition mode"""
        if self.model is not None and len(self.sequence_buffer) == SEQUENCE_LENGTH:
            predicted_letter, confidence = self.make_prediction()
            
            # Make a direct prediction without stability filtering, for debugging
            model_input = np.expand_dims(np.array(self.sequence_buffer), axis=0)
            model_input = (model_input - np.mean(model_input)) / (np.std(model_input) + 1e-7)
            prediction = self.model.predict(model_input, verbose=0)
            raw_class = np.argmax(prediction[0])
            raw_confidence = np.max(prediction[0])
            raw_letter = self.alphabet.get(raw_class, "?")
            
            # Get top 3 predictions
            top_indices = np.argsort(prediction[0])[-3:][::-1]
            top_letters = [self.alphabet.get(idx, "?") for idx in top_indices]
            top_confidences = [prediction[0][idx] for idx in top_indices]
            
            # Display prediction with confidence
            # Main prediction box
            if predicted_letter is not None:
                cv2.rectangle(frame, (0, 0), (150, 120), (245, 117, 16), -1)
                cv2.putText(frame, predicted_letter, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                cv2.putText(frame, f"{confidence:.2f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
            # Raw prediction (for debugging)
            cv2.putText(frame, f"Raw: {raw_letter} ({raw_confidence:.2f})", (frame.shape[1] - 250, 40), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Show top 3 predictions
            for i, (letter, conf) in enumerate(zip(top_letters, top_confidences)):
                cv2.putText(frame, f"{i+1}. {letter}: {conf:.2f}", (10, frame.shape[0] - 20 - i*30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Draw confidence bars
                bar_length = int(conf * 150)
                cv2.rectangle(frame, (180, frame.shape[0] - 25 - i*30), 
                            (180 + bar_length, frame.shape[0] - 10 - i*30), 
                            (0, 255, 0), -1)
                
        # Display FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (frame.shape[1] - 120, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                   
        # Display recording indicator
        cv2.putText(frame, "Recording", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.circle(frame, (120, 15), 10, (0, 0, 255), -1)
        
        return frame
    
    def save_collected_data(self):
        """Save collected data for training"""
        if not self.collected_sequences:
            print("❌ No data collected!")
            return
            
        # Create output directory
        output_dir = "data/alphabet"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each letter's data
        for letter, sequences in self.collected_sequences.items():
            if not sequences:
                continue
                
            # Stack all sequences for this letter
            stacked_data = np.stack(sequences)
            output_file = os.path.join(output_dir, f"{letter}.npy")
            np.save(output_file, stacked_data)
            print(f"✅ Saved {len(sequences)} samples for letter {letter} to {output_file}")
            
        print(f"✅ All collected data saved to {output_dir}")
    
    def run(self):
        """Main loop for alphabet recognition"""
        self.open_webcam()
        print(f"🔄 Running in {self.mode.upper()} mode. Press 'q' to quit.")
        
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to grab frame. Exiting...")
                    break
                    
                # Process frame
                frame, results, keypoints = self.process_frame(frame)
                
                # Draw landmarks
                frame = self.draw_landmarks(frame, results)
                
                # Handle based on mode
                if self.mode == "collection":
                    frame = self.handle_data_collection(frame, keypoints)
                else:
                    frame = self.handle_recognition(frame)
                
                # Show the frame
                cv2.imshow('ASL Alphabet Recognition', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                # Quit on 'q'
                if key == ord('q'):
                    break
                
                # Handle recognition mode key presses
                if self.mode == "recognition":
                    # C to capture the current frame for training
                    if key == ord('c'):
                        # Save the current frame as an image
                        os.makedirs("captured_frames", exist_ok=True)
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        capture_path = os.path.join("captured_frames", f"frame_{timestamp}.jpg")
                        cv2.imwrite(capture_path, frame)
                        print(f"✅ Frame captured to {capture_path}")
                        print("To use this image for training, run:")
                        print(f"python alphabet_recognition.py --convert-image {capture_path} --letter X  # Replace X with the correct letter")
                    
                    # 0-9 to adjust confidence threshold
                    if 48 <= key <= 57:  # ASCII for 0-9
                        threshold_val = (key - 48) / 10.0  # 0.0 to 0.9
                        global CONFIDENCE_THRESHOLD
                        CONFIDENCE_THRESHOLD = threshold_val
                        print(f"🔄 Confidence threshold set to {CONFIDENCE_THRESHOLD}")
                
                # Handle collection mode key presses
                if self.mode == "collection":
                    # A-Z keys for letter selection
                    if 97 <= key <= 122:  # ASCII for a-z
                        letter = chr(key).upper()
                        self.current_collection_letter = letter
                        print(f"🔤 Now collecting samples for letter {letter}")
                    
                    # Space to finish collection
                    elif key == 32:  # Space
                        self.current_collection_letter = None
                        print("⏸ Paused collection")
                        
                    # S to save collected data
                    elif key == ord('s'):
                        self.save_collected_data()
                
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("✅ Session ended.")
            
            # Save data if in collection mode
            if self.mode == "collection" and self.collected_sequences:
                save = input("Do you want to save the collected data? (y/n): ")
                if save.lower() == 'y':
                    self.save_collected_data()

def train_alphabet_model(data_dir="data/alphabet", output_model=MODEL_PATH):
    """Train a model on the collected alphabet data"""
    from tensorflow.keras.utils import to_categorical
    from src.lstm_model import create_lstm_model, compile_model
    
    print("🔄 Loading alphabet data...")
    
    # Load data
    X = []
    y = []
    
    # Map letters to indices based on available data, not the full alphabet
    alphabet = {}
    available_letters = []
    
    # Get available letters from data directory
    for i, letter_file in enumerate(sorted(os.listdir(data_dir))):
        if not letter_file.endswith('.npy'):
            continue
            
        letter = os.path.splitext(letter_file)[0].upper()
        alphabet[letter] = i
        available_letters.append(letter)
    
    print(f"📝 Found data for letters: {', '.join(available_letters)}")
    
    # Load each letter's data
    for letter, idx in alphabet.items():
        data_path = os.path.join(data_dir, f"{letter}.npy")
        sequences = np.load(data_path)
        
        # Add to dataset
        X.append(sequences)
        y.extend([idx] * len(sequences))
    
    if not X:
        print("❌ No data found!")
        return
        
    # Combine data
    X = np.vstack(X)
    y = np.array(y)
    
    # Convert labels to categorical
    num_classes = len(alphabet)
    y = to_categorical(y, num_classes=num_classes)
    
    print(f"✅ Loaded dataset: {X.shape[0]} samples, {X.shape[1]} frames, {X.shape[2]} features")
    print(f"✅ Using {num_classes} classes for letters: {', '.join(available_letters)}")
    
    # Save alphabet mapping
    alphabet_mapping = {idx: letter for letter, idx in alphabet.items()}
    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    mapping_path = os.path.join(os.path.dirname(output_model), "alphabet_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(alphabet_mapping, f)
    print(f"✅ Saved alphabet mapping to {mapping_path}")
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and compile model
    input_shape = (X.shape[1], X.shape[2])  # (frames, features)
    model = create_lstm_model(input_shape, num_classes, use_attention=True)
    model = compile_model(model, learning_rate=1e-4)
    
    # Train model
    print("🏋️ Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=16,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    )
    
    # Evaluate model
    loss, acc = model.evaluate(X_test, y_test)
    print(f"✅ Test accuracy: {acc:.4f}")
    
    # Save model
    model.save(output_model)
    print(f"✅ Model saved to {output_model}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig("alphabet_training_history.png")
    
    return model

def process_image_to_keypoints(image_path, letter, output_dir="data/alphabet"):
    """Convert a single image to keypoints and add to training data"""
    print(f"Processing image {image_path} for letter {letter}...")
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Error: Could not read image {image_path}")
            return False
            
        # Initialize MediaPipe
        mp_holistic = mp.solutions.holistic
        holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
            refine_face_landmarks=False
        )
        
        # Process image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = holistic.process(img_rgb)
        
        # Extract keypoints
        keypoints = extract_keypoints(results)
        if keypoints is None:
            print(f"❌ Error: No keypoints detected in {image_path}")
            return False
            
        # Create a sequence by repeating the same keypoints
        keypoints_sequence = np.array([keypoints] * SEQUENCE_LENGTH)
        
        # Load existing data or create new
        output_path = os.path.join(output_dir, f"{letter}.npy")
        if os.path.exists(output_path):
            existing_data = np.load(output_path)
            # Append new sequence
            combined_data = np.vstack([existing_data, np.expand_dims(keypoints_sequence, axis=0)])
            np.save(output_path, combined_data)
            print(f"✅ Added keypoints from {image_path} to existing data for letter {letter}")
        else:
            # Save as new file
            np.save(output_path, np.expand_dims(keypoints_sequence, axis=0))
            print(f"✅ Created new keypoints data for letter {letter} from {image_path}")
        
        # Visualize the detection
        img_copy = img.copy()
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                img_copy, 
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                img_copy, 
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
            )
            
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                img_copy, 
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
            )
        
        # Save visualization
        viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        cv2.imwrite(os.path.join(viz_dir, f"{letter}_{os.path.basename(image_path)}"), img_copy)
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ASL Alphabet Recognition")
    parser.add_argument('--train', action='store_true', help="Train model from collected data")
    parser.add_argument('--collect', action='store_true', help="Force collection mode even if model exists")
    parser.add_argument('--test', action='store_true', help="Run a simple test to verify the environment")
    parser.add_argument('--convert-image', type=str, help="Convert a single image to keypoints data")
    parser.add_argument('--letter', type=str, help="Letter label for the image being converted")
    parser.add_argument('--threshold', type=float, default=0.75, help="Confidence threshold for prediction (default: 0.75)")
    parser.add_argument('--check-compatibility', action='store_true', help="Check compatibility of installed package versions")
    args = parser.parse_args()
    
    # Check compatibility if requested
    if args.check_compatibility or args.test:
        check_compatibility()
    
    # Update global confidence threshold if specified
    if args.threshold != 0.75:
        CONFIDENCE_THRESHOLD = args.threshold
        print(f"🔄 Setting confidence threshold to {CONFIDENCE_THRESHOLD}")
    
    if args.convert_image:
        if not args.letter:
            print("❌ Error: --letter is required when using --convert-image")
            sys.exit(1)
        if not os.path.exists(args.convert_image):
            print(f"❌ Error: Image file {args.convert_image} does not exist")
            sys.exit(1)
        success = process_image_to_keypoints(args.convert_image, args.letter.upper())
        if success:
            print("✅ Image converted successfully")
            print("🔍 Now you can train the model with: python alphabet_recognition.py --train")
        else:
            print("❌ Failed to convert image")
        sys.exit(0)
    elif args.test:
        print("Running test mode to verify the environment...")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Python version: {sys.version}")
        print(f"OpenCV version: {cv2.__version__}")
        print(f"MediaPipe version: {mp.__version__}")
        print(f"TensorFlow version: {tf.__version__}")
        
        # Test OpenCV window creation
        print("Testing OpenCV window creation...")
        test_image = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.putText(test_image, "Test Window", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.namedWindow('Test Window', cv2.WINDOW_NORMAL)
        cv2.imshow('Test Window', test_image)
        print("Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("OpenCV window test completed.")
        
        # Test MediaPipe initialization
        print("Testing MediaPipe initialization...")
        mp_holistic = mp.solutions.holistic
        holistic = mp_holistic.Holistic(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        holistic.close()
        print("MediaPipe initialization test completed.")
        
        sys.exit(0)
    elif args.train:
        train_alphabet_model()
    elif args.collect or not os.path.exists(MODEL_PATH):
        print("🔤 Starting data collection mode")
        recognizer = AlphabetRecognizer()
        recognizer.mode = "collection"
        recognizer.run()
    else:
        print("🔤 Starting recognition mode")
        recognizer = AlphabetRecognizer()
        recognizer.run() 