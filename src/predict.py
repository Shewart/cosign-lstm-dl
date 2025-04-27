"""
predict.py
----------
This script loads the trained LSTM model and performs real-time sign language prediction using 
Mediapipe for keypoint extraction from webcam frames.
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp
from collections import deque
import json
from data_preprocessing_fixed import extract_keypoints  # Import from fixed version
import argparse
import pkg_resources
import platform
from dotenv import load_dotenv
import datetime as dt
from tqdm import tqdm

# Check compatibility of installed packages
def check_version_compatibility():
    print("Checking version compatibility...")
    
    warnings = []
    
    # Check TensorFlow version
    tf_version = tf.__version__
    expected_tf_version = "2.10.0"
    if tf_version != expected_tf_version:
        warnings.append(f"⚠️ TensorFlow version mismatch: found {tf_version}, expected {expected_tf_version}")
    
    # Check OpenCV version
    cv_version = cv2.__version__
    expected_cv_version = "4.8.0.76"
    if cv_version != expected_cv_version:
        warnings.append(f"⚠️ OpenCV version mismatch: found {cv_version}, expected {expected_cv_version}")
        # Specific contrib check
        try:
            # Just try to access a known contrib module to see if it's available
            _ = cv2.aruco
        except AttributeError:
            warnings.append("⚠️ OpenCV contrib modules not available but using opencv-python. You may need opencv-contrib-python.")
    
    # Check MediaPipe version
    try:
        mp_version = pkg_resources.get_distribution("mediapipe").version
        expected_mp_version = "0.10.8"
        if mp_version != expected_mp_version:
            warnings.append(f"⚠️ MediaPipe version mismatch: found {mp_version}, expected {expected_mp_version}")
    except (ImportError, pkg_resources.DistributionNotFound):
        warnings.append("⚠️ MediaPipe not installed properly")
    
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
        print("✅ All package versions match expected values")

# Run compatibility check if not imported as a module
if __name__ == "__main__":
    check_version_compatibility()

# Load environment variables
load_dotenv()

# Enable optimizations for inference
try:
    # Enable memory growth to avoid allocating all GPU memory
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("✅ GPU memory growth enabled")
except Exception as e:
    print(f"⚠️ GPU optimization warning: {e}")

def initialize_model(model_path="models/lstm_model.h5"):
    """Load and optimize the model for inference"""
    try:
        # Load the model with h5 format
        model = tf.keras.models.load_model(model_path)
        
        # Optimize for inference
        model = optimize_model_for_inference(model)
        
        print("✅ Model loaded and optimized successfully!")
        return model
    except Exception as e:
        print("❌ ERROR: Could not load model:", e)
        sys.exit(1)

def optimize_model_for_inference(model):
    """Apply TF optimizations for faster inference"""
    try:
        # Convert to TensorRT if available (much faster on NVIDIA GPUs)
        if hasattr(tf, 'experimental') and hasattr(tf.experimental, 'tensorrt'):
            try:
                conversion_params = tf.experimental.tensorrt.ConversionParams(
                    precision_mode='FP16',
                    maximum_cached_engines=100
                )
                converter = tf.experimental.tensorrt.Converter(
                    input_saved_model_dir=model,
                    conversion_params=conversion_params
                )
                model = converter.convert()
                print("✅ TensorRT optimization applied")
            except Exception as e:
                print(f"⚠️ TensorRT optimization not available: {e}")
    except Exception as e:
        print(f"⚠️ Inference optimization not available: {e}")
    
    return model

def initialize_mediapipe():
    """Initialize MediaPipe holistic model with performance configurations"""
    try:
        # Initialize MediaPipe
        mp_holistic = mp.solutions.holistic
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        
        # Configure holistic with optimized parameters for real-time
        holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,  # Use medium complexity for balance of speed and accuracy
            smooth_landmarks=True,  # Temporally filter landmarks to reduce jitter
            refine_face_landmarks=False  # Set to True only if needed 
        )
        
        return mp_holistic, mp_drawing, mp_drawing_styles, holistic
    except Exception as e:
        print(f"Error initializing MediaPipe: {e}")
        print("\nPossible causes:")
        print("1. MediaPipe version incompatibility")
        print("2. Missing OpenCV dependencies")
        print("3. Hardware/OS compatibility issues")
        print("\nTry running the compatibility check with '--check-compatibility'")
        sys.exit(1)

def open_webcam(device_id=0, width=640, height=480, fps=30):
    """Open webcam with optimized settings"""
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        print("❌ ERROR: Could not open webcam")
        sys.exit(1)
    
    # Set optimized webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering for real-time
    
    print(f"🎥 Webcam opened successfully at {width}x{height}@{fps}fps")
    return cap

def load_label_mapping(json_path="models/label_mapping.json"):
    """Load label mapping from JSON file for prediction display"""
    try:
        with open(json_path, "r") as f:
            label_to_gloss = json.load(f)
        
        print(f"✅ Loaded {len(label_to_gloss)} labels from mapping file")
        return label_to_gloss
    except Exception as e:
        print(f"⚠️ Warning: Could not load label mapping: {e}")
        
        # Try the old method using the WLASL JSON if label_mapping.json doesn't exist
        try:
            wlasl_path = "processed_vids/wlasl-video/WLASL_v0.3.json"
            with open(wlasl_path, "r") as f:
                data = json.load(f)
            
            # Extract unique glosses
            gloss_set = set()
            for entry in data:
                gloss = entry.get("gloss")
                if gloss:
                    gloss_set.add(gloss)
            
            # Create mapping
            gloss_to_label = {gloss: idx for idx, gloss in enumerate(sorted(gloss_set))}
            label_to_gloss = {str(idx): gloss for gloss, idx in gloss_to_label.items()}
            
            return label_to_gloss
        except Exception as e2:
            print(f"⚠️ Warning: Could not load fallback mapping: {e2}")
            return {str(i): f"Class {i}" for i in range(100)}  # Fallback

def main():
    # Initialize components
    model = initialize_model()
    mp_holistic, mp_drawing, mp_drawing_styles, holistic = initialize_mediapipe()
    cap = open_webcam()

    # Load label mapping
    label_to_gloss = load_label_mapping()
    
    # Create a named window
    cv2.namedWindow('Sign Language Recognition', cv2.WINDOW_NORMAL)
    
    # Sequence buffer with efficient frame collection
    sequence_buffer = deque(maxlen=50)  # Using deque for efficient fixed-size buffer
    SEQUENCE_LENGTH = 50
    
    # For prediction smoothing
    prediction_history = deque(maxlen=5)
    confidence_threshold = 0.7
    
    # FPS calculation
    fps_counter = 0
    fps_start_time = time.time()
    fps = 0

    print("⏱ Starting prediction loop. Press 'q' to exit.")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("❌ ERROR: Failed to grab frame. Exiting...")
                break

            # Calculate FPS
            fps_counter += 1
            if (time.time() - fps_start_time) > 1:
                fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()
            
            # Resize frame for faster processing while maintaining aspect ratio
            frame_height, frame_width = frame.shape[:2]
            scaling_factor = 320 / frame_height
            dim = (int(frame_width * scaling_factor), 320)
            small_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
            
            # Flip for mirror effect and convert to RGB for MediaPipe
            small_frame = cv2.flip(small_frame, 1)
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe (use smaller frame for faster processing)
            rgb_frame.flags.writeable = False  # Performance optimization
            results = holistic.process(rgb_frame)
            rgb_frame.flags.writeable = True

            # Draw landmarks on the small frame
            if results.face_landmarks:
                mp_drawing.draw_landmarks(small_frame, results.face_landmarks, 
                                        mp_holistic.FACEMESH_CONTOURS,
                                        landmark_drawing_spec=None,
                                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(small_frame, results.pose_landmarks, 
                                        mp_holistic.POSE_CONNECTIONS,
                                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(small_frame, results.left_hand_landmarks, 
                                        mp_holistic.HAND_CONNECTIONS,
                                        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(small_frame, results.right_hand_landmarks, 
                                        mp_holistic.HAND_CONNECTIONS,
                                        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())

            # Extract keypoints
            try:
                keypoints = extract_keypoints(results)
                if keypoints is not None:
                    sequence_buffer.append(keypoints)
            except Exception as e:
                print(f"❌ Error in extract_keypoints: {e}")
                continue

            # If we have enough frames, make a prediction
            if len(sequence_buffer) == SEQUENCE_LENGTH:
                # Create model input with batching dimension
                model_input = np.expand_dims(np.array(sequence_buffer), axis=0)
                
                # Normalize input (use same normalization as training)
                model_input = (model_input - np.mean(model_input)) / (np.std(model_input) + 1e-7)
                
                # Get prediction
                start_time = time.time()
                prediction = model.predict(model_input, verbose=0)
                inference_time = (time.time() - start_time) * 1000  # ms
                
                # Process prediction
                predicted_class = np.argmax(prediction[0])
                confidence = np.max(prediction[0])
                
                # Store prediction in history for smoothing
                prediction_history.append((predicted_class, confidence))
                
                # Compute smoothed prediction (majority vote with confidence threshold)
                if len(prediction_history) >= 3:
                    # Count occurrences of each class
                    class_counts = {}
                    for pred_class, conf in prediction_history:
                        if conf >= confidence_threshold:
                            class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
                    
                    # Get most frequent class
                    if class_counts:
                        smoothed_class = max(class_counts.items(), key=lambda x: x[1])[0]
                        # Get the gloss label for this class
                        gloss = label_to_gloss.get(str(smoothed_class), f"Unknown ({smoothed_class})")
                        prediction_text = f"Sign: {gloss}"
                    else:
                        prediction_text = "No confident prediction"
                else:
                    # Get the gloss label for this class
                    gloss = label_to_gloss.get(str(predicted_class), f"Unknown ({predicted_class})")
                    prediction_text = f"Sign: {gloss}"
                
                # Draw prediction info
                cv2.rectangle(small_frame, (0, 0), (640, 40), (245, 117, 16), -1)
                cv2.putText(small_frame, prediction_text, 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Display inference time and FPS
                cv2.putText(small_frame, f"Inference: {inference_time:.1f}ms FPS: {fps:.1f}", 
                           (small_frame.shape[1] - 230, small_frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Resize back to original size for display if needed
            display_frame = cv2.resize(small_frame, (frame_width, frame_height), 
                                      interpolation=cv2.INTER_LINEAR)

            # Show the frame
            cv2.imshow('Sign Language Recognition', display_frame)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("🔴 Exiting prediction loop...")
                break

    except Exception as e:
        print("❌ Unexpected error:", e)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Webcam closed successfully.")

if __name__ == "__main__":
    main()
