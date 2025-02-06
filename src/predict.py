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
from data_preprocessing import extract_keypoints  # Ensure this function is defined properly

def initialize_model(model_path="models/lstm_model.keras"):
    try:
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        print("❌ ERROR: Could not load model:", e)
        sys.exit(1)

def initialize_mediapipe():
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    return mp_holistic, mp_drawing, holistic

def open_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ ERROR: Could not open webcam")
        sys.exit(1)
    print("🎥 Webcam opened successfully!")
    return cap

def main():
    # Initialize
    model = initialize_model()
    mp_holistic, mp_drawing, holistic = initialize_mediapipe()
    cap = open_webcam()

    # Create a named window (WINDOW_NORMAL allows resizing)
    cv2.namedWindow('Sign Language Recognition', cv2.WINDOW_NORMAL)
    
    # Define action labels (for 30 classes as placeholders)
    actions = [f"Sign {i}" for i in range(30)]
    
    # For building a full sequence of frames
    sequence_buffer = []
    SEQUENCE_LENGTH = 30

    print("⏱ Starting prediction loop. Press 'q' to exit.")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("❌ ERROR: Failed to grab frame. Exiting...")
                break

            # Flip and preprocess the frame
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame with Mediapipe
            results = holistic.process(rgb_frame)

            # Optional: Draw landmarks for debugging
            if results.face_landmarks:
                mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # Extract keypoints (should return a 1D array)
            try:
                keypoints = extract_keypoints(results)
            except Exception as e:
                print("❌ Error in extract_keypoints:", e)
                continue

            # Accumulate keypoints into a sequence buffer
            sequence_buffer.append(keypoints)
            if len(sequence_buffer) > SEQUENCE_LENGTH:
                sequence_buffer.pop(0)

            # If a full sequence is ready, perform prediction
            if len(sequence_buffer) == SEQUENCE_LENGTH:
                try:
                    input_sequence = np.expand_dims(np.array(sequence_buffer), axis=0)
                    prediction = model.predict(input_sequence)
                    predicted_class = np.argmax(prediction)
                    confidence = np.max(prediction)
                    prediction_text = f"{actions[predicted_class]} ({confidence:.2f})"
                    print("🧠 Prediction:", prediction_text)
                    cv2.putText(frame, prediction_text, (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                except Exception as e:
                    print("❌ Prediction error:", e)

            # Show the frame
            cv2.imshow('Sign Language Recognition', frame)

            # Wait for a short interval; exit if 'q' is pressed
            if cv2.waitKey(10) & 0xFF == ord('q'):
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
