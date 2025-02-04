import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp
from data_preprocessing import extract_keypoints
import time

# Load trained model
MODEL_PATH = "models/lstm_model.keras"  # Ensure this matches your save format
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ ERROR: Could not load model:", e)
    exit()

# Load class labels (placeholder names for 30 classes)
actions = [f"Sign {i}" for i in range(30)]

# Initialize Mediapipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ ERROR: Could not open webcam")
    exit()
print("🎥 Webcam opened successfully!")

# Create a named window
cv2.namedWindow('Sign Language Recognition', cv2.WINDOW_NORMAL)

# Initialize sequence buffer for accumulating frames (for a sequence length of 30)
sequence_buffer = []
SEQUENCE_LENGTH = 30

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("❌ ERROR: Failed to grab frame. Exiting...")
            break

        # Flip frame horizontally
        frame = cv2.flip(frame, 1)

        # Convert frame to RGB for Mediapipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)

        # Optionally draw landmarks on the frame for visual feedback
        if results.face_landmarks:
            mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Extract keypoints from the current frame (this returns a 1D array)
        try:
            keypoints = extract_keypoints(results)
        except Exception as e:
            print("❌ ERROR in extract_keypoints:", e)
            continue

        # Append the keypoints to the sequence buffer
        sequence_buffer.append(keypoints)
        # Maintain a fixed sequence length by removing the oldest frame if needed
        if len(sequence_buffer) > SEQUENCE_LENGTH:
            sequence_buffer.pop(0)

        # Once we have a full sequence, make a prediction
        if len(sequence_buffer) == SEQUENCE_LENGTH:
            try:
                # Convert buffer to numpy array and add batch dimension: shape becomes (1, 30, features)
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

        # Display the frame
        cv2.imshow('Sign Language Recognition', frame)

        # Wait for 10ms; if 'q' is pressed, break the loop
        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("🔴 Exiting...")
            break

# Release resources and close the window
cap.release()
cv2.destroyAllWindows()
print("✅ Webcam closed successfully.")
