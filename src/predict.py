import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp
from data_preprocessing import extract_keypoints

# Load trained model
MODEL_PATH = "models/lstm_model.keras"  # Make sure this matches your save format
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# Load class labels
actions = [f"Sign {i}" for i in range(30)]  # Placeholder labels

# Initialize Mediapipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  # Open webcam
with mp_holistic.Holistic() as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)
        keypoints = extract_keypoints(results)
        
        # Make a prediction (only if keypoints exist)
        if np.any(keypoints):
            prediction = model.predict(np.expand_dims(keypoints, axis=0))
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)

            # Display Prediction
            cv2.putText(frame, f"{actions[predicted_class]} ({confidence:.2f})",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Sign Language Recognition', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
