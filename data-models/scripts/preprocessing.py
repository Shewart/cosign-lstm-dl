import cv2
import mediapipe as mp
import os
import json
import numpy as np

# Paths
# Paths
DATASET_DIR = "I:/wlasl-video/videos"  
OUTPUT_DIR = "I:/wlasl-video/processed_keypoints"  
ANNOTATIONS_FILE = "I:/wlasl-video/WLASL_v0.3.json" 


# Initialize Mediapipe models
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Process videos
video_files = [f for f in os.listdir(DATASET_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]

for video_file in video_files:
    cap = cv2.VideoCapture(os.path.join(DATASET_DIR, video_file))
    keypoints_sequence = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)

        # Extract keypoints (pose, face, hands)
        pose = results.pose_landmarks.landmark if results.pose_landmarks else []
        face = results.face_landmarks.landmark if results.face_landmarks else []
        lh = results.left_hand_landmarks.landmark if results.left_hand_landmarks else []
        rh = results.right_hand_landmarks.landmark if results.right_hand_landmarks else []

        # Flatten and store
        keypoints = [kp.x for kp in pose] + [kp.y for kp in pose] + \
                    [kp.x for kp in face] + [kp.y for kp in face] + \
                    [kp.x for kp in lh] + [kp.y for kp in lh] + \
                    [kp.x for kp in rh] + [kp.y for kp in rh]

        keypoints_sequence.append(keypoints)

    cap.release()
    
    # Save the processed keypoints
    np.save(os.path.join(OUTPUT_DIR, f"{video_file}.npy"), np.array(keypoints_sequence))

print("Preprocessing complete. Keypoints saved in:", OUTPUT_DIR)
