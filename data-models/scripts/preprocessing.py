import cv2
import mediapipe as mp
import os
import json
import numpy as np

# Paths
DATASET_DIR = "I:/wlasl-video/videos"  
OUTPUT_DIR = "I:/wlasl-video/processed_keypoints"  
ANNOTATIONS_FILE = "I:/wlasl-video/WLASL_v0.3.json"  # (if needed later)

print("Starting preprocessing script...")

# Initialize Mediapipe Holistic with confidence thresholds
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# List video files in the dataset directory
video_files = [f for f in os.listdir(DATASET_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]
print("Found video files:", video_files)

if not video_files:
    print("No video files found in", DATASET_DIR)
    exit()

# Process each video file
for video_file in video_files:
    video_path = os.path.join(DATASET_DIR, video_file)
    print(f"\nProcessing video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_file}")
        continue

    keypoints_sequence = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Convert frame to RGB for Mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)

        # Extract keypoints for pose, face, left hand, and right hand
        pose = results.pose_landmarks.landmark if results.pose_landmarks else []
        face = results.face_landmarks.landmark if results.face_landmarks else []
        lh = results.left_hand_landmarks.landmark if results.left_hand_landmarks else []
        rh = results.right_hand_landmarks.landmark if results.right_hand_landmarks else []

        # Flatten and combine keypoints into a single list
        keypoints = (
            [kp.x for kp in pose] + [kp.y for kp in pose] +
            [kp.x for kp in face] + [kp.y for kp in face] +
            [kp.x for kp in lh] + [kp.y for kp in lh] +
            [kp.x for kp in rh] + [kp.y for kp in rh]
        )
        keypoints_sequence.append(keypoints)

        # Print progress every 10 frames
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames for video {video_file}")

    cap.release()
    print(f"Finished processing {video_file}, total frames: {frame_count}")

    # Save the processed keypoints as a .npy file (using the video filename as identifier)
    output_file = os.path.join(OUTPUT_DIR, f"{os.path.splitext(video_file)[0]}.npy")
    np.save(output_file, np.array(keypoints_sequence))
    print(f"Saved processed data to {output_file}")

print("Preprocessing complete. Keypoints saved in:", os.path.abspath(OUTPUT_DIR))
