import cv2
import mediapipe as mp
import os
import json
import numpy as np
import sys

# Paths
DATASET_DIR = "I:/wlasl-video/videos"  
OUTPUT_DIR = "I:/wlasl-video/processed_keypoints"  
ANNOTATIONS_FILE = "I:/wlasl-video/WLASL_v0.3.json"  # (Not used in this script)

print("Starting preprocessing script...", flush=True)

# Initialize Mediapipe Holistic with thresholds
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Output directory ensured:", os.path.abspath(OUTPUT_DIR), flush=True)

# Fixed-length keypoint extraction function
def extract_fixed_keypoints(results):
    # Expected counts for each set of landmarks
    expected_pose = 33
    expected_face = 468
    expected_hand = 21

    def extract_landmarks(landmarks, expected):
        if landmarks is not None:
            kp_list = []
            for i in range(expected):
                if i < len(landmarks):
                    kp = landmarks[i]
                    kp_list.extend([kp.x, kp.y, kp.z])
                else:
                    kp_list.extend([0, 0, 0])
            return kp_list
        else:
            return [0, 0, 0] * expected

    pose = extract_landmarks(results.pose_landmarks.landmark if results.pose_landmarks else None, expected_pose)
    face = extract_landmarks(results.face_landmarks.landmark if results.face_landmarks else None, expected_face)
    lh = extract_landmarks(results.left_hand_landmarks.landmark if results.left_hand_landmarks else None, expected_hand)
    rh = extract_landmarks(results.right_hand_landmarks.landmark if results.right_hand_landmarks else None, expected_hand)

    return pose + face + lh + rh  # Total features = 33*3 + 468*3 + 21*3 + 21*3 = 1629

# List video files in the dataset directory
video_files = [f for f in os.listdir(DATASET_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]
print("Found video files:", video_files, flush=True)

if not video_files:
    print("No video files found in", DATASET_DIR, flush=True)
    sys.exit(1)

# Process each video file
for video_file in video_files:
    video_path = os.path.join(DATASET_DIR, video_file)
    print(f"\nProcessing video: {video_path}", flush=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_file}", flush=True)
        continue

    keypoints_sequence = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"Finished processing {video_file}", flush=True)
            break

        frame_count += 1
        print(f"Processing frame {frame_count} for {video_file}...", flush=True)

        # Convert frame to RGB for Mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)

        # Extract fixed keypoints and append to sequence
        keypoints = extract_fixed_keypoints(results)
        keypoints_sequence.append(keypoints)

    cap.release()
    print(f"Finished processing {video_file}, total frames: {frame_count}", flush=True)
    print(f"Total keypoint sequences collected: {len(keypoints_sequence)}", flush=True)
    
    # Save the processed keypoints as a .npy file
    output_file = os.path.join(OUTPUT_DIR, f"{os.path.splitext(video_file)[0]}.npy")
    try:
        np.save(output_file, np.array(keypoints_sequence))
        print(f"✅ Saved processed data to {output_file}", flush=True)
    except Exception as e:
        print("Error saving data for", video_file, ":", e, flush=True)

print("🎉 Preprocessing complete. Keypoints saved in:", os.path.abspath(OUTPUT_DIR), flush=True)
