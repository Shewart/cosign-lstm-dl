import os
import cv2
import json
import dotenv
from tqdm import tqdm
import mediapipe as mp
import sys

dotenv.load_dotenv()

print("Starting convert_videos_to_points.py", flush=True)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
print("Script directory:", script_dir, flush=True)

# Construct the absolute path for the videos directory
# Expected: root/data/signs/videos
videos_dir = os.path.join(script_dir, "..", "data", "signs", "videos")
print("Videos directory (absolute):", os.path.abspath(videos_dir), flush=True)

if not os.path.exists(videos_dir):
    print("ERROR: Videos directory does not exist!", flush=True)
    sys.exit(1)

videos = os.listdir(videos_dir)
print("Found video files:", videos, flush=True)

bar = tqdm(total=len(videos), desc="Processing videos")

# Process each video file
for file in videos:
    bar.update(1)
    video_path = os.path.join(videos_dir, file)
    word = file.split(".")[0]
    print(f"\nProcessing video: {video_path} with label: {word}", flush=True)
    
    data = []
    frame_number = 0
    capture = cv2.VideoCapture(video_path)
    
    if not capture.isOpened():
        print(f"ERROR: Unable to open video: {video_path}", flush=True)
        continue

    while True:
        success, frame = capture.read()
        if not success:
            break

        frame_number += 1

        # Resize frame to a fixed size
        frame = cv2.resize(frame, (640, 480))
        # Convert frame to RGB for Mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame with Mediapipe Pose and Hands models
        pose_model = mp.solutions.pose.Pose()
        hand_model = mp.solutions.hands.Hands()

        poses = pose_model.process(frame_rgb)
        hands = hand_model.process(frame_rgb)

        # Create a point structure: [frame_number, pose_landmarks, [left_hand, right_hand]]
        point = [frame_number, [], [[], []]]

        if poses.pose_landmarks:
            for idx, landmark in enumerate(poses.pose_landmarks.landmark):
                point[1].append([idx, landmark.x * 640, landmark.y * 480, landmark.z])
        
        if hands.multi_hand_landmarks:
            for landmarks in hands.multi_handmarks:
                hand = landmarks.landmark
                # Determine handedness based on wrist and thumb CMC positions
                if hand[mp.solutions.hands.HandLandmark.WRIST].x < hand[mp.solutions.hands.HandLandmark.THUMB_CMC].x:
                    handedness = 0
                else:
                    handedness = 1
                for idx, landmark in enumerate(hand):
                    point[2][handedness].append([idx, landmark.x * 640, landmark.y * 480, landmark.z])
        else:
            # If no hand landmarks are found, skip this frame
            continue

        data.append(point)

    capture.release()
    print(f"Processed {frame_number} frames for video {file}", flush=True)
    print(f"Total data points for {word}: {len(data)}", flush=True)
    
    # Define the output directory for processed data: root/data/processed_videos
    output_dir = os.path.join(script_dir, "..", "data", "processed_videos")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory ensured: {os.path.abspath(output_dir)}", flush=True)
    output_file = os.path.join(output_dir, f"{word}.json")
    
    try:
        with open(output_file, "w") as f:
            json.dump(data, f)
        print(f"Data for {word} saved to {output_file}", flush=True)
    except Exception as e:
        print("Error saving data:", e, flush=True)
        continue

bar.close()
print("All videos processed.", flush=True)
