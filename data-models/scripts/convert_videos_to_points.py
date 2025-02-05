import os
import cv2
import json
import dotenv
# import psycopg2
from tqdm import tqdm
import mediapipe as mp

dotenv.load_dotenv()

# Comment out DB code since we’re not using Postgres for now
# conn = psycopg2.connect(
#     database="signs",
#     host="localhost",
#     user="postgres",
#     password=os.getenv("POSTGRES_PASSWORD"),
#     port=5432,
# )
# cur = conn.cursor()

mp_drawing = mp.solutions.drawing_utils
pose_tools = mp.solutions.pose
pose_model = pose_tools.Pose()
hand_tools = mp.solutions.hands
hand_model = hand_tools.Hands()

# List videos from the specified directory
videos = os.listdir("../data/signs/videos/")
print("Found video files:", videos)  # Debug: list video files

bar = tqdm(total=len(videos))

try:
    for i, file in enumerate(videos):
        bar.update(1)
        video = os.path.join("../data/signs/videos/", file)
        word = file.split(".")[0]

        data = []
        frame_number = 0
        capture = cv2.VideoCapture(video)
        
        while capture.isOpened():
            success, frame = capture.read()
            if not success:
                break

            frame = cv2.resize(frame, (640, 480))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            poses = pose_model.process(frame_rgb)
            hands = hand_model.process(frame_rgb)

            frame_number += 1
            point = [
                frame_number,
                [],
                [[], []],
            ]  # [frame_number, pose_landmarks, [left_hand_landmarks, right_hand_landmarks]]

            if poses.pose_landmarks:
                for idx, landmark in enumerate(poses.pose_landmarks.landmark):
                    point[1].append([idx, landmark.x * 640, landmark.y * 480, landmark.z])

            if hands.multi_hand_landmarks:
                for landmarks in hands.multi_hand_landmarks:
                    hand = landmarks.landmark
                    if hand[hand_tools.HandLandmark.WRIST].x < hand[hand_tools.HandLandmark.THUMB_CMC].x:
                        handedness = 0
                    else:
                        handedness = 1

                    for idx, landmark in enumerate(hand):
                        point[2][handedness].append([idx, landmark.x * 640, landmark.y * 480, landmark.z])
            else:
                continue

            data.append(point)
        
        capture.release()
        print(f"Processed {frame_number} frames for video {file}")
        print(f"Total processed data points for {word}: {len(data)}")
        
        # Save the data as JSON
        output_dir = "../data/processed_videos"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory ensured: {output_dir}")
        output_file = os.path.join(output_dir, f"{word}.json")
        
        try:
            with open(output_file, "w") as f:
                json.dump(data, f)
            print(f"Data for {word} saved to {output_file}")
        except Exception as e:
            print("Error saving data:", e)
            continue

except Exception as e:
    print(e)
