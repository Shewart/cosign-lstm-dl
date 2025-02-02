import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    """
    Extract keypoints from MediaPipe results.
    """
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 3)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

def collect_data(video_source=0, num_sequences=30, sequence_length=30, save_path="data"):
    """
    Capture video and extract keypoints for each frame.
    """
    cap = cv2.VideoCapture(video_source)
    holistic = mp_holistic.Holistic()

    os.makedirs(save_path, exist_ok=True)

    for seq in range(num_sequences):
        sequence = []
        for frame_num in range(sequence_length):
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_frame)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

            # Display video feed
            cv2.imshow('Frame', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        np.save(os.path.join(save_path, f"seq_{seq}.npy"), np.array(sequence))

    cap.release()
    cv2.destroyAllWindows()
    print(f"Data saved in {save_path}/")

if __name__ == "__main__":
    collect_data()
