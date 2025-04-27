import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import sys
from speech_to_text import recognize_speech
from text_to_speech import speak_text

# Import the extract_keypoints function from data_preprocessing_fixed
from data_preprocessing_fixed import extract_keypoints

def sign_to_speech():
    """
    Captures sign language from the webcam, uses the LSTM model to predict the sign,
    and converts the prediction into audible speech.
    """
    # Load the trained model
    MODEL_PATH = "models/lstm_model.h5"
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print("❌ ERROR loading model:", e)
        sys.exit(1)
    actions = [f"Sign {i}" for i in range(30)]  # Adjust if you have a mapping dictionary

    # Initialize Mediapipe Holistic
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ ERROR: Could not open webcam")
        sys.exit(1)
    print("🎥 Webcam opened successfully!")

    # Create a named window
    cv2.namedWindow('Sign-to-Speech', cv2.WINDOW_NORMAL)

    # Buffer to collect a sequence of frames
    sequence_buffer = []
    SEQUENCE_LENGTH = 30

    print("⏳ Please perform a sign now...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("❌ ERROR: Failed to capture frame. Exiting...")
            break

        # Flip frame horizontally for a mirror view
        frame = cv2.flip(frame, 1)
        # Convert frame to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)

        try:
            keypoints = extract_keypoints(results)
        except Exception as e:
            print("Error extracting keypoints:", e)
            continue

        sequence_buffer.append(keypoints)
        if len(sequence_buffer) > SEQUENCE_LENGTH:
            sequence_buffer.pop(0)

        # Show the video feed for visual confirmation
        cv2.imshow('Sign-to-Speech', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("🔴 Exiting sign-to-speech mode")
            break

        # Once a full sequence is collected, perform prediction
        if len(sequence_buffer) == SEQUENCE_LENGTH:
            input_sequence = np.expand_dims(np.array(sequence_buffer), axis=0)  # shape: (1, 30, features)
            prediction = model.predict(input_sequence)
            predicted_class = int(np.argmax(prediction))
            confidence = float(np.max(prediction))
            predicted_sign = actions[predicted_class]
            print(f"🧠 Predicted sign: {predicted_sign} (Confidence: {confidence:.2f})")
            # Convert the prediction to speech
            speak_text(f"The predicted sign is {predicted_sign}")
            # Exit after one prediction (or you can continue in a loop)
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Sign-to-speech mode ended.")

def speech_to_sign():
    """
    Captures spoken input, transcribes it to text, and then echoes it back as a placeholder
    for a sign generation system.
    """
    print("🎤 Please speak now...")
    text = recognize_speech()
    if text:
        print("📝 Transcribed speech:", text)
        # For now, we'll just echo the text as the corresponding sign.
        speak_text(f"You said: {text}.")
    else:
        print("⚠️ No speech recognized.")

def main():
    print("=== Bidirectional Translation System ===")
    print("Select mode:")
    print("1: Sign-to-Speech Translation")
    print("2: Speech-to-Sign Translation")
    choice = input("Enter 1 or 2: ").strip()
    if choice == "1":
        sign_to_speech()
    elif choice == "2":
        speech_to_sign()
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()
