# src/speech_integration.py

from speech_to_text import recognize_speech
from text_to_speech import speak_text

def main():
    print("🚀 Starting speech integration demo...")
    text = recognize_speech()
    if text:
        print("📝 Transcribed text:", text)
        # Echo the recognized text using TTS
        speak_text(f"You said: {text}")
    else:
        print("⚠️ No speech recognized.")

if __name__ == "__main__":
    main()
