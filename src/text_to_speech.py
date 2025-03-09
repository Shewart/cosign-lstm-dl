# src/text_to_speech.py

import pyttsx3

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    print("🔊 Spoke:", text)

if __name__ == "__main__":
    speak_text("Hello, I am your text to speech system")
