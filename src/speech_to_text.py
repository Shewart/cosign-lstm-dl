# src/speech_to_text.py

import speech_recognition as sr

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Listening for speech...")
        audio = recognizer.listen(source)
    try:
        # Using Google's free speech recognition service
        text = recognizer.recognize_google(audio)
        print("🧠 Recognized speech:", text)
        return text
    except sr.UnknownValueError:
        print("❌ Speech Recognition could not understand audio")
        return ""
    except sr.RequestError as e:
        print("❌ Could not request results from Google Speech Recognition service; {0}".format(e))
        return ""

if __name__ == "__main__":
    recognize_speech()
