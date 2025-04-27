# src/speech_to_text.py

import speech_recognition as sr
import pkg_resources
import sys

def check_compatibility():
    """Check compatibility of speech recognition related packages"""
    print("Checking speech recognition compatibility...")
    
    warnings = []
    
    # Check SpeechRecognition version
    try:
        sr_version = pkg_resources.get_distribution("SpeechRecognition").version
        expected_sr_version = "3.10.0"
        if sr_version != expected_sr_version:
            warnings.append(f"⚠️ SpeechRecognition version mismatch: found {sr_version}, expected {expected_sr_version}")
    except (ImportError, pkg_resources.DistributionNotFound):
        warnings.append("⚠️ SpeechRecognition not installed properly")
    
    # Check PyAudio version
    try:
        pa_version = pkg_resources.get_distribution("PyAudio").version
        expected_pa_version = "0.2.13"
        if pa_version != expected_pa_version:
            warnings.append(f"⚠️ PyAudio version mismatch: found {pa_version}, expected {expected_pa_version}")
    except (ImportError, pkg_resources.DistributionNotFound):
        warnings.append("⚠️ PyAudio not installed. This is required for microphone support.")
    
    # Print warnings if any
    if warnings:
        print("\nSpeech recognition compatibility issues detected:")
        for warning in warnings:
            print(warning)
        print("\nThese mismatches might cause issues with speech recognition functionality.")
        # This is optional - only exit if microphone won't work at all
        if "PyAudio not installed" in str(warnings):
            print("Speech recognition requires PyAudio for microphone access.")
            if input("Continue anyway? (y/n): ").lower().strip() != 'y':
                print("Exiting...")
                sys.exit(1)
    else:
        print("✅ All speech recognition packages are compatible")

def recognize_speech():
    try:
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
    except Exception as e:
        print(f"❌ Error during speech recognition: {e}")
        print("This may be due to missing or incompatible PyAudio installation.")
        return ""

if __name__ == "__main__":
    check_compatibility()
    recognize_speech()
