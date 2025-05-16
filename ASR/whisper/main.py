import whisper
import os
import sys
import subprocess
import requests
import time
import soundfile as sf

# Main function
def main():
    # Print Python version
    print(f"Running with Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    try:
        print("Loading Whisper model...")
        model = whisper.load_model("base")
        print("Model loaded successfully.")
        
        print("Transcribing audio in Croatian...")
        start_time = time.time()
        result = model.transcribe("sample_voice.wav", language="hr")
        
        # Calculate total processing time
        total_time = time.time() - start_time
        
        # Get audio duration
        audio_info = sf.info("sample_voice.wav")
        audio_duration = audio_info.duration
        
        # Calculate the ratio
        ratio = total_time / audio_duration
        
        print("\nTranscription result:")
        print(result["text"])
        
        # Print performance results
        print("\nPerformance Results:")
        print(f"Model: Whisper Base")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Audio duration: {audio_duration:.2f} seconds")
        print(f"Performance ratio: {ratio:.2f}x (i.e., {ratio:.2f} seconds of processing time per 1 second of audio)")
        print(f"In other words: {total_time:.2f} seconds were needed to transcribe {audio_duration:.2f} seconds of speech")
    except Exception as e:
        print(f"Error during transcription: {e}")

if __name__ == "__main__":
    main()