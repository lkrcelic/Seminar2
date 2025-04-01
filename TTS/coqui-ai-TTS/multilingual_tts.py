import torch
import time
import soundfile as sf
import argparse
import os
from TTS.api import TTS

# Set environment variable to accept the license terms
os.environ["COQUI_TOS_AGREED"] = "1"

def generate_speech_multilingual(text, language="cs", output_file=None):
    """Generate speech using XTTS multilingual model"""
    # Create output filename with language prefix if not provided
    if output_file is None:
        output_file = f"{language}_output.wav"
    
    print(f"Generating speech using XTTS with language: {language}")
    
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize TTS with XTTS model
    start_time = time.time()
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    
    # Create directory for sample voices if it doesn't exist
    os.makedirs("sample_voices", exist_ok=True)
    
    # Check if we have a sample voice file, if not create one
    sample_voice_path = os.path.join("sample_voices", "sample_voice.wav")
    if not os.path.exists(sample_voice_path):
        print("Creating a sample voice file...")
        # Generate a simple sine wave as a sample voice
        import numpy as np
        sample_rate = 16000
        duration = 3  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        # Generate a 440 Hz sine wave
        data = 0.5 * np.sin(2 * np.pi * 440 * t)
        sf.write(sample_voice_path, data, sample_rate)
    
    # Generate speech
    print(f"Using sample voice from: {sample_voice_path}")
    tts.tts_to_file(text=text, speaker_wav=sample_voice_path, language=language, file_path=output_file)
    
    # Calculate performance metrics
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Get the duration of the generated audio
    audio_info = sf.info(output_file)
    audio_duration = audio_info.duration
    
    # Calculate the ratio
    ratio = processing_time / audio_duration
    
    # Print performance results
    print("\nPerformance Results:")
    print(f"Language: {language} (XTTS)")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Audio duration: {audio_duration:.2f} seconds")
    print(f"Performance ratio: {ratio:.2f}x (i.e., {ratio:.2f} seconds of processing time per 1 second of audio)")
    print(f"In other words: {processing_time:.2f} seconds were needed to synthesize {audio_duration:.2f} seconds of speech")
    
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate speech using XTTS multilingual model')
    parser.add_argument('--text', type=str, default="Dobar dan, kako si? Ja sam Lovro, a ti? Ideš li u dućan možda?", 
                        help='Text to convert to speech')
    parser.add_argument('--language', type=str, default="cs",
                        help='Language code for XTTS (e.g., cs, pl, ru)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename (default: {language}_output.wav)')
    
    args = parser.parse_args()
    
    generate_speech_multilingual(args.text, args.language, args.output)
