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
        output_file = f"bark_{language}_output.wav"
    
    print(f"Generating speech using XTTS with language: {language}")
    
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize TTS with XTTS v2 model
    tts = TTS("tts_models/multilingual/multi-dataset/bark").to(device)

    start_time = time.time()

    tts.tts_to_file(
        text="Ovo je test.",
        file_path=output_file
    )   

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
    print(f"Language: {language}")
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
