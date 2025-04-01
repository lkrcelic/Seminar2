import torch
import time
import soundfile as sf
import argparse
import os
from TTS.api import TTS

# Set environment variable to accept the license terms
os.environ["COQUI_TOS_AGREED"] = "1"

def generate_speech_croatian(text, output_file=None):
    """Generate speech using Croatian-specific model"""
    # Create output filename if not provided
    if output_file is None:
        output_file = "hr_output.wav"
    
    print("Generating speech using Croatian-specific model")
    
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    
    # Try to use Croatian model
    tts = TTS("tts_models/hr/cv/vits").to(device)
        
    # Initialize TTS with Croatian model
    start_time = time.time()
    
    # Generate speech
    tts.tts_to_file(text=text, file_path=output_file)
        
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
    print(f"Model: Croatian VITS")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Audio duration: {audio_duration:.2f} seconds")
    print(f"Performance ratio: {ratio:.2f}x (i.e., {ratio:.2f} seconds of processing time per 1 second of audio)")
    print(f"In other words: {processing_time:.2f} seconds were needed to synthesize {audio_duration:.2f} seconds of speech")
    
    return output_file
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Croatian speech using Coqui TTS')
    parser.add_argument('--text', type=str, default="Dobar dan, kako si? Ja sam Lovro, a ti? Ideš li u dućan možda?", 
                        help='Text to convert to speech (in Croatian)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename (default: hr_output.wav)')
    
    args = parser.parse_args()
    
    generate_speech_croatian(args.text, args.output)
