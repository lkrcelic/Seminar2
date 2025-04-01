import torch
import os
import time
import argparse
import soundfile as sf
from TTS.api import TTS

# Set environment variable to accept the license terms
os.environ["COQUI_TOS_AGREED"] = "1"

def voice_conversion(source_wav, output_file=None):
    """Convert voice from source to target using FreeVC24 model"""
    # Create output filename if not provided

    output_file = "converted_output.wav"
    target_wav = "sample_voices/sample_voice.wav"
    
    print("Performing voice conversion using FreeVC24 model")
    
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize TTS with voice conversion model
    tts = TTS("voice_conversion_models/multilingual/vctk/freevc24").to(device)
    
    # Perform voice conversion
    print(f"Converting voice from {source_wav} to match {target_wav}...")
    tts.voice_conversion_to_file(
        source_wav=source_wav,
        target_wav=target_wav,
        file_path=output_file
    )
    print(f"Voice conversion completed. Output saved to: {output_file}")
    
    return output_file

if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description='Voice Conversion using FreeVC24 model')
    parser.add_argument('--source', type=str, required=True,
                        help='Source audio file path')
    
    args = parser.parse_args()

    voice_conversion(args.source)
