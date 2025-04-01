import time
import subprocess
import os
import soundfile as sf
import argparse

def measure_performance(script_path, text, speaker_id, output_path):
    """Measure the performance of speech synthesis"""
    print(f"Measuring performance of {script_path}...")
    print(f"Text to synthesize: '{text}'")
    print(f"Speaker ID: {speaker_id}")
    
    # Start timing
    start_time = time.time()
    
    # Run the script
    cmd = [
        "py", "-3.12", script_path,
        "--text", text,
        "--speaker_id", str(speaker_id),
        "--output", output_path
    ]
    
    subprocess.run(cmd, check=True)
    
    # End timing
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Get the duration of the generated audio
    audio_info = sf.info(output_path)
    audio_duration = audio_info.duration
    
    # Calculate the ratio
    ratio = processing_time / audio_duration
    
    print("\nPerformance Results:")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Audio duration: {audio_duration:.2f} seconds")
    print(f"Performance ratio: {ratio:.2f}x (i.e., {ratio:.2f} seconds of processing time per 1 second of audio)")
    print(f"In other words: {processing_time:.2f} seconds were needed to synthesize {audio_duration:.2f} seconds of speech")
    
    return processing_time, audio_duration, ratio

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Measure performance of speech synthesis')
    parser.add_argument('--script', type=str, default="main.py", 
                        help='Script to measure (main.py or creator_approach.py)')
    parser.add_argument('--text', type=str, 
                        default="Dobar dan, kako si? Ja sam Lovro, a ti? ideu0161 li u duu0107an mou017eda?", 
                        help='Text to convert to speech (in Croatian)')
    parser.add_argument('--speaker_id', type=int, default=112744, 
                        help='Speaker ID to use')
    
    args = parser.parse_args()
    
    # Define output path
    output_filename = f"performance_test_{os.path.splitext(args.script)[0]}.wav"
    
    # Measure performance
    measure_performance(args.script, args.text, args.speaker_id, output_filename)
