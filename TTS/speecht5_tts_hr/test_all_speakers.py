import subprocess
import os
import time

def main():
    print("Testing speaker IDs 0 to 20...")
    
    # The text to use for all tests
    text = "Dobar dan, kako si? Ja sam Lovro, a ti? ideš li u dućan možda?"
    
    # Create a directory for the outputs if it doesn't exist
    output_dir = "speaker_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test each speaker ID from 0 to 20
    for speaker_id in range(21):  # 0 to 20 inclusive
        output_filename = f"output_speaker_{speaker_id}.wav"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"\nTesting speaker ID {speaker_id}...")
        
        # Run the main.py script with the current speaker ID
        cmd = [
            "py", "-3.12", "main.py",
            "--speaker_id", str(speaker_id),
            "--text", text,
            "--output", output_path
        ]
        
        # Execute the command
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully generated audio for speaker ID {speaker_id}")
        except subprocess.CalledProcessError as e:
            print(f"Error generating audio for speaker ID {speaker_id}: {e}")
        
        # Small delay to avoid overloading the system
        time.sleep(1)
    
    print("\nAll speaker IDs tested. Audio files saved in the 'speaker_samples' directory.")
    print("You can now listen to each file to determine which speaker ID sounds best.")

if __name__ == "__main__":
    main()
