import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import os
import soundfile as sf
import argparse

def generate_speech(text, speaker_id, output_filename="output.wav"):
    """Generate speech from text using the specified speaker ID"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load models exactly as the model creator did
        print("Loading models...")
        processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        model = SpeechT5ForTextToSpeech.from_pretrained("nikolab/speecht5_tts_hr", weights_only=True).to(device)
        vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan", weights_only=True).to(device)
        
        # Create directory for speaker embeddings if it doesn't exist
        os.makedirs("./speaker_embeddings", exist_ok=True)
        
        # Check if speaker embeddings exist, if not create them
        if not os.path.exists("./speaker_embeddings/speaker_embeddings_test.pt"):
            print("Creating test speaker embeddings...")
            # Set a seed for reproducibility
            torch.manual_seed(42)
            # Create normalized speaker embeddings
            num_speakers = 50
            speaker_embeddings = torch.zeros(num_speakers, 512)
            for i in range(num_speakers):
                vec = torch.randn(512)
                vec = vec / torch.norm(vec)
                speaker_embeddings[i] = vec
            torch.save(speaker_embeddings, "./speaker_embeddings/speaker_embeddings_test.pt")
        
        # Load speaker embeddings with weights_only=True as specified by the model creator
        print(f"Loading speaker embeddings with weights_only=True for speaker ID {speaker_id}...")
        speaker_embedding = torch.load("./speaker_embeddings/speaker_embeddings_test.pt", weights_only=True)[speaker_id].unsqueeze(0)
        
        print(f"Converting text to speech: '{text}'")
        
        # Prepare inputs
        inputs = processor(text=text, return_tensors="pt")
        
        # Generate speech exactly as the model creator did
        speech = model.generate_speech(
            inputs["input_ids"].to(device),
            speaker_embedding.to(device),
            vocoder=vocoder
        ).cpu().numpy()
        
        # Save the audio file
        output_path = os.path.join(os.getcwd(), output_filename)
        sf.write(output_path, speech, 16000)
        
        print(f"Speech generated successfully and saved to '{output_path}'")
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate Croatian speech using SpeechT5')
    parser.add_argument('--text', type=str, default="Dobar dan, kako si? Ja sam Lovro, a ti? ideš li u dućan možda?", 
                        help='Text to convert to speech (in Croatian)')
    parser.add_argument('--speaker_id', type=int, default=19, 
                        help='Speaker ID to use (0-49)')
    parser.add_argument('--output', type=str, default="output.wav", 
                        help='Output filename')
    
    args = parser.parse_args()
    
    # Generate speech with the specified parameters
    generate_speech(args.text, args.speaker_id, args.output)