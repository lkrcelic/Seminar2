import torch
import soundfile as sf
import os
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

# Set parameters
speaker_id = 27
device = torch.device('cpu')
output_file = "output.wav"
text = "Naravno! Danas je sunchan dan."

print(f"Using device: {device}")
print(f"Processing text: '{text}'")

# Load models
print("Loading models...")
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("nikolab/speecht5_tts_hr").to(device)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

# Create directory for speaker embeddings if it doesn't exist
os.makedirs("./speaker_embeddings", exist_ok=True)

# Load speaker embeddings
print(f"Loading speaker embeddings for speaker ID {speaker_id}...")
speaker_embeddings = torch.load("./speaker_embeddings/speaker_embeddings_test.pt", weights_only=True)[speaker_id].unsqueeze(0)

# Generate speech
print("Generating speech...")
inputs = processor(text=text, return_tensors="pt")
speech = model.generate_speech(
    inputs["input_ids"].to(device),
    speaker_embeddings.to(device),
    vocoder=vocoder
).cpu().numpy()

# Save the audio file
print(f"Saving audio to {output_file}...")
sf.write(output_file, speech, 16000)
print(f"Done! Audio saved to {output_file}")