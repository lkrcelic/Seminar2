from transformers import Wav2Vec2ProcessorWithLM, Wav2Vec2ForCTC
import soundfile as sf
import torch
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# load model and tokenizer
processor = Wav2Vec2ProcessorWithLM.from_pretrained(
    "classla/wav2vec2-large-slavic-parlaspeech-hr-lm")
model = Wav2Vec2ForCTC.from_pretrained("classla/wav2vec2-large-slavic-parlaspeech-hr-lm")
# download the example wav files:
# read the wav file 

print("Processing audio file...")
audio_file = "sample_voice.wav"
speech, sample_rate = sf.read(audio_file)
print(f"Original audio sample rate: {sample_rate} Hz")
    
# Resample if needed (model expects 16kHz)
if sample_rate != 16000:
    print(f"Resampling audio from {sample_rate}Hz to 16000Hz...")
    try:
        import librosa
        speech_resampled = librosa.resample(speech, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
        print("Resampling completed successfully.")
    except ImportError:
        print("Warning: librosa not installed. Attempting to proceed without resampling.")

input_values = processor(speech_resampled, sampling_rate=sample_rate, return_tensors="pt").input_values.cuda()
inputs = processor(speech_resampled, sampling_rate=sample_rate, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
transcription = processor.batch_decode(logits.numpy()).text[0]

print("\nTranscription result:")
print(transcription)
