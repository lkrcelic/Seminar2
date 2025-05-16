from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import soundfile as sf
import torch
import os
import requests
import sys
import time

# Set up error handling
def download_file(url, filename):
    try:
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {filename} successfully.")
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

# Check Python version
python_version = sys.version_info
print(f"Running with Python {python_version.major}.{python_version.minor}.{python_version.micro}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    # load model and tokenizer
    print("Loading model and tokenizer...")
    processor = Wav2Vec2Processor.from_pretrained(
        "classla/wav2vec2-xls-r-parlaspeech-hr")
    model = Wav2Vec2ForCTC.from_pretrained("classla/wav2vec2-xls-r-parlaspeech-hr")
    print("Model and tokenizer loaded successfully.")

    # download the example wav file
    # audio_url = "https://huggingface.co/classla/wav2vec2-xls-r-parlaspeech-hr/raw/main/00020570a.flac.wav"
   # audio_file = "00020570a.flac.wav"
    
    # if not download_file(audio_url, audio_file):
    #     print("Failed to download audio file. Exiting.")
    #     sys.exit(1)
    
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
    
    input_values = processor(speech_resampled, sampling_rate=sample_rate, return_tensors="pt").input_values.to(device)
    
    # retrieve logits
    print("Running inference...")
    start_time = time.time()
    logits = model.to(device)(input_values).logits
    
    # take argmax and decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0]).lower()
    
    # Calculate total processing time
    total_time = time.time() - start_time
    
    # Get audio duration
    audio_info = sf.info(audio_file)
    audio_duration = audio_info.duration
    
    # Calculate the ratio
    ratio = total_time / audio_duration
    
    print("\nTranscription result:")
    print(transcription)
    
    # Print performance results
    print("\nPerformance Results:")
    print(f"Model: Wav2Vec2 XLS-R ParlaSpeech HR")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Audio duration: {audio_duration:.2f} seconds")
    print(f"Performance ratio: {ratio:.2f}x (i.e., {ratio:.2f} seconds of processing time per 1 second of audio)")
    print(f"In other words: {total_time:.2f} seconds were needed to transcribe {audio_duration:.2f} seconds of speech")

except Exception as e:
    print(f"Error: {e}")
