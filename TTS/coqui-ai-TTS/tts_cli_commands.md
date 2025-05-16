# Using Coqui TTS from the Command Line

## List Available Models

To see all available models:

```
py -3.12 -m TTS.bin.synthesize --list_models
```

To see only XTTS models:

```
py -3.12 -m TTS.bin.synthesize --list_models | findstr xtts
```

## Generate Speech with XTTS v2

```
py -3.12 -m TTS.bin.synthesize --text "Dobar dan, kako si? Ja sam Lovro, a ti?" --model_name tts_models/multilingual/multi-dataset/xtts_v2 --language_idx hr --out_path xtts_output.wav
```

## Generate Speech with Bark

```
py -3.12 -m TTS.bin.synthesize --text "Dobar dan, kako si? Ja sam Lovro, a ti?" --model_name tts_models/multilingual/multi-dataset/bark --out_path bark_output.wav
```

## Generate Speech with Voice Cloning (XTTS v2)

```
py -3.12 -m TTS.bin.synthesize --text "Dobar dan, kako si? Ja sam Lovro, a ti?" --model_name tts_models/multilingual/multi-dataset/xtts_v2 --language_idx hr --speaker_wav path/to/reference_audio.wav --out_path cloned_output.wav
```

## Additional Options

- `--language_idx`: Language code (e.g., hr, en, de)
- `--speaker_wav`: Path to a reference audio file for voice cloning
- `--vocoder_name`: Specify a custom vocoder
- `--use_cuda`: Use GPU acceleration if available

## Help

For full list of options:

```
py -3.12 -m TTS.bin.synthesize --help
```

## Example Command for Croatian TTS

```
py -3.12 -m TTS.bin.synthesize --text "Dobar dan, kako si? Ja sam Lovro, a ti?" --model_name tts_models/multilingual/multi-dataset/xtts_v2 --language_idx hr --out_path hr_output.wav
```
