import os
from TTS.api import TTS

# Set environment variable to accept the license terms
os.environ["COQUI_TOS_AGREED"] = "1"

def list_available_models():
    """List all available TTS models"""
    print("Listing all available TTS models...")
    tts = TTS()
    models = tts.list_models()
    
    # Filter for Slavic language models
    slavic_languages = ["pl", "cs", "sk", "sl", "hr", "bs", "sr", "bg", "mk", "uk", "ru", "be"]
    slavic_models = []
    
    for model in models:
        for lang in slavic_languages:
            if f"/{lang}/" in model or f"_{lang}_" in model:
                slavic_models.append(model)
                break
    
    print("\nSlavic language models:")
    for model in slavic_models:
        print(f"- {model}")
    
    print("\nMultilingual models that support Slavic languages:")
    multilingual_models = [model for model in models if "multilingual" in model]
    for model in multilingual_models:
        print(f"- {model}")
    
    # Print supported languages for XTTS
    print("\nLanguages supported by XTTS v2:")
    xtts_languages = [
        "Arabic (ar)", "Chinese (zh-cn)", "Czech (cs)", "Dutch (nl)", 
        "English (en)", "French (fr)", "German (de)", "Hindi (hi)", 
        "Hungarian (hu)", "Italian (it)", "Japanese (ja)", "Korean (ko)", 
        "Polish (pl)", "Portuguese (pt)", "Russian (ru)", "Spanish (es)", 
        "Turkish (tr)"
    ]
    for lang in xtts_languages:
        print(f"- {lang}")
    
    return models

if __name__ == "__main__":
    list_available_models()
