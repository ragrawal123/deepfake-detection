import whisper
from whisper_normalizer.english import EnglishTextNormalizer
import sys
import os


def transcribe(file):
    normalize = EnglishTextNormalizer()
    model = whisper.load_model("base.en").to('cuda')
    audio = whisper.load_audio(file)
    result = whisper.transcribe(model=model, audio=audio, temperature=0.0)
    return normalize(result["text"])

def main():
    if len(sys.argv) < 2:
        print("Usage: python whisper.py <file>")
        sys.exit(1)
    file = sys.argv[1]
    print(transcribe(file))

if __name__=='__main__':
    main()