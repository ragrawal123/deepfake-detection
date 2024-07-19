from datasets import load_dataset
import time
#gs = load_dataset("speechcolab/gigaspeech", "xl", streaming=True, token="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")


gs = load_dataset("speechcolab/gigaspeech", "xl", split="train", streaming=True, token="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")

# see structure
print(gs)

# load audio sample on the fly
#data["text"] for transcription
#data["segment_id"] or data["audio_id"]
#data["audio"]["path"] for audio file to be transcribed
for data in gs:
    print(data)
    time.sleep(5)