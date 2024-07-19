from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import evaluate
import torch

def main(): 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = load_dataset("librispeech_asr", "all", split="test.clean", streaming=True)
    
    #Debugging Purposes
    ds = ds.take(512)

    processor = WhisperProcessor.from_pretrained("openai/whisper-base.en")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base.en")
    model = model.to('cuda')
    wer = evaluate.load("wer")

    predictions = []
    labels = []
    i = 0
    
    for data in ds:
        transcription = predict(data, processor, model)
        predictions.append(processor.tokenizer.normalize(transcription[0]))
        labels.append(processor.tokenizer.normalize(data["text"]))
        print(f"{i}: {transcription[0]}")
        i = i + 1
    
    wer_metric = wer.compute(references=labels, predictions=predictions)
    wer_metric = round(100 * wer_metric, 2)
    print(wer_metric)

def predict(data, processor, model):
    audio_sample = data["audio"]
    waveform = audio_sample["array"]
    sampling_rate = audio_sample["sampling_rate"]
    
    input_features = processor(
        waveform, sampling_rate=sampling_rate, return_tensors="pt"
    ).input_features
    input_features = input_features.to('cuda')
    
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription


if __name__=='__main__':
    main()