from faster_whisper import WhisperModel
import evaluate
import torch
import os
from whisper_normalizer.english import EnglishTextNormalizer
import json


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using: ', device)
    json_dir = '/media/raunak/1TB/data_entries/'
    wav_dir = '/media/raunak/1TB/yt_wav_files/'

    model = WhisperModel("base.en", device=device, compute_type="float16")
    wer = evaluate.load("wer")
    normalize = EnglishTextNormalizer()
    
    json_files = os.scandir(json_dir)
    predictions = []
    references = []

    noise = 'noisy_data.txt'
    if os.path.exists(noise):
        os.remove(noise)
    noisy_data = open(noise, 'a')
    checked_data = open('checked_data.txt', 'a')
    gigaspeech_garbage_utterance_tags = ['<SIL>', '<NOISE>', '<MUSIC>', '<OTHER>']
    
    for (counter, file) in enumerate(json_files):
        file_id = os.path.splitext(file.name)[0]
        entry = open(f"{json_dir}{file.name}")
        data = json.load(entry)
        
        if data['text'] in gigaspeech_garbage_utterance_tags:
            noisy_data.write(f"{file_id}\n")
            print(f"{counter}:Noise")
            continue
        
        segments, _ = model.transcribe(f"{wav_dir}{file_id}.wav", beam_size=5)
        prediction = ""
        for segment in segments:
            prediction += segment.text
        
        prediction = normalize(prediction)
        print(f"{counter}")
        checked_data.write(f"{file_id}\n")
        references.append(normalize(data['text']))
        predictions.append(prediction)
        entry.close()

    noisy_data.close()
    checked_data.close()

    wer_metric = wer.compute(references=references, predictions=predictions)
    wer_metric = round(100 * wer_metric, 2)
    print('WER Rate: ',wer_metric, '%')


    








if __name__=='__main__':
    main()