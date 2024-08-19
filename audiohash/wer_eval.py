from faster_whisper import WhisperModel
import evaluate
import torch
import os
from whisper_normalizer.english import EnglishTextNormalizer
import json

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using:', device)
    json_dir = '/media/storage/data_entries/'
    wav_dir = '/media/storage/yt_wav_files/'

    model_type = 'medium.en'

    model = WhisperModel(model_type, device=device, compute_type="float16")
    wer = evaluate.load("wer")
    cer = evaluate.load('cer')
    normalize = EnglishTextNormalizer()
    
    json_files = os.scandir(json_dir)
    predictions = []
    references = []
    checked_data_dict = dict()


    noise = 'noisy_data.txt'
    check = f'checked_data_{os.path.splitext(model_type)[0]}.txt'
    error = 'error_data.txt'
    
    if os.path.exists(noise):
        os.remove(noise)
    noisy_data = open(noise, 'a')

    if os.path.exists(error):
        os.remove(error)
    error_data = open(error, 'a')
    
    if os.path.exists(check):
        with open(check) as file:
            for line in file:
                line = line.strip('\n')
                id, pred = line.split(':')
                checked_data_dict[id] = pred
            file.close()
            
    checked_data = open(check, 'a')
    print(len(checked_data_dict))
    gigaspeech_garbage_utterance_tags = ['<SIL>', '<NOISE>', '<MUSIC>', '<OTHER>']
    
    for (counter, file) in enumerate(json_files):
        file_id = os.path.splitext(file.name)[0]
        entry = open(f"{json_dir}{file.name}")
        try:
            data = json.load(entry)
        except:
            print(f"{counter}: Error loading: {file.name}")
            error_data.write(f"{file_id}:Loading\n")
            continue
        
        if data['text'] in gigaspeech_garbage_utterance_tags:
            noisy_data.write(f"{file_id}\n")
            print(f"{counter}:Noise")
            continue
        
        data['text'] = normalize(data['text'])
        
        if file_id in checked_data_dict.keys():
            if checked_data_dict[file_id] == '' or data['text'] == '':
                print(f"{counter}: Error evaluating: {file.name}")
                error_data.write(f"{file_id}:Eval\n")
                continue    
            predictions.append(checked_data_dict[file_id])
            references.append(normalize(data['text']))
            print(f"{counter}: Checked: {file_id}")
            continue
        
        
        segments, _ = model.transcribe(f"{wav_dir}{file_id}.wav", beam_size=5)
        prediction = ""
        for segment in segments:
            prediction += segment.text
        
        prediction = normalize(prediction)
        if prediction == '' or data['text'] == '':
            print(f"{counter}: Error evaluating: {file.name}")
            error_data.write(f"{file_id}:Eval\n")
            continue
        
        print(f"{counter}: {file_id}")
        checked_data.write(f"{file_id}:{prediction}\n")
        references.append(normalize(data['text']))
        predictions.append(prediction)
        entry.close()
        
        

    noisy_data.close()
    error_data.close()
    checked_data.close()

    wer_metric = wer.compute(references=references, predictions=predictions)
    wer_metric = round(100 * wer_metric, 2)

    print('WER Rate: ',wer_metric, '%')


    








if __name__=='__main__':
    main()