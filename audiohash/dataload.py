from datasets import load_dataset
import evaluate
import torch
import scipy.io.wavfile as wav
import os
import codecs, json

def main():
    #Data contains data['source'] to grab source of audio, if source==2 --> Youtube
    youtube = 2

    sample_rate = 16000
    json_dir = '/media/storage/data_entries/'
    wav_dir = '/media/storage/yt_wav_files/'

    #Requires Hugging Face account & token due to being a gated dataset
    #Grabbing xl training subset, aiming to download 1100 hours of YouTube data
    gigaspeech_train = load_dataset("speechcolab/gigaspeech", "xl", split="train", streaming=True, token="")

    #Debugging Purposes
    #gigaspeech_train = gigaspeech_train.take(10)

    if not os.path.isdir(wav_dir):
        os.makedirs(wav_dir)
    if not os.path.isdir(json_dir):
        os.makedirs(json_dir)

    i = 0
    total_time = 0
    for data in gigaspeech_train:
        if data['source'] == youtube:
            file = os.path.basename(data['audio']['path'])
            json_file = f"{json_dir}{data['segment_id']}.json"
            #Should add a check to see if data already in data_entries/
            #And if wav file already in yt_wav_files/
            if os.path.exists(f"{json_file}") and os.path.exists(f"{wav_dir}{file}"):
                total_time += (data['end_time'] - data['begin_time'])
                print(f"{i}: Already Created")
                print(f"Time(s): {round(total_time,2)}")
                i = i + 1
                continue
            wav.write(f"{wav_dir}{file}", sample_rate, data['audio']['array'])
            data['audio']['array'] = data['audio']['array'].tolist().clear()
            data['audio']['path'] = f"{wav_dir}{file}"
            json.dump(data, codecs.open(json_file, 'w', encoding='utf-8'), 
            separators=(',', ':'), 
            sort_keys=True, 
            indent=4)
            total_time += (data['end_time'] - data['begin_time'])
            print(f"{i}: {data['segment_id']}.json created")
            print(f"Total Time (s): {total_time}")
            i = i + 1

if __name__=='__main__':
    main()
