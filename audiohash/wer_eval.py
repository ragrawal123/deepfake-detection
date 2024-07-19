from datasets import load_dataset
from faster_whisper import WhisperModel
import evaluate
import torch
from scipy.io.wavfile import write
import os
from whisper_normalizer.english import EnglishTextNormalizer
import codecs, json

def main():
    #Setup and Load Models/Data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sample_rate = 16000
    #Data contains data['source'] to grab source of audio, if source==2 --> Youtube
    youtube = 2

    #Requires Hugging Face account & token due to being a gated dataset
    gigaspeech_train = load_dataset("speechcolab/gigaspeech", "xl", split="train", streaming=True, token="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")

    model = WhisperModel("medium.en", device=device, compute_type="float16")
    wer = evaluate.load("wer")
    normalize = EnglishTextNormalizer()

    #Debugging Purposes
    gigaspeech_train = gigaspeech_train.take(50)

    if not os.path.isdir('yt_wav_files/'):
        os.makedirs('yt_wav_files/')
    if not os.path.isdir('data_entries/'):
        os.makedirs('data_entries/')

    #data["text"] for transcription
    #data["segment_id"] or data["audio_id"]
    #data["audio"]["path"] for audio file to be transcribed
    for data in gigaspeech_train:
        if data['source'] == youtube:
            file = os.path.basename(data['audio']['path'])
            json_file = f"data_entries/{data['segment_id']}.json"
            write(f"yt_wav_files/{file}", sample_rate, data['audio']['array'])
            data['audio']['array'] = data['audio']['array'].tolist()
            data['audio']['path'] = f"yt_wav_files/{file}"
            json.dump(data, codecs.open(json_file, 'w', encoding='utf-8'), 
            separators=(',', ':'), 
            sort_keys=True, 
            indent=4)


    # for data in ds:
    #         print(data)
    #         print(type(data['audio']['array']))
    #         file = "output.wav"
    #         write(file, 16000, data['audio']['array'])
    #         segments, _ = model.transcribe(file, beam_size=5)
    #         segments = list(segments)
    #         print(segments[0].text)

if __name__=='__main__':
    main()

# load data in streaming mode
# given data, iterate through it and obtain only data with source==2 --> Youtube
# given youtube source, get file name from path and obtain .wav file name, use
# scipy wavfile write to write it to a folder output of that file name, then use that
# in faster whisper to generate the transcription, combine the transcription into
#single long text if there are multiple segments, and then normalize, compare to normalize
#label in data, append both label and pred to lists. at end get wer result

#Get working with less data and then make work for train val and test of xl subset of gigspeech

#Also find a way to grab the entire data entry itself when it is youtube and create
#new data subset of it from gigspeech. So create gigspeech-xl youtube data subset.
#For every youtube entry, save the wav file in one folder and save the entire entry
#as a file in json format to another directory and change relevant information 
#like path name to our local machine path name