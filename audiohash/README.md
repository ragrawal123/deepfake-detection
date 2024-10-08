# Evaluating OpenAI's Whisper for Robustness Under Audio In-The-Wild
We are evaluating [OpenAI's whisper](https://github.com/openai/whisper/tree/main?tab=readme-ov-file) under audio in-the-wild using the WER metric to understand its robustness. The aim is to see if Whisper can be used as an additional feature to Mobile X Lab's VeriLight system by focusing on hashing live speech transcriptions as an additional form of verification to VeriLight's visual detection system.

[basic_eval](./basic_eval) contains a script to do a smaller, basic evaluation using OpenAI's whisper 'base' model under the [OpenSLR LibriSpeech](https://huggingface.co/datasets/openslr/librispeech_asr) clean subset.
## Setup
Create a conda environment with Python 3.9:

`conda create -n whispereval python=3.9`

Install the necessary packages:

`pip install -r requirements.txt`

## Model & Dataset
### Model
We are using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) in our evaluation as it is a reimplementation of OpenAI's whisper with a fast inference engine, so we are still using the same model architecture and receiving the same accuracy while taking less time and memory. We do this to emulate near-real-time speech transcription.
### Dataset
For our dataset, we prioritized evaluating whisper over data that can be considered realistic and random, or 'wild'. As such, we wanted data with audio recordings from multiple distances, angles, and with noisy or clean backgrounds. We chose to use the YouTube transcribed subset of [SpeechColab's GigaSpeech](https://github.com/SpeechColab/GigaSpeech) dataset, aiming to use 1100 hours of transcribed YouTube data for our evaluation from their xl train and test splits.

We chose the YouTube subset because of the various acoustic descriptions:
- Clean and noisy
- Indoor and outdoor
- Near-and-far field
- Reading and spontaneous
- Various ages and accents
<!-- end of list -->
And, a variety of YouTube videos tend to inherently be recorded at different angles.\
\
In order to use the YouTube subset, we wrote the script `dataload.py` that would download the audio segments locally, taking 480 GB. This is due to the dataset being gated on Hugging Face and requiring an HF account & token to use, which provides an unnecessary barrier.

In order to download the data, we ran: 

`<time> python dataload.py`

## Evaluation
To evaluate the YouTube data, we wrote the script `wer_eval.py` using [faster-whisper](https://github.com/SYSTRAN/faster-whisper), which outputs:
- Number of audio segment transcribed, possibly labeled with:
  
  "#:Noise", meaning that the audio segment fell under the `gigaspeech_garbage_utterance_tags` and was skipped over in transcription

  "#:Checked", meaning the audio segment was already transcribed in a previous script run and was saved into a file and loaded into a dict during runtime for easy access to the prediction rather than re-transcribing

  "#:Error", there was either a loading or evaluating error with that particular file, which could be caused by an empty reference or json file
 
- The WER metric % at the end of runtime
- Three text files:
  
   'noisy_data.txt': contains the file_id of the segments that were under the `gigaspeech_garbage_utterance_tags`
  
    'checked_data_{model}.txt': contains the file_id of all segments transcribed from that particular Whisper model type

   'error_data.txt': contains the file_id of all segments that caused errors and were skipped
<!-- end oflist -->
To run the script: 

`<time> python wer_eval.py`
