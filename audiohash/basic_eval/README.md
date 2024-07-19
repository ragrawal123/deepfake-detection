# Basic Evaluation
Evaluating Whisper's base model for the [WER metric](https://huggingface.co/spaces/evaluate-metric/wer) under the [OpenSLR LibriSpeech](https://huggingface.co/datasets/openslr/librispeech_asr) clean validation dataset.

## Setup & Run
We are using Hugging Face Transformers and Datasets to conduct this evalution. Install required libraries on a conda enviroment as such:

`pip install -r requirements.txt`

Run the evaluation script with the following:

`python eval.py`

If you would like to see the transcription of one of the files in the [test audio](basic_eval/test_audio/) directory, run:

`python filetranscribe.py test_audio/<file>`

The script will output every transcription happening and at the end will output the WER % metric.
