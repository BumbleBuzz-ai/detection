Automatic Audio tagging of large collection of soundscapes using PANNs.

## Requirements

Install the following dependencies (versions should not matter) using your favorite package manager (pip, conda, ...)

- huggingface_hub
- pandas
- pytorch
- torchaudio
- librosa
- numpy
- scipy
- torchinfo
- plotly (optional, for the dash app)
- dash (optional, for the dash app)

## How to use

First, create a directory with wav audio files named YYMMDD_HHMMSS.wav/flac or What_EVER_YYMMDD_HHMMSS.wav/flac.
Then, use process.py to perform audio tagging.
Finally, use the dash_app.py script to run the web application.

![plot](image.png)

## Audio processing

```
python process.py [-h] [--data_path DATA_PATH] [--save_path SAVE_PATH] [--name NAME] [--audio_format AUDIO_FORMAT] [--l LENGTH_AUDIO_SEGMENT] [--save_audio_flac SAVE_AUDIO_FLAC] 

Script to process sound files recorded by Audiomoth

options:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Path to a folder with wav / flac files
  --save_path SAVE_PATH
                        Path to save outputs and audio files. folder will be created if does not exist
  --name NAME           name of measurement - you can put whatever you want, for example the name of the site 
  --audio_format AUDIO_FORMAT
                        wav or flac
  --l LENGTH_AUDIO_SEGMENT
                        Window length in seconds, must be larger than 5
  --save_audio_flac SAVE_AUDIO_FLAC
                        Saving audio in flac format (needed to run visualization tool)

```
There are other options (such as the type of pretrained model to use), please check the source code if necessary.
### Example

```
python3 process.py --save_path example/metadata/ --data_path example/metadata/audio_0002/ --name 0004 --audio_format flac --l 5 
```

Tagging of all flac files in folder example/metadata/audio_002 in non-overlapping chunks of 5 seconds.

Each chunk is converted to flac and saved.

## Dash app

```
python dash_app.py [-h] [--save_path SAVE_PATH]

Script to display sound files recorded by Audiomoth

options:
  -h, --help            show this help message and exit
  --save_path SAVE_PATH
                        Path to folder output of process.py : has to be the one that was entered when launching process.py
  --name NAME           name of measurement : has to be the one that was entered when launching process.py
```

Code for Audioset Tagging CNN from [Qiu Qiang Kong](https://github.com/qiuqiangkong/audioset_tagging_cnn)
