import os
import torch
from torch.utils.data import Dataset

import librosa
import pandas as pd
from tqdm import tqdm
import datetime

import numpy as np
from scipy.signal import resample

from indices import compute_ecoacoustics
import torchaudio

class Silent_dataset(Dataset):
    def __init__(self, meta_dataloader, len_audio_s, savepath, save_audio = True):        
        self.meta = meta_dataloader        
        self.sr_tagging = 32000
        self.savepath = savepath
        self.len_audio_s = len_audio_s
        self.save_audio = save_audio

    def __getitem__(self, idx):

        
        filename = self.meta['filename'][idx]

        # Load audio file, convert to mono, and apply offset and duration
        wav_o, sr = librosa.load(filename, sr=None, mono=True,
                              offset=self.meta['start'][idx], duration=self.len_audio_s)
        wav = torch.tensor(wav_o).view(1, len(wav_o))
        
        if self.save_audio:
            torchaudio.save(os.path.join(self.savepath, self.meta['date'][idx].strftime('%Y%m%d_%H%M%S')+ '.flac'), wav, sr)

        # Compute ecoacoustic indices
        ecoac = compute_ecoacoustics(wav_o)
        
        # Resample to audio tagging model sample rate
        wav = resample(wav_o, int(self.len_audio_s*self.sr_tagging))
        wav = torch.tensor(wav)
        

        return (wav.view(int(self.len_audio_s*self.sr_tagging)), {'name': os.path.basename(filename), 'start': self.meta['start'][idx],
                                                        'date': self.meta['date'][idx].strftime('%Y%m%d_%H%M%S'),
                                                        'ecoac' : ecoac})
        #return([None, None])
    def __len__(self):
        return len(self.meta['filename'])
    
def get_dataloader_site(meta_site, savepath, len_audio_s , save_audio = True, batch_size=12,  num_workers=8):

    meta_dataloader = pd.DataFrame(
        columns=['filename', 'sr', 'start', 'stop'])

    for idx, wavfile in enumerate(tqdm(meta_site['filename'])):

        len_file = meta_site['length'][idx]
        sr_in = meta_site['sr'][idx]
        duration = len_file/sr_in
        nb_win = int(duration // len_audio_s )

        # cut into 10 sec segment length
        for win in range(nb_win):
            delta = datetime.timedelta(seconds=int((win*len_audio_s)))
            # meta_dataloader = pd.concat([meta_dataloader,pd.DataFrame({'filename': wavfile, 'sr': sr_in, 'start': (
            #     win*len_audio_s), 'stop': ((win+1)*len_audio_s), 'len': len_file, 'date': meta_site['datetime'][idx] + delta})], ignore_index=True)
            curmeta = pd.DataFrame.from_dict({'filename': [wavfile], 'sr': [sr_in], 'start': (
                    [win*len_audio_s]), 'stop': [((win+1)*len_audio_s)], 'len': [len_file], 'date': meta_site['datetime'][idx] + delta})
            meta_dataloader = pd.concat([meta_dataloader,curmeta], ignore_index=True)

    site_set = Silent_dataset(meta_dataloader.reset_index(drop=True),len_audio_s, savepath, save_audio)
    site_set = torch.utils.data.DataLoader(
        site_set,shuffle=False, batch_size=batch_size,  num_workers=num_workers)

    return site_set