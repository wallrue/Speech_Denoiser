# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 08:35:16 2023

@author: turner_le
"""

import torch.utils.data
import torchaudio
import random
import os
from natsort import natsorted
from torch.utils.data.distributed import DistributedSampler
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" #Fix error on computer

class DemandDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, cut_len=16000 * 4): #sample rate is usually sr=16000
        self.cut_len = cut_len
        self.clean_dir = os.path.join(data_dir, "clean")
        self.noisy_dir = os.path.join(data_dir, "noisy")
        
        # Shuffle data
        self.clean_wav_name = os.listdir(self.clean_dir)
        self.clean_wav_name = natsorted(self.clean_wav_name)
        
    def __len__(self):
        return len(self.clean_wav_name)    
    
    def __getitem__(self, idx):    
        # Get audio file
        clean_file = os.path.join(self.clean_dir, self.clean_wav_name[idx])
        noisy_file = os.path.join(self.noisy_dir, self.clean_wav_name[idx])

        # Load audio file
        clean_ds, _ = torchaudio.load(clean_file)
        noisy_ds, _ = torchaudio.load(noisy_file)
        clean_ds, noisy_ds = clean_ds.squeeze(), noisy_ds.squeeze()
        assert len(clean_ds) == len(noisy_ds)
        length = len(clean_ds)

        # Supposed we want to load a cut_len = 2 seconds not all of audio files    
        if length < self.cut_len: #repeats the audio until get length 2 seconds
            units = self.cut_len // length # take the quotient as the number of repeating
            clean_ds_final, noisy_ds_final = list(), list()
            for i in range(units):
                clean_ds_final.append(clean_ds)
                noisy_ds_final.append(noisy_ds)
            clean_ds_final.append(clean_ds[: self.cut_len % length])
            noisy_ds_final.append(noisy_ds[: self.cut_len % length])
            clean_ds = torch.cat(clean_ds_final, dim=-1)
            noisy_ds = torch.cat(noisy_ds_final, dim=-1)
        else: # randomly cut 2 seconds segment
            wav_start = random.randint(0, length - self.cut_len)
            noisy_ds = noisy_ds[wav_start : wav_start + self.cut_len]
            clean_ds = clean_ds[wav_start : wav_start + self.cut_len]

        return clean_ds, noisy_ds, length
    
    
def load_data(ds_dir, batch_size, n_cpu, cut_len, MULTI_DEVICES):
    torchaudio.set_audio_backend("soundfile")  # "sox_io" in linux, "soundfile" in windows
    train_ds = DemandDataset(os.path.join(ds_dir, "train"), cut_len)
    test_ds = DemandDataset(os.path.join(ds_dir, "test"), cut_len)

    train_dataset = torch.utils.data.DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        pin_memory=True if MULTI_DEVICES else False,
        shuffle=False,
        sampler=DistributedSampler(train_ds) if MULTI_DEVICES else None,
        drop_last=True,
        num_workers=n_cpu,
    )
    test_dataset = torch.utils.data.DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        pin_memory=True if MULTI_DEVICES else False,
        shuffle=False,
        sampler=DistributedSampler(test_ds) if MULTI_DEVICES else None,
        drop_last=False,
        num_workers=n_cpu,
    )

    return train_dataset, test_dataset
        