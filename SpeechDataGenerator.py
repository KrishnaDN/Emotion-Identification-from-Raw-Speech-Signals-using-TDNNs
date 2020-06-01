#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 14:09:31 2019

@author: Krishna
"""
import numpy as np
import torch
from utils import utils_wav

class SpeechDataGenerator():
    """Speech dataset."""

    def __init__(self, manifest, mode):
        """
        Read the textfile and get the paths
        """
        self.mode=mode
        self.audio_links = [line.rstrip('\n').split(' ')[0] for line in open(manifest)]
        self.emo_labels = [int(line.rstrip('\n').split(' ')[1]) for line in open(manifest)]
        self.gen_labels = [int(line.rstrip('\n').split(' ')[2]) for line in open(manifest)]
        

    def __len__(self):
        return len(self.audio_links)

    def __getitem__(self, idx):
        audio_link =self.audio_links[idx]
        class_id = self.emo_labels[idx]
        #lang_label=lang_id[self.audio_links[idx].split('/')[-2]]
        audio_data = utils_wav.load_data_wav(audio_link,min_dur_sec=10)
        sample = {'raw_speech': torch.from_numpy(np.ascontiguousarray(audio_data)), 'labels': torch.from_numpy(np.ascontiguousarray(class_id))}
        return sample
        
    
