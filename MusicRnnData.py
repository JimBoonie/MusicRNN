# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 21:25:25 2017

@author: bricklayer
"""

import numpy as np
from scipy.io import wavfile

class MusicRnnData(object):
    def __init__(self, track_list, bitrate=16, twos_comp=True):
        self.tracks = []
        for track in track_list:
            audio, sample_rate = self.__load_audio_from_wav(track)
            self.tracks.append(audio)
        self.sample_rate = sample_rate
    
    def __load_audio_from_wav(self, filename, bitrate=16, twos_comp=True):
        # read audio
        sample_rate, audio = wavfile.read(filename)
        assert(audio.dtype=='int16') # assume audio is int16 for now
        assert(sample_rate==44100) # assume sample_rate is 44100 for now

        # combine channels
        audio = np.mean(np.array(audio), 1)
        
        # normalize to [-1, 1]
        max_code = 2**bitrate
        norm_factor = max_code/2.0
        offset = (not twos_comp)*max_code
        normed_audio = (audio - offset)/norm_factor
        
        return normed_audio, sample_rate
    
    def __extract_segment(self, audio, n_x, n_y, start_idx=-1):
        n_samples = audio.shape[0]
        n_points = n_x + n_y
        
        if start_idx==-1:
            #   select random index from range(0, n_samples - n_points)
            start_idx = np.random.randint(0, n_samples - n_points, 1)[0]
        
        # extract segment
        x = audio[start_idx:start_idx+n_x]
        y = audio[start_idx+n_x:start_idx+n_x+n_y]
        return x, y
        
    def batch(self, n_x, n_y, batch_size):
        n_tracks = len(self.tracks)
        idxs = np.random.randint(0, n_tracks, batch_size)
        
        x_batch = np.zeros((batch_size, n_x))
        y_batch = np.zeros((batch_size, n_y))
        for i in range(batch_size):
            idx = idxs[i]
            x_i, y_i = self.__extract_segment(self.tracks[idx], n_x, n_y)
            x_batch[idx, :] = x_i
            y_batch[idx, :] = y_i
        
        return x_batch, y_batch