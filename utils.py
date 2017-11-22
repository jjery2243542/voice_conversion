import json
import h5py
import pickle
import os
from collections import defaultdict
from collections import namedtuple
import numpy as np
import math
import argparse
import random
import time
import torch
from torch.utils import data
from tensorboardX import SummaryWriter

class Hps(object):
    def __init__(self):
        self.hps = namedtuple('hps', [
            'lr',
            'alpha',
            'beta',
            'lambda_',
            'max_grad_norm',
            'max_step',
            'seg_len',
            'D_iterations',
            'batch_size',
            'pretrain_iterations',
            'iterations',
            ]
        )
        default = [5e-4, 1, 0.5, 10, 1, 5, 128, 5, 16, 10000, 15000]
        self._hps = self.hps._make(default)

    def get_tuple(self):
        return self._hps

    def load(self, path):
        with open(path, 'r') as f_json:
            hps_dict = json.load(f_json)
        self._hps = self.hps(**hps_dict)

    def dump(self, path):
        with open(path, 'w') as f_json:
            json.dump(self._hps._asdict(), f_json, indent=4, separators=(',', ': '))

class Sampler(object):
    def __init__(
        self, 
        h5_path, 
        speaker_sex_path='/storage/raw_feature/voice_conversion/train-clean-100-speaker-sex.txt', 
        utt_len_path='/storage/raw_feature/voice_conversion/utterence_length.txt', 
        max_step=5, 
        seg_len=128
    ):
        self.f_h5 = h5py.File(h5_path, 'r')
        self.max_step = max_step
        self.seg_len = seg_len
        self.read_sex_file(speaker_sex_path)
        self.utt2len = self.read_utt_len_file(utt_len_path)
        self.speakers = list(self.f_h5['train'].keys())
        self.speaker_used = self.speakers[:8]
        self.speaker2utts = {speaker:list(self.f_h5['train/{}'.format(speaker)].keys()) \
                for speaker in self.speakers}
        # remove too short utterence
        self.rm_too_short_utt()
        self.indexer = namedtuple('index', ['i', 'j', 't', 't_k', 't_k_prime', 't_j'])

    def read_utt_len_file(self, utt_len_path):
        with open(utt_len_path, 'r') as f:
            # header
            f.readline()
            # speaker, utt, length
            lines = [tuple(line.strip().split()) for line in f.readlines()]
            mapping = {(speaker, utt_id): int(length) for speaker, utt_id, length in lines}
        return mapping

    def rm_too_short_utt(self, limit=None):
        if not limit:
            limit = self.seg_len * 2
        for (speaker, utt_id), length in self.utt2len.items():
            if length < limit:
                self.speaker2utts[speaker].remove(utt_id)

    def read_sex_file(self, speaker_sex_path):
        with open(speaker_sex_path, 'r') as f:
            # Female
            f.readline()
            self.female_ids = f.readline().strip().split()
            # Male
            f.readline()
            self.male_ids = f.readline().strip().split()

    def sample_utt(self, speaker_id):
        # sample an utterence
        utt_id = self.rand(self.speaker2utts[speaker_id])
        length = self.f_h5['train/{}/{}/mel'.format(speaker_id, utt_id)].shape[0]
        return utt_id, length

    def rand(self, l):
        rand_idx = random.randint(0, len(l) - 1)
        return l[rand_idx] 

    def sample(self):
        seg_len = self.seg_len
        max_step = self.max_step
        # sample two speakers
        speakerA, speakerB = random.sample(self.speaker_used, 2)
        A_utt_id, A_len = self.sample_utt(speakerA)
        B_utt_id, B_len = self.sample_utt(speakerB)
        # sample t and t^k 
        t = random.randint(0, A_len - 2 * seg_len)  
        t_k = random.randint(t + seg_len, min(A_len - seg_len, t + max_step * seg_len))
        t_k_prime = random.randint(t + seg_len, min(A_len - seg_len, t + max_step * seg_len))
        # sample a segment from speakerB
        t_j = random.randint(0, B_len - seg_len)
        index_tuple = self.indexer(i='{}/{}'.format(speakerA, A_utt_id), \
                j='{}/{}'.format(speakerB, B_utt_id), \
                t=t, t_k=t_k, t_k_prime=t_k_prime, t_j=t_j)
        return index_tuple
        #return self.f_h5['train/{}/{}/mel'.format(speakerA, A_utt_id)][t:t + seg_len], \
        #    self.f_h5['train/{}/{}/lin'.format(speakerA, A_utt_id)][t:t + seg_len],\
        #    self.f_h5['train/{}/{}/mel'.format(speakerA, A_utt_id)][t_k:t_k + seg_len],\
        #    self.f_h5['train/{}/{}/lin'.format(speakerA, A_utt_id)][t_k:t_k + seg_len],\
        #    self.f_h5['train/{}/{}/mel'.format(speakerA, A_utt_id)][t_k_prime:t_k_prime + seg_len],\
        #    self.f_h5['train/{}/{}/lin'.format(speakerA, A_utt_id)][t_k_prime:t_k_prime + seg_len],\
        #    self.f_h5['train/{}/{}/mel'.format(speakerB, B_utt_id)][t_j:t_j + seg_len],\
        #    self.f_h5['train/{}/{}/lin'.format(speakerB, B_utt_id)][t_j:t_j + seg_len]

#class DataLoader(object):
#    def __init__(self, h5py_path, batch_size=16):
#        self.f_h5 = h5py.File(h5py_path)
#        self.keys = list(self.f_h5.keys())
#        self.index = 0
#        self.batch_size = batch_size
#
#    def __iter__(self):
#        return self
#
#    def __next__(self):
#        if self.index >= len(self.keys):
#            self.index = 0
#        key = self.keys[self.index]
#        data = (self.f_h5['{}/X_i_t/mel'.format(key)][0:self.batch_size],
#            self.f_h5['{}/X_i_tk/mel'.format(key)][0:self.batch_size],
#            self.f_h5['{}/X_i_tk_prime/mel'.format(key)][0:self.batch_size],
#            self.f_h5['{}/X_j/mel'.format(key)][0:self.batch_size])
#        self.index += 1
#        return data

class myDataset(data.Dataset):
    def __init__(self, h5_path, index_path, seg_len=128):
        self.h5 = h5py.File(h5_path, 'r')
        with open(index_path) as f_index:
            self.indexes = json.load(f_index)

        self.seg_len = seg_len

    def __getitem__(self, i):
        index = self.indexes[i]
        i, j,= index['i'], index['j'] 
        t, t_k, t_k_prime, t_j = index['t'], index['t_k'], index['t_k_prime'], index['t_j']
        seg_len = self.seg_len
        data = []
        data.append(self.h5['train/{}/mel'.format(i)][t:t+seg_len])
        data.append(self.h5['train/{}/mel'.format(i)][t_k:t_k+seg_len])
        data.append(self.h5['train/{}/mel'.format(i)][t_k_prime:t_k_prime+seg_len])
        data.append(self.h5['train/{}/mel'.format(j)][t_j:t_j+seg_len])
        return tuple(data)

    def __len__(self):
        return len(self.indexes)

class Logger(object):
    def __init__(self, log_dir='./log'):
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

if __name__ == '__main__':
    hps = Hps()
    hps.dump('./hps/v4.json')
    dataset = myDataset('/storage/raw_feature/voice_conversion/tacotron_feature/train-clean-100.h5',\
            '/storage/librispeech_index/200k.json')
    def merge(data_list):
        [data[0] for data in data_list]
    data_loader = data.DataLoader(
            dataset=dataset,
            batch_size=16,
            shuffle=False)
    for i, batch in enumerate(data_loader):
        print(batch)



