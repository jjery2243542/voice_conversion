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
        h5_path='/storage/raw_feature/voice_conversion/vctk/vctk.h5', 
        speaker_info_path='/storage/raw_feature/voice_conversion/vctk/speaker-info.txt', 
        utt_len_path='/storage/raw_feature/voice_conversion/vctk/vctk_length.txt', 
        max_step=5, 
        seg_len=64,
        n_speaker=8,
    ):
        self.f_h5 = h5py.File(h5_path, 'r')
        self.max_step = max_step
        self.seg_len = seg_len
        #self.read_sex_file(speaker_sex_path)
        self.read_vctk_speaker_file(speaker_info_path)
        self.utt2len = self.read_utt_len_file(utt_len_path)
        self.speakers = list(self.f_h5['train'].keys())
        self.n_speaker = n_speaker
        self.speaker_used = self.female_ids[:n_speaker // 2] + self.male_ids[:n_speaker // 2]
        self.speaker2utts = {speaker:list(self.f_h5['train/{}'.format(speaker)].keys()) \
                for speaker in self.speakers}
        # remove too short utterence
        self.rm_too_short_utt()
        self.indexer = namedtuple('index', ['speaker_i', 'speaker_j', \
                'i0', 'i1', 'j', 't', 't_k', 't_prime', 't_j'])

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

    def read_vctk_speaker_file(self, speaker_info_path):
        self.female_ids, self.male_ids = [], []
        with open(speaker_info_path, 'r') as f:
            lines = f.readlines()
            infos = [line.strip().split() for line in lines[1:]]
            for info in infos:
                if info[2] == 'F':
                    self.female_ids.append(info[0])
                else:
                    self.male_ids.append(info[0])
            
    def read_libre_sex_file(self, speaker_sex_path):
        with open(speaker_sex_path, 'r') as f:
            # Female
            f.readline()
            self.female_ids = f.readline().strip().split()
            # Male
            f.readline()
            self.male_ids = f.readline().strip().split()

    def sample_utt(self, speaker_id, n_samples=1):
        # sample an utterence
        utt_ids = random.sample(self.speaker2utts[speaker_id], n_samples)
        lengths = [self.f_h5[f'train/{speaker_id}/{utt_id}/mel'].shape[0] for utt_id in utt_ids]
        return [(utt_id, length) for utt_id, length in zip(utt_ids, lengths)]

    def rand(self, l):
        rand_idx = random.randint(0, len(l) - 1)
        return l[rand_idx] 

    def sample(self):
        seg_len = self.seg_len
        max_step = self.max_step
        # sample two speakers
        speakerA_idx, speakerB_idx = random.sample(range(len(self.speaker_used)), 2)
        speakerA, speakerB = self.speaker_used[speakerA_idx], self.speaker_used[speakerB_idx]
        (A_utt_id_0, A_len_0), (A_utt_id_1, A_len_1) = self.sample_utt(speakerA, 2)
        (B_utt_id, B_len), = self.sample_utt(speakerB, 1)
        # sample t and t^k 
        t = random.randint(0, A_len_0 - 2 * seg_len)  
        t_k = random.randint(t + seg_len, min(A_len_0 - seg_len, t + max_step * seg_len))
        t_prime = random.randint(0, A_len_1 - seg_len)
        # sample a segment from speakerB
        t_j = random.randint(0, B_len - seg_len)
        index_tuple = self.indexer(speaker_i=speakerA_idx, speaker_j=speakerB_idx,\
                i0=f'{speakerA}/{A_utt_id_0}', i1=f'{speakerB}/{A_utt_id_1}',\
                j=f'{speakerB}/{B_utt_id}', t=t, t_k=t_k, t_prime=t_prime, t_j=t_j)
        return index_tuple

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

class DataLoader(object):
    def __init__(self, dataset, batch_size=16):
        self.dataset = dataset
        self.batch_size = batch_size
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        batch = [[] for _ in range(5)]
        for i in range(self.batch_size):
            sample = self.dataset[self.index + i]
            for j in range(len(batch)):
                batch[j].append(sample[j])

        for j in range(len(batch)):
            data = torch.stack([torch.from_numpy(data) for data in batch[j]], dim=0)
            batch[j] = data

        if self.index + 2 * self.batch_size >= len(self.dataset):
            self.index = 0
        else:
            self.index += self.batch_size
        return tuple(batch)

class myDataset(data.Dataset):
    def __init__(self, h5_path, index_path, seg_len=64):
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
        data.append(self.h5['train/{}/lin'.format(i)][t_k:t_k+seg_len])
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
    #hps = Hps()
    #hps.dump('./hps/v4.json')
    #dataset = myDataset('/storage/raw_feature/voice_conversion/tacotron_feature/train-clean-100.h5',\
    #        '/storage/librispeech_index/200k.json')
    #data_loader = DataLoader(dataset)
    #for i, batch in enumerate(data_loader):
    #    for j in batch:
    #        print(torch.sum(j), end=', ')
    sampler = Sampler()
    for i in range(100):
        print(sampler.sample())



