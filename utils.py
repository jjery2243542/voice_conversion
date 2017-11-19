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
import tensorflow as tf

class Hps(object):
    def __init__(self):
        self.hps = namedtuple('hps', [
            'lr',
            'alpha',
            'beta',
            'lambda_',
            'max_grad_norm',
            'max_step',
            'batch_size',
            'iterations',
            ]
        )
        default = [5e-4, 1, 1, 1, 1, 5, 16, 15000]
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
        self.speaker2utts = {speaker:list(self.f_h5['train/{}'.format(speaker)].keys()) for speaker in self.speakers}
        # remove too short utterence
        self.rm_too_short_utt()

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
        speakerA, speakerB = random.sample(self.speakers, 2)
        A_utt_id, A_len = self.sample_utt(speakerA)
        B_utt_id, B_len = self.sample_utt(speakerB)
        # sample t and t^k 
        t = random.randint(0, A_len - 2 * seg_len)  
        t_k = random.randint(t + seg_len, min(A_len - seg_len, t + max_step * seg_len))
        t_k_prime = random.randint(t + seg_len, min(A_len - seg_len, t + max_step * seg_len))
        # sample a segment from speakerB
        t_j = random.randint(0, B_len - seg_len)
        return self.f_h5['train/{}/{}/mel'.format(speakerA, A_utt_id)][t:t + seg_len], \
            self.f_h5['train/{}/{}/lin'.format(speakerA, A_utt_id)][t:t + seg_len],\
            self.f_h5['train/{}/{}/mel'.format(speakerA, A_utt_id)][t_k:t_k + seg_len],\
            self.f_h5['train/{}/{}/lin'.format(speakerA, A_utt_id)][t_k:t_k + seg_len],\
            self.f_h5['train/{}/{}/mel'.format(speakerA, A_utt_id)][t_k_prime:t_k_prime + seg_len],\
            self.f_h5['train/{}/{}/lin'.format(speakerA, A_utt_id)][t_k_prime:t_k_prime + seg_len],\
            self.f_h5['train/{}/{}/mel'.format(speakerB, B_utt_id)][t_j:t_j + seg_len],\
            self.f_h5['train/{}/{}/lin'.format(speakerB, B_utt_id)][t_j:t_j + seg_len]

class DataLoader(object):
    def __init__(self, h5py_path, batch_size=16):
        self.f_h5 = h5py.File(h5py_path)
        self.keys = list(self.f_h5.keys())
        self.index = 0
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.keys):
            self.index = 0
        key = self.keys[index]
        data = (self.f_h5['{}/X_i_t/mel'.format(key)][0:self.batch_size],
            self.f_h5['{}/X_i_tk/mel'.format(key)][0:self.batch_size],
            self.f_h5['{}/X_i_tk_prime/mel'.format(key)][0:self.batch_size],
            self.f_h5['{}/X_j/mel'.format(key)][0:self.batch_size])
        self.index += 1
        return data

class Logger(object):
    def __init__(self, log_dir='./log'):
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

if __name__ == '__main__':
    #sampler = Sampler(h5_path='/storage/raw_feature/voice_conversion/tacotron_feature/train-clean-100.h5', \
    #    speaker_sex_path='/storage/raw_feature/voice_conversion/train-clean-100-speaker-sex.txt', \
    #    max_step=5,\
    #    seg_len=128
    #)

    #st = time.time()
    #for _ in range(500):
    #    datas = sampler.sample()
    #et = time.time()
    #print(et - st)
    hps = Hps()
    hps.dump('./hps/v4.json')
    #data_loader = DataLoader('/storage/raw_feature/voice_conversion/two_speaker_16_5.h5')
    #for _ in range(10):
    #    print(next(data_loader))
    #st = time.time()
    #for i in range(100):
    #    print(i)
    #    batch = next(data_loader)
    #et = time.time()
    #print(et - st)
    #logger = Logger()
    #for i in range(100):
    #    logger.scalar_summary('loss', np.random.randn(), i + 1)


