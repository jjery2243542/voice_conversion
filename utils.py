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
import tensorflow as tf
from torch.utils import data

class Hps(object):
    def __init__(self):
        self.hps = namedtuple('hps', [
            'lr',
            'alpha',
            'beta',
            'max_step',
            'batch_size',
            'iterations',
            ]
        )
        default = [2e-3, 1, 1e-4, 5, 16, 100000]
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
    def __init__(self, h5_path, speaker_sex_path, max_step=5):
        self.f_h5 = h5py.File(h5_path, 'r')
        self.max_step = max_step
        self.read_sex_file(speaker_sex_path)
        self.speakers = list(self.f_h5['train'].keys())
        self.speaker2utts = {speaker:list(self.f_h5['train/{}'.format(speaker)].keys()) for speaker in self.speakers}

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
        spec = self.f_h5['train/{}/{}'.format(speaker_id, utt_id)]
        return spec

    def rand(self, l):
        rand_idx = random.randint(0, len(l) - 1)
        return l[rand_idx] 

    def sample(self):
        # sample two speakers
        #speakerA, speakerB = random.sample(self.speakers, 2)
        #speakerA = self.rand(self.female_ids)
        #speakerB = self.rand(self.male_ids)
        speakerA = self.female_ids[0]
        speakerB = self.male_ids[0]
        specA = self.sample_utt(speakerA)
        # sample t and t^k 
        t = random.randint(0, specA.shape[0] - 2)
        t_k = random.randint(t, min(specA.shape[0] - 1, t + self.max_step))
        # sample a segment from speakerB
        specB = self.sample_utt(speakerB)
        j = random.randint(0, specB.shape[0] - 1)
        return specA[t][0:1], specA[t_k][0:1], specB[j][0:1] 

class DataLoader(object):
    def __init__(self, h5py_path):
        self.f_h5 = h5py.File(h5py_path)
        self.keys = list(self.f_h5.keys())
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.keys):
            self.index = 0
        return self.f_h5['{}/X_i_t'.format(self.index)],\
            self.f_h5['{}/X_i_tk'.format(self.index)],\
            self.f_h5['{}/X_j'.format(self.index)]

class Logger(object):
    def __init__(self, log_dir='./log'):
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

if __name__ == '__main__':
    hps = Hps()
    hps.dump('./hps/v1.json')
    #data_loader = DataLoader(
    #    '/storage/raw_feature/voice_conversion/libre_equal.h5',
    #    '/storage/raw_feature/voice_conversion/train-clean-100-speaker-sex.txt',
    #)
    #st = time.time()
    #for i in range(100):
    #    print(i)
    #    batch = next(data_loader)
    #et = time.time()
    #print(et - st)
    #logger = Logger()
    #for i in range(100):
    #    logger.scalar_summary('loss', np.random.randn(), i + 1)


