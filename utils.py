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

class Hps(object):
    def __init__(self):
        self.hps = namedtuple('hps', [
            'lr',
            'alpha',
            'beta',
            'max_step',
            'max_grad_norm',
            'batch_size',
            'iterations',
            ]
        )
        default = [2e-3, 1, 1e-4, 5, 2, 32, 100000]
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

class DataLoader(object):
    def __init__(self, h5_path, speaker_sex_path, max_step=5, batch_size=16):
        self.f_h5 = h5py.File(h5_path, 'r')
        self.batch_size = batch_size
        self.max_step = max_step
        self.read_sex_file(speaker_sex_path)
        self.speakers = list(self.f_h5['train'].keys())

    def read_sex_file(self, speaker_sex_path):
        with open(speaker_sex_path, 'r') as f:
            # Female
            f.readline()
            self.female_ids = f.readline().strip().split()
            # Male
            f.readline()
            self.male_ids = f.readline().strip().split()

    def sample_utt(self, speaker_id):
        utt_ids = list(self.f_h5['train/{}'.format(speaker_id)].keys())
        # sample an utterence
        utt_id = random.sample(utt_ids, 1)[0]
        spec = self.f_h5['train/{}/{}'.format(speaker_id, utt_id)]
        return spec
        
    def sample(self):
        # sample two speakers
        #speakerA, speakerB = random.sample(self.speakers, 2)
        speakerA = random.sample(self.female_ids, 1)[0]
        speakerB = random.sample(self.male_ids)[0]
        specA = self.sample_utt(speakerA)
        # sample t and t^k 
        t = random.randint(0, specA.shape[0] - 2)
        t_k = random.randint(t, min(specA.shape[0] - 1, t + self.max_step))
        # sample a segment from speakerB
        specB = self.sample_utt(speakerB)
        segB = random.sample(specB, 1)[0]
        return specA[t][0:1], specA[t_k][0:1], segB[0:1] 

    def next_batch(self):
        all_X = [[], [], []]
        for i in range(self.batch_size):
            for X, x in zip(all_X, self.sample()):
                X.append(x) 
        return [np.array(X, dtype=np.float32) for X in all_X]


if __name__ == '__main__':
    hps = Hps()
    hps.dump('./hps/v1.json')
    #data_loader = DataLoader(
    #    '/storage/raw_feature/voice_conversion/libre_equal.h5',
    #    '/storage/raw_feature/voice_conversion/train-clean-100-speaker-sex.txt',
    #)
    #for i in range(100):
    #    print(i)
    #    _ = data_loader.next_batch()

