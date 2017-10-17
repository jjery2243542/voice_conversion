import json
import h5py
import pickle
import os
from collections import defaultdict
from collections import namedtuple
import numpy as np
#import nltk
import math
import argparse
import random

class Hps(object):
    def __init__(self):
        self.hps = namedtuple('hps', [
            'lr',
            'alpha',
            'beta', 
            'max_grad_norm',
            'batch_size',
            'iterations',
            ]
        )
        default = [2e-3, 1, 1e-4, 2, 32, 100000]
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

if __name__ == '__main__':
    hps = Hps()
    hps.dump('./hps/v1.json')
