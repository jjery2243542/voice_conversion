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
from torch.autograd import Variable

def cc(net):
    if torch.cuda.is_available():
        return net.cuda()
    else:
        return net

def gen_noise(x_dim, y_dim):
    x = torch.randn(x_dim, 1) 
    y = torch.randn(1, y_dim)
    return x @ y

def to_var(x, requires_grad=True):
    x = Variable(x, requires_grad=requires_grad)
    return x.cuda() if torch.cuda.is_available() else x

def reset_grad(net_list):
    for net in net_list:
        net.zero_grad()

def grad_clip(net_list, max_grad_norm):
    for net in net_list:
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)

def calculate_gradients_penalty(netD, real_data, fake_data):
    alpha = torch.rand(real_data.size(0))
    alpha = alpha.view(real_data.size(0), 1, 1)
    alpha = alpha.cuda() if torch.cuda.is_available() else alpha
    alpha = Variable(alpha)
    interpolates = alpha * real_data + (1 - alpha) * fake_data

    disc_interpolates = netD(interpolates)

    use_cuda = torch.cuda.is_available()
    grad_outputs = torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(disc_interpolates.size())

    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients_penalty = (1. - torch.sqrt(1e-12 + torch.sum(gradients.view(gradients.size(0), -1)**2, dim=1))) ** 2
    gradients_penalty = torch.mean(gradients_penalty)
    return gradients_penalty

def cal_acc(logits, y_true):
    _, ind = torch.max(logits, dim=1)
    acc = torch.sum((ind == y_true).type(torch.FloatTensor)) / y_true.size(0)
    return acc

class Hps(object):
    def __init__(self):
        self.hps = namedtuple('hps', [
            'lr',
            'alpha_dis',
            'alpha_enc',
            'beta_dis', 
            'beta_gen', 
            'beta_clf',
            'lambda_',
            'ns', 
            'enc_dp', 
            'dis_dp', 
            'max_grad_norm',
            'seg_len',
            'emb_size',
            'n_speakers',
            'n_latent_steps',
            'n_patch_steps', 
            'batch_size',
            'lat_sched_iters',
            'enc_pretrain_iters',
            'dis_pretrain_iters',
            'patch_iters', 
            'iters',
            ]
        )
        default = \
            [1e-4, 1, 1e-4, 0, 0, 0, 10, 0.01, 0.5, 0.1, 5, 128, 128, 8, 5, 0, 32, 50000, 5000, 5000, 30000, 60000]
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
    def __init__(self, dataset, batch_size=16):
        self.dataset = dataset
        self.n_elements = len(self.dataset[0])
        self.batch_size = batch_size
        self.index = 0

    def all(self, size=1000):
        samples = [self.dataset[self.index + i] for i in range(size)]
        batch = [[s for s in sample] for sample in zip(*samples)]
        batch_tensor = [torch.from_numpy(np.array(data)) for data in batch]

        if self.index + 2 * self.batch_size >= len(self.dataset):
            self.index = 0
        else:
            self.index += self.batch_size
        return tuple(batch_tensor)

    def __iter__(self):
        return self

    def __next__(self):
        samples = [self.dataset[self.index + i] for i in range(self.batch_size)]
        batch = [[s for s in sample] for sample in zip(*samples)]
        batch_tensor = [torch.from_numpy(np.array(data)) for data in batch]

        if self.index + 2 * self.batch_size >= len(self.dataset):
            self.index = 0
        else:
            self.index += self.batch_size
        return tuple(batch_tensor)

class SingleDataset(data.Dataset):
    def __init__(self, h5_path, index_path, dset='train', seg_len=128):
        self.dataset = h5py.File(h5_path, 'r')
        with open(index_path) as f_index:
            self.indexes = json.load(f_index)
        self.indexer = namedtuple('index', ['speaker', 'i', 't'])
        self.seg_len = seg_len
        self.dset = dset

    def __getitem__(self, i):
        index = self.indexes[i]
        index = self.indexer(**index)
        speaker = index.speaker
        i, t = index.i, index.t
        seg_len = self.seg_len
        data = [speaker, self.dataset[f'{self.dset}/{i}'][t:t+seg_len]]
        return tuple(data)

    def __len__(self):
        return len(self.indexes)

class Logger(object):
    def __init__(self, log_dir='./log'):
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

