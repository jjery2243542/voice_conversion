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

def cal_mean_grad(net):
    grad = Variable(torch.FloatTensor([0])).cuda()
    for i, p in enumerate(net.parameters()):
        grad += torch.mean(p.grad)
    return grad.data[0] / (i + 1)

def to_var(x, requires_grad=True):
    x = Variable(x, requires_grad=requires_grad)
    return x.cuda() if torch.cuda.is_available() else x

def reset_grad(net_list):
    for net in net_list:
        net.zero_grad()

def grad_clip(net_list, max_grad_norm):
    for net in net_list:
        torch.nn.utils.clip_grad_norm(net.parameters(), max_grad_norm)

def calculate_gradients_penalty(netD, real_data, fake_data):
    alpha = torch.rand(real_data.size(0))
    alpha = alpha.view(real_data.size(0), 1, 1)
    alpha = alpha.cuda() if torch.cuda.is_available() else alpha
    alpha = Variable(alpha)
    interpolates = alpha * real_data + (1 - alpha) * fake_data

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(
        outputs=torch.mean(disc_interpolates),
        inputs=interpolates,
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients_penalty = (1. - torch.sqrt(1e-12 + torch.sum(gradients.view(gradients.size(0), -1)**2, dim=1))) ** 2
    gradients_penalty = torch.mean(gradients_penalty)
    return gradients_penalty

class Hps(object):
    def __init__(self):
        self.hps = namedtuple('hps', [
            'lr',
            'alpha_dis',
            'alpha_enc',
            'beta_dis', 
            'beta_dec', 
            'beta_clf',
            'lambda_',
            'ns', 
            'dp', 
            'max_grad_norm',
            'max_step',
            'seg_len',
            'emb_size',
            'n_latent_steps',
            'n_patch_steps', 
            'batch_size',
            'lat_sched_iters',
            'patch_start_iter', 
            'iters',
            ]
        )
        default = \
            [1e-4, 1e-2, 1e-4, 1e-3, 1e-4, 1e-4, 10, 0.01, 0.0, 5, 5, 128, 128, 5, 5, 32, 50000, 50000, 60000]
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
                i0=f'{speakerA}/{A_utt_id_0}', i1=f'{speakerA}/{A_utt_id_1}',\
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
        self.n_elements = len(self.dataset[0])
        self.batch_size = batch_size
        self.index = 0

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

class myDataset(data.Dataset):
    def __init__(self, h5_path, index_path, seg_len=64):
        self.h5 = h5py.File(h5_path, 'r')
        with open(index_path) as f_index:
            self.indexes = json.load(f_index)
        self.indexer = namedtuple('index', ['speaker_i', 'speaker_j', \
                'i0', 'i1', 'j', 't', 't_k', 't_prime', 't_j'])
        self.seg_len = seg_len

    def __getitem__(self, i):
        index = self.indexes[i]
        index = self.indexer(**index)
        speaker_i, speaker_j = index.speaker_i, index.speaker_j
        i0, i1, j = index.i0, index.i1, index.j
        t, t_k, t_prime, t_j = index.t, index.t_k, index.t_prime, index.t_j
        seg_len = self.seg_len
        data = [speaker_i, speaker_j]
        data.append(self.h5[f'train/{i0}/lin'][t:t+seg_len])
        data.append(self.h5[f'train/{i0}/lin'][t_k:t_k+seg_len])
        data.append(self.h5[f'train/{i1}/lin'][t_prime:t_prime+seg_len])
        data.append(self.h5[f'train/{j}/lin'][t_j:t_j+seg_len])
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
    hps.dump('./hps/v18.json')
    dataset = myDataset('/home_local/jjery2243542/voice_conversion/datasets/vctk/vctk.h5',\
            '/home_local/jjery2243542/voice_conversion/datasets/vctk/128_513_2000k.json')
    data_loader = DataLoader(dataset)
    for i, batch in enumerate(data_loader):
        print(torch.max(batch[2]))
    #sampler = Sampler()
    #for i in range(100):
    #    print(sampler.sample())
