import sys
import h5py
import numpy as np
import json
from collections import namedtuple
import random

class Sampler(object):
    def __init__(self, h5_path, dset, seg_len, used_speaker_path):
        self.dset = dset
        self.f_h5 = h5py.File(h5_path, 'r')
        self.seg_len = seg_len
        self.utt2len = self.get_utt_len()
        self.speakers = self.read_speakers(used_speaker_path)
        self.n_speaker = len(self.speakers)
        print(self.speakers)
        self.speaker2utts = {speaker:list(self.f_h5[f'{dset}/{speaker}'].keys()) for speaker in self.speakers}
        # remove too short utterence
        self.rm_too_short_utt(limit=self.seg_len)
        self.single_indexer = namedtuple('single_index', ['speaker', 'i', 't'])

    def get_utt_len(self):
        utt2len = {}
        for dset in ['train', 'test']:
            for speaker in self.f_h5[f'{dset}']:
                for utt_id in self.f_h5[f'{dset}/{speaker}']:
                    length = self.f_h5[f'{dset}/{speaker}/{utt_id}'][()].shape[0]
                    utt2len[(speaker, utt_id)] = length
        return utt2len

    def rm_too_short_utt(self, limit):
        for (speaker, utt_id), length in self.utt2len.items():
            if speaker in self.speakers and length <= limit and utt_id in self.speaker2utts[speaker]:
                self.speaker2utts[speaker].remove(utt_id)

    def read_speakers(self, path):
        with open(path) as f:
            speakers = [line.strip() for line in f]
            return speakers     

    def sample_utt(self, speaker_id, n_samples=1):
        # sample an utterence
        dset = self.dset
        utt_ids = random.sample(self.speaker2utts[speaker_id], n_samples)
        lengths = [self.f_h5[f'{dset}/{speaker_id}/{utt_id}'].shape[0] for utt_id in utt_ids]
        return [(utt_id, length) for utt_id, length in zip(utt_ids, lengths)]

    def rand(self, l):
        rand_idx = random.randint(0, len(l) - 1)
        return l[rand_idx]

    def sample_single(self):
        seg_len = self.seg_len
        speaker_idx, = random.sample(range(len(self.speakers)), 1)
        speaker = self.speakers[speaker_idx]
        (utt_id, utt_len), = self.sample_utt(speaker, 1)
        t = random.randint(0, utt_len - seg_len)  
        index_tuple = self.single_indexer(speaker=speaker_idx, i=f'{speaker}/{utt_id}', t=t)
        return index_tuple

if __name__ == '__main__':
    if len(sys.argv) < 6:
        print('usage: python3 make_single_samples.py [h5py path] [training sampled index path (.json)] '
                '[n_samples] [segment length] [used speaker file path]')
        exit(0)

    h5py_path = sys.argv[1]
    output_path = sys.argv[2]
    n_samples = int(sys.argv[3])
    segment_len = int(sys.argv[4])
    speaker_path = sys.argv[5]

    sampler = Sampler(h5_path=h5py_path, seg_len=segment_len, dset='train', used_speaker_path=speaker_path)
    samples = [sampler.sample_single()._asdict() for _ in range(n_samples)]
    with open(sys.argv[2], 'w') as f_json:
        json.dump(samples, f_json, indent=4, separators=(',', ': '))
