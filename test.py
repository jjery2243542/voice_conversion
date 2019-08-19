import torch
from torch import optim
from torch.autograd import Variable
import numpy as np
import pickle
from utils import Hps
from preprocess.tacotron.norm_utils import spectrogram2wav, get_spectrograms
from scipy.io.wavfile import write
import glob
import os
import argparse
from solver import Solver

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-hps', help='The path of hyper-parameter set', default='vctk.json')
    parser.add_argument('-model', '-m', help='The path of model checkpoint')
    parser.add_argument('-source', '-s', help='The path of source .wav file')
    parser.add_argument('-target', '-t', help='Target speaker id (integer). Same order as the speaker list when preprocessing (en_speaker_used.txt)')
    parser.add_argument('-output', '-o', help='output .wav path')
    parser.add_argument('-sample_rate', '-sr', default=16000, type=int)
    parser.add_argument('--use_gen', default=True, action='store_true')

    args = parser.parse_args()

    hps = Hps()
    hps.load(args.hps)
    hps_tuple = hps.get_tuple()
    solver = Solver(hps_tuple, None)
    solver.load_model(args.model)
    _, spec = get_spectrograms(args.source)
    spec_expand = np.expand_dims(spec, axis=0)
    spec_tensor = torch.from_numpy(spec_expand).type(torch.FloatTensor)
    c = Variable(torch.from_numpy(np.array([int(args.target)]))).cuda()
    result = solver.test_step(spec_tensor, c, gen=args.use_gen)
    result = result.squeeze(axis=0).transpose((1, 0))
    wav_data = spectrogram2wav(result)
    write(args.output, rate=args.sample_rate, data=wav_data)
