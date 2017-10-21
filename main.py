import torch
from torch import optim
from torch.autograd import Variable
import numpy as np
import pickle
from model import Encoder
from model import Discriminator
from utils import Hps
from utils import DataLoader
from utils import Logger
from solver import Solver
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    hps = Hps()
    hps.load('./hps/v1.json')
    hps_tuple = hps.get_tuple()
    data_loader = DataLoader(
        '/storage/raw_feature/voice_conversion/two_speaker_16_5.h5'
    )
    solver = Solver(hps_tuple, data_loader)
    if args.pretrain:
        solver.pretrain_discri('/storage/model/voice_conversion')
    if args.train:
        solver.train('/nfs/Mazu/jjery2243542/voice_conversion/model')
