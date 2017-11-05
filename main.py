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
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--load_model_path', default='/nfs/Mazu/jjery2243542/voice_conversion/'
    'model/pretrain.pkl-1300')
    args = parser.parse_args()
    hps = Hps()
    hps.load('./hps/v2.json')
    hps_tuple = hps.get_tuple()
    data_loader = DataLoader(
        '/nfs/Mazu/jjery2243542/voice_conversion/datasets/multi_sex_16_10.h5', 
        batch_size=hps_tuple.batch_size
    )
    solver = Solver(hps_tuple, data_loader)
    if args.load_model:
        solver.load_model(args.load_model_path)
    if args.pretrain:
        solver.train('/nfs/Mazu/jjery2243542/voice_conversion/model/pretrain.pkl', is_pretrain=True)
    if args.train:
        solver.train('/nfs/Mazu/jjery2243542/voice_conversion/model/model.pkl', is_pretrain=False)
