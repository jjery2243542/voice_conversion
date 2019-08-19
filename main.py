import torch
from torch import optim
from torch.autograd import Variable
import numpy as np
import pickle
from utils import Hps
from utils import DataLoader
from utils import Logger
from utils import SingleDataset
from solver import Solver
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', default=False, action='store_true')
    parser.add_argument('-flag', default='train')
    parser.add_argument('-hps_path')
    parser.add_argument('-load_model_path')
    parser.add_argument('-dataset_path')
    parser.add_argument('-index_path')
    parser.add_argument('-output_model_path')
    args = parser.parse_args()
    hps = Hps()
    hps.load(args.hps_path)
    hps_tuple = hps.get_tuple()
    dataset = SingleDataset(args.dataset_path, args.index_path, seg_len=hps_tuple.seg_len)
    data_loader = DataLoader(dataset)

    solver = Solver(hps_tuple, data_loader)
    if args.load_model:
        solver.load_model(args.load_model_path)

    solver.train(args.output_model_path, args.flag, mode='pretrain_G')
    solver.train(args.output_model_path, args.flag, mode='pretrain_D')
    solver.train(args.output_model_path, args.flag, mode='train')
    solver.train(args.output_model_path, args.flag, mode='patchGAN')
