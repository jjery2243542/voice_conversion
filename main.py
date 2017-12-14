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
from utils import myDataset
from solver import Solver
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=True, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--load_model', default=True, action='store_true')
    parser.add_argument('-load_model_path', default='/storage/model/voice_conversion/'
            'pretrain_model.pkl-19999')
    parser.add_argument('-dataset_path', default='/storage/raw_feature/voice_conversion/vctk/vctk.h5')
    parser.add_argument('-index_path', default='/storage/raw_feature/voice_conversion/vctk/128_513_2000k.json')
    parser.add_argument('-pretrain_model_path', default='/storage/model/voice_conversion/pretrain_model.pkl')
    parser.add_argument('-output_model_path', default='/storage/model/voice_conversion/model.pkl')
    args = parser.parse_args()
    hps = Hps()
    hps.load('./hps/v4.json')
    hps_tuple = hps.get_tuple()
    dataset = myDataset(args.dataset_path,
            args.index_path,
            seg_len=hps_tuple.seg_len)
    data_loader = DataLoader(dataset)

    solver = Solver(hps_tuple, data_loader)
    if args.load_model:
        solver.load_model(args.load_model_path)
    if args.train:
        solver.train(args.pretrain_model_path, pretrain=True)
        solver.train(args.output_model_path, pretrain=False)
