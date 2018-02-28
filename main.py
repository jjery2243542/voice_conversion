import torch
from torch import optim
from torch.autograd import Variable
import numpy as np
import pickle
from utils import Hps
from utils import DataLoader
from utils import Logger
from utils import myDataset
from utils import SingleDataset
from solver import Solver
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=True, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--single', default=False, action='store_true')
    parser.add_argument('--load_model', default=False, action='store_true')
    parser.add_argument('-flag', default='train')
    parser.add_argument('-hps_path', default='./hps/v7.json')
    parser.add_argument('-load_model_path', default='/storage/model/voice_conversion/'
            'pretrain_model.pkl-19999')
    parser.add_argument('-dataset_path', default='/storage/raw_feature/voice_conversion/vctk/vctk.h5')
    parser.add_argument('-index_path', default='/storage/raw_feature/voice_conversion/vctk/128_513_2000k.json')
    parser.add_argument('-output_model_path', default='/storage/model/voice_conversion/model.pkl')
    args = parser.parse_args()
    hps = Hps()
    hps.load(args.hps_path)
    hps_tuple = hps.get_tuple()
    if not args.single:
        dataset = myDataset(args.dataset_path,
                args.index_path,
                seg_len=hps_tuple.seg_len)
    else:
        dataset = SingleDataset(args.dataset_path,
                args.index_path,
                seg_len=hps_tuple.seg_len)
    data_loader = DataLoader(dataset)

    solver = Solver(hps_tuple, data_loader)
    if args.load_model:
        solver.load_model(args.load_model_path)
    if args.train:
        solver.train(args.output_model_path, args.flag, mode='pretrain_G')
        solver.train(args.output_model_path, args.flag, mode='pretrain_D')
        solver.train(args.output_model_path, args.flag, mode='train')
        solver.train(args.output_model_path, args.flag, mode='patchGAN')
