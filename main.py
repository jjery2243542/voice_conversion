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

#python main.py -dataset_path /data/home_ext/arshdeep/vctk_h5py/vctk_random20_setok2.h5 -index_path ./vctk_random20_setok2_index.json -output_model_path /data/home_ext/arshdeep/models2/single_sample_model.pkl > /data/home_ext/arshdeep/logs/train2_1219.txt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=True, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--single', default=True, action='store_true')
    parser.add_argument('--load_model', default=False, action='store_true')
<<<<<<< HEAD
    parser.add_argument('--is_h5', default=False, action='store_true')
=======
    parser.add_argument('--is_h5', default=True, action='store_true')
>>>>>>> 1fbdb8b8865b71091d2c4e69d0e0e77d0f96f13c
    parser.add_argument('-flag', default='train')
    parser.add_argument('-hps_path', default='./hps/vctk.json')
    parser.add_argument('-load_model_path', default='')
    parser.add_argument('-dataset_path', default='/data/home_ext/arshdeep/vctk_h5py/vctk_random20_setok2.h5')
    parser.add_argument('-index_path', default='vctk_random20_setok2_index.json')
    parser.add_argument('-output_model_path', default='./models/model_single_sample.pkl')
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
                seg_len=hps_tuple.seg_len, is_h5=args.is_h5)

    data_loader = DataLoader(dataset)

    solver = Solver(hps_tuple, data_loader)
    if args.load_model:
        solver.load_model(args.load_model_path)
    if args.train:
        solver.train(args.output_model_path, args.flag, mode='pretrain_G')
        solver.train(args.output_model_path, args.flag, mode='pretrain_D')
        solver.train(args.output_model_path, args.flag, mode='train')
        solver.train(args.output_model_path, args.flag, mode='patchGAN')
<<<<<<< HEAD
=======

>>>>>>> 1fbdb8b8865b71091d2c4e69d0e0e77d0f96f13c
