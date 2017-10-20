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

if __name__ == '__main__':
    hps = Hps()
    hps.load('./hps/v1.json')
    hps_tuple = hps.get_tuple()
    data_loader = DataLoader(
        '/nfs/Mazu/jjery2243542/voice_conversion/datasets/libre_equal.h5',
        '/nfs/Mazu/jjery2243542/voice_conversion/datasets/train-clean-100-speaker-sex.txt'
    )
    solver = Solver(hps_tuple, data_loader)
    solver.train('/nfs/Mazu/jjery2243542/voice_conversion/model')
