import torch
from torch import optim
from torch.autograd import Variable
import numpy as np
from model import Encoder
from model import Discriminator
from utils import Hps

class Solver(object):
    def __init__(self, hps, data_loader):
        self.hps = hps
        self.data_loader = data_loader
        self.Es = None
        self.Ec = None
        self.D = None
        self.C = None
        self.G_opt = None
        self.C_opt = None
        self.build_model()

    def build_model(self):
        self.Es = Encoder(2, 1)
        self.Ec = Encoder(2, 1)
        self.D = Encoder(2, 2)
        self.C = Discriminator(2)
        params = list(self.Es.parameters()) + list(self.Ec.parameters()) + list(self.D.parameters())
        self.G_opt = optim.Adam(params, lr=self.hps.lr)
        self.C_opt = optim.Adam(self.C.parameters(), lr=self.hps.lr)


if __name__ == '__main__':
    hps = Hps()
    hps.load('./hps/v1.json')
    hps_tuple = hps.get_tuple()
    Solver(hps_tuple, '') 

