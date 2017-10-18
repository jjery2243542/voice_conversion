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
        if torch.cuda.is_available():
            self.Es.cuda()
            self.Ec.cuda()
            self.D.cuda()
            self.C.cuda()
        params = list(self.Es.parameters()) + list(self.Ec.parameters()) + list(self.D.parameters())
        self.G_opt = optim.Adam(params, lr=self.hps.lr)
        params = list(self.Cc.parameters()) + list(self.Cs.parameters()
        self.C_opt = optim.Adam(params, lr=self.hps.lr)

    def reset_grad(self):
        self.Es.zero_grad()
        self.Ec.zero_grad()
        self.D.zero_grad()
        self.C.zero_grad()

    def to_var(self, x):
        if torch.cuda.is_available():
            x.cuda()
        return Variable(x)

    def train(self):
        


if __name__ == '__main__':
    hps = Hps()
    hps.load('./hps/v1.json')
    hps_tuple = hps.get_tuple()
    Solver(hps_tuple, '')

