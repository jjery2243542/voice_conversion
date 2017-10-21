import torch
from torch import optim
from torch.autograd import Variable
from torch import nn
import numpy as np
import pickle
from model import Encoder
from model import Discriminator
from utils import Hps
from utils import DataLoader
from utils import Logger
import os 

class Solver(object):
    def __init__(self, hps, data_loader, log_dir='./log/'):
        self.hps = hps
        self.data_loader = data_loader
        self.model_kept = []
        self.max_keep = 10
        self.Encoder_s = None
        self.Encoder_c = None
        self.Decoder = None
        self.Discriminator = None
        self.G_opt = None
        self.D_opt = None
        self.build_model()
        self.logger = Logger(log_dir)

    def build_model(self):
        self.Encoder_s = Encoder(1, 1)
        self.Encoder_c = Encoder(1, 1)
        self.Decoder = Encoder(2, 1)
        self.Discriminator = Discriminator(2)
        if torch.cuda.is_available():
            self.Encoder_s.cuda()
            self.Encoder_c.cuda()
            self.Decoder.cuda()
            self.Discriminator.cuda()
        params = list(self.Encoder_s.parameters()) \
            + list(self.Encoder_c.parameters())\
            + list(self.Decoder.parameters())
        self.G_opt = optim.Adam(params, lr=self.hps.lr)
        self.D_opt = optim.Adam(self.Discriminator.parameters(), lr=self.hps.lr)

    def to_var(self, x):
        x = Variable(torch.from_numpy(x), requires_grad=True)
        return x.cuda() if torch.cuda.is_available() else x

    def pretrain_discri(self, model_path):
        # use spectrum pretrain
        for iteration in range(self.hps.pretrain_iterations):
            X_i_t, X_i_tk, X_j = [self.to_var(x) for x in next(self.data_loader)]
            D_same = self.Discriminator(X_i_t, X_i_tk)
            D_diff = self.Discriminator(X_i_t, X_j)
            loss = -torch.mean(torch.log(D_same) + torch.log(1 - D_diff))
            self.reset_grad()
            loss.backward()
            self.D_opt.step()
            slot_value = (
                iteration + 1,
                self.hps.pretrain_iterations, 
                loss.data[0]
            )
            print('iteration: [%04d/%04d], loss: %.6f' % slot_value)
            info = {
                'pretrain_loss': loss.data[0],
            }
            for tag, value in info.items():
                self.logger.scalar_summary(tag, value, iteration + 1)

            if iteration % 500 == 0 or iteration + 1 >= self.hps.pretrain_iterations:
                self.save_model(model_path, iteration, dis_only=True)

    def save_model(self, model_path, iteration, dis_only=False):
        if not dis_only:
            Es_path = os.path.join(model_path, 'Es.pkl-{}'.format(iteration))
            Ec_path = os.path.join(model_path, 'Ec.pkl-{}'.format(iteration))
            Dec_path = os.path.join(model_path, 'Dec.pkl-{}'.format(iteration))
        Dis_path = os.path.join(model_path, 'Dis.pkl-{}'.format(iteration))
        if not dis_only:
            torch.save(self.Encoder_s.state_dict(), Es_path)
            torch.save(self.Encoder_c.state_dict(), Ec_path)
            torch.save(self.Decoder.state_dict(), Dec_path)
        torch.save(self.Discriminator.state_dict(), Dis_path)
        if not dis_only:
            self.model_kept.append([Es_path, Ec_path, Dec_path, Dis_path])
        else:
            self.model_kept.append([Dis_path])
        if len(self.model_kept) >= self.max_keep:
            for model_path in self.model_kept[0]:
                os.remove(model_path)
            self.model_kept.pop(0)

    def reset_grad(self):
        self.Encoder_s.zero_grad()
        self.Encoder_c.zero_grad()
        self.Decoder.zero_grad()
        self.Discriminator.zero_grad()

    def train(self, model_path):
        for iteration in range(self.hps.iterations):
            #===================== Train D =====================#
            X_i_t, X_i_tk, X_j = [self.to_var(x) for x in next(self.data_loader)]
            # encode
            Ec_i_t, Ec_i_tk, Ec_j = self.Encoder_c(X_i_t), self.Encoder_c(X_i_tk), self.Encoder_c(X_j)
            # train discriminator
            loss_adv_dis = -torch.mean(
                torch.log(self.Discriminator(Ec_i_t, Ec_i_tk)) +
                torch.log(1 - self.Discriminator(Ec_i_t, Ec_j))
            )
            self.reset_grad()
            (self.hps.beta * L_adv_dis).backward()
            self.D_opt.step()
            #===================== Train G =====================#
            # train encoder and decoder
            X_i_t, X_i_tk, X_j = [self.to_var(x) for x in next(self.data_loader)]
            # encode
            Es_i_t = self.Encoder_s(X_i_t)
            Es_i_tk = self.Encoder_s(X_i_tk)
            Ec_i_tk = self.Encoder_c(X_i_tk)
            loss_sim = torch.sum((Es_i_t - Es_i_tk) ** 2) / self.hps.batch_size
            E = torch.cat([Es_i_t, Ec_i_tk], dim=1)
            X_tilde = self.Decoder(E)
            loss_rec = torch.sum((X_tilde - X_i_tk) ** 2) / self.hps.batch_size
            loss_adv_enc = -torch.mean(
                0.5 * torch.log(self.Discriminator(Ec_i_tk, Ec_i_tk)) + 
                0.5 * torch.log(1 - self.Discriminator(Ec_i_tk, Ec_i_tk))
            )
            loss = loss_rec + self.hps.alpha * loss_sim + self.hps.beta * loss_adv_enc
            self.reset_grad()
            loss.backward()
            self.G_opt.step()
            # print info
            slot_value = (
                iteration,
                self.hps.iterations,
                loss_rec.data[0],
                loss_sim.data[0],
                loss_adv_dis.data[0],
                loss_adv_enc.data[0],
            )
            print(
                'iteration:[%06d/%06d], loss_rec=%.3f, loss_sim=%.3f, '
                'loss_adv_dis=%.3f, loss_adv_enc=%.3f' 
                % slot_value,
            )
            info = {
                'loss_rec':loss_rec.data[0],
                'loss_sim':loss_sim.data[0],
                'loss_adv_dis':loss_adv_dis.data[0],
                'loss_adv_enc':loss_adv_enc.data[0],
            }
            for tag, value in info.items():
                self.logger.scalar_summary(tag, value, iteration + 1)

            if iteration % 1000 == 0:
                self.save_model(model_path, iteration)

if __name__ == '__main__':
    hps = Hps()
    hps.load('./hps/v1.json')
    hps_tuple = hps.get_tuple()
    data_loader = DataLoader(
        '/nfs/Mazu/jjery2243542/voice_conversion/datasets/libre_equal.h5',
        '/nfs/Mazu/jjery2243542/voice_conversion/datasets/train-clean-100-speaker-sex.txt'
    )
    solver = Solver(hps_tuple, data_loader)

