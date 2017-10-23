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

def mean_grad(net):
    grad = Variable(torch.FloatTensor([0])).cuda()
    for i, p in enumerate(net.parameters()):
        grad += torch.mean(p.grad)
    return grad / (i + 1)

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

    def save_model(self, model_path, iteration, enc_only=True):
        if not enc_only:
            all_model = {
                'encoder_s': self.Encoder_s.state_dict(),
                'encoder_c': self.Encoder_c.state_dict(),
                'decoder': self.Decoder.state_dict(),
                'discriminator': self.Discriminator.state_dict(),
            }
        else:
            all_model = {
                'encoder_s': self.Encoder_s.state_dict(),
                'encoder_c': self.Encoder_c.state_dict(),
                'decoder': self.Decoder.state_dict(),
            }
        new_model_path = '{}-{}'.format(model_path, iteration)
        with open(new_model_path, 'wb') as f_out:
            pickle.dump(all_model, f_out)
        self.model_kept.append(new_model_path)

        if len(self.model_kept) >= self.max_keep:
            os.remove(self.model_kept[0])
            self.model_kept.pop(0)

    def reset_grad(self):
        self.Encoder_s.zero_grad()
        self.Encoder_c.zero_grad()
        self.Decoder.zero_grad()
        self.Discriminator.zero_grad()

    def load_model(self, model_path, enc_only=False):
        print('load model from {}'.format(model_path))
        with open(model_path, 'rb') as f_in:
            all_model = pickle.load(f_in)
            self.Encoder_s.load_state_dict(all_model['encoder_s'])
            self.Encoder_c.load_state_dict(all_model['encoder_c'])
            self.Decoder.load_state_dict(all_model['decoder'])
            if not enc_only:
            self.Discriminator.load_state_dict(all_model['discriminator'])


    def train(self, model_path, is_pretrain=False):
        batch_size = self.hps.batch_size
        pretrain_iterations, iterations = self.hps.pretrain_iterations, self.hps.iterations
        alpha, beta1, beta2 = self.hps.alpha, self.hps.beta1, self.hps.beta2
        for iteration in range(pretrain_iterations if is_pretrain else iterations):
            if not is_pretrain:
                #===================== Train D =====================#
                X_i_t, X_i_tk, X_j = [self.to_var(x) for x in next(self.data_loader)]
                # encode
                Ec_i_t, Ec_i_tk, Ec_j = self.Encoder_c(X_i_t), self.Encoder_c(X_i_tk), self.Encoder_c(X_j)
                # train discriminator
                loss_adv_dis = torch.mean(
                    (self.Discriminator(Ec_i_t, Ec_i_tk) - 1) ** 2 +
                    self.Discriminator(Ec_i_t, Ec_j) ** 2
                )
                self.reset_grad()
                (beta1 * loss_adv_dis).backward()
                self.D_opt.step()
                # print info
                slot_value = (
                    iteration,
                    iterations,
                    loss_adv_dis.data[0],
                )
                print(
                    'D-iteration:[%06d/%06d], loss_adv_dis=%.3f' 
                    % slot_value,
                )
                info = {
                    'loss_adv_dis':loss_adv_dis.data[0],
                }
                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value, iteration + 1)
            #===================== Train G =====================#
            X_i_t, X_i_tk, X_j = [self.to_var(x) for x in next(self.data_loader)]
            # encode
            Es_i_t = self.Encoder_s(X_i_t)
            Es_i_tk = self.Encoder_s(X_i_tk)
            Ec_i_tk = self.Encoder_c(X_i_tk)
            loss_sim = torch.sum((Es_i_t - Es_i_tk) ** 2) / batch_size
            E = torch.cat([Es_i_t, Ec_i_tk], dim=1)
            X_tilde = self.Decoder(E)
            loss_rec = torch.sum((X_tilde - X_i_tk) ** 2) / batch_size
            if not is_pretrain:
                Ec_val = self.Discriminator(Ec_i_tk, Ec_i_tk)
                mean_Ec_val = torch.mean(Ec_val)
                loss_adv_enc = torch.mean(
                    (Ec_val - 0.5) ** 2
                )
                loss = loss_rec + alpha * loss_sim + beta2 * loss_adv_enc
            else:
                loss = loss_rec + alpha * loss_sim
            self.reset_grad()
            loss.backward()
            self.G_opt.step()
            # print info
            slot_value = (
                iteration,
                pretrain_iterations if is_pretrain else iterations,
                loss_rec.data[0],
                loss_sim.data[0],
                0 if is_pretrain else loss_adv_enc.data[0],
                0 if is_pretrain else mean_Ec_val.data[0],
            )
            print(
                'G-iteration:[%06d/%06d], loss_rec=%.3f, loss_sim=%.3f, loss_adv_enc=%.3f, mean_val=%.3f' 
                % slot_value,
            )
            info = {
                'loss_rec':loss_rec.data[0],
                'loss_sim':loss_sim.data[0],
                'loss_adv_enc': 0 if is_pretrain else loss_adv_enc.data[0],
                'mean_val': 0 if is_pretrain else mean_Ec_val.data[0],
            }
            for tag, value in info.items():
                self.logger.scalar_summary(tag, value, iteration + 1)

            if iteration % 100 == 0:
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

