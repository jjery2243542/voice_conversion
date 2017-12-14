import torch
from torch import optim
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
import numpy as np
import pickle
from utils import myDataset
from model import Encoder
from model import Decoder
from model import Discriminator
from model import CBHG
import os
from utils import Hps
from utils import Logger
from utils import DataLoader
from preprocess.tacotron import utils

def cal_mean_grad(net):
    grad = Variable(torch.FloatTensor([0])).cuda()
    for i, p in enumerate(net.parameters()):
        grad += torch.mean(p.grad)
    return grad.data[0] / (i + 1)

def calculate_gradients_penalty(netD, real_data, fake_data):
    alpha = torch.rand(real_data.size(0))
    alpha = alpha.view(real_data.size(0), 1, 1)
    alpha = alpha.cuda() if torch.cuda.is_available() else alpha
    alpha = Variable(alpha)
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(
        outputs=torch.mean(disc_interpolates),
        inputs=interpolates,
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients_penalty = (1. - torch.sqrt(1e-12 + torch.sum(gradients.view(gradients.size(0), -1)**2, dim=1))) ** 2
    gradients_penalty = torch.mean(gradients_penalty)
    return gradients_penalty

class Solver(object):
    def __init__(self, hps, data_loader, log_dir='./log/'):
        self.hps = hps
        self.data_loader = data_loader
        self.model_kept = []
        self.max_keep = 10
        self.Encoder = None
        self.Decoder = None
        self.Discriminator = None
        #self.postnet = None
        self.G_opt = None
        self.D_opt = None
        self.build_model()
        self.logger = Logger(log_dir)

    def build_model(self):
        ns = self.hps.ns
        self.Encoder = Encoder(ns=ns)
        self.Decoder = Decoder(ns=ns)
        self.Discriminator = Discriminator(ns=ns)
        #self.postnet = CBHG()
        if torch.cuda.is_available():
            self.Encoder.cuda()
            self.Decoder.cuda()
            self.Discriminator.cuda()
            #self.postnet.cuda()
        params = list(self.Encoder.parameters()) + list(self.Decoder.parameters())
            #+ list(self.postnet.parameters())
        self.G_opt = optim.Adam(params, lr=self.hps.lr, betas=(0.5, 0.9))
        self.D_opt = optim.Adam(self.Discriminator.parameters(), lr=self.hps.lr, betas=(0.5, 0.9))

    def to_var(self, x, requires_grad=True):
        x = Variable(x, requires_grad=requires_grad)
        return x.cuda() if torch.cuda.is_available() else x

    def save_model(self, model_path, iteration, enc_only=True):
        if not enc_only:
            all_model = {
                'encoder': self.Encoder.state_dict(),
                'decoder': self.Decoder.state_dict(),
                'discriminator': self.Discriminator.state_dict(),
            }
        else:
            all_model = {
                'encoder': self.Encoder.state_dict(),
                'decoder': self.Decoder.state_dict(),
            }
        new_model_path = '{}-{}'.format(model_path, iteration)
        with open(new_model_path, 'wb') as f_out:
            torch.save(all_model, f_out)
        self.model_kept.append(new_model_path)

        if len(self.model_kept) >= self.max_keep:
            os.remove(self.model_kept[0])
            self.model_kept.pop(0)

    def reset_grad(self):
        self.Encoder.zero_grad()
        self.Decoder.zero_grad()
        self.Discriminator.zero_grad()

    def load_model(self, model_path, enc_only=True):
        print('load model from {}'.format(model_path))
        with open(model_path, 'rb') as f_in:
            all_model = torch.load(f_in)
            self.Encoder.load_state_dict(all_model['encoder'])
            self.Decoder.load_state_dict(all_model['decoder'])
            if not enc_only:
                self.Discriminator.load_state_dict(all_model['discriminator'])

    def grad_clip(self, net_list):
        max_grad_norm = self.hps.max_grad_norm
        for net in net_list:
            torch.nn.utils.clip_grad_norm(net.parameters(), max_grad_norm)

    def test_step(self, X, c):
        X = self.to_var(X).permute(0, 2, 1)
        E = self.Encoder(X)
        X_tilde = self.Decoder(E, c)
        return X_tilde.data.cpu().numpy()

    def train(self, model_path, pretrain=True):
        # load hyperparams
        batch_size = self.hps.batch_size
        D_iterations = self.hps.D_iterations
        pretrain_iterations = self.hps.pretrain_iterations
        iterations = self.hps.iterations
        max_grad_norm = self.hps.max_grad_norm
        alpha, lambda_ = self.hps.alpha, self.hps.lambda_
        flag = 'train'
        if pretrain:
            alpha, D_iterations = 0., 0
            iterations = pretrain_iterations
            flag = 'pretrain'
        for iteration in range(iterations):
            for j in range(D_iterations):
                #===================== Train D =====================#
                data = next(self.data_loader)
                X_i_t, X_i_tk, X_i_prime, X_j = \
                        [self.to_var(x).permute(0, 2, 1) for x in data[2:]]
                # encode
                E_i_t = self.Encoder(X_i_t)
                E_i_tk = self.Encoder(X_i_tk)
                E_i_prime = self.Encoder(X_i_prime)
                E_j = self.Encoder(X_j)
                same_pair = torch.cat([E_i_t, E_i_tk], dim=1)
                diff_pair = torch.cat([E_i_prime, E_j], dim=1)
                same_val = self.Discriminator(same_pair)
                diff_val = self.Discriminator(diff_pair)
                gradients_penalty = calculate_gradients_penalty(self.Discriminator, same_pair, diff_pair)
                w_distance = torch.mean(same_val - diff_val)
                D_loss = -alpha * w_distance + lambda_ * gradients_penalty
                self.reset_grad()
                D_loss.backward()
                self.grad_clip([self.Discriminator])
                self.D_opt.step()
                # print info
                info = {
                    f'{flag}/D_loss': D_loss.data[0],
                    f'{flag}/D_w_distance': w_distance.data[0],
                    f'{flag}/gradients_penalty': gradients_penalty.data[0],
                }
                slot_value = (j, iteration+1, iterations) + tuple([value for value in info.values()])
                print(
                    'D-%d:[%06d/%06d], D_loss=%.3f, w_distance=%.3f, gp=%.3f'
                    % slot_value,
                )
                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value, iteration + 1)
            #===================== Train G =====================#
            data = next(self.data_loader)
            c_i, c_j = [self.to_var(x, requires_grad=False) for x in data[:2]]
            X_i_t, X_i_tk, X_i_prime, X_j = \
                    [self.to_var(x).permute(0, 2, 1) for x in data[2:]]
            # encode
            E_i_t = self.Encoder(X_i_t)
            E_i_tk = self.Encoder(X_i_tk)
            E_i_prime = self.Encoder(X_i_prime)
            E_j = self.Encoder(X_j)
            # reconstruction
            X_tilde = self.Decoder(E_i_t, c_i)
            loss_rec = torch.mean(torch.abs(X_tilde - X_i_t))
            same_pair = torch.cat([E_i_t, E_i_tk], dim=1)
            diff_pair = torch.cat([E_i_prime, E_j], dim=1)
            same_val = self.Discriminator(same_pair)
            diff_val = self.Discriminator(diff_pair)
            w_distance = torch.mean(same_val - diff_val)
            G_loss = loss_rec + alpha * w_distance
            self.reset_grad()
            G_loss.backward()
            self.grad_clip([self.Encoder, self.Decoder])
            self.G_opt.step()
            info = {
                f'{flag}/loss_rec': loss_rec.data[0],
                f'{flag}/G_w_distance': w_distance.data[0],
            }
            slot_value = (iteration+1, iterations) + tuple([value for value in info.values()])
            print(
                'G:[%06d/%06d], loss_rec=%.3f, w_distance=%.3f'
                % slot_value,
            )
            for tag, value in info.items():
                self.logger.scalar_summary(tag, value, iteration + 1)
            if iteration % 100 == 0 or iteration + 1 == iterations:
                self.save_model(model_path, iteration)

if __name__ == '__main__':
    hps = Hps()
    hps.load('./hps/v4.json')
    hps_tuple = hps.get_tuple()
    dataset = myDataset('/storage/raw_feature/voice_conversion/vctk/vctk.h5',\
            '/storage/raw_feature/voice_conversion/vctk/64_513_2000k.json')
    data_loader = DataLoader(dataset)

    solver = Solver(hps_tuple, data_loader)
