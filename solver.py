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
from model import LatentDiscriminator
from model import PatchDiscriminator
from model import CBHG
import os
from utils import Hps
from utils import Logger
from utils import DataLoader
#from preprocess.tacotron import utils

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
        self.LatentDiscriminator = None
        self.PatchDiscriminator = None
        self.E_opt = None
        self.G_opt = None
        self.D_opt = None
        self.build_model()
        self.logger = Logger(log_dir)

    def build_model(self):
        ns = self.hps.ns
        self.Encoder = Encoder(ns=ns)
        self.Decoder = Decoder(ns=ns)
        self.LatentDiscriminator = LatentDiscriminator(ns=ns)
        self.PatchDiscriminator = PatchDiscriminator(ns=ns)
        if torch.cuda.is_available():
            self.Encoder.cuda()
            self.Decoder.cuda()
            self.LatentDiscriminator.cuda()
            self.PatchDiscriminator.cuda()
        betas = (0.5, 0.9)
        self.E_opt = optim.Adam(self.Encoder.parameters(), lr=self.hps.lr, betas=betas)
        self.G_opt = optim.Adam(self.Decoder.parameters(), lr=self.hps.lr, betas=betas)
        params = list(self.PatchDiscriminator.parameters()) + list(self.LatentDiscriminator.parameters())
        self.D_opt = optim.Adam(params, lr=self.hps.lr, betas=betas)

    def to_var(self, x, requires_grad=True):
        x = Variable(x, requires_grad=requires_grad)
        return x.cuda() if torch.cuda.is_available() else x

    def save_model(self, model_path, iteration, enc_only=True):
        if not enc_only:
            all_model = {
                'encoder': self.Encoder.state_dict(),
                'decoder': self.Decoder.state_dict(),
                'latent_discriminator': self.LatentDiscriminator.state_dict(),
                'patch_discriminator': self.PatchDiscriminator.state_dict(),
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
        self.PatchDiscriminator.zero_grad()
        self.LatentDiscriminator.zero_grad()

    def load_model(self, model_path, enc_only=True):
        print('load model from {}'.format(model_path))
        with open(model_path, 'rb') as f_in:
            all_model = torch.load(f_in)
            self.Encoder.load_state_dict(all_model['encoder'])
            self.Decoder.load_state_dict(all_model['decoder'])
            if not enc_only:
                self.LatentDiscriminator.load_state_dict(all_model['latent_discriminator'])
                self.PatchDiscriminator.load_state_dict(all_model['patch_discriminator'])

    def grad_clip(self, net_list):
        max_grad_norm = self.hps.max_grad_norm
        for net in net_list:
            torch.nn.utils.clip_grad_norm(net.parameters(), max_grad_norm)

    def test_step(self, X, c):
        X = self.to_var(X).permute(0, 2, 1)
        E = self.Encoder(X)
        X_tilde = self.Decoder(E, c)
        return X_tilde.data.cpu().numpy()

    def permute_data(self, data):
        C = [self.to_var(x, requires_grad=False) for x in data[:2]]
        X = [self.to_var(x).permute(0, 2, 1) for x in data[2:]]
        return C, X

    def encode_step(self, X_i_t, X_i_tk, X_i_prime, X_j):
        E_i_t = self.Encoder(X_i_t)
        E_i_tk = self.Encoder(X_i_tk)
        E_i_prime = self.Encoder(X_i_prime)
        E_j = self.Encoder(X_j)
        return E_i_t, E_i_tk, E_i_prime, E_j 

    def decode_step(self, E, c):
        X_tilde = self.Decoder(E, c)
        return X_tilde

    def latent_discriminate_step(self, E_i_t, E_i_tk, E_i_prime, E_j, cal_gp=True):
        same_pair = torch.cat([E_i_t, E_i_tk], dim=1)
        diff_pair = torch.cat([E_i_prime, E_j], dim=1)
        same_val = self.LatentDiscriminator(same_pair)
        diff_val = self.LatentDiscriminator(diff_pair)
        w_dis = torch.mean(same_val - diff_val)
        if cal_gp:
            gp = calculate_gradients_penalty(self.LatentDiscriminator, same_pair, diff_pair)
            return w_dis, gp
        else:
            return w_dis

    def patch_discriminate_step(self, X, X_tilde, cal_gp=True):
        D_real = self.PatchDiscriminator(X)
        D_fake = self.PatchDiscriminator(X_tilde)
        w_dis = torch.mean(D_real - D_fake)
        if cal_gp:
            gp = calculate_gradients_penalty(self.PatchDiscriminator, X, X_tilde)
            return w_dis, gp
        else:
            return w_dis

    def train(self, model_path, flag='train'):
        # load hyperparams
        hps = self.hps
        for iteration in range(hps.iterations):
            # calculate current alpha, beta
            if iteration + 1 < hps.scheduled_iterations:
                current_alpha = hps.alpha * (iteration + 1) / hps.scheduled_iterations
                current_beta = hps.beta * (iteration + 1) / hps.scheduled_iterations
            for j in range(hps.D_iterations):
                #===================== Train D =====================#
                data = next(self.data_loader)
                (c_i, c_j), (X_i_t, X_i_tk, X_i_prime, X_j) = self.permute_data(data)
                # encode
                E_i_t, E_i_tk, E_i_prime, E_j = self.encode_step(X_i_t, X_i_tk, X_i_prime, X_j)
                # decode 
                X_tilde = self.decode_step(E_i_t, c_i)
                # latent discriminate
                latent_w_dis, latent_gp = self.latent_discriminate_step(E_i_t, E_i_tk, E_i_prime, E_j)
                # patch discriminate
                patch_w_dis, patch_gp = self.patch_discriminate_step(X_i_t, X_tilde)
                D_loss = -current_alpha * latent_w_dis \
                         -current_beta * patch_w_dis + \
                         hps.lambda_ * (latent_gp + patch_gp)
                self.reset_grad()
                D_loss.backward()
                self.grad_clip([self.PatchDiscriminator, self.LatentDiscriminator])
                self.D_opt.step()
                # print info
                info = {
                    f'{flag}/D_loss': D_loss.data[0],
                    f'{flag}/D_latent_w_dis': latent_w_dis.data[0],
                    f'{flag}/D_patch_w_dis': patch_w_dis.data[0],
                    f'{flag}/latent_gp': latent_gp.data[0],
                    f'{flag}/patch_gp': patch_gp.data[0],
                }
                slot_value = (j, iteration+1, hps.iterations) + tuple([value for value in info.values()])
                print(
                    'D-%d:[%06d/%06d], D_loss=%.3f, latent_w_dis=%.3f, patch_w_dis=%.3f, '
                    'latent_gp=%.3f, patch_gp=%.3f'
                    % slot_value,
                )
                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value, iteration*hps.D_iterations + j)
            #===================== Train G =====================#
            data = next(self.data_loader)
            (c_i, c_j), (X_i_t, X_i_tk, X_i_prime, X_j) = self.permute_data(data)
            # encode
            E_i_t, E_i_tk, E_i_prime, E_j = self.encode_step(X_i_t, X_i_tk, X_i_prime, X_j)
            # decode E_i_t
            X_tilde = self.decode_step(E_i_t, c_i)
            loss_rec = torch.mean(torch.abs(X_tilde - X_i_t))
            # latent discriminate
            latent_w_dis = self.latent_discriminate_step(E_i_t, E_i_tk, E_i_prime, E_j, cal_gp=False)
            # patch discriminate
            patch_w_dis = self.patch_discriminate_step(X_i_t, X_tilde, cal_gp=False)
            E_loss = loss_rec + current_alpha * latent_w_dis
            self.reset_grad()
            E_loss.backward(retain_graph=True)
            self.grad_clip([self.Encoder])
            self.E_opt.step()
            # patch loss only update decoder
            G_loss = current_beta * patch_w_dis
            G_loss.backward()
            self.grad_clip([self.Decoder])
            self.G_opt.step()
            info = {
                f'{flag}/loss_rec': loss_rec.data[0],
                f'{flag}/G_latent_w_dis': latent_w_dis.data[0],
                f'{flag}/G_patch_w_dis': patch_w_dis.data[0],
                f'{flag}/alpha': current_alpha,
                f'{flag}/beta': current_beta,
            }
            slot_value = (iteration+1, hps.iterations) + tuple([value for value in info.values()])
            print(
                'G:[%06d/%06d], loss_rec=%.3f, latent_w_dis=%.3f, patch_w_dis=%.3f, alpha=%.2e, beta=%.2e'
                % slot_value,
            )
            for tag, value in info.items():
                self.logger.scalar_summary(tag, value, iteration + 1)
            if iteration % 1000 == 0 or iteration + 1 == hps.iterations:
                self.save_model(model_path, iteration)

if __name__ == '__main__':
    hps = Hps()
    hps.load('./hps/v7.json')
    hps_tuple = hps.get_tuple()
    dataset = myDataset('/storage/raw_feature/voice_conversion/vctk/vctk.h5',\
            '/storage/raw_feature/voice_conversion/vctk/64_513_2000k.json')
    data_loader = DataLoader(dataset)

    solver = Solver(hps_tuple, data_loader)
