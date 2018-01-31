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
from utils import to_var
from utils import reset_grad
from utils import grad_clip
from utils import calculate_gradients_penalty
#from preprocess.tacotron import utils

class Solver(object):
    def __init__(self, hps, data_loader, log_dir='./log/'):
        self.hps = hps
        self.data_loader = data_loader
        self.model_kept = []
        self.max_keep = 10
        self.build_model()
        self.logger = Logger(log_dir)

    def build_model(self):
        hps = self.hps
        ns = self.hps.ns
        emb_size = self.hps.emb_size
        self.Encoder = Encoder(ns=ns)
        self.Decoder = Decoder(ns=ns, c_a=hps.n_speakers, emb_size=emb_size)
        self.Generator = Decoder(ns=ns, c_a=hps.n_speakers, emb_size=emb_size)
        self.LatentDiscriminator = LatentDiscriminator(ns=ns)
        self.PatchDiscriminator = PatchDiscriminator(ns=ns, n_class=hps.n_speakers)
        if torch.cuda.is_available():
            self.Encoder.cuda()
            self.Decoder.cuda()
            self.Generator.cuda()
            self.LatentDiscriminator.cuda()
            self.PatchDiscriminator.cuda()
        betas = (0.5, 0.9)
        params = list(self.Encoder.parameters()) + list(self.Decoder.parameters())
        self.ae_opt = optim.Adam(params, lr=self.hps.lr, betas=betas)
        self.gen_opt = optim.Adam(self.Generator.parameters(), lr=self.hps.lr, betas=betas)
        self.lat_opt = optim.Adam(self.LatentDiscriminator.parameters(), lr=self.hps.lr, betas=betas)
        self.patch_opt = optim.Adam(self.PatchDiscriminator.parameters(), lr=self.hps.lr, betas=betas)

    def save_model(self, model_path, iteration, enc_only=True):
        if not enc_only:
            all_model = {
                'encoder': self.Encoder.state_dict(),
                'decoder': self.Decoder.state_dict(),
                'generator': self.Generator.state_dict(),
                'latent_discriminator': self.LatentDiscriminator.state_dict(),
                'patch_discriminator': self.PatchDiscriminator.state_dict(),
            }
        else:
            all_model = {
                'encoder': self.Encoder.state_dict(),
                'decoder': self.Decoder.state_dict(),
                'generator': self.Generator.state_dict(),
            }
        new_model_path = '{}-{}'.format(model_path, iteration)
        with open(new_model_path, 'wb') as f_out:
            torch.save(all_model, f_out)
        self.model_kept.append(new_model_path)

        if len(self.model_kept) >= self.max_keep:
            os.remove(self.model_kept[0])
            self.model_kept.pop(0)

    def load_model(self, model_path, enc_only=True):
        print('load model from {}'.format(model_path))
        with open(model_path, 'rb') as f_in:
            all_model = torch.load(f_in)
            self.Encoder.load_state_dict(all_model['encoder'])
            self.Decoder.load_state_dict(all_model['decoder'])
            #self.Genrator.load_state_dict(all_model['generator'])
            if not enc_only:
                self.LatentDiscriminator.load_state_dict(all_model['latent_discriminator'])
                self.PatchDiscriminator.load_state_dict(all_model['patch_discriminator'])

    def set_eval(self):
        self.Encoder.eval()
        self.Decoder.eval()
        self.Generator.eval()
        #self.LatentDiscriminator.eval()

    def test_step(self, x, c):
        self.set_eval()
        x = to_var(x).permute(0, 2, 1)
        enc = self.Encoder(x)
        x_tilde = self.Decoder(enc, c)
        return x_tilde.data.cpu().numpy()

    def permute_data(self, data):
        C = [to_var(c, requires_grad=False) for c in data[:2]]
        X = [to_var(x).permute(0, 2, 1) for x in data[2:]]
        return C, X

    def sample_c(self, size):
        c_sample = Variable(
                torch.multinomial(torch.ones(8), num_samples=size, replacement=True),  
                requires_grad=False)
        c_sample = c_sample.cuda() if torch.cuda.is_available() else c_sample
        return c_sample

    def cal_acc(self, logits, y_true):
        _, ind = torch.max(logits, dim=1)
        acc = torch.sum((ind == y_true).type(torch.FloatTensor)) / y_true.size(0)
        return acc

    def encode_step(self, *args):
        enc_list = []
        for x in args:
            enc = self.Encoder(x)
            enc_list.append(enc)
        return tuple(enc_list)

    def decode_step(self, enc, c):
        x_tilde = self.Decoder(enc, c)
        return x_tilde

    def latent_discriminate_step(self, enc_i_t, enc_i_tk, enc_i_prime, enc_j, cal_gp=True):
        same_pair = torch.cat([enc_i_t, enc_i_tk], dim=1)
        diff_pair = torch.cat([enc_i_prime, enc_j], dim=1)
        same_val = self.LatentDiscriminator(same_pair)
        diff_val = self.LatentDiscriminator(diff_pair)
        w_dis = torch.mean(same_val - diff_val)
        if cal_gp:
            gp = calculate_gradients_penalty(self.LatentDiscriminator, same_pair, diff_pair)
            return w_dis, gp
        else:
            return (w_dis,)

    def patch_discriminate_step(self, x, x_tilde, cal_gp=True):
        # w-distance
        D_real, real_logits = self.PatchDiscriminator(x, classify=True)
        D_fake, fake_logits = self.PatchDiscriminator(x_tilde, classify=True)
        w_dis = torch.mean(D_real - D_fake)
        if cal_gp:
            gp = calculate_gradients_penalty(self.PatchDiscriminator, x, x_tilde)
            return w_dis, real_logits, fake_logits, gp
        else:
            return w_dis, real_logits, fake_logits
    # backup
    #def classify():
    #    # aux clssify loss 
    #    criterion = nn.NLLLoss()
    #    c_loss = criterion(real_logits, c) + criterion(fake_logits, c_sample)
    #    real_acc = self.cal_acc(real_logits, c)
    #    fake_acc = self.cal_acc(fake_logits, c_sample)

    def train(self, model_path, flag='train'):
        # load hyperparams
        hps = self.hps
        for iteration in range(hps.iters):
            # calculate current alpha
            if iteration + 1 < hps.lat_sched_iters and iteration >= hps.pretrain_iters:
                current_alpha = hps.alpha_enc * (iteration + 1) / (hps.lat_sched_iters - hps.pretrain_iters)
            else:
                current_alpha = 0
            if iteration >= hps.pretrain_iters:
                for step in range(hps.n_latent_steps):
                    #===================== Train latent discriminator =====================#
                    data = next(self.data_loader)
                    (c_i, c_j), (x_i_t, x_i_tk, x_i_prime, x_j) = self.permute_data(data)
                    # encode
                    enc_i_t, enc_i_tk, enc_i_prime, enc_j = self.encode_step(x_i_t, x_i_tk, x_i_prime, x_j)
                    # latent discriminate
                    latent_w_dis, latent_gp = self.latent_discriminate_step(enc_i_t, enc_i_tk, enc_i_prime, enc_j)
                    lat_loss = -hps.alpha_dis * latent_w_dis + hps.lambda_ * latent_gp
                    reset_grad([self.LatentDiscriminator])
                    lat_loss.backward()
                    grad_clip([self.LatentDiscriminator], self.hps.max_grad_norm)
                    self.lat_opt.step()
                    # print info
                    info = {
                        f'{flag}/D_latent_w_dis': latent_w_dis.data[0],
                        f'{flag}/latent_gp': latent_gp.data[0], 
                    }
                    slot_value = (step, iteration + 1, hps.iters) + \
                            tuple([value for value in info.values()])
                    log = 'lat_D-%d:[%06d/%06d], w_dis=%.3f, gp=%.2f'
                    print(log % slot_value)
                    for tag, value in info.items():
                        self.logger.scalar_summary(tag, value, iteration)
            # two stage training
            if iteration >= hps.patch_start_iter:
                for step in range(hps.n_patch_steps):
                    #===================== Train patch discriminator =====================#
                    data = next(self.data_loader)
                    (c_i, _), (x_i_t, _, _, _) = self.permute_data(data)
                    # encode
                    enc_i_t, = self.encode_step(x_i_t)
                    c_sample = self.sample_c(x_i_t.size(0))
                    x_tilde = self.decode_step(enc_i_t, c_i)
                    # Aux classify loss
                    patch_w_dis, real_logits, fake_logits, patch_gp = \
                            self.patch_discriminate_step(x_i_t, x_tilde, cal_gp=True)
                    patch_loss = -hps.beta_dis * patch_w_dis + hps.lambda_ * patch_gp + hps.beta_clf * c_loss
                    reset_grad([self.PatchDiscriminator])
                    patch_loss.backward()
                    grad_clip([self.PatchDiscriminator], self.hps.max_grad_norm)
                    self.patch_opt.step()
                    # print info
                    info = {
                        f'{flag}/D_patch_w_dis': patch_w_dis.data[0],
                        f'{flag}/patch_gp': patch_gp.data[0],
                        f'{flag}/c_loss': c_loss.data[0],
                        f'{flag}/real_acc': real_acc,
                        f'{flag}/fake_acc': fake_acc,
                    }
                    slot_value = (step, iteration + 1, hps.iters) + \
                            tuple([value for value in info.values()])
                    log = 'patch_D-%d:[%06d/%06d], w_dis=%.3f, gp=%.2f, c_loss=%.3f, real_acc=%.2f, fake_acc=%.2f'
                    print(log % slot_value)
                    for tag, value in info.items():
                        self.logger.scalar_summary(tag, value, iteration)
            #===================== Train G =====================#
            data = next(self.data_loader)
            (c_i, c_j), (x_i_t, x_i_tk, x_i_prime, x_j) = self.permute_data(data)
            # encode
            enc_i_t, enc_i_tk, enc_i_prime, enc_j = self.encode_step(x_i_t, x_i_tk, x_i_prime, x_j)
            # decode
            x_tilde = self.decode_step(enc_i_t, c_i)
            loss_rec = torch.mean(torch.abs(x_tilde - x_i_t))
            # latent discriminate
            latent_w_dis, = self.latent_discriminate_step(
                    enc_i_t, enc_i_tk, enc_i_prime, enc_j, cal_gp=False)
            ae_loss = loss_rec + current_alpha * latent_w_dis
            reset_grad([self.Encoder, self.Decoder])
            retain_graph = True if hps.n_patch_steps > 0 else False
            ae_loss.backward(retain_graph=retain_graph)
            grad_clip([self.Encoder, self.Decoder], self.hps.max_grad_norm)
            self.ae_opt.step()
            info = {
                f'{flag}/loss_rec': loss_rec.data[0],
                f'{flag}/G_latent_w_dis': latent_w_dis.data[0],
                f'{flag}/alpha': current_alpha,
            }
            slot_value = (iteration+1, hps.iters) + tuple([value for value in info.values()])
            log = 'G:[%06d/%06d], loss_rec=%.2f, latent_w_dis=%.2f, alpha=%.2e'
            print(log % slot_value)
            for tag, value in info.items():
                self.logger.scalar_summary(tag, value, iteration + 1)
            # patch discriminate
            if hps.n_patch_steps > 0 and iteration >= hps.patch_start_iter:
                c_sample = self.sample_c(x_i_t.size(0))
                x_tilde = self.decode_step(enc_i_t, c_sample)
                patch_w_dis, real_logits, fake_logits = \
                        self.patch_discriminate_step(x_i_t, x_tilde, cal_gp=False)
                patch_loss = hps.beta_dec * patch_w_dis + hps.beta_clf * c_loss
                reset_grad([self.Decoder])
                patch_loss.backward()
                grad_clip([self.Decoder], self.hps.max_grad_norm)
                self.decoder_opt.step()
                info = {
                    f'{flag}/G_patch_w_dis': patch_w_dis.data[0],
                    f'{flag}/c_loss': c_loss.data[0],
                    f'{flag}/real_acc': real_acc,
                    f'{flag}/fake_acc': fake_acc,
                }
                slot_value = (iteration+1, hps.iters) + tuple([value for value in info.values()])
                log = 'G:[%06d/%06d]: patch_w_dis=%.2f, c_loss=%.2f, real_acc=%.2f, fake_acc=%.2f'
                print(log % slot_value)
                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value, iteration + 1)
            if iteration % 1000 == 0 or iteration + 1 == hps.iters:
                self.save_model(model_path, iteration)

if __name__ == '__main__':
    hps = Hps()
    hps.load('./hps/v7.json')
    hps_tuple = hps.get_tuple()
    dataset = myDataset('/storage/raw_feature/voice_conversion/vctk/vctk.h5',\
            '/storage/raw_feature/voice_conversion/vctk/64_513_2000k.json')
    data_loader = DataLoader(dataset)

    solver = Solver(hps_tuple, data_loader)
