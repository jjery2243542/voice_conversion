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
from model import SpeakerClassifier
from model import SpectrogramClassifier
#from model import WeakSpeakerClassifier
#from model import LatentDiscriminator
#from model import PatchDiscriminator
from model import CBHG
import os
from utils import Hps
from utils import Logger
from utils import DataLoader
from utils import to_var
from utils import reset_grad
from utils import multiply_grad
from utils import grad_clip
from utils import cal_acc
from utils import cc
from utils import calculate_gradients_penalty
#from preprocess.tacotron import utils

class Solver(object):
    def __init__(self, hps, data_loader, log_dir='./log/'):
        self.hps = hps
        self.data_loader = data_loader
        self.model_kept = []
        self.max_keep = 20
        self.build_model()
        self.logger = Logger(log_dir)

    def build_model(self):
        hps = self.hps
        ns = self.hps.ns
        emb_size = self.hps.emb_size
        self.Encoder = cc(Encoder(ns=ns, dp=hps.enc_dp))
        self.Decoder = cc(Decoder(ns=ns, c_a=hps.n_speakers, emb_size=emb_size))
        self.SpeakerClassifier = cc(SpeakerClassifier(ns=ns, n_class=hps.n_speakers, dp=hps.dis_dp))
        self.SpectrogramClassifier = cc(SpectrogramClassifier(ns=ns, n_class=hps.n_speakers, dp=hps.dis_dp))
        betas = (0.5, 0.9)
        params = list(self.Encoder.parameters()) + list(self.Decoder.parameters())
        self.ae_opt = optim.Adam(params, lr=self.hps.lr, betas=betas)
        self.clf_opt = optim.Adam(self.SpeakerClassifier.parameters(), lr=self.hps.lr, betas=betas)
        self.spec_clf_opt = optim.Adam(self.SpectrogramClassifier.parameters(), lr=self.hps.lr, betas=betas)

    def save_model(self, model_path, iteration, enc_only=True):
        if not enc_only:
            all_model = {
                'encoder': self.Encoder.state_dict(),
                'decoder': self.Decoder.state_dict(),
                'classifier': self.SpeakerClassifier.state_dict(),
                'spectrogram_classifier': self.SpectrogramClassifier.state_dict(),
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

    def load_model(self, model_path, enc_only=True):
        print('load model from {}'.format(model_path))
        with open(model_path, 'rb') as f_in:
            all_model = torch.load(f_in)
            self.Encoder.load_state_dict(all_model['encoder'])
            self.Decoder.load_state_dict(all_model['decoder'])
            #self.Generator.load_state_dict(all_model['generator'])
            if not enc_only:
                self.SpeakerClassifier.load_state_dict(all_model['classifier'])
                #self.PatchDiscriminator.load_state_dict(all_model['patch_discriminator'])

    def set_eval(self):
        self.Encoder.eval()
        self.Decoder.eval()
        self.SpeakerClassifier.eval()
        self.SpectrogramClassifier.eval()

    def test_step(self, x, c, gen=False):
        self.set_eval()
        x = to_var(x).permute(0, 2, 1)
        enc = self.Encoder(x)
        x_tilde = self.Decoder(enc, c)
        return x_tilde.data.cpu().numpy()

    def permute_data(self, data):
        C = to_var(data[0], requires_grad=False)
        X = to_var(data[1]).permute(0, 2, 1)
        return C, X

    def sample_c(self, size):
        n_speakers = self.hps.n_speakers
        c_sample = Variable(
                torch.multinomial(torch.ones(n_speakers), num_samples=size, replacement=True),  
                requires_grad=False)
        c_sample = c_sample.cuda() if torch.cuda.is_available() else c_sample
        return c_sample

    def encode_step(self, x):
        enc = self.Encoder(x)
        return enc

    def decode_step(self, enc, c):
        x_tilde = self.Decoder(enc, c)
        return x_tilde

    def patch_step(self, x, x_tilde, is_dis=True):
        D_real, real_logits = self.PatchDiscriminator(x, classify=True)
        D_fake, fake_logits = self.PatchDiscriminator(x_tilde, classify=True)
        if is_dis:
            w_dis = torch.mean(D_real - D_fake)
            gp = calculate_gradients_penalty(self.PatchDiscriminator, x, x_tilde)
            return w_dis, real_logits, gp
        else:
            mean_D_fake = torch.mean(D_fake)
            return mean_D_fake, fake_logits

    def spec_clf_step(self, x):
        logits = self.SpectrogramClassifier(x)
        return logits

    def gen_step(self, enc, c):
        x_gen = self.Decoder(enc, c) + self.Generator(enc, c)
        #x_gen = self.Generator(enc, c)
        return x_gen 

    def clf_step(self, enc):
        logits = self.SpeakerClassifier(enc)
        return logits

    def cal_loss(self, logits, y_true):
        # calculate loss 
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, y_true)
        return loss

    def train(self, model_path, flag='train', mode='train'):
        # load hyperparams
        hps = self.hps
        if mode == 'pretrain_G':
            for iteration in range(hps.enc_pretrain_iters):
                data = next(self.data_loader)
                c, x = self.permute_data(data)
                # encode
                enc = self.encode_step(x)
                x_tilde = self.decode_step(enc, c)
                loss_rec = torch.mean(torch.abs(x_tilde - x))
                reset_grad([self.Encoder, self.Decoder])
                loss_rec.backward()
                grad_clip([self.Encoder, self.Decoder], self.hps.max_grad_norm)
                self.ae_opt.step()
                # tb info
                info = {
                    f'{flag}/pre_loss_rec': loss_rec.data[0],
                }
                slot_value = (iteration + 1, hps.enc_pretrain_iters) + tuple([value for value in info.values()])
                log = 'pre_G:[%06d/%06d], loss_rec=%.2f'
                print(log % slot_value)
                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value, iteration + 1)
        elif mode == 'pretrain_D':
            for iteration in range(hps.dis_pretrain_iters):
                data = next(self.data_loader)
                c, x = self.permute_data(data)
                # encode
                enc = self.encode_step(x)
                # classify speaker
                logits = self.clf_step(enc)
                loss_clf = self.cal_loss(logits, c)
                # update 
                reset_grad([self.SpeakerClassifier])
                loss_clf.backward()
                grad_clip([self.SpeakerClassifier], self.hps.max_grad_norm)
                self.clf_opt.step()
                # calculate acc
                acc = cal_acc(logits, c)
                info = {
                    f'{flag}/pre_loss_clf': loss_clf.data[0],
                    f'{flag}/pre_acc': acc,
                }
                slot_value = (iteration + 1, hps.dis_pretrain_iters) + tuple([value for value in info.values()])
                log = 'pre_D:[%06d/%06d], loss_clf=%.2f, acc=%.2f'
                print(log % slot_value)
                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value, iteration + 1)
        elif mode == 'pretrain_clf':
            for iteration in range(hps.patch_iters):
                data = next(self.data_loader)
                c, x = self.permute_data(data)
                # classify speaker 
                logits = self.spec_clf_step(x)
                loss = self.cal_loss(logits, c)
                # update 
                reset_grad([self.SpectrogramClassifier])
                loss.backward()
                grad_clip([self.SpectrogramClassifier], self.hps.max_grad_norm)
                self.spec_clf_opt.step()
                # calculate acc
                acc = cal_acc(logits, c)
                info = {
                    f'{flag}/spec_loss_clf': loss.data[0],
                    f'{flag}/spec_acc': acc,
                }
                slot_value = (iteration + 1, hps.patch_iters) + tuple([value for value in info.values()])
                log = 'pre_clf:[%06d/%06d], loss_clf=%.2f, acc=%.2f'
                print(log % slot_value)
                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value, iteration + 1)
        elif mode == 'train':
            for iteration in range(hps.iters):
                # calculate current alpha
                if iteration < hps.lat_sched_iters:
                    current_alpha = hps.alpha_enc * (iteration / hps.lat_sched_iters)
                    current_beta = hps.beta_clf * (iteration / hps.lat_sched_iters)
                else:
                    current_alpha, current_beta = hps.alpha_enc, hps.beta_clf
                #==================train D==================#
                for step in range(hps.n_latent_steps):
                    data = next(self.data_loader)
                    c, x = self.permute_data(data)
                    # encode
                    enc = self.encode_step(x)
                    # classify speaker
                    logits = self.clf_step(enc)
                    loss_clf = self.cal_loss(logits, c)
                    loss = hps.alpha_dis * loss_clf
                    # update 
                    reset_grad([self.SpeakerClassifier])
                    loss.backward()
                    grad_clip([self.SpeakerClassifier], self.hps.max_grad_norm)
                    self.clf_opt.step()
                    # calculate acc
                    acc = cal_acc(logits, c)
                    info = {
                        f'{flag}/D_loss_clf': loss_clf.data[0],
                        f'{flag}/D_acc': acc,
                    }
                    slot_value = (step, iteration + 1, hps.iters) + tuple([value for value in info.values()])
                    log = 'D-%d:[%06d/%06d], loss_clf=%.2f, acc=%.2f'
                    print(log % slot_value)
                    for tag, value in info.items():
                        self.logger.scalar_summary(tag, value, iteration + 1)
                #==================train G==================#
                data = next(self.data_loader)
                c, x = self.permute_data(data)
                # encode
                enc = self.encode_step(x)
                # decode
                x_tilde = self.decode_step(enc, c)
                loss_rec = torch.mean(torch.abs(x_tilde - x))
                # permute speakers
                enc_detached = enc.detach()
                c_prime = self.sample_c(enc.size(0))
                x_prime = self.decode_step(enc_detached, c_prime)
                permuted_logits = self.SpectrogramClassifier(x_prime)
                loss_spec_clf = self.cal_loss(permuted_logits, c_prime)
                spec_acc = cal_acc(permuted_logits, c_prime)
                # classify speaker
                logits = self.clf_step(enc)
                acc = cal_acc(logits, c)
                loss_clf = self.cal_loss(logits, c)
                # maximize classification loss
                loss = loss_rec + current_beta * loss_spec_clf - current_alpha * loss_clf
                reset_grad([self.Encoder, self.Decoder])
                loss.backward()
                grad_clip([self.Encoder, self.Decoder], self.hps.max_grad_norm)
                self.ae_opt.step()
                info = {
                    f'{flag}/loss_rec': loss_rec.data[0],
                    f'{flag}/G_loss_clf': loss_clf.data[0],
                    f'{flag}/spec_loss_clf': loss_spec_clf.data[0],
                    f'{flag}/alpha': current_alpha,
                    f'{flag}/beta': current_beta,
                    f'{flag}/G_acc': acc,
                    f'{flag}/spec_acc': spec_acc,
                }
                slot_value = (iteration + 1, hps.iters) + tuple([value for value in info.values()])
                log = 'G:[%06d/%06d], loss_rec=%.2f, loss_clf=%.2f, loss_clf2=%.2f, alpha=%.2e, beta=%.2f, ' \
                    'acc=%.2f, acc2=%.2f'
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
