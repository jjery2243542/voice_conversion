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
#from model import LatentDiscriminator
#from model import PatchDiscriminator
from model import CBHG
import os
from utils import Hps
from utils import Logger
from utils import DataLoader
from utils import to_var
from utils import reset_grad
from utils import grad_clip
from utils import cal_acc
#from utils import calculate_gradients_penalty
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
        self.Encoder = Encoder(ns=ns, dp=hps.enc_dp)
        self.Decoder = Decoder(ns=ns, c_a=hps.n_speakers, emb_size=emb_size)
        #self.Generator = Decoder(ns=ns, c_a=hps.n_speakers, emb_size=emb_size)
        #self.LatentDiscriminator = LatentDiscriminator(ns=ns, dp=hps.dis_dp)
        self.SpeakerClassifier = SpeakerClassifier(ns=ns, n_class=hps.n_speakers, dp=hps.dis_dp) 
        #self.PatchDiscriminator = PatchDiscriminator(ns=ns, n_class=hps.n_speakers)
        if torch.cuda.is_available():
            self.Encoder.cuda()
            self.Decoder.cuda()
            self.SpeakerClassifier.cuda()
        betas = (0.5, 0.9)
        params = list(self.Encoder.parameters()) + list(self.Decoder.parameters()) + list(self.SpeakerClassifier.parameters())
        self.opt = optim.Adam(params, lr=self.hps.lr, betas=betas)

    def save_model(self, model_path, iteration, enc_only=True):
        if not enc_only:
            all_model = {
                'encoder': self.Encoder.state_dict(),
                'decoder': self.Decoder.state_dict(),
                'classifier': self.SpeakerClassifier.state_dict(),
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
            if not enc_only:
                self.SpeakerClassifier.load_state_dict(all_model['classifier'])

    def set_eval(self):
        self.Encoder.eval()
        self.Decoder.eval()
        self.SpeakerClassifier.eval()

    def test_step(self, x, c):
        self.set_eval()
        x = to_var(x).permute(0, 2, 1)
        enc = self.Encoder(x)
        x_tilde = self.Decoder(enc, c)
        return x_tilde.data.cpu().numpy()

    def permute_data(self, data):
        C = to_var(data[0], requires_grad=False)
        X = to_var(data[2]).permute(0, 2, 1)
        return C, X

    def sample_c(self, size):
        c_sample = Variable(
                torch.multinomial(torch.ones(8), num_samples=size, replacement=True),  
                requires_grad=False)
        c_sample = c_sample.cuda() if torch.cuda.is_available() else c_sample
        return c_sample

    def encode_step(self, x):
        enc = self.Encoder(x)
        return enc

    def decode_step(self, enc, c):
        x_tilde = self.Decoder(enc, c)
        return x_tilde

    def clf_step(self, enc, _lambda=0.0001, gr=True):
        logits = self.SpeakerClassifier(enc, _lambda=_lambda, gr=gr)
        return logits

    def cal_loss(self, logits, y_true):
        # calculate loss 
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, y_true)
        return loss

    def train(self, model_path, flag='train'):
        # load hyperparams
        hps = self.hps
        for iteration in range(hps.iters):
            # calculate current alpha
            if iteration < hps.enc_pretrain_iters:
                current_alpha = 0
            elif iteration < hps.enc_pretrain_iters + hps.lat_sched_iters:
                current_alpha = hps.alpha_enc * (iteration - hps.enc_pretrain_iters) / hps.lat_sched_iters
            else:
                current_alpha = hps.alpha_enc
            data = next(self.data_loader)
            c, x = self.permute_data(data)
            # encode
            enc = self.encode_step(x)
            reset_grad([self.Encoder, self.Decoder, self.SpeakerClassifier])
            if iteration >= hps.enc_pretrain_iters:
                # classify speaker
                logits = self.clf_step(enc, current_alpha, gr=True)
                loss_clf = self.cal_loss(logits, c)
                # update 
                loss_clf.backward(retain_graph=True)
                acc = cal_acc(logits, c)
            # decode
            x_tilde = self.decode_step(enc, c)
            loss_rec = torch.mean(torch.abs(x_tilde - x))
            loss_rec.backward()
            grad_clip([self.Encoder, self.Decoder, self.SpeakerClassifier], self.hps.max_grad_norm)
            self.opt.step()
            info = {
                f'{flag}/loss_rec': loss_rec.data[0],
                f'{flag}/loss_clf': loss_clf.data[0] if iteration >= hps.enc_pretrain_iters else 0,
                f'{flag}/alpha': current_alpha,
                f'{flag}/acc': acc if iteration >= hps.enc_pretrain_iters else 0,
            }
            slot_value = (iteration+1, hps.iters) + tuple([value for value in info.values()])
            log = 'G:[%06d/%06d], loss_rec=%.2f, loss_clf=%.2f, alpha=%.2e, acc=%.2f'
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
