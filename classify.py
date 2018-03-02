import torch
from torch import optim
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
import numpy as np
import pickle
from utils import myDataset
from utils import SingleDataset
from model import SpeakerClassifier
from model import Encoder
import os
from utils import Hps
from utils import Logger
from utils import DataLoader
from utils import to_var
from utils import reset_grad
from utils import grad_clip
from utils import calculate_gradients_penalty
from utils import cal_acc
import argparse
#from preprocess.tacotron import utils

class Classifier(object):
    def __init__(self, hps, data_loader, valid_data_loader, log_dir='./log/'):
        self.hps = hps
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.model_kept = []
        self.max_keep = 10
        self.build_model()
        self.logger = Logger(log_dir)

    def build_model(self):
        hps = self.hps
        self.SpeakerClassifier = SpeakerClassifier(ns=hps.ns, dp=hps.dis_dp, n_class=hps.n_speakers)
        self.Encoder = Encoder(ns=hps.ns, dp=hps.enc_dp)
        if torch.cuda.is_available():
            self.SpeakerClassifier.cuda()
            self.Encoder.cuda()
        betas = (0.5, 0.9)
        self.opt = optim.Adam(self.SpeakerClassifier.parameters(), lr=self.hps.lr, betas=betas)

    def load_encoder(self, model_path):
        print('load model from {}'.format(model_path))
        with open(model_path, 'rb') as f_in:
            all_model = torch.load(f_in)
            self.Encoder.load_state_dict(all_model['encoder'])

    def save_model(self, model_path, iteration):
        new_model_path = '{}-{}'.format(model_path, iteration)
        torch.save(self.SpeakerClassifier.state_dict(), new_model_path)
        self.model_kept.append(new_model_path)
        if len(self.model_kept) >= self.max_keep:
            os.remove(self.model_kept[0])
            self.model_kept.pop(0)

    def load_model(self, model_path):
        print('load model from {}'.format(model_path))
        self.SpeakerClassifier.load_state_dict(torch.load(model_path))

    def set_eval(self):
        self.SpeakerClassifier.eval()

    def set_train(self):
        self.SpeakerClassifier.train()

    def permute_data(self, data):
        C = to_var(data[0], requires_grad=False)
        X = to_var(data[2]).permute(0, 2, 1)
        return C, X

    def encode_step(self, x):
        enc = self.Encoder(x)
        return enc

    def forward_step(self, enc):
        logits = self.SpeakerClassifier(enc)
        return logits

    def cal_loss(self, logits, y_true):
        # calculate loss 
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, y_true)
        return loss

    def valid(self, n_batches=10):
        # input: valid data, output: (loss, acc)
        total_loss, total_acc = 0., 0.
        self.set_eval()
        for i in range(n_batches):
            data = next(self.valid_data_loader)
            y, x = self.permute_data(data)
            enc = self.Encoder(x)
            logits = self.SpeakerClassifier(enc)
            loss = self.cal_loss(logits, y)
            acc = cal_acc(logits, y)
            total_loss += loss.data[0]
            total_acc += acc  
        self.set_train()
        return total_loss / n_batches, total_acc / n_batches

    def train(self, model_path, flag='train'):
        # load hyperparams
        hps = self.hps
        for iteration in range(hps.iters):
            data = next(self.data_loader)
            y, x = self.permute_data(data)
            # encode
            enc = self.encode_step(x)
            # forward to classifier
            logits = self.forward_step(enc)
            # calculate loss
            loss = self.cal_loss(logits, y)
            # optimize
            reset_grad([self.SpeakerClassifier])
            loss.backward()
            grad_clip([self.SpeakerClassifier], self.hps.max_grad_norm)
            self.opt.step()
            # calculate acc
            acc = cal_acc(logits, y)
            # print info
            info = {
                f'{flag}/loss': loss.data[0], 
                f'{flag}/acc': acc,
            }
            slot_value = (iteration + 1, hps.iters) + tuple([value for value in info.values()])
            log = 'iter:[%06d/%06d], loss=%.3f, acc=%.3f'
            print(log % slot_value, end='\r')
            for tag, value in info.items():
                self.logger.scalar_summary(tag, value, iteration)
            if iteration % 1000 == 0 or iteration + 1 == hps.iters:
                valid_loss, valid_acc = self.valid(n_batches=10)
                # print info
                info = {
                    f'{flag}/valid_loss': valid_loss, 
                    f'{flag}/valid_acc': valid_acc,
                }
                slot_value = (iteration + 1, hps.iters) + \
                        tuple([value for value in info.values()])
                log = 'iter:[%06d/%06d], valid_loss=%.3f, valid_acc=%.3f'
                print(log % slot_value)
                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value, iteration)
                self.save_model(model_path, iteration)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=True, action='store_true')
    parser.add_argument('--load_model', default=False, action='store_true')
    parser.add_argument('-flag', default='train')
    parser.add_argument('-hps_path', default='./hps/clf.json')
    parser.add_argument('-load_model_path', default='/storage/model/voice_conversion/'
            'pretrain_model.pkl-19999')
    parser.add_argument('-encoder_model_path', default='/storage/model/voice_conversion/v20/1000_model.pkl')
    parser.add_argument('-dataset_path', default='/storage/feature/voice_conversion/vcc/trim_log.pkl')
    parser.add_argument('-train_index_path', \
            default='/storage/feature/voice_conversion/vctk/128_513_2000k.json')
    parser.add_argument('-valid_index_path', \
            default='/storage/feature/voice_conversion/vctk/128_513_2000k.json')
    parser.add_argument('-output_model_path', default='/storage/model/voice_conversion/model.pkl')
    args = parser.parse_args()
    hps = Hps()
    hps.load(args.hps_path)
    hps_tuple = hps.get_tuple()
    train_dataset = SingleDataset(args.dataset_path,
            args.train_index_path,
            seg_len=hps_tuple.seg_len)
    valid_dataset = SingleDataset(args.dataset_path,
            args.valid_index_path,
            dset='test',
            seg_len=hps_tuple.seg_len)
    data_loader = DataLoader(train_dataset)
    valid_data_loader = DataLoader(valid_dataset, batch_size=100)
    classifier = Classifier(hps_tuple, data_loader, valid_data_loader)
    classifier.load_encoder(args.encoder_model_path)
    if args.train:
        classifier.train(args.output_model_path, args.flag)
