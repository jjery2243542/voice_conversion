import torch
from torch import optim
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
import numpy as np
import pickle
from model import Encoder
from model import Decoder
from model import Discriminator
from utils import Hps
from utils import DataLoader
from utils import Logger
from postprocess.utils import ispecgram
from scipy.io import wavfile
import os 

def cal_mean_grad(net):
    grad = Variable(torch.FloatTensor([0])).cuda()
    for i, p in enumerate(net.parameters()):
        grad += torch.mean(p.grad)
    return grad.data[0] / (i + 1)

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

    def test(self, specs_A, specs_B, wav_filename='test.wav', rate=16000):
        # extract speaker_A's representation
        Es_A = self.Encoder_s(specs_A)
        avg_Es_A = torch.mean(Es_A, dim=0, keepdim=True)
        # extract speaker_B's content
        Ec_B = self.Encoder_c(specs_B)
        # combine speaker_A and content_B
        torch.cat(avg_Es_A, )
        # transpose and reshape
        result_specs = np.transpose(result_specs, [1, 0, 2, 3])
        result_specs = np.reshape([257, -1, 1])
        print('spec shape={}'.format(result_specs.shape))
        wav_arr = ispecgram(result_specs)
        wavfile.write(wav_filename, rate=rate, data=wav_arr)

    def build_model(self):
        self.Encoder_s = Encoder(1, 1)
        self.Encoder_c = Encoder(1, 1)
        self.Decoder = Decoder(2, 1)
        self.Discriminator = Discriminator(2)
        if torch.cuda.is_available():
            self.Encoder_s.cuda()
            self.Encoder_c.cuda()
            self.Decoder.cuda()
            self.Discriminator.cuda()
        params = list(self.Encoder_s.parameters()) \
            + list(self.Encoder_c.parameters())\
            + list(self.Decoder.parameters())
        self.G_opt = optim.Adagrad(params, lr=self.hps.lr)
        self.D_opt = optim.Adagrad(self.Discriminator.parameters(), lr=self.hps.lr)

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
            torch.save(all_model, f_out)
        self.model_kept.append(new_model_path)

        if len(self.model_kept) >= self.max_keep:
            os.remove(self.model_kept[0])
            self.model_kept.pop(0)

    def reset_grad(self):
        self.Encoder_s.zero_grad()
        self.Encoder_c.zero_grad()
        self.Decoder.zero_grad()
        self.Discriminator.zero_grad()

    def load_model(self, model_path, enc_only=True):
        print('load model from {}'.format(model_path))
        with open(model_path, 'rb') as f_in:
            all_model = torch.load(f_in)
            self.Encoder_s.load_state_dict(all_model['encoder_s'])
            self.Encoder_c.load_state_dict(all_model['encoder_c'])
            self.Decoder.load_state_dict(all_model['decoder'])
            if not enc_only:
                self.Discriminator.load_state_dict(all_model['discriminator'])

    def grad_clip(self, net_list):
        max_grad_norm = self.hps.max_grad_norm
        for net in net_list:
            torch.nn.utils.clip_grad_norm(net.parameters(), max_grad_norm)

    def train(self, model_path, is_pretrain=False):
        batch_size = self.hps.batch_size
        pretrain_iterations, iterations = self.hps.pretrain_iterations, self.hps.iterations
        g_iterations = self.hps.g_iterations
        max_grad_norm = self.hps.max_grad_norm
        alpha, beta1, beta2, beta3 = self.hps.alpha, self.hps.beta1, self.hps.beta2, self.hps.beta3
        margin = self.hps.margin
        for iteration in range(pretrain_iterations if is_pretrain else iterations):
            if not is_pretrain:
                #===================== Train D =====================#
                X_i_t, X_i_tk, _, X_j = [self.to_var(x) for x in next(self.data_loader)]
                # encode
                Ec_i_t, Ec_i_tk, Ec_j = self.Encoder_c(X_i_t), self.Encoder_c(X_i_tk), self.Encoder_c(X_j)
                same_prob = self.Discriminator(Ec_i_t, Ec_i_tk)
                diff_prob = self.Discriminator(Ec_i_t, Ec_j)
                # train discriminator
                loss_same = -torch.mean(torch.log(same_prob))
                loss_diff = -torch.mean(torch.log(1 - diff_prob))
                loss_dis = beta1 * loss_same + beta2 * loss_diff
                self.reset_grad()
                loss_dis.backward()
                self.grad_clip([self.Discriminator])
                self.D_opt.step()
                # calculate accuracy
                same_acc = torch.mean(torch.ge(same_prob, 0.5).type(torch.FloatTensor))
                diff_acc = torch.mean(torch.lt(diff_prob, 0.5).type(torch.FloatTensor))
                same_val = torch.mean(same_prob)
                diff_val = torch.mean(diff_prob)
                # print info
                slot_value = (
                    iteration + 1,
                    iterations,
                    loss_dis.data[0],
                    same_val.data[0],
                    diff_val.data[0],
                    same_acc.data[0],
                    diff_acc.data[0],
                )
                print(
                    'D-iteration:[%06d/%06d], loss_dis=%.3f, same_val=%.3f, diff_val=%.3f, '
                    'same_acc=%.3f, diff_acc=%.3f'
                    % slot_value,
                )
                info = {
                    'loss_dis':loss_dis.data[0],
                    'same_val': same_val.data[0],
                    'diff_val': diff_val.data[0],
                    'same_acc': same_acc.data[0],
                    'diff_acc': diff_acc.data[0],
                }
                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value, iteration + 1)
            #===================== Train G =====================#
            for j in range(1 if is_pretrain else g_iterations):
                X_i_t, X_i_tk, X_i_tk_prime, X_j = [self.to_var(x) for x in next(self.data_loader)]
                # encode
                Es_i_t = self.Encoder_s(X_i_t)
                Es_i_tk = self.Encoder_s(X_i_tk)
                Es_j = self.Encoder_s(X_j)
                Ec_i_t = self.Encoder_c(X_i_t)
                Ec_i_tk = self.Encoder_c(X_i_tk)
                Ec_i_tk_prime = self.Encoder_c(X_i_tk_prime)
                Es_i_t_flat = Es_i_t.view(Es_i_t.size()[0], -1)
                Es_i_tk_flat = Es_i_tk.view(Es_i_tk.size()[0], -1)
                Es_j_flat = Es_j.view(Es_j.size()[0], -1)
                # max margin loss
                same_sim = F.cosine_similarity(Es_i_t_flat, Es_i_tk_flat)
                diff_sim = F.cosine_similarity(Es_i_t_flat, Es_j_flat)
                loss_sim = torch.mean(F.relu(margin - same_sim + diff_sim))
                # Reconstruct 2 batches
                E_tk = torch.cat([Es_i_t, Ec_i_tk], dim=1)
                X_tilde1 = self.Decoder(E_tk)
                loss_rec1 = torch.mean((X_tilde1 - X_i_tk) ** 2)
                E_tk_prime = torch.cat([Es_i_t, Ec_i_tk_prime], dim=1)
                X_tilde2 = self.Decoder(E_tk_prime)
                loss_rec2 = torch.mean((X_tilde2 - X_i_tk_prime) ** 2)
                loss_rec = (loss_rec1 + loss_rec2) / 2
                loss_other = loss_rec + alpha * loss_sim
                if not is_pretrain:
                    Ec_val1 = self.Discriminator(Ec_i_t, Ec_i_tk)
                    Ec_val2 = self.Discriminator(Ec_i_t, Ec_i_tk_prime)
                    Ec_val = torch.cat([Ec_val1, Ec_val2], dim=0)
                    # BS-GAN loss
                    #loss_adv_enc = 0.5 * torch.mean((torch.log(Ec_val) - torch.log(1 - Ec_val)) ** 2)
                    # LS-GAN loss
                    #loss_adv_enc = torch.mean(Ec_val ** 2)
                    # normal-GAN loss
                    #loss_adv_enc = beta3 * -torch.mean(
                    #    0.5 * torch.log(Ec_val) + 
                    #)
                    mean_Ec_val = torch.mean(Ec_val)
                if not is_pretrain:
                    loss = loss_other + loss_adv_enc
                else:
                    loss = loss_other
                self.reset_grad()
                loss.backward()
                self.grad_clip([self.Encoder_c, self.Encoder_s, self.Decoder])
                self.G_opt.step()
                slot_value = (
                    loss_rec.data[0],
                    loss_sim.data[0],
                    diff_sim.data[0],
                    loss_adv_enc.data[0],
                    mean_Ec_val.data[0],
                )
                # print info
                print_slot_value = (j+1, iteration+1, iterations) + slot_value
                print(
                    'G-iteration-%02d:[%06d/%06d], loss_rec=%.3f, loss_sim=%.3f, diff_sim=%.3f, '
                    'loss_adv=%.3f, mean_val=%.3f'
                    % print_slot_value,
                )
            keys = ['loss_rec', 'loss_sim', 'diff_sim', 'loss_adv_enc', 'mean_Ec_val']
            info = {key:value for key, value in zip(keys, slot_value)}
            for tag, value in info.items():
                self.logger.scalar_summary(tag, value, iteration + 1)

            if iteration % 100 == 0 or iteration + 1 == iterations:
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

