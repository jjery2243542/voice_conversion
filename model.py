import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

def pad_layer(inp, layer):
    if type(layer.kernel_size) == tuple:
        kernel_size = layer.kernel_size[0]
    else:
        kernel_size = layer.kernel_size
    if kernel_size % 2 == 0:
        pad = (0, 0, kernel_size//2, kernel_size//2 - 1)
    else:
        pad = (0, 0, kernel_size//2, kernel_size//2)
    # padding
    inp = F.pad(
        torch.unsqueeze(inp, dim=3),
        pad=pad,
        mode='constant',
        value=0.)
    inp = inp.squeeze(dim=3)
    out = layer(inp)
    return out

def upsample(x, scale_factor=2):
    # reshape
    x = x.unsqueeze(dim=3)
    x_up = F.upsample(x, scale_factor=2, mode='nearest')[:, :, :, 0]
    return x_up

def GLU(inp, layer, res=True):
    kernel_size = layer.kernel_size[0]
    channels = layer.out_channels // 2
    # padding
    out = F.pad(inp.unsqueeze(dim=3), pad=(0, 0, kernel_size//2, kernel_size//2), mode='constant', value=0.)
    out = out.squeeze(dim=3)
    out = layer(out)
    # gated
    if res:
        A = out[:, :channels, :] + inp
    else:
        A = out[:, :channels, :]
    B = F.sigmoid(out[:, channels:, :])
    H = A * B
    return H

def highway(inp, layers, gates, act):
    # permute
    batch_size = inp.size(0)
    seq_len = inp.size(2)
    inp_permuted = inp.permute(0, 2, 1)
    # merge dim
    out_expand = inp_permuted.contiguous().view(batch_size*seq_len, inp_permuted.size(2))
    for l, g in zip(layers, gates):
        H = l(out_expand)
        H = act(H)
        T = g(out_expand)
        T = F.sigmoid(T)
        out_expand = H * T + out_expand * (1. - T)
    out_permuted = out_expand.view(batch_size, seq_len, out_expand.size(1))
    out = out_permuted.permute(0, 2, 1)
    return out

def RNN(inp, layer):
    inp_permuted = inp.permute(2, 0, 1)
    state_mul = (int(layer.bidirectional) + 1) * layer.num_layers
    zero_state = Variable(torch.zeros(state_mul, inp.size(0), layer.hidden_size))
    zero_state = zero_state.cuda() if torch.cuda.is_available() else zero_state
    out_permuted, _ = layer(inp_permuted, zero_state)
    out_rnn = out_permuted.permute(1, 2, 0)
    return out_rnn

def linear(inp, layer):
    batch_size = inp.size(0)
    hidden_dim = inp.size(1)
    seq_len = inp.size(2)
    inp_permuted = inp.permute(0, 2, 1)
    inp_expand = inp_permuted.contiguous().view(batch_size*seq_len, hidden_dim)
    out_expand = layer(inp_expand)
    out_permuted = out_expand.view(batch_size, seq_len, out_expand.size(1))
    out = out_permuted.permute(0, 2, 1)
    return out

class Discriminator(nn.Module):
    def __init__(self, c_in=1024, c_h=256):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv1d(c_in, c_h*2, kernel_size=3, stride=2)
        self.conv2 = nn.Conv1d(c_h, c_h*2, kernel_size=3, stride=2)
        self.conv3 = nn.Conv1d(c_h, 1, kernel_size=128//4)

    def forward(self, x):
        out = GLU(x, self.conv1, res=False)
        out = GLU(out, self.conv2, res=False)
        out = self.conv3(out)
        out = out.view(out.size()[0], -1)
        return out

class CBHG(nn.Module):
    def __init__(self, c_in=80, c_out=1025):
        super(CBHG, self).__init__()
        self.conv1s = nn.ModuleList(
                [nn.Conv1d(c_in, 128, kernel_size=k) for k in range(1, 9)]
                )
        self.bn1s = nn.ModuleList([nn.BatchNorm1d(128) for _ in range(1, 9)])
        self.mp1 = nn.MaxPool1d(kernel_size=2, stride=1)
        self.conv2 = nn.Conv1d(len(self.conv1s)*128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 80, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(80)
        # highway network
        self.linear1 = nn.Linear(80, 128)
        self.layers = nn.ModuleList([nn.Linear(128, 128) for _ in range(4)])
        self.gates = nn.ModuleList([nn.Linear(128, 128) for _ in range(4)])
        self.RNN = nn.GRU(input_size=128, hidden_size=128, num_layers=1, bidirectional=True)
        self.linear2 = nn.Linear(256, c_out) 
        
    def forward(self, x):
        outs = []
        for l in self.conv1s:
            out = pad_layer(x, l)
            out = F.relu(out)
            outs.append(out)
        bn_outs = []
        for out, bn in zip(outs, self.bn1s):
           out = bn(out) 
           bn_outs.append(out)
        out = torch.cat(bn_outs, dim=1)
        out = pad_layer(out, self.mp1)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = out + x
        out = linear(out, self.linear1)
        out = highway(out, self.layers, self.gates, F.relu)
        out_rnn = RNN(out, self.RNN)
        out = linear(out_rnn, self.linear2)
        return out

class Decoder(nn.Module):
    def __init__(self, c_in=1024, c_out=80, c_h=512):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv1d(c_in, 2*c_h, kernel_size=3)
        self.conv2 = nn.Conv1d(c_h, 2*c_h, kernel_size=3)
        self.conv3 = nn.Conv1d(c_h, 2*c_h, kernel_size=3)
        self.conv4 = nn.Conv1d(c_h, 2*c_out, kernel_size=3)

    def forward(self, x):
        out = GLU(x, self.conv1, res=False)
        out = GLU(out, self.conv2, res=True)
        out = GLU(out, self.conv3, res=True)
        out = GLU(out, self.conv4, res=False)
        return out

class Encoder(nn.Module):
    def __init__(self, c_in=80, c_h1=128, c_h2=512, c_h3=256):
        super(Encoder, self).__init__()
        self.conv1s = nn.ModuleList(
                [nn.Conv1d(c_in, c_h1, kernel_size=k) for k in range(1, 15)]
            )
        self.conv2 = nn.Conv1d(len(self.conv1s)*c_h1 + c_in, c_h2*2, kernel_size=3)
        self.conv3 = nn.Conv1d(c_h2, c_h2*2, kernel_size=3)
        self.layers = nn.ModuleList([nn.Linear(c_h2, c_h2) for _ in range(4)])
        self.gates = nn.ModuleList([nn.Linear(c_h2, c_h2) for _ in range(4)])
        self.RNN = nn.GRU(input_size=c_h2, hidden_size=c_h3, num_layers=2, bidirectional=True)

    def forward(self, x):
        outs = []
        for l in self.conv1s:
            out = pad_layer(x, l)
            outs.append(out)
        out = torch.cat(outs + [x], dim=1)
        out = F.leaky_relu(out)
        out = GLU(out, self.conv2, res=False)
        out = GLU(out, self.conv3, res=True)
        out = highway(out, self.layers, self.gates, F.leaky_relu)
        out_rnn = RNN(out, self.RNN)
        out = out + out_rnn
        return out

if __name__ == '__main__':
    E1, E2 = Encoder(80).cuda(), Encoder(80).cuda()
    D = Decoder().cuda()
    C = Discriminator().cuda()
    cbhg = CBHG().cuda()
    inp = Variable(torch.randn(16, 80, 128)).cuda()
    e1 = E1(inp)
    e2 = E2(inp)
    e = torch.cat([e1, e2], dim=1)
    d = D(e)
    d2 = cbhg(d)
    c = C(torch.cat([e2,e2],dim=1))
