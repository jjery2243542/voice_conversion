import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

def pad_layer(inp, layer, is_2d=False):
    if type(layer.kernel_size) == tuple:
        kernel_size = layer.kernel_size[0]
    else:
        kernel_size = layer.kernel_size
    if not is_2d:
        if kernel_size % 2 == 0:
            pad = (kernel_size//2, kernel_size//2 - 1)
        else:
            pad = (kernel_size//2, kernel_size//2)
    else:
        if kernel_size % 2 == 0:
            pad = (kernel_size//2, kernel_size//2 - 1, kernel_size//2, kernel_size//2 - 1)
        else:
            pad = (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2)
    # padding
    inp = F.pad(inp, 
            pad=pad,
            mode='reflect')
    out = layer(inp)
    return out

def upsample(x, scale_factor=2):
    x_up = F.upsample(x, scale_factor=2, mode='nearest')
    return x_up

def GLU(inp, layer, res=True):
    kernel_size = layer.kernel_size[0]
    channels = layer.out_channels // 2
    # padding
    out = F.pad(inp.unsqueeze(dim=3), pad=(0, 0, kernel_size//2, kernel_size//2), mode='constant', value=0.)
    out = out.squeeze(dim=3)
    out = layer(out)
    # gated
    A = out[:, :channels, :]
    B = F.sigmoid(out[:, channels:, :])
    if res:
        H = A * B + inp
    else:
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

def append_emb(inp, layer, expand_size, output):
    emb = layer(inp)
    emb = emb.unsqueeze(dim=2)
    emb_expand = emb.expand(emb.size(0), emb.size(1), expand_size)
    output = torch.cat([output, emb_expand], dim=1)
    return output

class PatchDiscriminator(nn.Module):
    def __init__(self, c_in=513, n_class=8, ns=0.2, dp=0.3):
        super(PatchDiscriminator, self).__init__()
        self.ns = ns
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2)
        self.conv5 = nn.Conv2d(512, 1, kernel_size=8)
        self.conv_classify = nn.Conv2d(512, n_class, kernel_size=(33, 8))
        self.drop1 = nn.Dropout(p=dp)
        self.drop2 = nn.Dropout(p=dp)
        self.drop3 = nn.Dropout(p=dp)
        self.drop4 = nn.Dropout(p=dp)

    def forward(self, x, classify=False):
        x = torch.unsqueeze(x, dim=1)
        out = pad_layer(x, self.conv1, is_2d=True)
        out = self.drop1(out)
        out = F.leaky_relu(out, negative_slope=self.ns)
        out = pad_layer(out, self.conv2, is_2d=True)
        out = self.drop2(out)
        out = F.leaky_relu(out, negative_slope=self.ns)
        out = pad_layer(out, self.conv3, is_2d=True)
        out = self.drop3(out)
        out = F.leaky_relu(out, negative_slope=self.ns)
        out = pad_layer(out, self.conv4, is_2d=True)
        out = self.drop4(out)
        out = F.leaky_relu(out, negative_slope=self.ns)
        # GAN output value
        val = pad_layer(out, self.conv5, is_2d=True)
        val = val.view(val.size(0), -1)
        mean_val = torch.mean(val, dim=1)
        if classify:
            # classify
            logits = self.conv_classify(out)
            logits = logits.view(logits.size()[0], -1)
            logits = F.log_softmax(logits, dim=1)
            return mean_val, logits
        else:
            return mean_val

class LatentDiscriminator(nn.Module):
    def __init__(self, c_in=1024, c_h=256, ns=0.2, dp=0.3):
        super(LatentDiscriminator, self).__init__()
        self.ns = ns
        self.conv1 = nn.Conv1d(c_in, c_h, kernel_size=5, stride=2)
        self.conv2 = nn.Conv1d(c_h, c_h, kernel_size=5, stride=2)
        self.conv3 = nn.Conv1d(c_h, c_h, kernel_size=5, stride=2)
        self.conv4 = nn.Conv1d(c_h, c_h, kernel_size=5, stride=2)
        self.conv5 = nn.Conv1d(c_h, 1, kernel_size=1)
        self.drop1 = nn.Dropout(p=dp)
        self.drop2 = nn.Dropout(p=dp)
        self.drop3 = nn.Dropout(p=dp)
        self.drop4 = nn.Dropout(p=dp)

    def forward(self, x):
        out = pad_layer(x, self.conv1)
        out = self.drop1(out)
        out = F.leaky_relu(out, negative_slope=self.ns)
        out = pad_layer(out, self.conv2)
        out = self.drop2(out)
        out = F.leaky_relu(out, negative_slope=self.ns)
        out = pad_layer(out, self.conv3)
        out = self.drop3(out)
        out = F.leaky_relu(out, negative_slope=self.ns)
        out = pad_layer(out, self.conv4)
        out = self.drop4(out)
        out = F.leaky_relu(out, negative_slope=self.ns)
        out = self.conv5(out)
        out = out.view(out.size()[0], -1)
        mean_value = torch.mean(out, dim=1)
        return mean_value

class CBHG(nn.Module):
    def __init__(self, c_in=80, c_out=513):
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
    def __init__(self, c_in=512, c_out=513, c_h=512, c_a=8, emb_size=128, ns=0.2):
        super(Decoder, self).__init__()
        self.ns = ns
        self.conv1 = nn.Conv1d(c_in + emb_size, c_h, kernel_size=5)
        self.conv2 = nn.Conv1d(c_h + emb_size, c_h, kernel_size=5)
        self.conv3 = nn.Conv1d(c_h + emb_size, c_h, kernel_size=5)
        self.conv4 = nn.Conv1d(c_h + emb_size, c_h, kernel_size=5)
        self.conv5 = nn.Conv1d(c_h + emb_size, c_h, kernel_size=5)
        self.dense1 = nn.Linear(c_h, c_h)
        self.dense2 = nn.Linear(c_h, c_h)
        self.dense3 = nn.Linear(c_h, c_h)
        self.dense4 = nn.Linear(c_h, c_h)
        self.RNN = nn.GRU(input_size=c_h + emb_size, hidden_size=c_h//2, num_layers=1, bidirectional=True)
        self.emb = nn.Embedding(c_a, emb_size)
        self.linear = nn.Linear(2*c_h + emb_size, c_out)

    def forward(self, x, c):
        # conv layer
        inp = append_emb(c, self.emb, x.size(2), x)
        inp = upsample(inp)
        out1 = pad_layer(inp, self.conv1)
        out1 = F.leaky_relu(out1, negative_slope=self.ns)
        out2 = append_emb(c, self.emb, out1.size(2), out1)
        out2 = upsample(out2)
        out2 = pad_layer(out2, self.conv2)
        out2 = F.leaky_relu(out2, negative_slope=self.ns)
        out3 = append_emb(c, self.emb, out2.size(2), out2)
        out3 = upsample(out3)
        out3 = pad_layer(out3, self.conv3)
        out3 = F.leaky_relu(out3, negative_slope=self.ns)
        out4 = append_emb(c, self.emb, out3.size(2), out3)
        out4 = pad_layer(out4, self.conv4)
        out4 = F.leaky_relu(out4, negative_slope=self.ns)
        out5 = append_emb(c, self.emb, out4.size(2), out4)
        out5 = pad_layer(out5, self.conv5)
        out5 = F.leaky_relu(out5, negative_slope=self.ns)
        out = out5 + out3
        # dense layer
        out_dense1 = linear(out, self.dense1)
        out_dense1 = F.leaky_relu(out_dense1, negative_slope=self.ns)
        out_dense2 = linear(out_dense1, self.dense2)
        out_dense2 = F.leaky_relu(out_dense2, negative_slope=self.ns)
        out_dense2 = out_dense2 + out
        out_dense3 = linear(out_dense2, self.dense3)
        out_dense3 = F.leaky_relu(out_dense3, negative_slope=self.ns)
        out_dense4 = linear(out_dense3, self.dense4)
        out_dense4 = F.leaky_relu(out_dense4, negative_slope=self.ns)
        out = out_dense4 + out_dense2
        out = append_emb(c, self.emb, out.size(2), out)
        out_rnn = RNN(out, self.RNN)
        out = torch.cat([out, out_rnn], dim=1)
        out = linear(out, self.linear)
        return out

class Encoder(nn.Module):
    def __init__(self, c_in=513, c_h1=128, c_h2=512, c_h3=256, ns=0.2):
        super(Encoder, self).__init__()
        self.ns = ns
        self.conv1s = nn.ModuleList(
                [nn.Conv1d(c_in, c_h1, kernel_size=k) for k in range(1, 16)]
            )
        self.conv2 = nn.Conv1d(len(self.conv1s)*c_h1 + c_in, c_h2, kernel_size=3)
        self.conv3 = nn.Conv1d(c_h2, c_h2, kernel_size=5, stride=2)
        self.conv4 = nn.Conv1d(c_h2, c_h2, kernel_size=5, stride=2)
        self.conv5 = nn.Conv1d(c_h2, c_h2, kernel_size=5, stride=2)
        self.dense1 = nn.Linear(c_h2, c_h2)
        self.dense2 = nn.Linear(c_h2, c_h2)
        self.dense3 = nn.Linear(c_h2, c_h2)
        self.dense4 = nn.Linear(c_h2, c_h2)
        self.RNN = nn.GRU(input_size=c_h2, hidden_size=c_h3, num_layers=2, bidirectional=True)
        self.linear = nn.Linear(c_h2 + 2*c_h3, c_h2)

    def forward(self, x):
        outs = []
        for l in self.conv1s:
            out = pad_layer(x, l)
            outs.append(out)
        out = torch.cat(outs + [x], dim=1)
        out = F.leaky_relu(out, negative_slope=self.ns)
        out = pad_layer(out, self.conv2)
        out = F.leaky_relu(out, negative_slope=self.ns)
        out = pad_layer(out, self.conv3)
        out = F.leaky_relu(out, negative_slope=self.ns)
        out = pad_layer(out, self.conv4)
        out = F.leaky_relu(out, negative_slope=self.ns)
        out = pad_layer(out, self.conv5)
        out = F.leaky_relu(out, negative_slope=self.ns)
        out_dense1 = linear(out, self.dense1)
        out_dense1 = F.leaky_relu(out_dense1, negative_slope=self.ns)
        out_dense2 = linear(out_dense1, self.dense2)
        out_dense2 = F.leaky_relu(out_dense2, negative_slope=self.ns)
        out_dense2 = out_dense2 + out
        out_dense3 = linear(out_dense2, self.dense3)
        out_dense3 = F.leaky_relu(out_dense3, negative_slope=self.ns)
        out_dense4 = linear(out_dense3, self.dense4)
        out_dense4 = F.leaky_relu(out_dense4, negative_slope=self.ns)
        out = out_dense4 + out_dense2
        out_rnn = RNN(out, self.RNN)
        out = torch.cat([out, out_rnn], dim=1)
        out = linear(out, self.linear) 
        return out

if __name__ == '__main__':
    E1, E2 = Encoder(513).cuda(), Encoder(513).cuda()
    D = Decoder().cuda()
    C = LatentDiscriminator().cuda()
    P = PatchDiscriminator().cuda()
    cbhg = CBHG().cuda()
    inp = Variable(torch.randn(16, 513, 128)).cuda()
    e1 = E1(inp)
    e2 = E2(inp)
    c = Variable(torch.from_numpy(np.random.randint(8, size=(16)))).cuda()
    d = D(e1, c)
    print(d.size())
    p1, p2 = P(d, classify=True)
    print(p1.size(), p2.size())
    c = C(torch.cat([e2,e2],dim=1))
    print(c.size())
