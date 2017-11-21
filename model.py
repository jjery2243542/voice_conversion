import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

def conv(inp, layer, pad=True, act=True):
    kernel_size = layer.kernel_size[0]
    # padding
    if pad:
        inp = F.pad(
            torch.unsqueeze(inp, dim=3),
            pad=(0, 0, kernel_size//2, kernel_size//2),
            mode='constant',
            value=0.)
        inp = inp.squeeze(dim=3)
    out = layer(inp)
    if act:
        out = F.leaky_relu(out)
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
    A = out[:, :channels, :]
    B = F.sigmoid(out[:, channels:, :])
    if res:
        H = A * B + inp
    else:
        H = A * B
    return H

class Discriminator(nn.Module):
    def __init__(self, c_in=512, c_h=256):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv1d(c_in, c_h*2, kernel_size=3, stride=2)
        self.conv2 = nn.Conv1d(c_h, c_h*2, kernel_size=3, stride=2)
        self.conv3 = nn.Conv1d(c_h, 1, kernel_size=128//4)

    def forward(self, x):
        out = GLU(x, self.conv1, res=False)
        out = GLU(out, self.conv2, res=False)
        out = conv(out, self.conv3, pad=False, act=False)
        out = out.view(out.size()[0], -1)
        return out

class Decoder(nn.Module):
    def __init__(self, c_in=512, c_out=80, c_h=512):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv1d(c_in, 2*c_h, kernel_size=3)
        self.conv2 = nn.Conv1d(c_h, 2*c_out, kernel_size=3)

    def forward(self, x):
        print(x.size())
        out = GLU(x, self.conv1, res=False)
        out = GLU(out, self.conv2, res=False)
        return out

class Encoder(nn.Module):
    def __init__(self, c_in=80, c_h1=128, c_h2=256):
        super(Encoder, self).__init__()
        self.convs = nn.ModuleList(
                [nn.Conv1d(c_in, c_h1, kernel_size=k, padding=k//2) for k in range(1, 13, 2)]
            )
        self.conv2 = nn.Conv1d(len(self.convs) * c_h1 + c_in, c_h2 * 2, kernel_size=3)
        self.conv3 = nn.Conv1d(c_h2, c_h2*2, kernel_size=3) 

    def forward(self, x):
        outs = []
        for l in self.convs:
            out = l(x)
            outs.append(out)
        out = torch.cat(outs + [x], dim=1)
        out = GLU(out, self.conv2, res=False)
        out = GLU(out, self.conv3, res=True)
        return out


if __name__ == '__main__':
    E1, E2 = Encoder(80), Encoder(80)
    D = Decoder()
    C = Discriminator()
    inp = Variable(torch.randn(16, 80, 128))
    e1 = E1(inp)
    e2 = E2(inp)
    e = torch.cat([e1, e2], dim=1)
    d = D(e)
    c = C(torch.cat([e2,e2],dim=1))
