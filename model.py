import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

def get_grad(net):
    for p in net.parameters():
       print(torch.sum(p.grad), p.size())
    return

def conv(inp, layer, pad=True, act=True):
    kernel_size = layer.kernel_size[0]
    # padding
    if pad:
        inp = F.pad(torch.unsqueeze(inp, dim=3), pad=(0, 0, kernel_size//2, kernel_size//2), mode='reflect')
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
    out = F.pad(inp.unsqueeze(dim=3), pad=(0, 0, kernel_size//2, kernel_size//2), mode='reflect')
    out = out.squeeze(dim=3)
    # conv
    out = layer(out)
    # gated
    if res:
        A = out[:, :channels, :] + inp
    else:
        A = out[:, :channels, :]
    B = F.sigmoid(out[:, channels:, :])
    H = A * B
    return H

class Discriminator(nn.Module):
    def __init__(self, c_in=64, c_h=256):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv1d(c_in, c_h*2, kernel_size=5, stride=2)
        self.conv2 = nn.Conv1d(c_h, c_h*2, kernel_size=5, stride=2)
        self.conv3 = nn.Conv1d(c_h, c_h*2, kernel_size=5, stride=2)
        self.conv4 = nn.Conv1d(c_h, 1, kernel_size=4)

    def forward(self, x):
        out = GLU(x, self.conv1, res=False)
        out = GLU(out, self.conv2, res=False)
        out = GLU(out, self.conv3, res=False)
        out = conv(out, self.conv4, pad=False, act=False)
        out = out.view(out.size()[0], -1)
        return out

class Decoder(nn.Module):
    def __init__(self, c_in=128, c_out=80, c_h=256):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv1d(c_in, 2*c_h, kernel_size=5)
        self.conv2 = nn.Conv1d(c_h, 2*c_h, kernel_size=5)
        self.conv3 = nn.Conv1d(c_h, 2*c_h, kernel_size=5)
        self.conv4 = nn.Conv1d(c_h, 2*c_h, kernel_size=5)
        self.conv5 = nn.Conv1d(c_h, 2*c_h, kernel_size=5)
        self.conv6 = nn.Conv1d(c_h, 2*c_out, kernel_size=5)

    def forward(self, x):
        x = upsample(x)
        out = GLU(x, self.conv1, res=False)
        out = upsample(out)
        out = GLU(out, self.conv2)
        out = GLU(out, self.conv3)
        out = GLU(out, self.conv4)
        out = GLU(out, self.conv5)
        out = GLU(out, self.conv6, res=False)
        return out

class Encoder(nn.Module):
    def __init__(self, c_in=80, c_h=256):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(c_in, c_h, kernel_size=1)
        self.conv2 = nn.Conv1d(c_h, 2 * c_h, kernel_size=11)
        self.conv3 = nn.Conv1d(c_h, 2 * c_h, kernel_size=11)
        self.conv4 = nn.Conv1d(c_h, 2 * c_h, kernel_size=11)
        self.conv5 = nn.Conv1d(c_h, c_h, kernel_size=15, stride=2)
        self.conv6 = nn.Conv1d(c_h // 2, c_h // 2, kernel_size=19, stride=2)

    def forward(self, x):
        out = conv(x, self.conv1)
        out = GLU(out, self.conv2)
        out = GLU(out, self.conv3)
        out = GLU(out, self.conv4)
        out = GLU(out, self.conv5, res=False)
        out = GLU(out, self.conv6, res=False)
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
    c = C(e2)
    print(c.size())
