import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

def get_grad(net):
    for p in net.parameters():
       print(torch.sum(p.grad), p.size())
    return

def conv(inp, layer):
    kernel_size = layer.kernel_size[0]
    # padding
    out = F.pad(torch.unsqueeze(inp, dim=3), pad=(0, 0, kernel_size//2, kernel_size//2), mode='reflect')
    out = out.squeeze(dim=3)
    out = layer(out)
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

'''DEPRECATE
def conv(c_in, c_out, kernel_size, stride=1, pad=True, bn=True):
    layers = []
    if pad:
        layers.append(
            nn.ReflectionPad2d(kernel_size[0] // 2)
        )
    layers.append(nn.Conv2d(c_in, c_out, kernel_size, stride=stride))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

class Decoder(nn.Module):
    def __init__(self, c_in, c_out):
        super(Decoder, self).__init__()
        self.conv1 = conv(c_in, 128, kernel_size=(5, 5), stride=1, pad=True, bn=False)
        self.conv2 = conv(128 + c_in, 256, kernel_size=(5, 5), stride=1, pad=True, bn=False)
        self.conv3 = conv(256 + c_in, 256, kernel_size=(5, 5), stride=1, pad=True, bn=False)
        self.conv4 = conv(256 + c_in, 128, kernel_size=(5, 5), stride=1, pad=True, bn=False)
        self.conv5 = conv(128 + c_in, 128, kernel_size=(5, 5), stride=1, pad=True, bn=False)
        self.conv6 = conv(128 + c_in, c_out, kernel_size=(5, 5), stride=1, pad=True, bn=False)

    def forward(self, x):
        out = self.conv1(x)
        out = F.leaky_relu(out)
        out = torch.cat((out, x), 1)
        out = self.conv2(out)
        out = F.leaky_relu(out)
        out = torch.cat((out, x), 1)
        out = self.conv3(out)
        out = F.leaky_relu(out)
        out = torch.cat((out, x), 1)
        out = self.conv4(out)
        out = F.leaky_relu(out)
        out = torch.cat((out, x), 1)
        out = self.conv5(out)
        out = F.leaky_relu(out)
        out = torch.cat((out, x), 1)
        out = self.conv6(out)
        out = F.sigmoid(out)
        return out


class Encoder(nn.Module):
    def __init__(self, c_in, c_out):
        super(Encoder, self).__init__()
        self.conv1 = conv(c_in, 128, kernel_size=(5, 5), stride=1, pad=True, bn=False)
        self.conv2 = conv(128 + c_in, 256, kernel_size=(5, 5), stride=1, pad=True, bn=False)
        self.conv3 = conv(256 + c_in, 128, kernel_size=(5, 5), stride=1, pad=True, bn=False)
        self.conv4 = conv(128, c_out, kernel_size=(5, 5), stride=1, pad=True, bn=False)

    def forward(self, x):
        out = self.conv1(x)
        out = F.leaky_relu(out)
        out = torch.cat((out, x), 1)
        out = self.conv2(out)
        out = F.leaky_relu(out)
        out = torch.cat((out, x), 1)
        out = self.conv3(out)
        out = F.leaky_relu(out)
        out = self.conv4(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, c_in, image_size=(257, 64)):
        super(Discriminator, self).__init__()
        self.conv1 = conv(c_in, 64, kernel_size=(5, 5), stride=2, bn=False, pad=True)
        self.conv2 = conv(64, 128, kernel_size=(5, 5), stride=2, bn=True, pad=True)
        self.conv3 = conv(128, 256, kernel_size=(5, 5), stride=2, bn=True, pad=True)
        self.conv4 = conv(256, 512, kernel_size=(5, 5), stride=2, bn=True, pad=True)
        self.fc = conv(512, 1, (int(image_size[0] / 16 + 1), int(image_size[1] / 16)), bn=False, pad=False)

    def forward(self, e1, e2):
        e = torch.cat([e1, e2], dim=1)
        out = self.conv1(e)
        out = F.leaky_relu(out)
        out = self.conv2(out)
        out = F.leaky_relu(out)
        out = self.conv3(out)
        out = F.leaky_relu(out)
        out = self.conv4(out)
        out = F.leaky_relu(out)
        out = self.fc(out).squeeze()
        out = F.sigmoid(out)
        return out
'''

if __name__ == '__main__':
    E1, E2 = Encoder(80), Encoder(80)
    D = Decoder()
    inp = Variable(torch.randn(16, 80, 128))
    e1 = E1(inp)
    e2 = E2(inp)
    print(e1.size())
    e = torch.cat([e1, e2], dim=1)
    d = D(e)
    print(d.size())
