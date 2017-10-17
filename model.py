import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def conv(c_in, c_out, kernel_size):
    layers = []
    layers.append(nn.ReflectionPad2d((kernel_size[0] - 1, kernel_size[1] - 1)))
    layers.append(nn.ConvTranspose2d(c_in, c_out, kernel_size))
    return nn.Sequential(*layers)

class Encoder(nn.Module):
    def __init__(self, c_in, c_out):
        super(Encoder, self).__init__()
        self.conv1 = conv(c_in, 128, kernel_size=(5, 5))
        self.conv2 = conv(128, 256, kernel_size=(5, 5))
        self.conv3 = conv(256, 128, kernel_size=(5, 5))
        self.conv4 = conv(128, c_out, kernel_size=(1, 1))

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = torch.cat((out, x), 1)
        out = self.conv2(out)
        out = F.relu(out)
        out = torch.cat((out, x), 1)
        out = self.conv3(out)
        out = F.relu(out)
        out = torch.cat((out, x), 1)
        out = self.conv4(out)
        out = F.relu(out)
        return out

def dconv(c_in, c_out, kernel_size, bn=True, pad=False, stride=2):
    layers = []
    if pad:
        layers.append(nn.ReflectionPad2d((kernel_size[0] - 1, kernel_size[1] - 1)))
    layers.append(nn.ConvTranspose2d(c_in, c_out, kernel_size, stride=stride))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

class Discriminator(nn.Module):
    def __init__(self, c_in, image_size=(257, 64)):
        super(Discriminator, self).__init__()
        self.dconv1 = dconv(c_in, 64, kernel_size=(5, 5))
        self.dconv2 = dconv(64, 128, kernel_size=(5, 5), bn=True, pad=True)
        self.dconv3 = dconv(128, 256, kernel_size=(5, 5), bn=True, pad=True)
        self.dconv4 = dconv(256, 512, kernel_size=(5, 5), bn=True, pad=True)
        self.fc = dconv(512, 1, (int(image_size[0] / 16), int(image_size[1] / 16)), bn=False, pad=False)

    def forward(self, e1, e2):
        e = torch.cat([e1, e2], axis=1)
        out = self.dconv1(e)
        out = F.leaky_relu(out)
        out = self.dconv2(out)
        out = F.leaky_relu(out)
        out = self.dconv3(out)
        out = F.leaky_relu(out)
        out = self.dconv4(out)
        out = F.leaky_relu(out)
        out = self.fc(out).squeeze()
        out = F.sigmoid(out)
        return out

if __name__ == '__main__':
    E_s = Encoder(2, 1)
    E_c = Encoder(2, 1)
    D = Encoder(2, 2)
    C = Discriminator(2)
