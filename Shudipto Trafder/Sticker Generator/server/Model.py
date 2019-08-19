import os

import torch.nn as nn
import torch
import math
from torchvision.utils import make_grid
import numpy as np
import base64
import imageio


batch_size = 128
nc = 3
nz = 100
ngf = 64


# Define Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # num of z input
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # second layer
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # third layer
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # fourth layer
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # last layer
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


def init():
    # Create the generator
    netG = Generator()
    path = os.path.join('model', 'NetG' + "." + 'pth')
    netG.load_state_dict(torch.load(path, map_location='cpu'))

    return netG


def images(model, num=16):
    nrow = int(math.sqrt(num))
    noise = torch.randn(num, nz, 1, 1)
    gen_image = model(noise).to("cpu").clone().detach().squeeze(0)
    grid = (make_grid(gen_image, nrow=nrow, padding=10, normalize=True))
    img = np.transpose(grid.numpy(), (1, 2, 0))
    return img


def process(model, num):
    num = int(num)
    img = images(model, num)
    imageio.imwrite('output.png', img)
    img = open('output.png', 'rb').read()
    encode = base64.encodebytes(img)
    return encode.decode()
