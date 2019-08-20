#############
# Libraries #
#############

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(7)

MODEL_PATH = '../model/art-flower.pth'

def g_block(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1, final = False, negative_slope = 0.2):
    """
    Create a generator block.
    
    returns
    -------
    A block.
    """
    layers = []
    conv_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias = False)
    layers.append(conv_layer)
    
    if final:
        layers.append(nn.Tanh())
    else:
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(negative_slope))
    
    return nn.Sequential(*layers)

class Generator(nn.Module):
    def __init__(self, z_size, conv_dim, img_size):
        super(Generator, self).__init__()
        self.conv_dim = conv_dim
        self.img_size = img_size//(16)
        self.fc = nn.Linear(z_size, conv_dim * 6 * self.img_size * self.img_size, bias = False)
        self.conv1 = g_block(conv_dim * 6, conv_dim * 4)
        self.conv2 = g_block(conv_dim * 4, conv_dim * 2)
        self.conv3 = g_block(conv_dim * 2, conv_dim)
        self.conv4 = g_block(conv_dim, 3, final = True)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.conv_dim * 6, self.img_size, self.img_size)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x