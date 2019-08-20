# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 15:04:52 2019

@author: myidispg
"""
#import numpy as np
#import torch
import torch.nn as nn
import torchvision.models as models
from models.helper import init, make_block

# Extract the first 10 layers of the VGG-19 model.
class VGGFeatureExtractor(nn.Module):
    def __init__(self, use_bn=True):  # Original implementation doesn't use BN
        super(VGGFeatureExtractor, self).__init__()
        if use_bn:
            vgg = models.vgg19(pretrained=True)
            layers_to_use = list(list(vgg.children())[0].children())[:23]
        else:
            vgg = models.vgg19_bn(pretrained=True)
            layers_to_use = list(list(vgg.children())[0].children())[:33]
        self.vgg = nn.Sequential(*layers_to_use)
        self.feature_extractor = nn.Sequential(make_block(512, 256, 3),
                                               make_block(256, 128, 3))
        init(self.feature_extractor)

    def forward(self, x):
        x = self.vgg(x)
        x = self.feature_extractor(x)
        return x
    
#vgg = VGGFeatureExtractor()
#input_ = torch.from_numpy(np.zeros((1, 3, 368, 368))).float()
#output = vgg(input_).detach().numpy()