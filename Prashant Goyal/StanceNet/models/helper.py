# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 17:06:09 2019

@author: myidispg
"""

import torch.nn as nn
import math


def init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

def make_block(features_in, features_out, kernel, stride=1, padding=1, relu=True):
    layers = []
    layers += [nn.Conv2d(features_in, features_out, kernel, stride, padding)]
    if relu:
        layers += [nn.ReLU(inplace=True)]

    return nn.Sequential(*layers)

#def make_standard_block(feat_in, feat_out, kernel, stride=1, padding=1, use_bn=False):
#    layers = []
#    layers += [nn.Conv2d(feat_in, feat_out, kernel, stride, padding)]
#    if use_bn:
#        layers += [nn.BatchNorm2d(feat_out, eps=1e-05, momentum=0.1, affine=True,
#                                  track_running_stats=True)]
#    layers += [nn.ReLU(inplace=True)]
#    return nn.Sequential(*layers)