# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 16:59:52 2019

@author: myidispg
"""

import torch
import torch.nn as nn

from models.vgg_model import VGGFeatureExtractor
from models.helper import make_block, init


def make_block_stage1(inp_feats, output_feats):
    layers = [make_block(inp_feats, 128, 3),
              make_block(128, 128, 3),
              make_block(128, 128, 3),
              make_block(128, 512, 1, 1, 0,)]
    layers += [nn.Conv2d(512, output_feats, 1, 1, 0)]
    return nn.Sequential(*layers)


def make_block_stage2(inp_feats, output_feats):
    layers = [make_block(inp_feats, 128, 7, 1, 3),
              make_block(128, 128, 7, 1, 3),
              make_block(128, 128, 7, 1, 3),
              make_block(128, 128, 7, 1, 3),
              make_block(128, 128, 7, 1, 3),
              make_block(128, 128, 1, 1, 0)]
    layers += [nn.Conv2d(128, output_feats, 1, 1, 0)]
    return nn.Sequential(*layers)

class StanceNet(nn.Module):
    def __init__(self, n_joints, n_limbs):
        
        super(StanceNet, self).__init__()
        
        self.vgg = VGGFeatureExtractor()
        
        # PAF Section
        self.stage_1_1 = make_block_stage1(128, n_limbs)
        self.stage_2_1 = make_block_stage2(185, n_limbs)
        self.stage_3_1 = make_block_stage2(185, n_limbs)
        self.stage_4_1 = make_block_stage2(185, n_limbs)
        self.stage_5_1 = make_block_stage2(185, n_limbs)
        self.stage_6_1 = make_block_stage2(185, n_limbs)
        
        # Confidence Maps Section
        self.stage_1_2 = make_block_stage1(128, n_joints + 1)
        self.stage_2_2 = make_block_stage2(185, n_joints + 1)
        self.stage_3_2 = make_block_stage2(185, n_joints + 1)
        self.stage_4_2 = make_block_stage2(185, n_joints + 1)
        self.stage_5_2 = make_block_stage2(185, n_joints + 1)
        self.stage_6_2 = make_block_stage2(185, n_joints + 1)

    def forward(self, x):
        vgg_out = self.vgg(x)
#        print(f'vgg_out: {vgg_out.shape}')
        out1_1 = self.stage_1_1(vgg_out)
        out1_2 = self.stage_1_2(vgg_out)
        out2 = torch.cat([out1_1, out1_2, vgg_out], 1)
#        print(f'out1_1: {out1_1.shape}; out1_2: {out1_2.shape}')
        out2_1 = self.stage_2_1(out2)
        out2_2 = self.stage_2_2(out2)
        out3 = torch.cat([out2_1, out2_2, vgg_out], 1)

        out3_1 = self.stage_3_1(out3)
        out3_2 = self.stage_3_2(out3)
        out4 = torch.cat([out3_1, out3_2, vgg_out], 1)

        out4_1 = self.stage_4_1(out4)
        out4_2 = self.stage_4_2(out4)
        out5 = torch.cat([out4_1, out4_2, vgg_out], 1)

        out5_1 = self.stage_5_1(out5)
        out5_2 = self.stage_5_2(out5)
        out6 = torch.cat([out5_1, out5_2, vgg_out], 1)

        out6_1 = self.stage_6_1(out6)
        out6_2 = self.stage_6_2(out6)
        
        # 1 is PAF, 2 is Conf map
        return out6_1, out6_2
    
#model = StanceNet(19, 38)