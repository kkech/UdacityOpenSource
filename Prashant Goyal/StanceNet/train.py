# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 18:33:28 2019

@author: myidispg
"""

import os
import torch
import numpy as np

from pycocotools.coco import COCO

from models.full_model import OpenPoseModel

import utilities.constants as constants
import utilities.helper as helper
from training_utilities.train_utils import train_epoch, train
from training_utilities.stancenet_dataset import StanceNetDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Loading training COCO Annotations used for mask generation. Might take time.')
coco_train = COCO(os.path.join(os.path.join(os.getcwd(), 'Coco_Dataset'),
                       'annotations', 'person_keypoints_train2017.json'))
#coco_val = COCO(os.path.join(os.path.join(os.getcwd(), 'Coco_Dataset'),
#                       'annotations', 'person_keypoints_val2017.json'))
print('Annotation load complete.')

train_data = StanceNetDataset(coco_train, os.path.join(constants.dataset_dir,
                                                       'train2017'))
#val_data = StanceNetDataset(coco_val, os.path.join(constants.dataset_dir,
#                                                       'val2017'))

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=1,
                                               shuffle=True)
#val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=1,
#                                               shuffle=True)
status = train(train_dataloader, device, num_epochs=20, val_every=False,
               print_every=50, resume=False)
if status == None:
    print('There was some issue in the training process. Please check.')

