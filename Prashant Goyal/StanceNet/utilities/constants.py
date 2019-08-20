# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:41:36 2019

@author: myidispg
"""

import os

img_size = 400 # All the images will be resized to this size. PAFs, mask etc. will be adjusted accordingly.

# IMAGE NET CONSTANTS
MEAN = [0.485, 0.456, 0.406],
STD = [0.229, 0.224, 0.225]

transform_scale = 8

num_joints = 18
num_limbs = 19

# Used in creating confidence maps.
threshold = 0.15

dataset_dir = os.path.join(os.getcwd(), 'Coco_Dataset')
model_path = os.path.join(os.getcwd(), 'trained_models')

# Since I had to use a pre-trained model for inference as training was taking 
# a very long time on my machine, I had to map the keypoints order in coco data
# to the joint indexes used in the pre-trained model.
# in the pre-trained model's skeleton, joint 1 is the neck joint.
# First is coco index, second is required index
joint_map_coco = [(0, 0), (17, 1), (6, 2), (8, 3), (10, 4), (5, 5), (7, 6), 
                   (9, 7), (12, 8), (14, 9), (16, 10), (11, 11), (13, 12),
                   (15, 13), (2, 14), (1, 15), (4, 16), (3, 17)]

skeleton_limb_indices = [(1, 8), # Neck - right waist
                         (8, 9), # Right waist - right knee
                         (9, 10), # Right knee - right foot
                         (1, 11), # Neck - left waist
                         (11, 12), # Left waist - left knee
                         (12, 13), # left knee - left foot
                         (1, 2), # neck - right shoulder
                         (2, 3), # Right shoulder - left elbow
                         (3, 4), # Right elbow - left arm
                         (2, 16), # Right shoulder - right ear
                         (1, 5), # neck - left shoulder
                         (5, 6), # left shoulder - left elbow
                         (6, 7), # left elbow - left arm
                         (5, 17), # left shoulder - left ear
                         (1, 0), # neck - nose
                         (0, 14), # nose - right eye
                         (0, 15), # nose - left eye
                         (14, 16), # right eye - right ear
                         (15, 17)] # left eye - left ear

# These are indices without the shoulder ear connections. Used while inferencing
BODY_PARTS = [(1, 8), # Neck - right waist
              (8, 9), # Right waist - right knee
              (9, 10), # Right knee - right foot
              (1, 11), # Neck - left waist
              (11, 12), # Left waist - left knee
              (12, 13), # left knee - left foot
              (1, 2), # neck - right shoulder
              (2, 3), # Right shoulder - left elbow
              (3, 4), # Right elbow - left arm
              (1, 5), # neck - left shoulder
              (5, 6), # left shoulder - left elbow
              (6, 7), # left elbow - left arm
              (1, 0), # neck - nose
              (0, 14), # nose - right eye
              (0, 15), # nose - left eye
              (14, 16), # right eye - right ear
              (15, 17)] # left eye - left ear

