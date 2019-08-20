# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 17:31:55 2019

@author: myidi
"""

import numpy as np

from PIL import Image
import cv2

import os
import json


dataset_dir = os.path.join('C:\Machine Learning Projects\OpenPose', 'Coco_Dataset')

with open(os.path.join(dataset_dir,
                       'annotations', 'person_keypoints_val2017.json'), 'r') as JSON:
    val_dict = json.load(JSON)

with open(os.path.join(dataset_dir,
                       'annotations', 'person_keypoints_train2017.json'), 'r') as JSON:
    train_dict = json.load(JSON)

print(f'The length of train annotations is: {len(train_dict["annotations"])}')
print(f'The length of validation annotations is: {len(val_dict["annotations"])}')

skeleton_limb_indices = [(3,5), (3,2), (2, 4), (7,6), (7,9), (9,11), (6,8),
                         (8,10), (7,13), (6,12), (13,15), (12,14), (15,17),
                         (14, 16), (13, 12)]

def display_im_keypoints(annotation_dict, index, skeleton_limb_indices, val=False):
    """
    Takes in the index of the image from the validation annotations,
    returns the keypoints from that image that are labeled and shows image
    with keypoints.
    The keypoints are in the following format: (x, y, v)
    v=0: not labeled, v=1: labeled but not visible, v=2: labeled and visible.
    """
    
    image_id = annotation_dict['annotations'][index]['image_id']
    
    # Find the number of zeros to append before img_id
    num_zeros = 12 - len(str(image_id))
    zeros = '0' * num_zeros
    image_name = zeros + str(image_id)
    
    print(image_name)
    
    if val:
        image_path = os.path.join(dataset_dir, 
                                    'val2017', 
                                    f"{image_name}.jpg")
    else:
        image_path = os.path.join(dataset_dir,
                                  'train2017',
                                  f"{image_name}.jpg")
    
    keypoints = annotation_dict['annotations'][index]['keypoints']
    
    arranged_keypoints = list()
    
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Group keypoints into a list of tuples. Each tuple has: (x, y, v)
    for i in range(0, len(keypoints), 3):
        arranged_keypoints.append((keypoints[i], keypoints[i+1], keypoints[i+2]))
    
    # Display points for joints that are visible.
    for keypoint in arranged_keypoints:
        if keypoint[2] != 0:
            cv2.circle(img, (keypoint[0], keypoint[1]), 0, (0, 255, 255), 6)
    
    
    # Draw limbs for the visible joints
    # Note: 1 is subtracted because indices start from 0.
    for joint_index in skeleton_limb_indices:
        if arranged_keypoints[joint_index[0]-1][2] != 0:
            if arranged_keypoints[joint_index[1]-1][2] != 0:
                x1 = arranged_keypoints[joint_index[0]-1][0]
                y1 = arranged_keypoints[joint_index[0]-1][1]
                x2 = arranged_keypoints[joint_index[1]-1][0]
                y2 = arranged_keypoints[joint_index[1]-1][1]
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 4)

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return image_id, arranged_keypoints
    
for i in range(0, 10):
    img_id, keypoints = display_im_keypoints(train_dict, i, skeleton_limb_indices)

index = None
for i in range(len(val_dict['annotations'])):
    if val_dict['annotations'][i]['image_id'] == 785:
        index = i
        break
