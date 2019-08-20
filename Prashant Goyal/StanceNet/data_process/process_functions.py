# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 09:42:17 2019

@author: myidispg
"""

import cv2
import numpy as np

from utilities.constants import dataset_dir, img_size, transform_scale 
from utilities.constants import MEAN, STD, joint_map_coco
from utilities.constants import im_height, im_width, num_joints, skeleton_limb_indices

def group_keypoints(keypoints):
    """
    Given a list of 51 keypoints, groups the keypoints into a list of 17 tuples.
    Each tuple is (x, y, v) where 
    v=0: not labeled, v=1: labeled but not visible, v=2: labeled and visible.
    """
    arranged_keypoints = list()
    for i in range(0, len(keypoints), 3):
        arranged_keypoints.append((keypoints[i], keypoints[i+1], keypoints[i+2]))
    
    return arranged_keypoints

def normalize(img):
    img = (img - MEAN) / STD
    return img

def adjust_keypoints(keypoints, original_shape):
  """
  Adjust keypoints according to the new image size.
  Also, changes the indexes of the keypoints according to the pre-trained model's
  specification of joint and limb arrangement.
  """
  new_keypoints = []
  # For a sublist in keypoints
  for list_ in range(len(keypoints)):
      sub_list = []
      for i in range(0, len(keypoints[list_]), 3):
          sub_list.append(int((keypoints[list_][i]/original_shape[0]) * img_size))
          sub_list.append(int((keypoints[list_][i+1]/original_shape[1]) * img_size))
          sub_list.append(keypoints[list_][i+2])
#      sub_list[i] = int((keypoints[list_][i]/original_shape[0]) * img_size)
#      keypoints[list_][i+1] = int((keypoints[list_][i+1]/original_shape[1]) * img_size)
      new_keypoints.append(sub_list)
      
  # Change the arrangement of the keypoints to map the new schema
  arranged_keypoints = []
  for list_ in new_keypoints:
      sub_list = []
      for pair in joint_map_coco:
          x_index = list_[pair[0] * 3]
          y_index = list_[(pair[0] * 3) + 1]
          visibility = list_[(pair[0] * 3) + 2]
        
          sub_list.append(x_index)
          sub_list.append(y_index)
          sub_list.append(visibility)
      arranged_keypoints.append(sub_list)

  return arranged_keypoints

def add_neck_joint(keypoints):
    """
    Takes in the list of all keypoints for an image and returns a list of all the neck
    """
    for list_ in range(len(keypoints)):
        if keypoints[list_][(6*3) + 2] != 0 and keypoints[list_][(5*3) + 2] != 0:
            neck_x = (keypoints[list_][6*3] + keypoints[list_][5*3]) // 2
            neck_y = (keypoints[list_][(6*3) + 1] + keypoints[list_][(5*3) + 1]) // 2
            keypoints[list_].append(neck_x) 
            keypoints[list_].append(neck_y)
            keypoints[list_].append(2)
        else:
            # Append 0 with no visibility to make sure there are 54 elements in the list.
            keypoints[list_].append(0)
            keypoints[list_].append(0)
            keypoints[list_].append(0)
        
    return keypoints

def do_affine_transform(img, scale=0.25):
    """
    Takes an image and perform scaling affine transformation on it.
    Inputs: 
        img: the image on which to apply the transform
        scale: The factor by which to scale. Default 0.25 as required by model.
    Outputs:
        tranformed image is returned.
    """
    transformation_matrix = [[scale, 0, 0], [0, scale, 0]]
    rows, cols = img.shape[0], img.shape[1]
    
#    print(f'rows: {rows}, cols: {cols}')
    
    transformation_matrix = np.asarray(transformation_matrix, dtype=np.float32)
    return cv2.warpAffine(np.float32(img), transformation_matrix, (int(cols*scale), int(rows*scale)))

def generate_confidence_maps(keypoints, sigma=12):
    """
    Generate confidence maps for a batch of indices.
    The generated confidence_maps are of shape: 
        (batch_size, im_width, im_height, num_joints)
    Input:
        keypoints: The list of keypoints labeled in the image. If multiple people,
            there can be sub-lists too.
        sigma: used to control the spread of the peak.
    Output:
        conf_map: A numpy array of shape: (batch_size, im_width, im_height, num_joints)
    """
    conf_map = np.zeros((img_size // transform_scale,
                             img_size // transform_scale, num_joints + 1), np.float32)
    
    # For sub list in keypoints:
    for list_ in keypoints:
        part_num = 0
        for i in range(0, len(list_), 3):
            x_index = list_[i]
            y_index = list_[i+1]
            visibility = list_[i+2]
            
            if visibility != 0:
                x_ind, y_ind = np.meshgrid(np.arange(img_size), 
                                          np.arange(img_size))
                numerator = (-(x_ind-x_index)**2) + (-(y_ind-y_index)**2)
                heatmap_joint = np.exp(numerator/sigma).transpose()
                heatmap_joint = cv2.resize(heatmap_joint,
                                               (img_size // transform_scale,
                                                img_size // transform_scale),
                                                cv2.INTER_AREA)
                conf_map[:, :, part_num] = np.maximum(heatmap_joint, conf_map[:, :, part_num])
            
            part_num += 1
    
    
    # Add the sum of all heatmaps to the last index
    for i in range(18):
        conf_map[:, :, 18] += conf_map[:, :, i]
    
    return conf_map
    

def generate_paf(keypoints, sigma=5):
    """
    Generate Part Affinity Fields given a batch of keypoints.
    Inputs:
         keypoints: The list of keypoints labeled in the image. If multiple people,
            there can be sub-lists too.
        sigma: The width of a limb in pixels.
    Outputs:
        paf: A parts affinity fields map of shape: 
            (batch_size, im_width, im_height, 2, num_joints)
    """
    
    paf = np.zeros((img_size // transform_scale, img_size // transform_scale,
                    len(skeleton_limb_indices), 2), np.float32)
    # For sub list in keypoints
    for list_ in keypoints:
        for limb in range(len(skeleton_limb_indices)):
            limb_indices = skeleton_limb_indices[limb]
            # Convert part num given in skeleton_limb_indices to index in keypoints
            joint_one_index = (limb_indices[0]) * 3
            joint_two_index = (limb_indices[1]) * 3
            
            # If there is 0 for visibility, skip the limb
            if list_[joint_one_index + 2] != 0 and list_[joint_two_index + 2] != 0:
                joint_one_loc = np.asarray((list_[joint_one_index], list_[joint_one_index+1]))
                joint_two_loc = np.asarray((list_[joint_two_index], list_[joint_two_index+1]))
                part_line_segment = joint_two_loc - joint_one_loc
                norm = np.linalg.norm(part_line_segment)
                
                norm = norm + 1e-8 if norm == 0 else norm # To make sure it is not equal to zero.
                
                vector = part_line_segment / norm
                # https://gamedev.stackexchange.com/questions/70075/how-can-i-find-the-perpendicular-to-a-2d-vector
                perpendicular_vec = [-vector[1], vector[0]]
                x, y = np.meshgrid(np.arange(img_size), np.arange(img_size))
                 # Formula according to the paper
                along_limb = vector[0] * (x-joint_one_loc[0]) + vector[1] * (y-joint_one_loc[1])
                across_limb = np.abs(perpendicular_vec[0] * (x-joint_one_loc[0]) + perpendicular_vec[1] * (y - joint_one_loc[1])) 
                
                # Check the given conditions
                cond_1 = along_limb >= 0
                cond_2 = along_limb <= norm
                cond_3 = across_limb <= sigma
                mask = (cond_1 & cond_2 & cond_3).astype(np.float32).transpose()
                
                
                # put the values
                mask = cv2.resize(mask, (int(img_size / transform_scale),
                                         int(img_size / transform_scale)),
                                        cv2.INTER_AREA)
                paf[:, :, limb, 0] += mask * vector[0]
                paf[:, :, limb, 1] += mask * vector[1]
                    
    return paf
                
                
                