# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 15:46:42 2019

@author: myidispg
"""

"""
This file stores all the helper functions for the project.
"""

import cv2
import os
import numpy as np

from utilities.constants import dataset_dir, im_width, im_height
from data_process.process_functions import generate_confidence_maps, generate_paf

def get_image_name(image_id):
    """
    Given an image id, adds zeros before it so that the image name length is 
    of 12 digits as required by the database.
    Input:
        image_id: the image id without zeros.
    Output:
        image_name: image name with zeros added and the .jpg extension
    """
    num_zeros = 12 - len(str(image_id))
    zeros = '0' * num_zeros
    image_name = zeros + str(image_id) + '.jpg'
    
    return image_name

def get_image_id_from_filename(filename):
    """
    Get the image_id from a filename with .jpg extension
    """    
    return int(filename.split('.')[0])

def gen_data(all_keypoints, batch_size = 64, val=False, affine_transform=True):
    """
    Generate batches of training data. 
    Inputs:
        all_keypoints: 
    """
    batch_count = len(all_keypoints.keys()) // batch_size
    
    count = 0
    
    # Loop over all keypoints in batches
    for batch in range(1, batch_count * batch_size + 1, batch_size):
        
        count += 1
        
        images = np.zeros((batch_size, im_width, im_height, 3), dtype=np.uint8)
        
        # Loop over all individual indices in a batch
        for image_id in range(batch, batch + batch_size):
            img_name = get_image_name(image_id-1)
            
            if val:
                img = cv2.imread(os.path.join(dataset_dir,'new_val2017',img_name))
            else:
                img = cv2.imread(os.path.join(dataset_dir,'new_train2017',img_name))
            
            images[image_id % batch] = img
        
        conf_maps = generate_confidence_maps(all_keypoints, range(batch-1, batch+batch_size-1))
        pafs = generate_paf(all_keypoints, range(batch-1, batch+batch_size-1))
    
        yield images, conf_maps, pafs
    
    # Handle cases where the total size is not a multiple of batch_size
    
    if len(all_keypoints.keys()) % batch_size != 0:
        
        start_index = batch_size * batch_count
        final_index = list(all_keypoints.keys())[-1]
        
#        print(final_index + 1 - start_index)
        
        images = np.zeros((final_index + 1 - start_index, im_width, im_height, 3), dtype=np.uint8)
        
        for image_id in range(start_index, final_index + 1):
            img_name = get_image_name(image_id)
            
            if val:
                img = cv2.imread(os.path.join(dataset_dir, 'new_val2017', img_name))
            else:
                img = cv2.imread(os.path.join(dataset_dir, 'new_train2017', img_name))
            
            images[image_id % batch_size] = img
            
        conf_maps = generate_confidence_maps(all_keypoints, range(start_index, final_index + 1))
        pafs = generate_paf(all_keypoints, range(start_index, final_index + 1))
        
        yield images, conf_maps, pafs
        
# -------EVALUATION FUNCTIONS-------------------------
def find_joints(confidence_maps, threshold = 0.7):
    """
    Finds the list of peaks with the confidence scores from a confidence map 
    for an image. The confidence map has all the heatmaps for all joints.
    Inputs:
        confidence_map: A confidence map for all joints of a single image.
            expected shape: (1, im_width, im_height, num_joints)
        threshold: A number less than one used to eliminate low probability 
            detections.
    Output:
        joints_list: A list of all the detected joints. 
        It is a list of (num_joints) where each element is a list of detected 
        joints. Each joint is of following format: (x, y, confidence_score)
    """
    
    import numpy as np
    from scipy.ndimage.filters import maximum_filter
    from scipy.ndimage.morphology import generate_binary_structure
    
    # Check if the input is of expected shape
    assert len(confidence_maps.shape) == 3, "Wrong input confidence map shape."
    
    joints_list = []
    
    for joint_num in range(confidence_maps.shape[2]):
        # Detected joints for this type
        joints = []
        
        # Get the map for the joint and reshape to 2-d
        conf_map = confidence_maps[:, :, joint_num].reshape(confidence_maps.shape[0], confidence_maps.shape[1])
        # Threshold the map to eliminate low probability locations.
        conf_map = np.where(conf_map >= threshold, conf_map, 0)
        # Apply a 2x1 convolution(kind of).
        # This replaces a 2x1 window with the max of that window.
        peaks = maximum_filter(conf_map.astype(np.float64), footprint=generate_binary_structure(2, 1))
        peaks = np.where(peaks == 0, 0.1, peaks)
        # Now equate the peaks with joint_one.
        # This works because the maxima in joint_one will be at a single place and
        # equating with peaks will result in a single peak with all others as 0. 
        peaks = np.where(peaks == conf_map, peaks, 0)
        y_indices, x_indices = np.where(peaks != 0)

        for x, y in zip(x_indices, y_indices):
            joints.append((x, y, conf_map[y, x]))
        
        joints_list.append(joints)
        
    return 
