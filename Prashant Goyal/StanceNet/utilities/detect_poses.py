# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 17:47:52 2019

@author: myidispg
"""

import numpy as np
from scipy.ndimage.filters import gaussian_filter
import cv2

from utilities.constants import BODY_PARTS

def find_joint_peaks(heatmap, orig_img_shape, threshold):
    """
    Given a heatmap, find the peaks of the detected joints with 
    confidence score > threshold.
    Inputs:
        heatmap: The heatmaps for all the joints. It is of shape: h, w, num_joints.
        orig_img_shape: The shape of the original image: (x, y, num_channels)
        threshold: The minimum score for each joint
    Output:
        A list of the detected joints. There is a sublist for each joint 
        type (18 in total) and each sublist contains a tuple with 4 values:
        (x, y, confidence score, unique_id). The number of tuples is equal
        to the number of joints detected of that type. For example, if there
        are 3 nose detected (joint type 0), then there will be 3 tuples in the 
        nose sublist            
    """
    joints_list = list()
    counter = 0 # This will serve as unique id for all joints
    heatmap = cv2.resize(heatmap, (orig_img_shape[1], orig_img_shape[0]),
                                    interpolation=cv2.INTER_CUBIC)
    for i in range(heatmap.shape[2]-1):
        sub_list = list()
        joint_map = heatmap[:, :, i] # the heatmap for the particular joint type
        # Smoothen the heatmap
        joint_map = gaussian_filter(joint_map, sigma=3)
        map_left = np.zeros(joint_map.shape)
        map_left[1:, :] = joint_map[:-1, :]
        map_right = np.zeros(joint_map.shape)
        map_right[:-1, :] = joint_map[1:, :]
        map_up = np.zeros(joint_map.shape)
        map_up[:, 1:] = joint_map[:, :-1]
        map_down = np.zeros(joint_map.shape)
        map_down[:, :-1] = joint_map[:, 1:]
        
        peaks_binary = np.logical_and.reduce((joint_map >= map_left,
                                              joint_map >= map_right,
                                              joint_map >= map_up,
                                              joint_map >= map_down,
                                              joint_map > threshold))
        x_index, y_index = np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]
        for x, y in zip(x_index, y_index):
            confidence = joint_map[y, x]
            sub_list.append((x, y, confidence, counter))
            counter += 1
        
        joints_list.append(sub_list)
    return joints_list
       
def get_connected_joints(upsampled_paf, joints_list, num_inter_pts = 10):
    """
    For every type of limb (eg: forearm, shin, etc.), look for every potential
    pair of joints (eg: every wrist-elbow combination) and evaluate the PAFs to
    determine which pairs are indeed body limbs.
    Inputs:
        upsampled_paf: PAFs upsampled to the original image size.
        joints_list: The ist of joints made by the find_joints_peaks()
        num_inter_pts: The number of points to consider to integrate the PAFs
            and give score to connection candidate
    """
    
    connected_limbs = []
    
    limb_intermed_coords = np.empty((4, num_inter_pts), dtype=np.int)
    for limb_type in range(len(BODY_PARTS)):
        # List of all joints of source
        joints_src = joints_list[BODY_PARTS[limb_type][0]]
        # List of all joints of destination
        joints_dest = joints_list[BODY_PARTS[limb_type][1]]
        if len(joints_src) == 0 or len(joints_dest) == 0:
            # No limbs of this type found. For example if no left knee or no left waist
            # found, then there is no limb connection possible.
            connected_limbs.append([])
        else: 
            connection_candidates = [] # list for all candiates of this type.
            # Specify the paf index that contains the x-coord of the paf for
            # this limb
            limb_intermed_coords[2, :] = 2*limb_type
            # And the y-coord paf index
            limb_intermed_coords[3, :] = 2*limb_type + 1
            
            for i, joint_src in enumerate(joints_src):
                for j, joint_dest in enumerate(joints_dest):
                    # Subtract the position of both joints to obtain the
                    # direction of the potential limb
                    x_src, y_src = joint_src[0], joint_src[1]
                    x_dest, y_dest = joint_dest[0], joint_dest[1]
                    limb_dir = np.asarray([x_dest - x_src, y_dest - y_src])
                    # Compute the norm of the potential limb
                    limb_dist = np.sqrt(np.sum(limb_dir ** 2)) + 1e-8
                    limb_dir = limb_dir / limb_dist  # Normalize limb_dir to be a unit vector
                    
                    # Get intermediate points between the source and dest
                    # For x coordinate
                    limb_intermed_coords[0, :] = np.round(np.linspace(
                            x_src, x_dest, num=num_inter_pts))
                    # For y coordinate
                    limb_intermed_coords[1, :] = np.round(np.linspace(
                            y_src, y_dest, num=num_inter_pts))
                    # Get all the intermediate points
                    intermed_paf = upsampled_paf[limb_intermed_coords[1,: ],
                                                 limb_intermed_coords[0,:],
                                                 limb_intermed_coords[2:4, :]
                                                 ].transpose()
                    score_intermed_points = intermed_paf.dot(limb_dir)
                    score_penalizing_long_dist = score_intermed_points.mean() + min(0.5 * upsampled_paf.shape[0] / limb_dist - 1, 0)
                    
#                    # Criterion 1: At least 80% of the intermediate points have
#                    # a score higher than thre2
#                    criterion1 = (np.count_nonzero(
#                        score_intermed_points > 0.05) > 0.8 * num_inter_pts)
#                    # Criterion 2: Mean score, penalized for large limb
#                    # distances (larger than half the image height), is
#                    # positive
#                    criterion2 = (score_penalizing_long_dist > 0)
#                    if criterion1 and criterion2:
#                        # Last value is the combined paf(+limb_dist) + heatmap
#                        # scores of both joints
#                        connection_candidates.append([joint_src[3], joint_dest[3],
#                                                      score_penalizing_long_dist,
#                                                      (x_src, y_src),
#                                                      (x_dest, y_dest)])

#                    
                    connection_candidates.append([joint_src[3], joint_dest[3],
                                                      score_penalizing_long_dist,
                                                      (x_src, y_src),
                                                      (x_dest, y_dest)])
                    
#            Sort the connections based on their score_penalizing_long_distance
#            Key is used to specify which element to consider while sorting.
            connection_candidates = sorted(connection_candidates, key=lambda x: x[2],
                                           reverse=True) # Reverse to use descending order.
            
            used_idx1 = [] # A list to keep track of the used src joints
            used_idx2 = [] # A list to keep track of the used dest joints
            
            connection = [] # A list to store all the final connection of limb type.
            
            for potential_connection in connection_candidates:
                if potential_connection[0] in used_idx1 or potential_connection[1] in used_idx2:
                    continue
                
                connection.append(potential_connection)
                used_idx1.append(potential_connection[0])
                used_idx2.append( potential_connection[1])
            
            connected_limbs.append(connection)
    return connected_limbs

