# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 14:28:39 2019

@author: myidispg
"""

from torch.utils.data import Dataset

import numpy as np
import os
import cv2

from utilities.constants import img_size, transform_scale
from data_process.process_functions import generate_confidence_maps, normalize, adjust_keypoints
from data_process.process_functions import generate_paf, do_affine_transform, add_neck_joint
from utilities.helper import get_image_name


class StanceNetDataset(Dataset):
    """
    Custom Dataset to get objects in batches while training. 
    Must be used along with Dataloaders.
    """
    
    def __init__(self, coco, img_dir):
        """
        Args:
            coco: A COCO object for the annotations. Used for mask generation.
            img_dir: The path to the images directory. Used to differentiate
                between train and validation images.
        """
        self.img_dir = img_dir
        self.coco = coco
        
        # We have a COCO data object for a person. Get the category id for a person.
        person_ids = self.coco.getCatIds(catNms=['person'])
        # Get the ids of all images with a person in it.
        self.img_indices = sorted(self.coco.getImgIds(catIds=person_ids))
    
    def __len__(self):
        # The length will be equal to count of those images that have a person.
        return len(self.img_indices)
    
    def __getitem__(self, idx):
        """
        Given the id of an image, return the image and the corresponding 
        confidence maps and the parts affinity fields.
        """
        # Get a specific image id from the list of image ids.
        img_index = self.img_indices[idx]
        
        # Load the image
        img_name = os.path.join(self.img_dir, get_image_name(img_index))
        img = cv2.imread(img_name).transpose(1, 0, 2)/255
        original_shape = img.shape[:2]
        # Resize image to 400x400 dimensions.
        img = cv2.resize(normalize(img), (img_size, img_size))
        # Get the annotation id of the annotaions about the image.
        annotations_indices = self.coco.getAnnIds(img_index)
        # Load the annotations from the annotaion ids.
        annotations = self.coco.loadAnns(annotations_indices) 
        keypoints = []
        mask = np.zeros((img_size // transform_scale, img_size // transform_scale),
                        np.uint8)
        for annotation in annotations:
            if annotation['num_keypoints'] != 0:
                keypoints.append(annotation['keypoints'])
            mask = mask | cv2.resize(self.coco.annToMask(annotation),
                                     (img_size // transform_scale,
                                      img_size // transform_scale))
            
        # Add neck joints to the list of keypoints
        keypoints = add_neck_joint(keypoints)
        # Adjust keypoints according to resized images.
        keypoints = adjust_keypoints(keypoints, original_shape)
        
        conf_maps = generate_confidence_maps(keypoints)
        paf = generate_paf(keypoints)
        paf = paf.reshape(paf.shape[0], paf.shape[1], paf.shape[2] * paf.shape[3])
        
        return img, conf_maps, paf, mask.transpose()
        
        