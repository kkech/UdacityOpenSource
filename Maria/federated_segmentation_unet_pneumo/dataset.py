import os
import numpy as np
import pandas as pd
from PIL import Image
import pydicom
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    """
    A class for importing Dicom files and RLE masks into a 
    PyTorch Dataset for semantic segmentation with images and 
    binary masks in PIL.Image format.
    Params:
    root: directory with input Dicom files and the RLE masks file
    rle_filename: the name of the CSV file containing RLE masks with
    corresponding image file basenames
    """
    def __init__(self, root, rle_filename, input_transform=None, target_transform=None):
        self.images_root = root
        self.rle_filename = rle_filename
        
        self.id_to_rle = {}
        
        self.data = []
        self.targets = []
        
        self._fill_id2rle()
        self._fill_data()

        self.input_transform = input_transform
        self.target_transform = target_transform
        
    def _transform_to_mask(self, x):
        if x == "-1":
            return np.zeros((1024, 1024))
        else: 
            return self.rle2mask(x, 1024, 1024) 

    @staticmethod
    def rle2mask(rle, width, height):
        # Taken from mask_functions.py 
        # @ https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/data 
        
        mask= np.zeros(width* height)
        array = np.asarray([int(x) for x in rle.split()])
        starts = array[0::2]
        lengths = array[1::2]

        current_position = 0
        for index, start in enumerate(starts):
            current_position += start
            mask[current_position:current_position+lengths[index]] = 255
            current_position += lengths[index]
        return mask.reshape(width, height)
        
        
    def _fill_id2rle(self):
        df = pd.read_csv(os.path.join(self.images_root, self.rle_filename), 
                         names=["id", "rle"], header=None)
        for i, row in df.iterrows():
            self.id_to_rle[row["id"]] = row["rle"]
        

    def _fill_data(self):
        for image_id, rle in self.id_to_rle.items():
            file_path = os.path.join(self.images_root, f"{image_id}.dcm")
            dataset = pydicom.dcmread(file_path)
            
            inputs = np.stack((dataset.pixel_array,)*3, axis=-1)
            inputs = Image.fromarray(inputs)
            self.data.append(inputs) 
            
            mask = self._transform_to_mask(rle)
            mask[mask == 255] = 1
            mask = Image.fromarray(mask)
            self.targets.append(mask)
    
    
    def __getitem__(self, index):
        image = self.data[index]
        target = self.targets[index]

        if self.input_transform is not None:
            image = self.input_transform(image)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target
    
    def __len__(self):
        return len(self.data)
