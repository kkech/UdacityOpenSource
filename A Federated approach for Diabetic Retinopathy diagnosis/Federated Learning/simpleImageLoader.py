from __future__ import print_function, division
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform
import numpy as np

class simpleImageLoader(Dataset):
    """Eye disease dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.labels.iloc[idx, 0])
        image = io.imread(img_name +'.jpg')
        label = self.labels.iloc[idx, 1:]
        label = np.array([label])
        label = label.astype('int')

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.from_numpy(label), self.labels.iloc[idx, 0]
