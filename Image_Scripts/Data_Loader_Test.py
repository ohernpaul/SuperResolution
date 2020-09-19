# -*- coding: utf-8 -*-
"""
This script is an exp script for dataloaders
"""
import os
import numpy as np
from skimage import io, transform
import cv2
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

sr_destination_base = 'Z:/SuperResolution/Labeled_Tiled_Datasets/BSDS100\Scale_4/'

class ImageLabelDataset(Dataset):
    
    def __init__(self, data_path, resize=True, transform=None):
        self.x_dir = data_path + 'Inputs/'
        self.y_dir = data_path + 'Labels/'
        self.x_imgs = os.listdir(self.x_dir)
        self.y_imgs = os.listdir(self.y_dir)
        self.transform = transform
        self.resize = resize
        
        
    def __len__(self):
        assert len(self.x_imgs) == len(self.y_imgs), "Sizes of Labels and Inputs are not the same!"
        return len(self.y_imgs)
        
    def __getitem__(self, idx):
        x = io.imread(self.x_dir + self.x_imgs[idx])
        y = io.imread(self.y_dir + self.y_imgs[idx])
        
        if self.resize:
            #bi-cubic interpolation to target size
            x = cv2.resize(x, dsize=(y.shape[0], y.shape[1]), interpolation=cv2.INTER_CUBIC)
        
        sample = {'input':x, 'label':y}
        
        if self.transform:
            #TODO: need to apply transform to each key
            #rather than the 'sample'
            sample = self.transform(sample)
        
        return sample
    
batch_size = 64
    
sr_dataset = ImageLabelDataset(sr_destination_base, transform=None)

sr_dataloader = DataLoader(sr_dataset, batch_size=batch_size,
                           shuffle=True, num_workers=0)
            

t_batch = next(iter(sr_dataloader))

plt.imshow(np.array(t_batch['input'][0]))
plt.imshow(np.array(t_batch['label'][0]))