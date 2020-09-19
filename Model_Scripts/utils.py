# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
from skimage import io
from torch.utils.data import Dataset
import PIL.Image as pil_image
from PIL import Image
import matplotlib.pyplot as plt

def plotLosses(train, test, psnr, ssim, path):
    fig_l, ax_l = plt.subplots(4)
    
    ax_l[0].plot(train, color='blue')
    ax_l[0].set_title("Train Loss")
    
    ax_l[1].plot(test, color='red')
    ax_l[1].set_title("Test Loss")
    
    ax_l[2].plot(psnr)
    ax_l[2].set_title("Test Avg PSNR")
    
    ax_l[3].plot(ssim)
    ax_l[3].set_title("Test Avg SSIM")
    
    fig_l.tight_layout()
    fig_l.savefig(path + "test_metrics" + '.png', dpi=800)

class ImageLabelDataset(Dataset):
    """
    Reads in input and label pairs from a directory, creates the sample dict,
    and applies any transforms that were passed in.
    
    -The input images have already been downscaled and (early) upscaled in a
    creation of the tile dataset.
    """
    
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
        #normalize RGB to 0-1
        x = io.imread(self.x_dir + self.x_imgs[idx])/255
        y = io.imread(self.y_dir + self.y_imgs[idx])/255
        
        if self.resize:
            #bi-cubic interpolation to target size if wasnt done in tile step
            x = cv2.resize(x, dsize=(y.shape[0], y.shape[1]), interpolation=cv2.INTER_CUBIC)
        
        sample = {'input':x.astype(np.float32), 'label':y.astype(np.float32)}
        
        if self.transform:
            #TODO: set up for torch tranform compositions
            for k in sample.keys():
                sample[k] = self.transform(sample[k])
        
        return sample
    
def cropCenter(img,cropx,cropy):
    """Used when padding isnt used and output is smaller than input/target """
    b,c,x,y = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[:,:,starty:starty+cropy,startx:startx+cropx]

def checkFolder(path):
    if not os.path.exists(path):
        os.mkdir(path)

def tileInferenceImage(img_path, scale=3, tile=41, n_channels=3):
    """
    Does the same processing as in dataset creation.
    
    -Read image and downscale to create low resolution
    -Rescale Low Resolution image size of HR
    -Pass both images into tiling function
    """
    
    def getTiles(im):
        tiles = []
        stride = tile
        for i in range(0, int(hr.shape[0]/stride)):
            start_i = (i * stride)
            stop_i = (start_i + tile)
            
        
            for j in range(0, int(hr.shape[1]/stride)):
    
                start_j = (j * stride)
                stop_j = (start_j + tile)
            
                temp_tile = im[start_i: stop_i, start_j: stop_j, :]
                
                #TODO: Create option for zero padding (doesnt collect partial tiles)
                if temp_tile.shape[0] != temp_tile.shape[1] or temp_tile.shape != (tile, tile, n_channels):
                    continue
                
                tiles.append(temp_tile)
        
        return tiles
        
    
    im = Image.open(img_path)
    
    hr_width = (im.width // scale) * scale
    hr_height = (im.height // scale) * scale
    hr = im.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
    lr = hr.resize((hr_width // scale, hr_height // scale), resample=pil_image.BICUBIC)
    lr = lr.resize((lr.width * scale, lr.height * scale), resample=pil_image.BICUBIC)
    hr = np.array(hr).astype(np.float32)/255
    lr = np.array(lr).astype(np.float32)/255
    
    return getTiles(hr), getTiles(lr), (im.width, im.height)


def untileInferenceImage(tile_arr, og_shape):
    """
    Recreates original size of input image
    """
    #im = Image.open(img_path)
    
    w = og_shape[0]
    
    tile_shape = tile_arr[0].shape[0]
    w_lim = (w//tile_shape) - 1
        
    temp_row = None
    final_out = None
    
    row_count = 0
    
    for tile in tile_arr:
        if temp_row is None:
            temp_row = tile
            row_count += 1
            
        elif temp_row is not None and row_count < w_lim:
            temp_row = np.concatenate([temp_row, tile], axis=1)
            row_count += 1
            
        else:
            if final_out is None:
                final_out = temp_row
            else:
                final_out = np.concatenate([final_out, temp_row], axis=0)
            temp_row = None
            row_count = 0
        
    return final_out
    