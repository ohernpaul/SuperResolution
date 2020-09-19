# -*- coding: utf-8 -*-

"""
This script is used to create the low resolution inputs to the model
from the high resolution images (that are used as labels in SR)

Multiple datasets are created from different scaling factors

All rescaled images use anti-aliasing and bi-cubic interpolation

"""

import os
from skimage import io
import PIL.Image as pil_image
import numpy as np

def checkFolder(path):
    if not os.path.exists(path):
        os.mkdir(path)
        
def saveTiles(src_path, im_name, im, scale, stride=21, tile=41, n_channels=3):
    stride = int(tile/2)
    im_name, file_type = im_name.split('.')[:]
    count = 0
    
    hr_width = (im.width // scale) * scale
    hr_height = (im.height // scale) * scale
    hr = im.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
    lr = hr.resize((hr_width // scale, hr_height // scale), resample=pil_image.BICUBIC)
    lr = lr.resize((lr.width * scale, lr.height * scale), resample=pil_image.BICUBIC)
    hr = np.array(hr).astype(np.float32)
    lr = np.array(lr).astype(np.float32)    
    
    for i in range(0, int(hr.shape[0]/stride)):
        start_i = (i * stride)
        stop_i = (start_i + tile)
        
    
        for j in range(0, int(hr.shape[1]/stride)):

            start_j = (j * stride)
            stop_j = (start_j + tile)
            
            temp_tile_hr = hr[start_i: stop_i, start_j: stop_j, :]
            temp_tile_lr = lr[start_i: stop_i, start_j: stop_j, :]
            
            if temp_tile_hr.shape[0] != temp_tile_hr.shape[1] or temp_tile_hr.shape != (tile, tile, n_channels):
                continue
            
            io.imsave(src_path + 'Inputs/' + im_name + '_' + str(count) + '.' + file_type, temp_tile_lr)
            io.imsave(src_path + 'Labels/' + im_name + '_' + str(count) + '.' + file_type, temp_tile_hr)
            
            count += 1

    return
        
#=================       

#================= 
    
stride = 21
tile = 41

scale_factors = {'2':2, '3':3, '4':4}

#parent SR source data dir
sr_source_base = 'Z:/SuperResolution/SR_Datasets/'

#list all folders
source_folders = os.listdir(sr_source_base)
source_folders = [x for x in source_folders if x != 'Pre_Labeled']

#parent destination dir
sr_destination_base = 'Z:/SuperResolution/Labeled_Tiled_Datasets_Fix/'

checkFolder(sr_destination_base)


#for each folder
for ds in source_folders:
    
    temp_ds_path = sr_destination_base + ds + '/'
    
    checkFolder(temp_ds_path)
    
    #create folders for each scale factor
    for k,scale in scale_factors.items():
        
        temp_scale_path = temp_ds_path + 'Scale_' + k + '/'
        
        checkFolder(temp_scale_path)
        
        #create new destination subfolders (SR/LR)
        checkFolder(temp_scale_path + 'Labels')
        checkFolder(temp_scale_path + 'Inputs')
        
        #get images in source dataset
        temp_src_imgs = os.listdir(sr_source_base + ds + '/')
        
        #for each img
        for im_name in temp_src_imgs:
            im_src_path = sr_source_base + ds + '/' + im_name
            try:
                og_img = pil_image.open(im_src_path).convert('RGB')
            except:
                print()
            
            #send HR and LR to tile function
            saveTiles(temp_scale_path, im_name, og_img, scale, stride=stride, tile=tile)
    
