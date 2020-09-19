# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 21:14:13 2020

@author: Bah
"""
import matplotlib.pyplot as plt
import os
from skimage import transform


sr_data_path = 'Z:\\SuperResolution\\SR_Datasets\\'

general_100_path = sr_data_path + 'General100\\'

g_100_imgs = os.listdir(general_100_path)

#get a single image
temp_img = plt.imread(general_100_path + g_100_imgs[0])

plt.imshow(temp_img)

#downsample it with imresize function
temp_img_resized = transform.rescale(temp_img, 0.24, multichannel=True, anti_aliasing=True, )
                                                       
fig, ax = plt.subplots(2)
ax[0].imshow(temp_img)
ax[1].imshow(temp_img_resized)