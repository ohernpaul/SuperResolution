# -*- coding: utf-8 -*-

from PIL import Image
import os

im_folder = 'Z:\SuperResolution\Outputs\DRCNN_Baisc\\'

ims = os.listdir(im_folder)

ims = [Image.open(im_folder + x) for x in ims if 'test' not in x and int(x.split('.')[0]) % 2 == 0]

ims[0].save(im_folder + 'test_1.gif',
               save_all=True, append_images=ims[1:], optimize=False, duration=700, loop=0)