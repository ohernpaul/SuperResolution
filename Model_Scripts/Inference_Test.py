# -*- coding: utf-8 -*-

import torch
from models import Skip_SRCNN
from utils import tileInferenceImage, untileInferenceImage
import matplotlib.pyplot as plt

img_path = 'Z:\SuperResolution\SR_Datasets\BSDS100\\37073.png'
img_path = 'Z:\SuperResolution\SR_Datasets\BSDS100\\103070.png'

hr_tiles, lr_tiles, og_shape = tileInferenceImage(img_path)

hr_out = untileInferenceImage(hr_tiles, og_shape)

model_type = 'Skip_SRCNN'
    
fig_path = 'Z:\SuperResolution\Outputs\\' + model_type + '\\' 

model_path = 'Z:\SuperResolution\Models\Skip_SRCNN\\' + model_type + '.pt' + '\\' 

model = Skip_SRCNN(num_channels=3)
model.load_state_dict(torch.load(model_path))
model.eval()

outs = []
for tile in lr_tiles:
    tile = torch.Tensor(tile).permute(2,0,1).unsqueeze(0)
    temp_out = model(tile)
    outs.append(temp_out.squeeze().permute(1,2,0).detach().cpu().numpy())
    
model_out = untileInferenceImage(outs, og_shape)
lr_out = untileInferenceImage(lr_tiles, og_shape)

fig, axs = plt.subplots(3)
axs[0].set_title("HR")
axs[0].imshow(hr_out)
axs[1].set_title("Model Out")
axs[1].imshow(model_out)
axs[2].set_title("LR")
axs[2].imshow(lr_out)

fig.tight_layout()
plt.savefig(fig_path + 'Tile_Test', fig_size=(15,15), dpi=1500)
