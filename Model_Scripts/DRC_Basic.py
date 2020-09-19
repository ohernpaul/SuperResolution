# -*- coding: utf-8 -*-

import os

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from torch.utils.data import DataLoader
from torch import nn
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from torch.optim import lr_scheduler

from models import Basic_DRC, advancedLoss, advancedLoss_Norm
from utils import ImageLabelDataset, checkFolder

def do_test():

    device = "cuda:0"
    
    #set device
    
    sr_destination_base = 'Z:/SuperResolution/Labeled_Tiled_Datasets_Fix/BSDS200\Scale_3/'
    
    test_base_200 = 'Z:/SuperResolution/Labeled_Tiled_Datasets_Fix/BSDS100\Scale_3/'
    
    out_base = 'Z:\SuperResolution\Outputs\DRCNN_Baisc\\'
    
    checkFolder(out_base)
    
        
    #training online for a whole day    
    batch_size = 32
    epochs = 500
    momentum = 0.9
    decay = 0.0001
    
    workers = 4
        
    sr_dataset = ImageLabelDataset(sr_destination_base, transform=transforms.ToTensor(), resize=False)
    
    sr_dataloader = DataLoader(sr_dataset, batch_size=batch_size,
                               shuffle=True, num_workers=workers, drop_last=True)
    
    test_dataset_200 = ImageLabelDataset(test_base_200, transform=transforms.ToTensor(), resize=False)
    
    test_dataloader_200 = DataLoader(test_dataset_200, batch_size=batch_size,
                               shuffle=False, num_workers=workers, drop_last=True)
    
            
    model = Basic_DRC(n_recursions = 8, n_channels = 3)
    model = model.to(device)
        
    mse = nn.MSELoss()
    
    
    #SGD optimizer where each layer has their own weights
    opt = torch.optim.SGD(params=[{'params': model.parameters(), 'lr': 0.01},],
                          momentum=momentum,
                          weight_decay=decay)
    
    
    sched = lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.001, patience=5, min_lr=10e-6)
    
    
    
    avg_loss = 0
    avg_test_loss = 0
    
    train_loss_list = []
    test_loss_list = []
    test_psnr_list = []
    test_ssim_list = []
    
    for e in range(epochs):
        #print("Train Epoch: " + str(e))
        for i, sample in tqdm(enumerate(sr_dataloader, 0), total=len(sr_dataloader)):
            model.train()
            
            x = sample['input'].to(device)
            y = sample['label'].to(device)
            
            opt.zero_grad()
            
            out = model(x)
            
            loss = mse(out, y).to(device)
            
            
            avg_loss += loss.item()
            
            loss.backward()
            opt.step()
            
        epoch_train_loss = avg_loss/len(sr_dataloader)
        train_loss_list.append(epoch_train_loss)
        print("Train Loss: " + str(epoch_train_loss))
        avg_loss = 0
        avg_psnr = 0
        avg_ssim = 0
    
        force_test = False
        if e % 10 == 0 or force_test:
            with torch.no_grad():
                print("Testing Epoch: " + str(e))
                for i, sample in tqdm(enumerate(test_dataloader_200, 0), total=len(test_dataloader_200)):
                    model.eval()
                    
                    x = sample['input'].to(device)
                    y = sample['label'].to(device)
                    
                    opt.zero_grad()
                    
                    out = model(x)
                    
                    test_loss = mse(out, y).to(device)
                    sched.step(test_loss)
                    
                    if out.dtype != y.dtype:
                        print("Dtype mixmatch")
                    if out.shape != y.shape:
                        print("shape mismatch")
                    
                    avg_test_loss += test_loss.item()
                    
                    avg_ssim += ssim(y.permute(0,2,3,1).detach().cpu().numpy(), out.permute(0,2,3,1).detach().cpu().numpy(), multichannel=True)
                    avg_psnr += psnr(y.detach().cpu().numpy(), out.detach().cpu().numpy())
                    
                    if i == 50:
                        t_o = out[0].permute(1,2,0).detach().cpu().numpy()
                        
                        t_y = y[0].permute(1,2,0).detach().cpu().numpy()
                        t_x = x[0].permute(1,2,0).detach().cpu().numpy()
        
                epoch_test_loss = avg_test_loss/len(test_dataloader_200)
                
                avg_ssim /= len(test_dataloader_200)
                avg_psnr /= len(test_dataloader_200)
                
                
                test_loss_list.append(epoch_test_loss)
                test_psnr_list.append(avg_psnr)
                test_ssim_list.append(avg_ssim)
                
                print("Test Loss: " + str(epoch_test_loss))
                print("Avg SSIM: " + str(avg_ssim))
                print("Avg PSNR: " + str(avg_psnr))
                
                avg_test_loss = 0
                
                fig, ax = plt.subplots(3)
                        
                ax[0].imshow(t_y)
                ax[1].imshow(t_x)
                ax[2].imshow(t_o)
                
                
                nb_out = len(os.listdir(out_base))
                fig.savefig(out_base + str(nb_out) + '.png', dpi=800)
                
                fig_l, ax_l = plt.subplots(4)
                
                ax_l[0].plot(train_loss_list, color='blue')
                ax_l[0].set_title("Train Loss")
                
                ax_l[1].plot(test_loss_list, color='red')
                ax_l[1].set_title("Test Loss")
                
                ax_l[2].plot(test_psnr_list)
                ax_l[2].set_title("Test Avg PSNR")
                
                ax_l[3].plot(test_ssim_list)
                ax_l[3].set_title("Test Avg SSIM")
                
                fig_l.tight_layout()
                
                fig_l.savefig(out_base + "test_metrics" + '.png', dpi=800)

if __name__ == '__main__':
    do_test()



