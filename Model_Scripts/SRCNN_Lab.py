# -*- coding: utf-8 -*-

import os
from torch.utils.data import DataLoader
from torch import nn
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from utils import ImageLabelDataset, checkFolder, plotLosses
from models import SRCNN
from torchvision import transforms

def do_test():

    device = "cuda:0"
    
    #set device
    
    sr_destination_base = 'Z:/SuperResolution/Labeled_Tiled_Datasets_Fix/BSDS100\Scale_3/'
    
    test_base_200 = 'Z:/SuperResolution/Labeled_Tiled_Datasets_Fix/BSDS200\Scale_3/'
    
    model_type = 'Skip_SRCNN'
    
    out_base = 'Z:\SuperResolution\Outputs\\' + model_type + '\\' 
    model_base = 'Z:\SuperResolution\\Models\\' + model_type + '\\' 
    
    checkFolder(out_base)
    checkFolder(model_base)
    
    #Hyper Params
    #===================
    epochs = 500
    batch_size = 64
    workers = 0
    
    force_test = False
    test_output = 10
    #==================
    
    train_trans = transforms.Compose([
                                      #transforms.ToPILImage(),
                                      #transforms.ToTensor(),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.RandomRotation(90),
                                      transforms.ToTensor(),
                                    ])
        
    sr_dataset = ImageLabelDataset(sr_destination_base, transform=train_trans, resize=False)
    
    sr_dataloader = DataLoader(sr_dataset, batch_size=batch_size,
                               shuffle=True, num_workers=workers, drop_last=True)
    
    test_dataset_200 = ImageLabelDataset(test_base_200, transform=transforms.ToTensor(), resize=False)
    
    test_dataloader_200 = DataLoader(test_dataset_200, batch_size=batch_size,
                               shuffle=False, num_workers=workers, drop_last=True)
    #==================    
    
    #model = Basic_DRC(n_channels = 3)
    model = SRCNN(num_channels=3, do_skip=True)
    model = model.to(device)
    
    mse = nn.MSELoss()
    
    #SGD optimizer where each layer has their own weights
    # opt = torch.optim.SGD(params=[{'params': model.conv1.parameters(), 'lr': 10e-4},
    #                               {'params': model.conv2.parameters(), 'lr': 10e-4},
    #                               {'params': model.conv3.parameters(), 'lr': 10e-5}],
    #                       momentum=0.9)
    
    opt = torch.optim.Adam(params=[{'params': model.conv1.parameters(), 'lr': 1e-4},
                                  {'params': model.conv2.parameters(), 'lr': 1e-4},
                                  {'params': model.conv3.parameters(), 'lr': 1e-5}])
    
    #==================
    
    train_loss_list = []
    test_loss_list = []
    test_psnr_list = []
    test_ssim_list = []
    
    avg_loss = 0
    
    epoch_imgs = []
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
            
        epoch_imgs.append(out[0].permute(1,2,0).detach().cpu().numpy())
            
        epoch_train_loss = avg_loss/len(sr_dataloader)
        train_loss_list.append(epoch_train_loss)
        print("Train Loss: " + str(epoch_train_loss))
        
        avg_test_loss = 0
        avg_psnr = 0
        avg_ssim = 0
        avg_loss = 0
    
    
        
        if e % test_output == 0 or force_test:
            print("Testing Epoch: " + str(e))
            for i, sample in tqdm(enumerate(test_dataloader_200, 0), total=len(test_dataloader_200)):
                model.eval()
                
                x = sample['input'].to(device)
                y = sample['label'].to(device)
                
                opt.zero_grad()
                
                out = model(x)
                
                if out.dtype != y.dtype:
                    print("Dtype mixmatch")
                if out.shape != y.shape:
                    print("shape mismatch")
                
                loss = mse(out, y)
                
                avg_test_loss += loss.item()
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
            
            plotLosses(train_loss_list,
                       test_loss_list,
                       test_psnr_list,
                       test_ssim_list,
                       out_base
                       )
    
            torch.save(model.state_dict(), model_base + model_type +'.pt')

if __name__ == '__main__':
    do_test()


