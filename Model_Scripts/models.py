# -*- coding: utf-8 -*-
import torch
from torch import nn

class Basic_DRC(nn.Module):
    
    class Block(nn.Module):
        def __init__(self, n_channels, kernel_size, filter_size, stride):
            self.block = nn.Sequential(nn.Conv2d(n_channels, filter_size, kernel_size = kernel_size, stride=stride, padding=kernel_size[0]//2),
                                           nn.ReLU(),
                                           nn.Conv2d(filter_size, filter_size, kernel_size = kernel_size, stride=stride, padding=kernel_size[0]//2),
                                           nn.ReLU()
                                           )
        
        def forward(self, x):
            return self.block(x)
    
    def __init__(self, n_recursions, n_channels, filter_size=256, kernel_size=(3,3), stride=(1,1), do_init=True):
        super(Basic_DRC, self).__init__()
        
        def initWeights(model):
            if type(model) == nn.Conv2d:
                nn.init.kaiming_normal_(model.weight)
                
        self.n_recursions = n_recursions
        self.n_channels = n_channels
        
        self.embed_net = self.Block(n_channels, kernel_size, filter_size, stride)
        
        self.inference_net = self.Block(filter_size, kernel_size, filter_size, stride)
        
        self.recon_net = self.Block(filter_size, kernel_size, filter_size, stride)
        
        #init_weights
        if do_init:
            self.embed_net.apply(initWeights)
            self.recon_net.apply(initWeights)
        
        
    def forward(self, x):
        
        h_0 = self.embed_net(x)
        
        for n in range(self.n_recursions):
            if n == 0:
                h_d = self.inference_net(h_0)
            else:
                h_d = self.inference_net(h_d)
        
        h_d1 = self.recon_net(h_d)
        
        return h_d1
    
class Advanced_DRC(nn.Module):
    
    class Block(nn.Module):
        def __init__(self, n_channels, kernel_size, filter_size, stride):
            self.block = nn.Sequential(nn.Conv2d(n_channels, filter_size, kernel_size = kernel_size, stride=stride, padding=kernel_size[0]//2),
                                           nn.ReLU(),
                                           nn.Conv2d(filter_size, filter_size, kernel_size = kernel_size, stride=stride, padding=kernel_size[0]//2),
                                           nn.ReLU()
                                           )
        
        def forward(self, x):
            return self.block(x)
    
    def __init__(self, n_recursions, n_channels, filter_size=256, kernel_size=(3,3), stride=(1,1), do_init=True):
        super(Advanced_DRC, self).__init__()
        
        def initWeights(model):
            if type(model) == nn.Conv2d:
                nn.init.kaiming_normal_(model.weight)
                
        self.n_recursions = n_recursions
        self.n_channels = n_channels
        
        self.embed_net = self.Block(n_channels, kernel_size, filter_size, stride)
        
        self.inference_net = self.Block(filter_size, kernel_size, filter_size, stride)
        
        self.recon_net = self.Block(filter_size, kernel_size, filter_size, stride)
        
        #init_weights
        if do_init:
            self.embed_net.apply(initWeights)
            self.recon_net.apply(initWeights)
        
        
        #use nn.init.kaiming_normal_ (He Normal) on all params in each sequential
        self.embed_net.apply(initWeights)
        self.recon_net.apply(initWeights)
        
        self.avg_out = nn.Conv2d(self.n_recursions * n_channels, n_channels, kernel_size = (1,1), stride=stride)
        
        
        
        
    def forward(self, x):
        identity = x
        
        outs = torch.Tensor()
        
        h_0 = self.embed_net(x)
        
        #Each recursive output goes through reconstruction and gets skip connection
        for n in range(self.n_recursions):
            
            if n == 0:
                inf_out = self.inference_net(h_0)
                outs = self.recon_net(inf_out) + identity
                #outs = outs.unsqueeze(0)
            else:
                inf_out = self.inference_net(inf_out)
                #t_out = 
                #outs = torch.cat((outs, t_out.unsqueeze(0)), axis=1)
                outs = torch.cat((outs, self.recon_net(self.inference_net(inf_out)) + identity), axis=1)
            
        
        #each output image is summed at the end
        out = outs.reshape(int(outs.shape[1]/self.n_channels), outs.shape[0], self.n_channels, outs.shape[-1], outs.shape[-1]).mean(axis=0)
        #print(out.shape)
        
        #out = self.avg_out(outs)
            
        return out, outs.reshape(int(outs.shape[1]/self.n_channels), outs.shape[0], self.n_channels, outs.shape[-1], outs.shape[-1])

class SRCNN(nn.Module):
    
    def __init__(self, num_channels=1, init_mean=0, init_std=10e-4, bias_val=0, normal_init=False, do_skip=False):
        super(SRCNN, self).__init__()
        self.init_mean = init_mean
        self.init_std = init_std
        self.normal_init = normal_init
        self.do_skip = do_skip
        
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9//2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5//2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5//2)
        self.relu = nn.ReLU(inplace=True)
        
        if normal_init:
            self.initWeights()
            
    def initWeights(self):
        
        #Gaussian Init - 0 bias vals
        params = [self.conv1, self.conv2, self.conv3]
        
        [torch.nn.init.normal_(x.weight,
                                mean=self.init_mean, std=self.init_std) for x in params]
        
        [x.bias.data.fill_(self.bias_val) for x in params]
        
    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        if self.do_skip:
            x3 = self.conv3(x2) + x
        else:
            x3 = self.conv3(x2)
            
        return x3
    
        
def advancedLoss(out, rec_out, labels, alpha):
    #might not need to add decay to loss since opt already implements it on theta
    l1 = nn.L1Loss(reduction='mean')
    l2 = nn.MSELoss()

    tl1 = l1(labels.unsqueeze(0), rec_out)
    tl2 = l2(labels, out)
    
    final_loss = (alpha * tl1) + ((1-alpha)*tl2)

    return final_loss

def advancedLoss_Norm(out, rec_out, labels, alpha, beta, model):
    #might not need to add decay to loss since opt already implements it on theta
    l1 = nn.L1Loss(reduction='mean')
    l2 = nn.MSELoss()
    
    norm = 0
    for n, w in model.named_parameters():
        if 'weight' in n:
            norm += torch.norm(w, 1)
    
    tl1 = l1(labels.unsqueeze(0), rec_out)
    tl2 = l2(labels, out)
    
    final_loss = (alpha * tl1) + ((1-alpha)*tl2) + beta*norm


    return final_loss
        