#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: Discriminator_SN_FC.py
# Created Date: Saturday April 11th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Thursday, 16th April 2020 11:18:52 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################


import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn import utils
# from ResBlock import ResBlock
# from  DeConv   import DeConv

class Discriminator(nn.Module):
    def __init__(self, chn=32, k_size=5):
        super().__init__()
        # padding_size = int((k_size -1)/2)
        slop         = 0.02
        feature_size = 1
        self.block = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(in_channels= 3,
                            out_channels= chn, kernel_size= k_size, stride= 2, bias= True)), # 1/2
            nn.LeakyReLU(slop),
            utils.spectral_norm(nn.Conv2d(in_channels= chn,
                            out_channels = chn*2, kernel_size= k_size, stride= 2, bias= True)),# 1/4
            nn.LeakyReLU(slop),
        )
        self.classifier1 = utils.spectral_norm(nn.Conv2d(in_channels= chn*2, 
                            out_channels= 1, stride= 1, kernel_size= k_size))
        self.block1 = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(in_channels= chn*2,
                            out_channels = chn*4, kernel_size= k_size, stride= 2, bias= True)),# 1/8
            nn.LeakyReLU(slop),
            utils.spectral_norm(nn.Conv2d(in_channels= chn*4,
                            out_channels= chn*8, kernel_size= k_size, stride= 2, bias= True)),# 1/16
            nn.LeakyReLU(slop),
            utils.spectral_norm(nn.Conv2d(in_channels= chn*8,
                            out_channels= chn*16, kernel_size= k_size, stride= 2, bias= True)),# 1/32
            nn.LeakyReLU(slop),
            # nn.AdaptiveAvgPool2d(1)
            utils.spectral_norm(nn.Conv2d(in_channels= chn*16,
                            out_channels= 1, stride= 1, kernel_size= 3)), # 1/64
            # utils.spectral_norm(nn.Conv2d(in_channels= chn*8,
            #                 out_channels= chn*16, stride= 2, kernel_size= k_size)), # 1/64
            # nn.LeakyReLU(slop),
            # utils.spectral_norm(nn.Conv2d(in_channels= chn*16,
            #                 out_channels= chn*16, stride= 1, kernel_size= k_size)), # 1/64
        )
        currentDim = feature_size**2 * chn*8
        self.fc_adv1 = utils.spectral_norm(nn.Linear(currentDim, 1))
        self.fc_adv = utils.spectral_norm(nn.Linear(currentDim, 1))
        self.__weights_init__()

    def __weights_init__(self):
        print("Init weights")
        for m in self.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, input):
        
        h   = self.block(input)
        per1= self.classifier1(h)
        h1  = self.block1(h)
        # h   = h.view(h.size()[0], -1)
        # h   = self.fc_adv(h)
        return h1

if __name__ == "__main__":
    wocao = Discriminator().cuda()
    from torchsummary import summary
    summary(wocao, input_size=(3, 512, 512))