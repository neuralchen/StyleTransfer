#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: Discriminator_SN1.py
# Created Date: Saturday April 11th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Wednesday, 15th April 2020 12:54:07 am
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
        self.block = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(in_channels= 3,
                            out_channels= chn, kernel_size= k_size, stride= 2, bias= False)),
            nn.LeakyReLU(slop),
            utils.spectral_norm(nn.Conv2d(in_channels= chn,
                            out_channels = chn*2, kernel_size= k_size, stride= 2, bias= False)),
            nn.LeakyReLU(slop),
            utils.spectral_norm(nn.Conv2d(in_channels= chn*2,
                            out_channels = chn*4, kernel_size= k_size, stride= 2, bias= False)),
            nn.LeakyReLU(slop),
            utils.spectral_norm(nn.Conv2d(in_channels= chn*4,
                            out_channels= chn*8, kernel_size= k_size, stride= 2, bias= False)),
            nn.LeakyReLU(slop),
            utils.spectral_norm(nn.Conv2d(in_channels= chn*8,
                            out_channels= chn*16, kernel_size= k_size, stride= 2, bias= False)),
            nn.LeakyReLU(slop),
            # add more downsampling layer
            utils.spectral_norm(nn.Conv2d(in_channels= chn*16,
                            out_channels= chn*16, kernel_size= k_size, stride= 2, bias= False)),
            nn.LeakyReLU(slop)
        )
        self.classfier = utils.spectral_norm(nn.Conv2d(in_channels= chn*16, out_channels= 1 , kernel_size= 3))
        self.__weights_init__()

    def __weights_init__(self):
        print("Init weights")
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, input):
        
        h       = self.block(input)
        out     = self.classfier(h)
        return out

if __name__ == "__main__":
    wocao = Discriminator()