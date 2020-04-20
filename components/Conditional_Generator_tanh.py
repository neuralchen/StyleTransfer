#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: Conditional_Generator_tanh.py
# Created Date: Saturday April 18th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Monday, 20th April 2020 6:56:45 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################


import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F

from components.ResBlock import ResBlock
from components.DeConv   import DeConv
from components.Conditional_ResBlock import Conditional_ResBlock

class Generator(nn.Module):
    def __init__(
                self, chn=32,
                k_size=3,
                res_num = 5,
                class_num = 3
                ):
        super().__init__()
        padding_size = int((k_size -1)/2)
        self.resblock_list = []
        self.n_class    = class_num
        self.encoder = nn.Sequential(
            nn.ReplicationPad2d(15),
            # nn.InstanceNorm2d(3, affine=True),
            # nn.ReflectionPad2d(padding_size),
            nn.Conv2d(in_channels = 3 , out_channels = chn , kernel_size= k_size, bias= False),
            nn.InstanceNorm2d(chn, affine=True, momentum=0),
            nn.ReLU(),
            # nn.ReflectionPad2d(padding_size),
            nn.Conv2d(in_channels = chn , out_channels = chn*2, kernel_size= k_size, stride=2, bias =False), # 
            nn.InstanceNorm2d(chn*2, affine=True, momentum=0),
            nn.ReLU(),
            # nn.ReflectionPad2d(padding_size),
            nn.Conv2d(in_channels = chn*2, out_channels = chn * 4, kernel_size= k_size, stride=2, bias =False),
            nn.InstanceNorm2d(chn * 4, affine=True, momentum=0),
            nn.ReLU(),
            # nn.ReflectionPad2d(padding_size),
            nn.Conv2d(in_channels = chn*4  , out_channels = chn * 8, kernel_size= k_size, stride=2, bias =False),
            nn.InstanceNorm2d(chn * 8, affine=True, momentum=0),
            nn.ReLU(),
            # # nn.ReflectionPad2d(padding_size),
            nn.Conv2d(in_channels = chn * 8, out_channels = chn * 8, kernel_size= k_size, stride=2, bias =False),
            nn.InstanceNorm2d(chn * 8, affine=True, momentum=0),
            nn.ReLU(),
        )
        res_size = chn * 8
        for _ in range(res_num-1):
            self.resblock_list += [ResBlock(res_size,k_size),]
        self.resblocks = nn.Sequential(*self.resblock_list)
        self.conditional_res = Conditional_ResBlock(res_size, k_size, class_num)
        self.decoder = nn.Sequential(
            DeConv(in_channels = chn * 8, out_channels = chn * 8, kernel_size= k_size),
            nn.InstanceNorm2d(chn * 8, affine=True, momentum=0),
            nn.ReLU(),
            DeConv(in_channels = chn * 8, out_channels = chn *4, kernel_size= k_size),
            nn.InstanceNorm2d(chn *4, affine=True, momentum=0),
            nn.ReLU(),
            DeConv(in_channels = chn * 4, out_channels = chn * 2 , kernel_size= k_size),
            nn.InstanceNorm2d(chn*2, affine=True, momentum=0),
            nn.ReLU(),
            DeConv(in_channels = chn * 2, out_channels = chn, kernel_size= k_size),
            nn.InstanceNorm2d(chn, affine=True, momentum=0),
            nn.ReLU(),
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels = chn, out_channels = 3, kernel_size= 7),
            nn.Tanh()
        )
        self.__weights_init__()

    def __weights_init__(self):
        for layer in self.encoder:
            if isinstance(layer,nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, input, condition=None,get_feature = False):
        feature = self.encoder(input)
        if get_feature:
            return feature
        out = self.resblocks(feature)
        out = self.conditional_res(out, condition)
        # n, _,h,w = out.size()
        # attr = condition.view((n, self.n_class, 1, 1)).expand((n, self.n_class, h, w))
        # out = torch.cat([out, attr], dim=1)
        out = self.decoder(out)
        return out,feature