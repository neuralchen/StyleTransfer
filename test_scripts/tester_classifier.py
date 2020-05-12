#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: tester_final.py
# Created Date: Friday November 8th 2019
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Thursday, 16th April 2020 12:55:52 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2019 Shanghai Jiao Tong University
#############################################################


import os
import time
import datetime
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from utilities.utilities import denorm
# from utilities.Reporter import Reporter
from tqdm import tqdm
from data_tools.test_data_loader import TestDataset

class Tester(object):
    def __init__(self, config,sampleloaders, reporter):
        
        self.config     = config
        # logger
        self.reporter   = reporter
        self.sampleloaders =sampleloaders

    def test(self):
        
        test_img    = self.config["testImgRoot"]
        save_dir    = self.config["testSamples"]
        batch_size  = self.config["batchSize"]

        # data
        sample_loader= self.sampleloaders 
        rounds= (307*16//32)

        # SpecifiedImages = None
        # if self.config["useSpecifiedImg"]:
        #     SpecifiedImages = self.config["specifiedTestImg"]
        test_data = TestDataset(test_img,self.config["imCropSize"],batch_size)
        total     = len(test_data)
                            
        # models
        package = __import__("components."+self.config["dScriptName"], fromlist=True)
        DClass  = getattr(package, 'Discriminator')
        
        #Gen     = GClass(self.config["GConvDim"], self.config["GKS"], self.config["resNum"])
        Dis     = DClass(self.config["DConvDim"], self.config["DKS"])
        if self.config["cuda"] >=0:
            Dis = Dis.cuda()
        Dis.load_state_dict(torch.load(self.config["ckp_name"]))
        print('loaded trained models {}...!'.format(self.config["ckp_name"]))
        
        start_time = time.time()
        Dis.eval()
        acc_1_total = 0
        acc_2_total = 0
        acc_3_total = 0
        with torch.no_grad():
            for iii in tqdm(range(rounds)):
                style_images,label  =  sample_loader.next() 
                out_logit = Dis(style_images)
                temp1 = (torch.argmax(out_logit[0], dim=1)==label).float().mean().cpu().numpy()
                temp2 = (torch.argmax(out_logit[1], dim=1)==label).float().mean().cpu().numpy()
                temp3 = (torch.argmax(out_logit[2], dim=1)==label).float().mean().cpu().numpy()
                acc_1_total=acc_1_total+temp1
                acc_2_total=acc_2_total+temp2
                acc_3_total=acc_3_total+temp3
        acc_1_total=acc_1_total/rounds
        acc_2_total=acc_2_total/rounds
        acc_3_total=acc_3_total/rounds
        acc_aver=(acc_1_total+acc_2_total+acc_3_total)/3
        print("acc_1: {:.4f}, acc_2: {:.4f}, acc_3: {:.4f}, acc_aver: {:.4f}".format(acc_1_total, acc_2_total, acc_3_total,acc_aver))


            

        elapsed = time.time() - start_time
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed [{}]".format(elapsed))