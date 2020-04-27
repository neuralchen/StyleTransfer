#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: tester_final.py
# Created Date: Friday November 8th 2019
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Friday, 24th April 2020 12:13:01 am
# Modified By: Chen Xuanhong
# Copyright (c) 2019 Shanghai Jiao Tong University
#############################################################


import os
import time
import datetime
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from utilities.utilities import denorm
# from utilities.Reporter import Reporter
from tqdm import tqdm
from data_tools.test_data_loader import TestDataset

class Tester(object):
    def __init__(self, config, reporter):
        
        self.config     = config
        # logger
        self.reporter   = reporter

    def test(self):
        
        test_img    = self.config["testImgRoot"]
        save_dir    = self.config["testSamples"]
        batch_size  = self.config["batchSize"]
        # data
        
        # SpecifiedImages = None
        # if self.config["useSpecifiedImg"]:
        #     SpecifiedImages = self.config["specifiedTestImg"]
        test_data = TestDataset(test_img,self.config["imCropSize"],batch_size)
        total     = len(test_data)
                            
        # models
        package = __import__(self.config["com_base"]+self.config["gScriptName"], fromlist=True)
        GClass  = getattr(package, 'Generator')
        
        Gen     = GClass(self.config["GConvDim"], self.config["GKS"], self.config["resNum"])
        if self.config["cuda"] >=0:
            Gen = Gen.cuda()
        Gen.load_state_dict(torch.load(self.config["ckp_name"]))
        print('loaded trained models {}...!'.format(self.config["ckp_name"]))
        
        start_time = time.time()
        Gen.eval()
        with torch.no_grad():
            for iii in tqdm(range(total//batch_size)):
                content,img_name = test_data()
                if self.config["cuda"] >=0:
                    content = content.cuda()       
                res,_ = Gen(content)
                print("Save test data")
                tensordata = res[0,:,:,:].data.cpu().numpy()
                np.savez(os.path.join(save_dir, '{}_step{}_v_{}.npz'.format(img_name,self.config["checkpointStep"],self.config["version"])),tensordata)
                save_image(denorm(res.data),
                            os.path.join(save_dir, '{}_step{}_v_{}.png'.format(img_name,self.config["checkpointStep"],self.config["version"])),nrow=batch_size)#,nrow=self.batch_size)
        elapsed = time.time() - start_time
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed [{}]".format(elapsed))