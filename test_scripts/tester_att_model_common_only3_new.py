#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: tester_condition.py
# Created Date: Friday November 8th 2019
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Thursday, 23rd April 2020 11:10:02 am
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
import numpy as np

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
        n_class     = len(self.config["selectedStyleDir"])

        sample_loader= self.sampleloaders 
        print("%d classes"%n_class)
        # data
        
        # SpecifiedImages = None
        # if self.config["useSpecifiedImg"]:
        #     SpecifiedImages = self.config["specifiedTestImg"]
        test_data = TestDataset(test_img,self.config["imCropSize"],batch_size)
        total     = 1000
                            
        # models
        package = __import__(self.config["com_base"]+self.config["gScriptName"], fromlist=True)
        GClass  = getattr(package, 'Generator')

                # models
        package = __import__("components.Conditional_Discriminator_att", fromlist=True)
        ClassifyClass  = getattr(package, 'Discriminator')
        
        Gen     = GClass(self.config["GConvDim"], self.config["GKS"], self.config["resNum"])
        Classify     = ClassifyClass(32, 5)
        if self.config["cuda"] >=0:
            Gen = Gen.cuda()
            Classify =Classify.cuda()
        Gen.load_state_dict(torch.load(self.config["ckp_name"]))
        Classify.load_state_dict(torch.load('/home/gdp/CXH/lny/StyleTransfer-master/train_logs/512_classy_5_3/checkpoints/252600_classify.pth'))
        print('loaded trained Gen models {}...!'.format(self.config["ckp_name"]))
        print('loaded trained Classify models ...!')
        condition_labels = torch.ones((n_class, batch_size, 1)).long()
        condition_labels_test = torch.ones((n_class, batch_size, 1)).long()
        for i in range(n_class):
            condition_labels[i,:,:] = condition_labels[i,:,:]*i

        for i in range(n_class):
            condition_labels_test[i,:,:] = condition_labels_test[i,:,:]*(2-i)
        # for i in range(n_class):
        #     if i ==0 :
        #         condition_labels[i,:,:] = condition_labels[i,:,:]*10
        #     elif i ==1:
        #         condition_labels[i,:,:] = condition_labels[i,:,:]*9
        #     elif i ==2:
        #         condition_labels[i,:,:] = condition_labels[i,:,:]*8

        


        # print(condition_labels)
        #condition_labels = torch.from_numpy(np.array([[[10]],[[9]],[[8]]])).long()
        if self.config["cuda"] >=0:
            condition_labels = condition_labels.cuda()

            condition_labels_test = condition_labels_test.cuda()
        #print(condition_labels)




        start_time = time.time()
        Gen.eval()
        Classify.eval()
        acc_12_total = 0
        acc_13_total = 0
        acc_22_total = 0
        acc_23_total = 0
        acc_32_total = 0
        acc_33_total = 0
        with torch.no_grad():
            for _ in tqdm(range(total//batch_size)):
            #for _ in range(total//batch_size):
                content =  sample_loader.next() 
                final_res = None
                
                for i in range(n_class):
                    if self.config["cuda"] >=0:
                        content = content.cuda()
                    #print(condition_labels[i,:,:].cpu().numpy())
                    res,_ = Gen(content, condition_labels[i,0,:])
                    #print(condition_labels[i,0,:])
                    #print("Save test data")
                    
                    if i ==0:
                        final_res = res
                        y2 = F.interpolate(res, size=[512, 512], mode="bilinear")
                        out_logit = Classify(y2)
                        # print(torch.squeeze(condition_labels_test[i,:,:]).cpu().numpy())
                        # print(torch.argmax(out_logit[1][:,8:], dim=1))

                        temp2 = (torch.argmax(out_logit[1][:,8:], dim=1)==torch.squeeze(condition_labels_test[i,:,:])).float().mean().cpu().numpy()
                        temp3 = (torch.argmax(out_logit[2][:,8:], dim=1)==torch.squeeze(condition_labels_test[i,:,:])).float().mean().cpu().numpy()
                        acc_12_total=acc_12_total+temp2
                        acc_13_total=acc_13_total+temp3
                    elif i ==1:
                        final_res = torch.cat([final_res,res],0)
                        y2 = F.interpolate(res, size=[512, 512], mode="bilinear")
                        out_logit = Classify(y2)
                        temp2 = (torch.argmax(out_logit[1][:,8:], dim=1)==torch.squeeze(condition_labels_test[i,:,:])).float().mean().cpu().numpy()
                        temp3 = (torch.argmax(out_logit[2][:,8:], dim=1)==torch.squeeze(condition_labels_test[i,:,:])).float().mean().cpu().numpy()
                        acc_22_total=acc_22_total+temp2
                        acc_23_total=acc_23_total+temp3
                    elif i ==2:
                        final_res = torch.cat([final_res,res],0)
                        y2 = F.interpolate(res, size=[512, 512], mode="bilinear")
                        out_logit = Classify(y2)
                        temp2 = (torch.argmax(out_logit[1][:,8:], dim=1)==torch.squeeze(condition_labels_test[i,:,:])).float().mean().cpu().numpy()
                        temp3 = (torch.argmax(out_logit[2][:,8:], dim=1)==torch.squeeze(condition_labels_test[i,:,:])).float().mean().cpu().numpy()
                        acc_32_total=acc_32_total+temp2
                        acc_33_total=acc_33_total+temp3

                save_image(denorm(final_res.data),
                            os.path.join(save_dir, '{}_step{}_v_{}.png'.format(i,self.config["checkpointStep"],self.config["version"])),nrow=n_class)#,nrow=self.batch_size)
        acc_12_total = acc_12_total/(total//batch_size)
        acc_13_total = acc_13_total/(total//batch_size)
        acc_22_total = acc_22_total/(total//batch_size)
        acc_23_total = acc_23_total/(total//batch_size)
        acc_32_total = acc_32_total/(total//batch_size)
        acc_33_total = acc_33_total/(total//batch_size)
        elapsed = time.time() - start_time
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("vangogh ：  acc_2: {:.4f}, acc_3: {:.4f}, acc_aver: {:.4f}".format(acc_12_total,acc_13_total,(acc_12_total+acc_13_total)/2))
        print("samuel  ：  acc_2: {:.4f}, acc_3: {:.4f}, acc_aver: {:.4f}".format(acc_22_total,acc_23_total,(acc_22_total+acc_23_total)/2))
        print("picasso ：  acc_2: {:.4f}, acc_3: {:.4f}, acc_aver: {:.4f}".format(acc_32_total,acc_33_total,(acc_32_total+acc_33_total)/2))
        print("Elapsed [{}]".format(elapsed))