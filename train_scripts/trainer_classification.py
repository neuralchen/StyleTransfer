#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: trainer_condition_SN_multiscale.py
# Created Date: Saturday April 18th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Monday, 27th April 2020 11:11:28 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################


import  os
import  time
import  datetime

import  torch
import  torch.nn as nn
import  torch.nn.functional as F
from    torch.autograd     import Variable
from    torchvision.utils  import save_image
from    functools import partial

#from    components.Transform import Transform_block
from    utilities.utilities import denorm
from    utilities import losses
import  numpy as np

class Trainer(object):
    def __init__(self, config, dataloaders_list,sampleloaders, reporter):

        self.config     = config
        # logger
        self.reporter   = reporter
        # Data loader
        self.dataloaders= dataloaders_list
        self.sampleloaders =sampleloaders

    def train(self):
        
        ckpt_dir    = self.config["projectCheckpoints"]
        sample_dir  = self.config["projectSamples"]
        total_step  = self.config["totalStep"]
        log_frep    = self.config["logStep"]
        sample_freq = self.config["sampleStep"]
        model_freq  = self.config["modelSaveStep"]
        lr_base     = self.config["gLr"]
        beta1       = self.config["beta1"]
        beta2       = self.config["beta2"]
        # lrDecayStep = self.config["lrDecayStep"]
        batch_size  = self.config["batchSize"]
        n_class     = len(self.config["selectedStyleDir"])
        # prep_weights= self.config["layersWeight"]
        #feature_w   = self.config["featureWeight"]
        #transform_w = self.config["transformWeight"]
        dStep       = self.config["dStep"]
        #gStep       = self.config["gStep"]
        total_loader= self.dataloaders
        sample_loader= self.sampleloaders 

        if self.config["useTensorboard"]:
            from utilities.utilities import build_tensorboard
            tensorboard_writer = build_tensorboard(self.config["projectSummary"])
        
        print("build models...")

        if self.config["mode"] == "train":
            # package = __import__("components."+self.config["gScriptName"], fromlist=True)
            # GClass  = getattr(package, 'Generator')
            package = __import__("components."+self.config["dScriptName"], fromlist=True)
            DClass  = getattr(package, 'Discriminator')
        elif self.config["mode"] == "finetune":
            print("finetune load scripts from %s"%self.config["com_base"])
            # package = __import__(self.config["com_base"]+self.config["gScriptName"], fromlist=True)
            # GClass  = getattr(package, 'Generator')
            package = __import__(self.config["com_base"]+self.config["dScriptName"], fromlist=True)
            DClass  = getattr(package, 'Discriminator')

        #Gen     = GClass(self.config["GConvDim"], self.config["GKS"], self.config["resNum"])
        Dis     = DClass(self.config["DConvDim"], self.config["DKS"])
        
        # self.reporter.writeInfo("Generator structure:")
        # self.reporter.writeModel(Gen.__str__())
        # print(self.Decoder)
        self.reporter.writeInfo("Discriminator structure:")
        self.reporter.writeModel(Dis.__str__())
        
        #Transform   = Transform_block().cuda()
        #Gen         = Gen.cuda()
        Dis         = Dis.cuda()

        if self.config["mode"] == "finetune":
            # model_path = os.path.join(self.config["projectCheckpoints"], "%d_Generator.pth"%self.config["checkpointStep"])
            # Gen.load_state_dict(torch.load(model_path))
            # print('loaded trained Generator model step {}...!'.format(self.config["checkpointStep"]))
            model_path = os.path.join(self.config["projectCheckpoints"], "%d_classify.pth"%self.config["checkpointStep"])
            Dis.load_state_dict(torch.load(model_path))
            print('loaded trained Discriminator model step {}...!'.format(self.config["checkpointStep"]))
        
        print("build the optimizer...")
        # Loss and optimizer
        # g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, 
        #                             Gen.parameters()), lr_base, [beta1, beta2])

        d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, 
                                    Dis.parameters()), lr_base, [beta1, beta2])
        L1_loss     = torch.nn.L1Loss()
        MSE_loss    = torch.nn.MSELoss()
        Hinge_loss  = torch.nn.ReLU().cuda()
        # L1_loss     = torch.nn.SmoothL1Loss()

        # Start with trained model
        if self.config["mode"] == "finetune":
            start = self.config["checkpointStep"]
        else:
            start = 0

        output_size = Dis.get_outputs_len()
        
        # Data iterator
        print("prepare the dataloaders...")
        # total_iter  = iter(total_loader)
        # prefetcher = data_prefetcher(total_loader)
        # input, target = prefetcher.next()
        # style_iter      = iter(style_loader)

        print("prepare the fixed labels...")
        fix_label   = [i for i in range(n_class)]
        fix_label   = torch.tensor(fix_label).long().cuda()
        # fix_label       = fix_label.view(n_class,1)
        # fix_label       = torch.zeros(n_class, n_class).cuda().scatter_(1, fix_label, 1)

        # Start time
        import datetime
        print("Start to train at %s"%(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        
        print('Start   ======  training...')
        start_time = time.time()
        Dis.train()
        self.classification_loss_type = "cross-entropy"
        classifierLoss = self.getClassifierLoss()
        softmax_fun=nn.Softmax(dim=1)
        for step in range(start, total_step):
            Dis.train()
            style_images,label  = total_loader.next()
            label           = label.long()
            D_logit = Dis(style_images)
            D_logit_total = 0

            conditionnp=np.full((batch_size, n_class), 0.0)
            for index , id in enumerate(label):
                conditionnp[index][id]=1.0
            
            condition=torch.from_numpy(conditionnp)
            #condition = condition.type(torch.LongTensor)
            #print(condition)
            condition=torch.tensor(condition, dtype=torch.float32)
            condition = condition.cuda()
            # print(condition.size())
            # print(D_logit[0].size())

            temp1 = classifierLoss(softmax_fun(D_logit[0]), condition)
            temp2 = classifierLoss(softmax_fun(D_logit[1]), condition)
            temp3 = classifierLoss(softmax_fun(D_logit[2]), condition)
            # temp *= prep_weights[i]
            D_logit_total = temp1+temp2+temp3



            # Backward + Optimize
            d_loss = D_logit_total 
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            elapsed = time.time() - start_time
            elapsed = str(datetime.timedelta(seconds=elapsed))
            print("inference time %s"%elapsed)
            



                #CS_loss = losses.mh_loss(D_fake, label)
            

            # Print out log info
            if (step + 1) % log_frep == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                epochinformation="[{}], Elapsed [{}], Step [{}/{}], d_total: {:.4f}, d_1: {:.4f}, d_2: {:.4f}, d_3: {:.4f}".format(self.config["version"], elapsed, step + 1, total_step, 
                            D_logit_total.item(),temp1.item(), temp2.item(), temp3.item())
                print(epochinformation)
                self.reporter.write_epochInf(epochinformation)
                
                if self.config["useTensorboard"]:
                    tensorboard_writer.add_scalar('data/d_total', D_logit_total.item(),(step + 1))
                    tensorboard_writer.add_scalar('data/d_1', temp1.item(),(step + 1))
                    tensorboard_writer.add_scalar('data/d_2', temp2.item(),(step + 1))
                    tensorboard_writer.add_scalar('data/d_3', temp3.item(), (step + 1))

            if (step+1) % model_freq==0:
                print("Save step %d model checkpoints!"%(step+1))
                torch.save(Dis.state_dict(),
                           os.path.join(ckpt_dir, '{}_classify.pth'.format(step + 1)))

            # Sample images
            if (step + 1) % sample_freq == 0:
                epochinformation='Sample time'
                print(epochinformation)
                self.reporter.write_epochInf(epochinformation)
                Dis.eval()
                with torch.no_grad():
                    #for index in range(self.SampleImgNum):
                    style_images,label  =  sample_loader.next() 
                    out_logit = Dis(style_images)
                    temp1 = (torch.argmax(out_logit[0], dim=1)==label).float().mean().cpu().numpy()
                    temp2 = (torch.argmax(out_logit[1], dim=1)==label).float().mean().cpu().numpy()
                    temp3 = (torch.argmax(out_logit[2], dim=1)==label).float().mean().cpu().numpy()

                    epochinformation="[{}], Elapsed [{}], Step [{}/{}], acc_1: {:.4f}, acc_2: {:.4f}, acc_3: {:.4f}".format(self.config["version"], elapsed, step + 1, total_step,temp1.item(), temp2.item(), temp3.item())
                    print(epochinformation)
                    self.reporter.write_epochInf(epochinformation)

    
    def classificationLoss(self, logit, target):
        """Compute binary cross entropy loss."""
        return F.binary_cross_entropy(logit, target, reduction='sum')/logit.size(0)
    
    def hingeLoss(self, logit, label):
        return nn.ReLU()(1.0 - label * logit).mean()

    def getClassifierLoss(self):
        if self.classification_loss_type == "hinge":
            return self.hingeLoss
        elif self.classification_loss_type == "cross-entropy":
            return self.classificationLoss