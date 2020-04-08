#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: train_styleaware.py
# Created Date: Monday April 6th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Thursday, 9th April 2020 12:10:52 am
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

from    data_tools.data_loader import getLoader
from    components.Transform import Transform_block
from    utilities.utilities import denorm


class Trainer(object):
    def __init__(self, config, reporter):

        self.config     = config
        # logger
        self.reporter   = reporter
        # Data loader
        

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
        lrDecayStep = self.config["lrDecayStep"]
        batch_size  = self.config["batchSize"]
        prep_weights= self.config["layersWeight"]
        feature_w   = self.config["featureWeight"]
        transform_w = self.config["transformWeight"]
        workers     = self.config["dataloader_workers"]

        if self.config["useTensorboard"]:
            from utilities.utilities import build_tensorboard
            tensorboard_writer = build_tensorboard(self.config["projectSummary"])

        print("prepare the dataloader...")
        content_loader  = getLoader(self.config["content"],self.config["selectedContentDir"],
                            self.config["imCropSize"],batch_size,"Content",workers)
        style_loader    = getLoader(self.config["style"],self.config["selectedStyleDir"],
                            self.config["imCropSize"],batch_size,"Style",workers)
        
        print("build models...")

        package  = __import__("components."+self.config["gScriptName"], fromlist=True)
        GClass   = getattr(package, 'Generator')
        package  = __import__("components."+self.config["dScriptName"], fromlist=True)
        DClass   = getattr(package, 'Discriminator')

        Gen     = GClass(self.config["GConvDim"], self.config["GKS"], self.config["resNum"])
        Dis     = DClass(self.config["DConvDim"], self.config["DKS"])

        self.reporter.writeInfo("Generator structure:")
        self.reporter.writeModel(Gen.__str__())
        # print(self.Decoder)
        self.reporter.writeInfo("Discriminator structure:")
        self.reporter.writeModel(Dis.__str__())
        
        Transform = Transform_block().cuda()
        Gen     = Gen.cuda()
        Dis     = Dis.cuda()
        
        print("build the optimizer...")
        # Loss and optimizer
        g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, 
                                    Gen.parameters()), lr_base, [beta1, beta2])

        d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, 
                                    Dis.parameters()), lr_base, [beta1, beta2])
        L1_loss = torch.nn.L1Loss()
        MSE_loss= torch.nn.MSELoss()
        C_loss  = torch.nn.BCEWithLogitsLoss()
        # L1_loss     = torch.nn.SmoothL1Loss()
        # Hinge_loss  = torch.nn.ReLU()

        # Start with trained model
        if self.config["mode"] == "finetune":
            start = self.config["finetuneCheckpoint"]
        else:
            start = 0

        win_rate      = 0.8
        discr_success = 0.8
        alpha         = 0.05
        real_labels = []
        fake_labels = []
        label_size = [[batch_size,1,760,760],[batch_size,1,371,371],[batch_size,1,83,83],[batch_size,1,15,15],[batch_size,1,6,6]]
        for i in range(5):
            real_label = torch.ones(label_size[i]).cuda()
            fake_label = torch.zeros(label_size[i]).cuda()
            # threshold = torch.zeros(size[i], device=self.device)
            real_labels.append(real_label)
            fake_labels.append(fake_label)
        
        output_size = len(label_size)
        output_size_float = float(output_size)
        
        # Data iterator
        print("prepare the dataloaders...")
        content_iter    = iter(content_loader)
        style_iter      = iter(style_loader)

        # Start time
        print('Start   ======  training...')
        start_time = time.time()
        for step in range(start, total_step):
            Dis.train()
            Gen.train()
            try:
                content_images =next(content_iter)
                style_images = next(style_iter)
            except:
                style_iter      = iter(style_loader)
                content_iter    = iter(content_loader)
                style_images = next(style_iter)
                content_images = next(content_iter)
            style_images    = style_images.cuda()
            content_images  = content_images.cuda()
            # ================== Train D ================== #
            # Compute loss with real images
            if discr_success < win_rate:
                
                real_out = Dis(style_images)
                d_loss_real = 0
                real_acc = 0
                for i in range(output_size):
                    temp = C_loss(real_out[i],real_labels[i]).mean()
                    real_acc +=  torch.gt(real_out[i],0).type(torch.float).mean()
                    temp *= prep_weights[i]
                    d_loss_real += temp
                real_acc /= output_size_float

                d_loss_photo = 0
                photo_out = Dis(content_images)
                photo_acc = 0
                for i in range(output_size):
                    temp = C_loss(photo_out[i],fake_labels[i])
                    photo_acc +=  torch.lt(photo_out[i],0).type(torch.float).mean()
                    temp *= prep_weights[i]
                    d_loss_photo += temp
                photo_acc /= output_size_float 

                fake_image,_ = Gen(content_images)
                fake_out = Dis(fake_image.detach())
                d_loss_fake = 0
                fake_acc = 0
                for i in range(output_size):
                    temp = C_loss(fake_out[i],fake_labels[i]).mean()
                    fake_acc +=  torch.lt(fake_out[i],0).type(torch.float).mean()
                    temp *= prep_weights[i]
                    d_loss_fake += temp
                fake_acc /= output_size_float

                d_acc = ((real_acc + photo_acc + fake_acc)/3).item()
                discr_success = discr_success * (1. - alpha) + alpha * d_acc
                # Backward + Optimize
                d_loss = d_loss_real + d_loss_photo + d_loss_fake
                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()
            else:
                # ================== Train G ================== #   
                #      
                g_fake_image,real_feature= Gen(content_images)
                fake_feature            = Gen(g_fake_image, get_feature = True)
                g_fake_out              = Dis(g_fake_image)
                g_feature_loss          = L1_loss(fake_feature,real_feature)
                g_transform_loss        = MSE_loss(Transform(content_images), Transform(g_fake_image))
                g_loss_fake = 0
                g_acc = 0
                for i in range(output_size):
                    temp = C_loss(g_fake_out[i],real_labels[i]).mean()
                    g_acc +=  torch.gt(g_fake_out[i],0).type(torch.float).mean()
                    temp *= prep_weights[i]
                    g_loss_fake += temp
                g_acc /= output_size_float
                
                discr_success = discr_success * (1. - alpha) + alpha * (1.0 - g_acc)

                # backward & optimize
                g_loss_fake = g_loss_fake + g_feature_loss* feature_w + g_transform_loss* transform_w
                g_optimizer.zero_grad()
                g_loss_fake.backward()
                g_optimizer.step()
            

            # Print out log info
            if (step + 1) % log_frep == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print("Elapsed [{}], G_step [{}/{}], D_step[{}/{}], d_out_real: {:.4f}, d_out_fake: {:.4f}, g_loss_fake: {:.4f}".
                      format(elapsed, step + 1, total_step, (step + 1),
                             total_step , d_loss_real.item(), d_loss_fake.item(), g_loss_fake.item()))
                
                if self.config["useTensorboard"]:
                    tensorboard_writer.add_scalar('data/d_loss_real', d_loss_real.item(),(step + 1))
                    tensorboard_writer.add_scalar('data/d_loss_fake', d_loss_fake.item(),(step + 1))
                    tensorboard_writer.add_scalar('data/d_loss', d_loss.item(), (step + 1))
                    tensorboard_writer.add_scalar('data/g_loss', g_loss_fake.item(), (step + 1))
                    tensorboard_writer.add_scalar('data/g_feature_loss', g_feature_loss, (step + 1))
                    tensorboard_writer.add_scalar('data/g_transform_loss', g_transform_loss, (step + 1))
                    tensorboard_writer.add_scalar('acc/real_acc', real_acc.item(), (step + 1))
                    tensorboard_writer.add_scalar('acc/photo_acc', photo_acc.item(), (step + 1))
                    tensorboard_writer.add_scalar('acc/fake_acc', fake_acc.item(), (step + 1))
                    tensorboard_writer.add_scalar('acc/disc_acc', d_acc, (step + 1))
                    tensorboard_writer.add_scalar('acc/g_acc', g_acc, (step + 1))
                    tensorboard_writer.add_scalar("acc/discr_success",discr_success,(step+1))

            # Sample images
            if (step + 1) % sample_freq == 0:
                print('Sample images {}_fake.jpg'.format(step + 1))
                fake_images,_ = Gen(content_images)
                saved_image1 = torch.cat([denorm(content_images),denorm(fake_images.data)],3)
                saved_image2 = torch.cat([denorm(style_images),denorm(fake_images.data)],3)
                wocao        = torch.cat([saved_image1,saved_image2],2)
                save_image(wocao,
                           os.path.join(sample_dir, '{}_fake.jpg'.format(step + 1)))
                # print("Transfer validation images")
                # num = 1
                # for val_img in self.validation_data:
                #     print("testing no.%d img"%num)
                #     val_img = val_img.cuda()
                #     fake_images,_ = Gen(val_img)
                #     saved_val_image = torch.cat([denorm(val_img),denorm(fake_images)],3)
                #     save_image(saved_val_image,
                #            os.path.join(self.valres_path, '%d_%d.jpg'%((step+1),num)))
                #     num +=1
                # save_image(denorm(displaymask.data),os.path.join(self.sample_path, '{}_mask.png'.format(step + 1)))

            if (step+1) % model_freq==0:
                torch.save(Gen.state_dict(),
                           os.path.join(ckpt_dir, '{}_Generator.pth'.format(step + 1)))
                torch.save(Dis.state_dict(),
                           os.path.join(ckpt_dir, '{}_Discriminator.pth'.format(step + 1)))