#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: train_multiscale.py
# Created Date: Monday April 6th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Sunday, 12th April 2020 1:48:47 am
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
from    components.Generator import Generator
from    components.Discriminator import Discriminator

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
        dStep       = self.config["dStep"]
        gStep       = self.config["gStep"]

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
        Dis0    = DClass(self.config["DConvDim"], self.config["DKS"])
        Dis1    = DClass(self.config["DConvDim"], self.config["DKS"])
        Dis2    = DClass(self.config["DConvDim"], self.config["DKS"])
        self.reporter.writeInfo("Generator structure:")
        self.reporter.writeModel(Gen.__str__())
        # print(self.Decoder)
        self.reporter.writeInfo("Discriminator structure:")
        self.reporter.writeModel(Dis0.__str__())
        
        Transform = Transform_block().cuda()
        Downscale = nn.AvgPool2d(3,2,1).cuda()
        Gen     = Gen.cuda()
        Dis0    = Dis0.cuda()
        Dis1    = Dis1.cuda()
        Dis2    = Dis2.cuda()

        if self.config["mode"] == "finetune":
            model_path = os.path.join(self.config["projectCheckpoints"], "%d_Generator.pth"%self.config["checkpointStep"])
            Gen.load_state_dict(torch.load(model_path))
            print('loaded trained Generator model step {}...!'.format(self.config["checkpointStep"]))
            model_path = os.path.join(self.config["projectCheckpoints"], "%d_Discriminator0.pth"%self.config["checkpointStep"])
            Dis0.load_state_dict(torch.load(model_path))
            print('loaded trained Discriminator model step {}...!'.format(self.config["checkpointStep"]))
            model_path = os.path.join(self.config["projectCheckpoints"], "%d_Discriminator1.pth"%self.config["checkpointStep"])
            Dis1.load_state_dict(torch.load(model_path))
            print('loaded trained Discriminator model step {}...!'.format(self.config["checkpointStep"]))
            model_path = os.path.join(self.config["projectCheckpoints"], "%d_Discriminator2.pth"%self.config["checkpointStep"])
            Dis2.load_state_dict(torch.load(model_path))
            print('loaded trained Discriminator model step {}...!'.format(self.config["checkpointStep"]))
        
        print("build the optimizer...")
        # Loss and optimizer
        g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, 
                                    Gen.parameters()), lr_base, [beta1, beta2])

        d0_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, 
                                    Dis0.parameters()), lr_base, [beta1, beta2])
        d1_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, 
                                    Dis1.parameters()), lr_base, [beta1, beta2])
        d2_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, 
                                    Dis2.parameters()), lr_base, [beta1, beta2])
        L1_loss = torch.nn.L1Loss()
        MSE_loss= torch.nn.MSELoss()
        # C_loss  = torch.nn.BCEWithLogitsLoss()
        # L1_loss     = torch.nn.SmoothL1Loss()
        Hinge_loss  = torch.nn.ReLU()

        # Start with trained model
        if self.config["mode"] == "finetune":
            start = self.config["checkpointStep"]
        else:
            start = 0
        total_step = total_step//(gStep+dStep)
        
        # Data iterator
        print("prepare the dataloaders...")
        content_iter    = iter(content_loader)
        style_iter      = iter(style_loader)

        # Start time
        print('Start   ======  training...')
        start_time = time.time()
        for step in range(start, total_step):
            Dis0.train()
            Dis1.train()
            Dis2.train()
            Gen.train()
            
            # ================== Train D ================== #
            # Compute loss with real images
            for _ in range(dStep):
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
                # scale 0
                d_out = Dis0(style_images)
                d0_loss_real = Hinge_loss(1 - d_out).mean()
                # scale 1
                style_images = Downscale(style_images)
                d_out = Dis1(style_images.detach())
                d1_loss_real = Hinge_loss(1 - d_out).mean()
                # scale 2
                style_images = Downscale(style_images)
                d_out = Dis2(style_images.detach())
                d2_loss_real = Hinge_loss(1 - d_out).mean()
                # scale 0
                d_out = Dis0(content_images)
                d0_loss_photo = Hinge_loss(1 + d_out).mean()
                # scale 1
                content_images_1 = Downscale(content_images)
                d_out = Dis1(content_images_1.detach())
                d1_loss_photo = Hinge_loss(1 + d_out).mean()
                # scale 2
                content_images_1 = Downscale(content_images_1)
                d_out = Dis2(content_images_1.detach())
                d2_loss_photo = Hinge_loss(1 + d_out).mean()

                # scale 0
                fake_image,_ = Gen(content_images)
                d_out = Dis0(fake_image.detach())
                d0_loss_fake  = Hinge_loss(1 + d_out).mean()
                # scale 1
                fake_image   = Downscale(fake_image)
                d_out = Dis0(fake_image.detach())
                d1_loss_fake = Hinge_loss(1 + d_out).mean()
                # scale 2
                fake_image   = Downscale(fake_image)
                d_out = Dis0(fake_image.detach())
                d2_loss_fake = Hinge_loss(1 + d_out).mean()
                
                # Backward + Optimize
                d0_loss = d0_loss_real + d0_loss_photo + d0_loss_fake
                d0_optimizer.zero_grad()
                d0_loss.backward()
                d0_optimizer.step()

                d1_loss = d1_loss_real + d1_loss_photo + d1_loss_fake
                d1_optimizer.zero_grad()
                d1_loss.backward()
                d1_optimizer.step()

                d2_loss = d2_loss_real + d2_loss_photo + d2_loss_fake
                d2_optimizer.zero_grad()
                d2_loss.backward()
                d2_optimizer.step()
                

            # ================== Train G ================== #
            for _ in range(gStep):
                try:
                    content_images =next(content_iter)
                except:
                    content_iter    = iter(content_loader)
                    content_images  = next(content_iter)
                content_images  = content_images.cuda()
                
                fake_image, real_feature= Gen(content_images)
                fake_feature            = Gen(fake_image, get_feature = True)

                g_feature_loss          = L1_loss(fake_feature,real_feature)
                g_transform_loss        = MSE_loss(Transform(content_images), Transform(fake_image))
                # scale 0
                fake_out0               = Dis0(fake_image)
                # scale 1
                fake_image              = Downscale(fake_image)
                fake_out1               = Dis1(fake_image)
                # scale 2
                fake_image              = Downscale(fake_image)
                fake_out2               = Dis2(fake_image)

                

                g_loss_fake = - (fake_out0.mean()+fake_out1.mean()+fake_out2.mean())
                g_loss_fake = g_loss_fake + g_feature_loss* feature_w + g_transform_loss* transform_w
                g_optimizer.zero_grad()
                g_loss_fake.backward()
                g_optimizer.step()
            

            # Print out log info
            if (step + 1) % log_frep == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print("Elapsed [{}], step [{}/{}], d0_out_real: {:.4f}, d1_out_real: {:.4f}, d2_out_real: {:.4f},d0_out_fake: {:.4f}, d1_out_fake: {:.4f}, d2_out_fake: {:.4f}, g_loss_fake: {:.4f}".format(\
                            elapsed, step + 1, total_step,d0_loss_real.item(),d1_loss_real.item(),d2_loss_real.item(),
                                    d0_loss_fake.item(), d1_loss_fake.item(), d2_loss_fake.item(), g_loss_fake.item()))
                
                if self.config["useTensorboard"]:
                    tensorboard_writer.add_scalar('data/d_loss_real', d0_loss_real.item(),(step + 1))
                    tensorboard_writer.add_scalar('data/d_loss_fake', d0_loss_fake.item(),(step + 1))
                    # tensorboard_writer.add_scalar('data/d_loss', d_loss.item(), (step + 1))
                    tensorboard_writer.add_scalar('data/g_loss', g_loss_fake.item(), (step + 1))
                    tensorboard_writer.add_scalar('data/g_feature_loss', g_feature_loss, (step + 1))
                    tensorboard_writer.add_scalar('data/g_transform_loss', g_transform_loss, (step + 1))

            # Sample images
            if (step + 1) % sample_freq == 0:
                print('Sample images {}_fake.jpg'.format(step + 1))
                Gen.eval()
                with torch.no_grad():
                    fake_images,_ = Gen(content_images)
                    saved_image1 = torch.cat([denorm(content_images),denorm(fake_images.data)],3)
                    # saved_image2 = torch.cat([denorm(style_images),denorm(fake_images.data)],3)
                    # wocao        = torch.cat([saved_image1,saved_image2],2)
                    save_image(saved_image1,
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
                torch.save(Dis0.state_dict(),
                           os.path.join(ckpt_dir, '{}_Discriminator0.pth'.format(step + 1)))
                torch.save(Dis1.state_dict(),
                           os.path.join(ckpt_dir, '{}_Discriminator1.pth'.format(step + 1)))
                torch.save(Dis2.state_dict(),
                           os.path.join(ckpt_dir, '{}_Discriminator2.pth'.format(step + 1)))