#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: trainer_condition_SN_multiscale.py
# Created Date: Saturday April 18th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Friday, 8th May 2020 10:49:54 pm
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

from    components.Transform import Transform_block
from    utilities.utilities import denorm

class Trainer(object):
    def __init__(self, config, dataloaders_list, reporter):

        self.config     = config
        # logger
        self.reporter   = reporter
        # Data loader
        self.dataloaders= dataloaders_list

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
        feature_w   = self.config["featureWeight"]
        transform_w = self.config["transformWeight"]
        dStep       = self.config["dStep"]
        gStep       = self.config["gStep"]
        total_loader= self.dataloaders

        if self.config["useTensorboard"]:
            from utilities.utilities import build_tensorboard
            tensorboard_writer = build_tensorboard(self.config["projectSummary"])
        
        print("build models...")

        if self.config["mode"] == "train":
            package = __import__("components."+self.config["gScriptName"], fromlist=True)
            GClass  = getattr(package, 'Generator')
            package = __import__("components."+self.config["dScriptName"], fromlist=True)
            DClass  = getattr(package, 'Discriminator')
        elif self.config["mode"] == "finetune":
            print("finetune load scripts from %s"%self.config["com_base"])
            package = __import__(self.config["com_base"]+self.config["gScriptName"], fromlist=True)
            GClass  = getattr(package, 'Generator')
            package = __import__(self.config["com_base"]+self.config["dScriptName"], fromlist=True)
            DClass  = getattr(package, 'Discriminator')

        Gen     = GClass(self.config["GConvDim"], self.config["GKS"], self.config["resNum"])
        Dis     = DClass(self.config["DConvDim"], self.config["DKS"])
        
        self.reporter.writeInfo("Generator structure:")
        self.reporter.writeModel(Gen.__str__())
        # print(self.Decoder)
        self.reporter.writeInfo("Discriminator structure:")
        self.reporter.writeModel(Dis.__str__())
        
        Transform   = Transform_block().cuda()
        Gen         = Gen.cuda()
        Dis         = Dis.cuda()

        if self.config["mode"] == "finetune":
            model_path = os.path.join(self.config["projectCheckpoints"], "%d_Generator.pth"%self.config["checkpointStep"])
            Gen.load_state_dict(torch.load(model_path))
            print('loaded trained Generator model step {}...!'.format(self.config["checkpointStep"]))
            model_path = os.path.join(self.config["projectCheckpoints"], "%d_Discriminator.pth"%self.config["checkpointStep"])
            Dis.load_state_dict(torch.load(model_path))
            print('loaded trained Discriminator model step {}...!'.format(self.config["checkpointStep"]))
        
        print("build the optimizer...")
        # Loss and optimizer
        g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, 
                                    Gen.parameters()), lr_base, [beta1, beta2])

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
        for step in range(start, total_step):
            Dis.train()
            Gen.train()
            
            # ================== Train D ================== #
            # Compute loss with real images
            for _ in range(dStep):
                # start_time = time.time()
                # try:
                #     # content_images      = next(content_iter)
                #     # style_images,label  = next(style_iter)
                #     content_images,style_images,label  = next(total_iter) 
                # except:
                #     # style_iter          = iter(style_loader)
                #     # content_iter        = iter(content_loader)
                #     # style_images,label  = next(style_iter)
                #     # content_images      = next(content_iter)
                #     total_iter    = iter(total_loader)
                #     content_images,style_images,label  = next(total_iter) 
                # label           = label.view(batch_size,1)
                content_images,style_images,label  = total_loader.next()
                label           = label.long()
                # style_images    = style_images.cuda()
                # content_images  = content_images.cuda()
                # label           = label.cuda()
                # elapsed = time.time() - start_time
                # elapsed = str(datetime.timedelta(seconds=elapsed))
                # print("data load time %s"%elapsed)
                
                # start_time = time.time()
                d_out = Dis(style_images,label)
                d_loss_real = 0
                for i in range(output_size):
                    temp = Hinge_loss(1 - d_out[i]).mean()
                    # temp *= prep_weights[i]
                    d_loss_real += temp

                d_loss_photo = 0
                d_out = Dis(content_images,label)
                for i in range(output_size):
                    temp = Hinge_loss(1 + d_out[i]).mean()
                    # temp *= prep_weights[i]
                    d_loss_photo += temp

                # label        = label.view(batch_size,1)
                # style_labels = torch.zeros(batch_size, n_class).cuda().scatter_(1, label, 1)
                fake_image,_ = Gen(content_images,label)
                d_out = Dis(fake_image.detach(),label)
                d_loss_fake = 0
                for i in range(output_size):
                    temp = Hinge_loss(1 + d_out[i]).mean()
                    # temp *= prep_weights[i]
                    d_loss_fake += temp

                # Backward + Optimize
                d_loss = d_loss_real + d_loss_photo + d_loss_fake
                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()
                # elapsed = time.time() - start_time
                # elapsed = str(datetime.timedelta(seconds=elapsed))
                # print("inference time %s"%elapsed)
            
            # ================== Train G ================== #
            for _ in range(gStep):
                # try:
                #     # content_images      = next(content_iter)
                #     # style_images,label  = next(style_iter)
                #     content_images,_,_  = next(total_iter) 
                # except:
                #     # style_iter          = iter(style_loader)
                #     # content_iter        = iter(content_loader)
                #     # style_images,label  = next(style_iter)
                #     # content_images      = next(content_iter)
                #     total_iter    = iter(total_loader)
                #     content_images,_,_  = next(total_iter) 
                content_images,_,_  = total_loader.next()
                # content_images  = content_images.cuda()
                # label     = label.view(batch_size,1)
                # style_labels = torch.zeros(batch_size, n_class).cuda().scatter_(1, label, 1)
                # fake_image,real_feature = Gen(content_images,style_labels)
                fake_image,real_feature = Gen(content_images,label)
                fake_feature            = Gen(fake_image, get_feature=True)
                d_out                   = Dis(fake_image,label.long())
                
                g_feature_loss          = L1_loss(fake_feature,real_feature)
                g_transform_loss        = MSE_loss(Transform(content_images), Transform(fake_image))
                g_loss_fake = 0
                for i in range(output_size):
                    temp = -d_out[i].mean()
                    # temp *= prep_weights[i]
                    g_loss_fake += temp

                # backward & optimize
                g_loss_fake = g_loss_fake + g_feature_loss* feature_w + g_transform_loss* transform_w
                g_optimizer.zero_grad()
                g_loss_fake.backward()
                g_optimizer.step()
            

            # Print out log info
            if (step + 1) % log_frep == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print("[{}], Elapsed [{}], Step [{}/{}], d_out_real: {:.4f}, d_out_fake: {:.4f}, g_loss_fake: {:.4f}".
                      format(self.config["version"], elapsed, step + 1, total_step, 
                            d_loss_real.item(), d_loss_fake.item(), g_loss_fake.item()))
                
                if self.config["useTensorboard"]:
                    tensorboard_writer.add_scalar('data/d_loss_real', d_loss_real.item(),(step + 1))
                    tensorboard_writer.add_scalar('data/d_loss_fake', d_loss_fake.item(),(step + 1))
                    tensorboard_writer.add_scalar('data/d_loss', d_loss.item(), (step + 1))
                    tensorboard_writer.add_scalar('data/g_loss', g_loss_fake.item(), (step + 1))
                    tensorboard_writer.add_scalar('data/g_feature_loss', g_feature_loss, (step + 1))
                    tensorboard_writer.add_scalar('data/g_transform_loss', g_transform_loss, (step + 1))

            # Sample images
            # if (step +1 ) % sample_freq == 0:
            if (step + 1) % sample_freq == 0:
                print('Sample images {}_fake.jpg'.format(step + 1))
                Gen.eval()
                with torch.no_grad():
                    for batch_item in range(batch_size):
                        for class_item in range(n_class):
                            fake_images,_ = Gen(content_images[[batch_item],:,:,:], fix_label[:,class_item])
                            if class_item == 0:
                                oneimg_res = torch.cat([denorm(content_images[[batch_item],:,:,:]),denorm(fake_images.data)],3)
                            else:
                                oneimg_res = torch.cat([oneimg_res,denorm(fake_images.data)],3)
                        if batch_item == 0:
                            batch_image = oneimg_res
                        else:
                            batch_image = torch.cat([batch_image,oneimg_res],2)
                    save_image(batch_image,
                            os.path.join(sample_dir, '{}_fake.jpg'.format(step + 1)),nrow=1)
                    del batch_image
                    del oneimg_res
                    del fake_images
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
                print("Save step %d model checkpoints!"%(step+1))
                torch.save(Gen.state_dict(),
                           os.path.join(ckpt_dir, '{}_Generator.pth'.format(step + 1)))
                torch.save(Dis.state_dict(),
                           os.path.join(ckpt_dir, '{}_Discriminator.pth'.format(step + 1)))