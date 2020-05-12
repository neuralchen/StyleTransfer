#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: train_norm.py
# Created Date: Monday April 6th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Wednesday, 8th April 2020 2:33:11 am
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################


import os
import time
import torch
import datetime

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd     import Variable
from torchvision.utils  import save_image

from models.Discriminator import Discriminator
from models.Generator     import Generator
from ops.Transform      import Transform_block
from utils.utils        import *
import logging
from torchvision import transforms
from data_helper.validation_dataloader import Validation_Data_Loader


class Trainer(object):
    def __init__(self, style_data_loader, content_data_loader, config):

        self.log_file           = os.path.join(config.log_path, config.version,config.version+"_log.log")
        self.report_file        = os.path.join(config.log_path, config.version,config.version+"_report.log")
        logging.basicConfig(filename=self.report_file,
            format='[%(asctime)s-%(levelname)s:%(message)s]', 
                level = logging.DEBUG,filemode='w',
                    datefmt='%Y-%m-%d%I:%M:%S %p')

        self.Experiment_description = config.experiment_description
        logging.info("Experiment description: \n%s"%self.Experiment_description)
        # Data loader
        self.style_data_loader = style_data_loader
        self.content_data_loader = content_data_loader

        # exact loss
        self.adv_loss = config.adv_loss
        logging.info("loss: %s"%self.adv_loss)

        # Model hyper-parameters
        self.imsize         = config.imsize
        logging.info("image size: %d"%self.imsize)
        self.batch_size     = config.batch_size
        logging.info("Batch size: %d"%self.batch_size)

        logging.info("Is shuffle: {}".format(config.is_shuffle))
        logging.info("Image center crop size: {}".format(config.center_crop))

        self.res_num        = config.res_num
        logging.info("resblock number: %d"%self.res_num)
        self.g_conv_dim     = config.g_conv_dim
        logging.info("generator convolution initial channel: %d"%self.g_conv_dim)
        self.d_conv_dim     = config.d_conv_dim
        logging.info("discriminator convolution initial channel: %d"%self.d_conv_dim)
        self.parallel       = config.parallel
        logging.info("Is multi-GPU parallel: %s"%str(self.parallel))
        self.gpus           = config.gpus
        logging.info("GPU number: %s"%self.gpus)
        self.total_step     = config.total_step
        logging.info("Total step: %d"%self.total_step)
        self.d_iters        = config.d_iters
        self.g_iters        = config.g_iters
        self.total_iters_ratio=config.total_iters_ratio
        
        self.num_workers    = config.num_workers

        self.g_lr           = config.g_lr
        logging.info("Generator learning rate: %f"%self.g_lr)
        self.d_lr           = config.d_lr
        logging.info("Discriminator learning rate: %f"%self.d_lr)
        self.lr_decay       = config.lr_decay
        logging.info("Learning rate decay: %f"%self.lr_decay)
        self.beta1          = config.beta1
        logging.info("Adam opitimizer beta1: %f"%self.beta1)
        self.beta2          = config.beta2
        logging.info("Adam opitimizer beta2: %f"%self.beta2)

        self.pretrained_model   = config.pretrained_model
        self.use_pretrained_model = config.use_pretrained_model
        logging.info("Use pretrained model: %s"%str(self.pretrained_model))

        self.use_tensorboard    = config.use_tensorboard
        logging.info("Use tensorboard: %s"%str(self.use_tensorboard))

        self.check_point_path   = config.check_point_path
        self.sample_path        = config.sample_path
        self.summary_path       = config.summary_path
        self.validation_path    = config.validation
        # val_dataloader          = Validation_Data_Loader(self.validation_path,self.imsize)
        # self.validation_data    = val_dataloader.load_validation_images()
        # valres_path = os.path.join(config.log_path, config.version, "valres")
        # if not os.path.exists(valres_path):
        #     os.makedirs(valres_path)
        # self.valres_path = valres_path

        self.log_step           = config.log_step
        self.sample_step        = config.sample_step
        self.model_save_step    = config.model_save_step
        self.prep_weights       = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.transform_loss_w   = config.transform_loss_w
        logging.info("transform loss weight: %f"%self.transform_loss_w)
        self.feature_loss_w     = config.feature_loss_w
        logging.info("feature loss weight: %f"%self.feature_loss_w)
        self.style_class        = config.style_class
        self.real_prep_threshold= config.real_prep_threshold
        logging.info("real label threshold: %f"%self.real_prep_threshold)
        # self.TVLossWeight       = config.TV_loss_weight
        # logging.info("TV loss weight: %f"%self.TVLossWeight)


        self.discr_success_rate = config.discr_success_rate
        logging.info("discriminator success rate: %f"%self.discr_success_rate)

        logging.info("Is conditional generating: %s"%str(config.condition_model))

        self.device = torch.device('cuda:%s'%config.default_GPU if torch.cuda.is_available() else 'cpu')

        print('build_model...')
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.use_pretrained_model:
            print('load_pretrained_model...')

    def train(self):

        # Data iterator
        style_iter      = iter(self.style_data_loader)
        content_iter    = iter(self.content_data_loader)

        step_per_epoch  = len(self.style_data_loader)
        model_save_step = int(self.model_save_step)

        # Fixed input for debugging

        # Start with trained model
        if self.use_pretrained_model:
            start = self.pretrained_model + 1
        else:
            start = 0
        alternately_iter     = 0
        self.d_iters         = self.d_iters * self.total_iters_ratio
        max_alternately_iter = self.d_iters + self.total_iters_ratio * self.g_iters
        d_acc         = 0
        real_acc      = 0
        photo_acc     = 0
        fake_acc      = 0
        win_rate      = self.discr_success_rate
        discr_success = self.discr_success_rate
        alpha         = 0.05


        real_labels = []
        fake_labels = []
        # size = [[self.batch_size,122*122],[self.batch_size,58*58],[self.batch_size,10*10],[self.batch_size,2*2],[self.batch_size,2*2]]
        size = [[self.batch_size,1,760,760],[self.batch_size,1,371,371],[self.batch_size,1,83,83],[self.batch_size,1,11,11],[self.batch_size,1,6,6]]
        for i in range(5):
            real_label = torch.ones(size[i], device=self.device)
            fake_label = torch.zeros(size[i], device=self.device)
            # threshold = torch.zeros(size[i], device=self.device)
            real_labels.append(real_label)
            fake_labels.append(fake_label)

        # Start time
        print('Start   ======  training...')
        start_time = time.time()
        for step in range(start, self.total_step):
            self.Discriminator.train()
            self.Generator.train()
            # self.Decoder.train()
            try:
                content_images =next(content_iter)
                style_images = next(style_iter)
            except:
                style_iter      = iter(self.style_data_loader)
                content_iter    = iter(self.content_data_loader)
                style_images = next(style_iter)
                content_images = next(content_iter)
            style_images    = style_images.to(self.device)
            content_images  = content_images.to(self.device)
            # ================== Train D ================== #
            # Compute loss with real images
            if discr_success < win_rate:
                real_out = self.Discriminator(style_images)
                d_loss_real = 0
                real_acc = 0
                for i in range(len(real_out)):
                    temp = self.C_loss(real_out[i],real_labels[i]).mean()
                    real_acc +=  torch.gt(real_out[i],0).type(torch.float).mean()
                    temp *= self.prep_weights[i]
                    d_loss_real += temp
                real_acc /= len(real_out)

                d_loss_photo = 0
                photo_out = self.Discriminator(content_images)
                photo_acc = 0
                for i in range(len(photo_out)):
                    temp = self.C_loss(photo_out[i],fake_labels[i])
                    photo_acc +=  torch.lt(photo_out[i],0).type(torch.float).mean()
                    temp *= self.prep_weights[i]
                    d_loss_photo += temp
                photo_acc /= len(photo_out) 

                fake_image,_ = self.Generator(content_images)
                fake_out = self.Discriminator(fake_image.detach())
                d_loss_fake = 0
                fake_acc = 0
                for i in range(len(fake_out)):
                    temp = self.C_loss(fake_out[i],fake_labels[i]).mean()
                    fake_acc +=  torch.lt(fake_out[i],0).type(torch.float).mean()
                    temp *= self.prep_weights[i]
                    d_loss_fake += temp
                fake_acc /= len(fake_out) 
                d_acc = ((real_acc + photo_acc + fake_acc)/3).item()
                discr_success = discr_success * (1. - alpha) + alpha * d_acc
                # Backward + Optimize
                d_loss = d_loss_real + d_loss_photo + d_loss_fake
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()
            else:
                # ================== Train G ================== #   
                #      
                fake_image, real_feature= self.Generator(content_images)
                fake_feature            = self.Generator(fake_image, get_feature = True)
                fake_out                = self.Discriminator(fake_image)
                g_feature_loss          = self.L1_loss(fake_feature,real_feature)
                g_transform_loss        = self.MSE_loss(self.Transform(content_images),self.Transform(fake_image))
                g_loss_fake = 0
                g_acc = 0
                for i in range(len(fake_out)):
                    temp = self.C_loss(fake_out[i],real_labels[i]).mean()
                    g_acc +=  torch.gt(fake_out[i],0).type(torch.float).mean()
                    temp *= self.prep_weights[i]
                    g_loss_fake += temp
                g_acc /= len(fake_out)
                g_loss_fake = g_loss_fake + g_feature_loss*self.feature_loss_w + \
                                        g_transform_loss*self.transform_loss_w
                discr_success = discr_success * (1. - alpha) + alpha * (1.0 - g_acc)
                self.reset_grad()
                g_loss_fake.backward()
                self.g_optimizer.step()
                # self.decoder_optimizer.step()
            

            # Print out log info
            if (step + 1) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print("Elapsed [{}], G_step [{}/{}], D_step[{}/{}], d_out_real: {:.4f}, d_out_fake: {:.4f}, g_loss_fake: {:.4f}".
                      format(elapsed, step + 1, self.total_step, (step + 1),
                             self.total_step , d_loss_real.item(), d_loss_fake.item(), g_loss_fake.item()))
                
                if self.use_tensorboard:
                    self.writer.add_scalar('data/d_loss_real', d_loss_real.item(),(step + 1))
                    self.writer.add_scalar('data/d_loss_fake', d_loss_fake.item(),(step + 1))
                    self.writer.add_scalar('data/d_loss', d_loss.item(), (step + 1))
                    self.writer.add_scalar('data/g_loss', g_loss_fake.item(), (step + 1))
                    self.writer.add_scalar('data/g_feature_loss', g_feature_loss, (step + 1))
                    self.writer.add_scalar('data/g_transform_loss', g_transform_loss, (step + 1))
                    # self.writer.add_scalar('data/g_tv_loss', g_tv_loss, (step + 1))
                    self.writer.add_scalar('acc/real_acc', real_acc.item(), (step + 1))
                    self.writer.add_scalar('acc/photo_acc', photo_acc.item(), (step + 1))
                    self.writer.add_scalar('acc/fake_acc', fake_acc.item(), (step + 1))
                    self.writer.add_scalar('acc/disc_acc', d_acc, (step + 1))
                    self.writer.add_scalar('acc/g_acc', g_acc, (step + 1))
                    self.writer.add_scalar("acc/discr_success",discr_success,(step+1))
                    

            # Sample images
            if (step + 1) % self.sample_step == 0:
                print('Sample images {}_fake.png'.format(step + 1))
                fake_images,_ = self.Generator(content_images)
                saved_image1 = torch.cat([denorm(content_images),denorm(fake_images.data)],3)
                saved_image2 = torch.cat([denorm(style_images),denorm(fake_images.data)],3)
                wocao        = torch.cat([saved_image1,saved_image2],2)
                save_image(wocao,
                           os.path.join(self.sample_path, '{}_fake.jpg'.format(step + 1)))
                # print("Transfer validation images")
                # num = 1
                # for val_img in self.validation_data:
                #     print("testing no.%d img"%num)
                #     val_img = val_img.to(self.device)
                #     fake_images,_ = self.Generator(val_img)
                #     saved_val_image = torch.cat([denorm(val_img),denorm(fake_images)],3)
                #     save_image(saved_val_image,
                #            os.path.join(self.valres_path, '%d_%d.jpg'%((step+1),num)))
                #     num +=1
                # save_image(denorm(displaymask.data),os.path.join(self.sample_path, '{}_mask.png'.format(step + 1)))

            if (step+1) % model_save_step==0:
                torch.save(self.Generator.state_dict(),
                           os.path.join(self.check_point_path , '{}_Generator.pth'.format(step + 1)))
                torch.save(self.Discriminator.state_dict(),
                           os.path.join(self.check_point_path , '{}_Discriminator.pth'.format(step + 1)))
            # alternately_iter += 1
            # alternately_iter %= max_alternately_iter
            
            

    def build_model(self):
        # code_dim=100, n_class=1000
        self.Generator = Generator(chn=self.g_conv_dim, k_size= 3, res_num= self.res_num).to(self.device)
        self.Discriminator = Discriminator(chn=self.d_conv_dim, k_size= 3).to(self.device)
        self.Transform = Transform_block().to(self.device)
        if self.parallel:

            print('use parallel...')
            print('gpuids ', self.gpus)
            gpus = [int(i) for i in self.gpus.split(',')]
    
            self.Generator      = nn.DataParallel(self.Generator, device_ids=gpus)
            self.Discriminator  = nn.DataParallel(self.Discriminator, device_ids=gpus)
            self.Transform      = nn.DataParallel(self.Transform, device_ids=gpus)

        # self.G.apply(weights_init)
        # self.D.apply(weights_init)

        # Loss and optimizer
        # self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])

        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, 
                                    self.Generator.parameters()), self.g_lr, [self.beta1, self.beta2])
        # self.decoder_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, 
        #                             self.Decoder.parameters()), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, 
                                    self.Discriminator.parameters()), self.d_lr, [self.beta1, self.beta2])
        # self.L1_loss = torch.nn.L1Loss()
        self.MSE_loss = torch.nn.MSELoss()
        self.L1_loss = torch.nn.SmoothL1Loss()
        self.C_loss = torch.nn.BCEWithLogitsLoss()
        # self.TV_loss = TVLoss(self.TVLossWeight,self.imsize,self.batch_size)
        
        # print networks
        logging.info("Generator structure:")
        logging.info(self.Generator)
        # print(self.Decoder)
        logging.info("Discriminator structure:")
        logging.info(self.Discriminator)

    def build_tensorboard(self):
        from tensorboardX import SummaryWriter
        # from logger import Logger
        # self.logger = Logger(self.log_path)
        self.writer = SummaryWriter(log_dir=self.summary_path)


    def load_pretrained_model(self):
        self.Generator.load_state_dict(torch.load(os.path.join(
            self.check_point_path , '{}_Generator.pth'.format(self.pretrained_model))))
        self.Discriminator.load_state_dict(torch.load(os.path.join(
            self.check_point_path , '{}_Discriminator.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def reset_grad(self):
        self.g_optimizer.zero_grad()
        # self.decoder_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def save_sample(self, data_iter):
        real_images, _ = next(data_iter)
        save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))