#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: data_loader_modify.py
# Created Date: Saturday April 4th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Thursday, 16th April 2020 12:55:16 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################

from torch.utils import data
from torchvision import transforms as T
from PIL import Image
import torch
import os
import random
from pathlib import Path

class TestDataset(data.Dataset):
    """Dataset class for the Artworks dataset."""

    def __init__(self, image_dir, crop_size=768,batch_size=1, subffix=['jpg','png']):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir  = image_dir
        self.batch_size = batch_size
        self.subffix    = subffix
        self.pointer    = 0
        self.dataset    = []
        self.preprocess()
        self.num_images = len(self.dataset)
        self.crop_size  = float(crop_size)
        transform       = []
        transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        self.transforms = T.Compose(transform)

    def preprocess(self):
        """Preprocess the images for testing."""
        for item_subffix in self.subffix:
            images = Path(self.image_dir).glob('*.%s'%(item_subffix))
            for item in images:
                self.dataset.append(item)
        # self.dataset = images
        print('Finished preprocessing the test dataset, total image number: %d...'%len(self.dataset))

    def __call__(self):
        """Return one image."""
        # image = Image.open(os.path.join(self.image_dir, filename))
        if self.pointer>=self.num_images:
            raise Exception("Reader stop exception!")
        if (self.pointer+self.batch_size) > self.num_images:
            end = self.num_images
        else:
            end = self.pointer+self.batch_size
        for i in range(self.pointer, end):
            filename = self.dataset[i]
            image = Image.open(filename)
            # alpha = self.crop_size / float(min(image.size))
            # image = image.resize((int(image.size[0]*alpha), int(image.size[1]*alpha)))
            if (i-self.pointer) == 0:
                res   = self.transforms(image).unsqueeze(0)
            else:
                torch.cat((res,self.transforms(image).unsqueeze(0)),0)
        self.pointer = end
        return res,Path(filename).name
    def __len__(self):
        """Return the number of images."""
        return self.num_images

if __name__ == "__main__":
    image_dir = "D:\\PatchFace\\PleaseWork\\Benchmark\\styletransfer"
    test_data = TestDataset(image_dir,1)
    for i in range(40):
        images = test_data()
        print(images.shape)