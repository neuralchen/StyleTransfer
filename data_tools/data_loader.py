#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: data_loader_modify.py
# Created Date: Saturday April 4th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Friday, 10th April 2020 3:10:42 pm
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

class ArtDataset(data.Dataset):
    """Dataset class for the Artworks dataset."""

    def __init__(self, image_dir, selectedClass, transform, subffix='jpg', random_seed=1234):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir  = image_dir
        self.transform  = transform
        self.selectedClass = selectedClass
        self.subffix    = subffix
        self.dataset    = []
        self.random_seed= random_seed
        self.preprocess()
        self.num_images = len(self.dataset)

    def preprocess(self):
        """Preprocess the Artworks dataset."""
        images = Path(self.image_dir).glob('%s/*.%s'%(self.selectedClass, self.subffix))
        for item in images:
            self.dataset.append(item)
        random.seed(self.random_seed)
        random.shuffle(self.dataset)
        # self.dataset = images
        print('Finished preprocessing the Art Works dataset, total image number: %d...'%len(self.dataset))

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        # image = Image.open(os.path.join(self.image_dir, filename))
        filename = self.dataset[index]
        image = Image.open(filename)
        # print("original size",image.size)
        if max(image.size) > 1800:
            alpha = 1800. / float(min(image.size))
            image = image.resize((int(image.size[0]*alpha), int(image.size[1]*alpha)))
        if max(image.size) < 800:
            # Resize the smallest side of the image to 800px
            alpha = 800. / float(min(image.size))
            if alpha < 4.:
                image = image.resize((int(image.size[0]*alpha), int(image.size[1]*alpha)))
            else:
                image = image.resize((800,800))
        if min(image.size) < 800:
            # Resize the smallest side of the image to 800px
            alpha = 800. / float(min(image.size))
            if alpha < 4.:
                image = image.resize((int(image.size[0]*alpha), int(image.size[1]*alpha)))
            else:
                image = image.resize((800,800))
        # print("after resize",image.size)
        res   = self.transform(image)
        return res

    def __len__(self):
        """Return the number of images."""
        return self.num_images

class ContentDataset(data.Dataset):
    """Dataset class for the Content dataset."""

    def __init__(self, image_dir, selectedClass, transform, subffix='jpg', random_seed=1234):
        """Initialize and preprocess the Content dataset."""
        self.image_dir  = image_dir
        self.transform  = transform
        self.selectedClass = selectedClass
        self.subffix    = subffix
        self.dataset    = []
        self.random_seed= random_seed
        self.preprocess()
        self.num_images = len(self.dataset)

    def preprocess(self):
        """Preprocess the Content dataset."""
        for dir_item in self.selectedClass:
            join_path = Path(self.image_dir,dir_item)
            if join_path.exists():
                print("processing %s"%dir_item,end='\r')
                images = join_path.glob('*.%s'%(self.subffix))
                for item in images:
                    self.dataset.append(item)
            else:
                print("%s dir does not exist!"%dir_item,end='\r')
        # self.dataset = images
        random.seed(self.random_seed)
        random.shuffle(self.dataset)
        print('Finished preprocessing the Content dataset, total image number: %d...'%len(self.dataset))

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        filename = self.dataset[index]
        image = Image.open(filename)
        image = image.resize((image.size[0]*2,image.size[1]*2))
        # print("original size",image.size)
        if max(image.size) > 1800:
            alpha = 1800. / float(min(image.size))
            image = image.resize((int(image.size[0]*alpha), int(image.size[1]*alpha)))
        if max(image.size) < 800:
            # Resize the smallest side of the image to 800px
            alpha = 800. / float(min(image.size))
            if alpha < 4.:
                image = image.resize((int(image.size[0]*alpha), int(image.size[1]*alpha)))
            else:
                image = image.resize((800,800))
        if min(image.size) < 800:
            # Resize the smallest side of the image to 800px
            alpha = 800. / float(min(image.size))
            if alpha < 4.:
                image = image.resize((int(image.size[0]*alpha), int(image.size[1]*alpha)))
            else:
                image = image.resize((800,800))
        # print("after resize",image.size)
        res   = self.transform(image)
        return res

    def __len__(self):
        """Return the number of images."""
        return self.num_images

def getLoader(image_dir, selected_dir, crop_size=178, batch_size=16, dataset_name='Style', num_workers=8, colorJitterEnable=False):
    """Build and return a data loader."""
    transforms = []
    transforms.append(T.RandomCrop(crop_size))
    transforms.append(T.RandomHorizontalFlip())
    transforms.append(T.RandomVerticalFlip())
    # colorBrightness = 0.01

    # if colorJitterEnable:
    #     transforms.append(T.ColorJitter(brightness=colorBrightness,\
    #                         contrast=colorContrast,saturation=colorSaturation, hue=colorHue))
    # transform.append(T.Resize(image_size,interpolation=PIL.Image.BICUBIC))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transforms = T.Compose(transforms)

    if dataset_name.lower() == 'style':
        dataset = ArtDataset(image_dir, selected_dir, transforms)
    if dataset_name.lower() == 'content':
        dataset = ContentDataset(image_dir, selected_dir, transforms)
    # elif dataset.lower() == 'Content':
    #     dataset = CelebA(image_dir, attr_path, selected_attrs, 
    #         transform, mode,batch_size,image_size,toPatch,microPatchSize)
    data_loader = data.DataLoader(dataset=dataset,batch_size=batch_size,drop_last=True,shuffle=True,num_workers=num_workers,pin_memory=True)
    return data_loader

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

if __name__ == "__main__":
    from torchvision.utils import save_image
    selected_attrs  = "vangogh"
    categories_names = \
        ['a/abbey', 'a/arch', 'a/amphitheater', 'a/aqueduct', 'a/arena/rodeo', 'a/athletic_field/outdoor',
         'b/badlands', 'b/balcony/exterior', 'b/bamboo_forest', 'b/barn', 'b/barndoor', 'b/baseball_field',
         'b/basilica', 'b/bayou', 'b/beach', 'b/beach_house', 'b/beer_garden', 'b/boardwalk', 'b/boathouse',
         'b/botanical_garden', 'b/bullring', 'b/butte', 'c/cabin/outdoor', 'c/campsite', 'c/campus',
         'c/canal/natural', 'c/canal/urban', 'c/canyon', 'c/castle', 'c/church/outdoor', 'c/chalet',
         'c/cliff', 'c/coast', 'c/corn_field', 'c/corral', 'c/cottage', 'c/courtyard', 'c/crevasse',
         'd/dam', 'd/desert/vegetation', 'd/desert_road', 'd/doorway/outdoor', 'f/farm', 'f/fairway',
         'f/field/cultivated', 'f/field/wild', 'f/field_road', 'f/fishpond', 'f/florist_shop/indoor',
         'f/forest/broadleaf', 'f/forest_path', 'f/forest_road', 'f/formal_garden', 'g/gazebo/exterior',
         'g/glacier', 'g/golf_course', 'g/greenhouse/indoor', 'g/greenhouse/outdoor', 'g/grotto', 'g/gorge',
         'h/hayfield', 'h/herb_garden', 'h/hot_spring', 'h/house', 'h/hunting_lodge/outdoor', 'i/ice_floe',
         'i/ice_shelf', 'i/iceberg', 'i/inn/outdoor', 'i/islet', 'j/japanese_garden', 'k/kasbah',
         'k/kennel/outdoor', 'l/lagoon', 'l/lake/natural', 'l/lawn', 'l/library/outdoor', 'l/lighthouse',
         'm/mansion', 'm/marsh', 'm/mausoleum', 'm/moat/water', 'm/mosque/outdoor', 'm/mountain',
         'm/mountain_path', 'm/mountain_snowy', 'o/oast_house', 'o/ocean', 'o/orchard', 'p/park',
         'p/pasture', 'p/pavilion', 'p/picnic_area', 'p/pier', 'p/pond', 'r/raft', 'r/railroad_track',
         'r/rainforest', 'r/rice_paddy', 'r/river', 'r/rock_arch', 'r/roof_garden', 'r/rope_bridge',
         'r/ruin', 's/schoolhouse', 's/sky', 's/snowfield', 's/swamp', 's/swimming_hole',
         's/synagogue/outdoor', 't/temple/asia', 't/topiary_garden', 't/tree_farm', 't/tree_house',
         'u/underwater/ocean_deep', 'u/utility_room', 'v/valley', 'v/vegetable_garden', 'v/viaduct',
         'v/village', 'v/vineyard', 'v/volcano', 'w/waterfall', 'w/watering_hole', 'w/wave',
         'w/wheat_field', 'z/zen_garden', 'a/alcove', 'a/apartment-building/outdoor', 'a/artists_loft',
         'b/building_facade', 'c/cemetery']

    datapath        = "D:\\F_Disk\\data_set\\Art_Data\\data_art"
    # contentdatapath = "D:\\迅雷下载\\data_large"
    imsize          = 768
    datasetloader   = getLoader(datapath, selected_attrs, imsize,1,'Style',0)
    wocao           = iter(datasetloader)
    for i in range(30000):
        print("new batch")
        image       = next(wocao)
        # save_image(denorm(image), "./wocao/%d-content.jpg"%i, nrow=4, padding=1)
    pass
    # import cv2
    # import os
    # for dir_item in categories_names:
    #     join_path = Path(contentdatapath,dir_item)
    #     if join_path.exists():
    #         print("processing %s"%dir_item,end='\r')
    #         images = join_path.glob('*.%s'%("jpg"))
    #         for item in images:
    #             temp_path = str(item)
    #             # temp = cv2.imread(temp_path)
    #             temp = Image.open(temp_path)
    #             if temp.layers<3:
    #                 print("remove broken image...")
    #                 print("image name:%s"%temp_path)
    #                 del temp
    #                 os.remove(item)