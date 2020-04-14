#############################################################
# File: parameters.py
# Created Date: Monday April 6th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Wednesday, 15th April 2020 12:55:32 am
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################

import argparse

def str2bool(v):
    return v.lower() in ('true')

def getParameters():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--mode', type=str, default="train", choices=['train', 'finetune','test','debug'])
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--dataloader_workers', type=int, default=4)
    parser.add_argument('--checkpoint', type=int, default=126000)
    # training
    parser.add_argument('--version', type=str, default='SN3')
    parser.add_argument('--experimentDescription', type=str, default="modify the discriminator to a SN one and replace loss with the hingle loss D:G 3:1, add color jitter, add another downsampling layer in D")
    parser.add_argument('--trainYaml', type=str, default="train_SN1.yaml")

    # test
    parser.add_argument('--testScriptsName', type=str, default='common')
    parser.add_argument('--nodeName', type=str, default='localhost',choices=['localhost', '4card', '8card','lyh','loc','localhost'])
    parser.add_argument('--testBatchSize', type=int, default=1)
    parser.add_argument('--totalImg', type=int, default=20)
    parser.add_argument('--saveTestImg', type=str2bool, default=True)
    parser.add_argument('--testImgRoot', type=str, default="D:\\PatchFace\\PleaseWork\\Benchmark\\styletransfer")
    parser.add_argument('--useSpecifiedImg', type=str2bool, default=False)
    parser.add_argument('--specifiedTestImages', nargs='+', help='selected images for validation', 
            # '000121.jpg','000124.jpg','000129.jpg','000132.jpg','000135.jpg','001210.jpg','001316.jpg', 
            default=[183947])
            
    return parser.parse_args()