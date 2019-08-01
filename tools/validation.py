# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:23:47 2019

@author: Molly
"""

import sys, os, argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
import data.BlindnessDataSet as BlindnessDataSet
import pdb


def validation(val_loader,log_dir_time, model_):
  
#    print('Loading data.')
#
#    transformations = transforms.Compose([transforms.Resize((224,224)),
#    transforms.CenterCrop(224), transforms.ToTensor(),
#    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
#
#    dataset = BlindnessDataSet.BlindnessDataSet(csv_path, transformations)
#    
#    test_loader = torch.utils.data.DataLoader(dataset=dataset,
#                                               batch_size=16)

    print('Ready to validate network.')

    # Test the Model
    model_.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    
    val_loss = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()
    
            predict = model_(images)
         #   pdb.set_trace()
            val_loss += torch.sum((predict - labels)**2).item()/len(val_loader)
                
        # Save first image in batch with pose cube or axis.     
    text = 'Validation loss  %.4f ' %  (val_loss)
    print(text)
    with open(log_dir_time+'\\' + 'validation_log.txt','a') as f:
        f.write("{}\n".format(text))