# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:26:33 2019

@author: Wink
"""



import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import transforms
import data.BlindnessDataSet as BlindnessDataSet
import pdb
import models.net as net 
import pandas as pd


def getResult(predict):
    for i in range(len(predict)):
        if predict[i]<=0.5:
            predict[i] = 0
        elif predict[i]<1.5:
            predict[i] = 1
        elif predict[i]<2.5:
            predict[i] = 2
        elif predict[i]<3.5:
            predict[i] = 3 
        else:
            predict[i] = 4
    return predict

def test(csv_path, snapshot):
  
    print('Loading data.')

    transformations = transforms.Compose([transforms.Resize((224,224)),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset = BlindnessDataSet.BlindnessDataSet(csv_path, transformations, mode = 'test')
    
    test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=16,
                                               num_workers=2)


    print('Loading model.')
    model = net.ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 1) 
    saved_state_dict = torch.load(snapshot)
    new_state_dict = {k[7:]:v for k,v in saved_state_dict.items()}   

# load params
    model.load_state_dict(new_state_dict)
    model.cuda()
    
    
    print('Ready to test network.')

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

    result = np.zeros((len(dataset), 1))
    with torch.no_grad():
        for i, images in enumerate(test_loader):
            images = Variable(images).cuda()
    
            predict = model(images)
         #   pdb.set_trace()
            # predictions
            print(i)
      #      _, pred = torch.max(predict.data, 1)
           # pdb.set_trace()
            result[i*16:(i+1)*16] = predict.cpu().numpy().reshape(-1,1)  
            
          #  pdb.set_trace()
    result = getResult(result)
    sample = pd.read_csv(r'E:\kaggle\sample_submission.csv')
    sample.diagnosis = result.astype(int)
    sample.to_csv(r'E:\kaggle\submission1.csv', index=False)
if __name__ == '__main__':
    csv_path = r'E:\kaggle\test.csv'
    snapshot = r'C:\Users\Molly\Desktop\snapshot_epoch_16.pkl'
    test(csv_path, snapshot)