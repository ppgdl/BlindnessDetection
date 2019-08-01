# -*- coding: utf-8 -*-

import sys, os, argparse
sys.path.append('../')
from time import strftime,localtime
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import models.ResNet50 as ResNet50
import models.SEResNet as SEResNet50
from models.efficientnet_pytorch import *
from tools.validation import validation
import data.BlindnessDataSet as BlindnessDataSet
import torch.utils.model_zoo as model_zoo
from utils.plot_curve import plot_loss
from torch.optim import lr_scheduler
from sklearn import metrics
from torch.autograd import Function
import pdb
from sklearn.model_selection import train_test_split
import pandas as pd
ROOT = os.path.join(os.getcwd(), '..')



def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='BlindnessDetection baseline.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default='0', type=str)
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
            default=50, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
            default=8, type=int)
    parser.add_argument('--train_path', dest='train_path', help='The path of train_list',
            default='', type=str)
    parser.add_argument('--lr', dest='lr', help='Base learning rate.',
            default=0.001, type=float)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
            default='', type=str)
    parser.add_argument('--net', dest='model', help='Model type',
            default='ResNet50', type=str)
    parser.add_argument('--log', dest='log', help='Log directory',
            default='logs', type=str)
    parser.add_argument('--output', dest='output', help='snapshot directory',
            default='./output/snapshot', type=str)

    args,_ = parser.parse_known_args()

    return args


def make_model(model_type):
    if model_type is None:
        print("model_type must be defined!")

    if model_type == 'ResNet50':
        return ResNet50.make_model(model_type)
    elif model_type == 'SEResNet50':
        return SEResNet50.make_model(model_type)     
    else:
        print("model_type: {:} is not existed now!")

def load_filtered_state_dict(model, snapshot_path):
    
    snapshot = torch.load(snapshot_path)
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict and k!='fc.weight' and k!='fc.bias'}
 #   pdb.set_trace()
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)

class kappaLoss(nn.Module):
    def __init__(self):
        super(kappaLoss, self).__init__()
        return
        
    def forward(self, X, y):
        X = X.numpy().cpu()
        y = y.numpy().cpu()
        for i, pred in enumerate(X):
            if pred < 0.5:
                X[i] = 0
            elif pred < 1.5:
                X[i] = 1
            elif pred < 2.5:
                X[i] = 2
            elif pred < 3.5:
                X[i] = 3
            else:
                X[i] = 4

        kappa_loss = metrics.cohen_kappa_score(y, X, weights='quadratic')
        return torch.tensor(kappa_loss, require_grad = True).cuda()
    
    def backward(grad):
        return grad

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu_id = args.gpu_id
    model_type = args.model
    train_path = args.train_path
 #   pdb.set_trace()
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    gpu =[int(i) for i in gpu_id.split(',')] 


    # train data
    if train_path == '':
        train_path = r'E:\kaggle\train.csv'
    if not os.path.exists(train_path):
        raise ValueError(
            "train path: {:} is not existsed".format(train_path)
        )

    # log directory
    log_dir = args.log
    log_dir = os.path.join(ROOT, args.log)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    output_snapshot_dir = args.output
    if not os.path.exists(os.path.join(ROOT, 'output', output_snapshot_dir)):
        os.makedirs(os.path.join(ROOT, 'output', output_snapshot_dir))
    output_string = 'blindness_' + strftime("%Y-%m-%d-%H-%M-%S", localtime())
#    model = make_model(model_type)

    model = model.EfficientNet.from_name('efficientnet-b0')
    model.load_state_dict(torch.load(r'E:\BlindnessDetection\output\efficientnet-b0-08094119.pth'))
    in_features = model._fc.in_features
    model._fc = nn.Linear(in_features, 1)
    
    if len(gpu) > 1:
        print(' ---  using multi-GPU')
        model = nn.DataParallel(model).cuda()
    else:
        model.cuda()
#    # ResNet50 structure
#    model = net.ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 1)
 
#    if args.snapshot == '':
#        load_filtered_state_dict(model, r'E:\BlindnessDetection\output\snapshots\resnet50-19c8e357.pth')
#    else:
#        saved_state_dict = torch.load(args.snapshot)
#        try:         
#            for k, v in model.state_dict.items():
#                name = 'module.'+k  # add `module.`
#                saved_state_dict[name] = v
#            # load params
#            model.load_state_dict(saved_state_dict)
#        except Exception as e:
#            print(e)

    print('Loading data.')

    transformations = transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation((-120, 120)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_csv = pd.read_csv(train_path)
    train_df, val_df = train_test_split(train_csv, test_size=0.1, random_state=2019, stratify=train_csv.diagnosis)
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    dataset_train = BlindnessDataSet.BlindnessDataSet(train_df, transformations)  
    dataset_val = BlindnessDataSet.BlindnessDataSet(val_df, transformations) 

    train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                               batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=dataset_val,
                                               batch_size=batch_size,
                                               shuffle=False)    
    criterion = nn.MSELoss().cuda()
 
#    criterion = kappaLoss().cuda(gpu)

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10)
    loss_log = []
    print('Ready to train network.')
    for epoch in range(num_epochs):
         
        scheduler.step()
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images).cuda()           
            labels = Variable(labels).float().cuda()
            labels = labels.view(-1,1)
            # Forward pass
            y_predict = model(images)
            
            # Cross entropy loss
            loss = criterion(y_predict, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 10 == 0:
                
                text = 'Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' %(epoch+1, num_epochs, i+1, len(dataset_train)//batch_size, loss.item())
                print(text)
                log_dir_time = os.path.join(log_dir, output_string)
                if not os.path.exists(log_dir_time):
                    os.mkdir(log_dir_time)
                with open(log_dir_time+'\\'+'train_log.txt','a') as f:
                    f.write("{}\n".format(text))
                loss_log.append(loss)

        # Save models at numbered epochs.
        if epoch % 5 == 0 and epoch < num_epochs:
            
            print('Taking snapshot...')
            output_snapshot = os.path.join(output_snapshot_dir, output_string)
            if not os.path.exists(output_snapshot):
                os.mkdir(output_snapshot)
                
            torch.save(model.state_dict(), output_snapshot + '\snapshot_epoch_'+ str(epoch+1) + '.pkl')
            
            validation(val_loader,log_dir_time, model)

    ###  now the validation data is the same as train . modify next time
       
    iter_list = [j for j in range(len(loss_log))]
    plot_loss(log_dir_time+'\loss_epoch_'   + str(epoch) + '.png', iter_list, loss_log)