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
import models.net as net 
import data.BlindnessDataSet as BlindnessDataSet
import torch.utils.model_zoo as model_zoo
import pdb
from validation import validation
from utils.utils import plot_loss


log_dir = r'E:\BlindnessDetection\output\log'
output_string = 'blindness_' + strftime("%Y-%m-%d-%H-%M-%S", localtime())
output_snapshot_dir = r'E:\BlindnessDetection\output\snapshots'
if not os.path.exists(r'E:\BlindnessDetection\output'):
    os.mkdir(r'E:\BlindnessDetection\output')
if not os.path.exists(output_snapshot_dir):
    os.mkdir(output_snapshot_dir)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='BlindnessDetection baseline.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
          default=50, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=16, type=int)
    parser.add_argument('--csv_path', dest='csv_path', help='The path of train_csv',
          default=r'E:\kaggle\train.csv', type=str)
    parser.add_argument('--lr', dest='lr', help='Base learning rate.',
          default=0.001, type=float)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
          default='', type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = args.gpu_id

    # ResNet50 structure
    model = net.ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
 
    if args.snapshot != '':
        saved_state_dict = torch.load(args.snapshot)
        try:         
            for k, v in model.state_dict.items():
                name = 'module.'+k  # add `module.`
                saved_state_dict[name] = v
            # load params
            model.load_state_dict(saved_state_dict)
    #        self.net.load_state_dict(new_state_dict)
        except Exception as e:
            print(e)

    print('Loading data.')

    transformations = transforms.Compose([transforms.Resize((224,224)),
    transforms.RandomCrop(224), transforms.ToTensor()])
    dataset = BlindnessDataSet.BlindnessDataSet(args.csv_path, transformations)

    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=2)

    model.cuda(gpu)
    
    criterion = nn.CrossEntropyLoss().cuda(gpu)

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    loss_log = []
    print('Ready to train network.')
    for epoch in range(num_epochs):

        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images).cuda(gpu)           
            labels = Variable(labels).cuda(gpu)

            # Forward pass
            y_predict = model(images)

            # Cross entropy loss
            loss = criterion(y_predict, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 10 == 0:
                
                text = 'Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' %(epoch+1, num_epochs, i+1, len(dataset)//batch_size, loss.item())
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
            
            validation(args.csv_path,log_dir_time, model)                    
    ###  now the validation data is the same as train . modify next time
       
    iter_list = [j for j in range(len(loss_log))]
    plot_loss(log_dir_time+'\loss_epoch_'   + str(epoch) + '.png', iter_list, loss_log)