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
import data.BlindnessDataSet as BlindnessDataSet
import torch.utils.model_zoo as model_zoo
from validation import validation
from utils.plot_curve import plot_loss

<<<<<<< HEAD:tools/train.py
ROOT = os.path.join(os.getcwd(), '..')
=======


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='BlindnessDetection baseline.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
            default=50, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
            default=16, type=int)
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
            default='snapshot', type=str)

    args = parser.parse_args()

    return args

<<<<<<< HEAD:tools/train.py

def make_model(model_type):
    if model_type is None:
        print("model_type must be defined!")

    if model_type == 'ResNet50':
        return ResNet50.make_model(model_type)
    elif model_type == 'SEResNet50':
        return SEResNet50.make_model(model_type)
    else:
        print("model_type: {:} is not existed now!")
=======
def load_filtered_state_dict(model, snapshot_path):
    
    snapshot = torch.load(snapshot_path)
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict and k!='fc.weight' and k!='fc.bias'}
 #   pdb.set_trace()
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)
>>>>>>> baseline_wink:models/train.py

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = args.gpu_id
<<<<<<< HEAD:tools/train.py
    model_type = args.model
    train_path = args.train_path

    # train data
    if train_path == '':
        train_path = os.path.join(ROOT, 'data', 'blindness', 'train_list.txt')
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
    model = make_model(model_type)

    # load snapshot
    if args.snapshot != '':
=======

    # ResNet50 structure
    model = net.ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 1)
 
    if args.snapshot == '':
        load_filtered_state_dict(model, r'E:\BlindnessDetection\output\snapshots\resnet50-19c8e357.pth')
    else:
>>>>>>> baseline_wink:models/train.py
        saved_state_dict = torch.load(args.snapshot)
        try:         
            for k, v in model.state_dict.items():
                name = 'module.'+k  # add `module.`
                saved_state_dict[name] = v
            # load params
            model.load_state_dict(saved_state_dict)
        except Exception as e:
            print(e)

    print('Loading data.')

    transformations = transforms.Compose([transforms.Resize((224,224)),

    transforms.RandomCrop(224), transforms.ToTensor()],
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    dataset = BlindnessDataSet.BlindnessDataSet(csv_path, transformations)


    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    model.cuda(gpu)
    
    criterion = nn.MSELoss().cuda(gpu)

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    loss_log = []
    print('Ready to train network.')
    for epoch in range(num_epochs):

        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images).cuda(gpu)           
            labels = Variable(labels).float().cuda(gpu)
            labels = labels.view(-1,1)
            # Forward pass
            y_predict = model(images)
            
            # Cross entropy loss
            loss = criterion(y_predict, labels)
            pdb.set_trace()
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