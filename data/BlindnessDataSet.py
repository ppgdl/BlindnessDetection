import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import cv2
ROOT = os.getcwd()
import numpy as np


def crop_image1(img,tol=7):
    # img is image data
    # tol  is tolerance
        
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]

def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img
#The Code from: https://www.kaggle.com/ratthachat/aptos-updated-albumentation-meets-grad-cam


class BlindnessDataSet1(Dataset):
    def __init__(self, train_list, transform=None, mode='train'):
        self.mode = mode
        self.train_list = train_list

        if not os.path.exists(train_list):
            raise ValueError(
                'train list: {} is not exists!'.format(train_list)
            )

        self.train_lists = []
        lines = open(train_list, 'r').readlines()
        for i in range(len(lines)):
            line = lines[i]
            img_path = line.strip().split(' ')[0] + '.png'
            label = line.strip().split(' ')[1]
            self.train_lists.append([img_path, label])

        self.transform = transform

    def __getitem__(self, index):
        #print('index: {}, path: {}'.format(index, self.train_lists[index]))
        image_path = self.train_lists[index][0]
        label = self.train_lists[index][1]
        image = Image.open(image_path)
        if self.mode == 'train':
            if self.transform is not None:
                image = self.transform(image)
            label = int(label)
        return image, label
    def __len__(self):
        return len(self.train_list)


class BlindnessDataSet(Dataset):
    def __init__(self, inputData, transform=None):
        self.data = inputData
        self.transform = transform

    def __getitem__(self, index):
        dirpath = r'E:\kaggle\trainImage'
        imgpath = os.path.join(dirpath, self.data.loc[index, 'id_code'] + '.png')
        image = cv2.imread(imgpath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = crop_image_from_gray(image)
        image = cv2.resize(image, (256, 256))
        image = cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , 30) ,-4 ,128)
        img = transforms.ToPILImage()(image)
        if self.transform != None:
            img = self.transform(img)
        label = self.data.loc[index, 'diagnosis']
        
        return img, label
    def __len__(self):
        return len(self.data)


		
		
		
if __name__ == '__main__':
    """
    Test
    """
    csv_path = r'E:\kaggle\train.csv'
    transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = BlindnessDataSet(csv_path, transform)
    train_data_loader = DataLoader(dataset, batch_size = 32, shuffle = True, num_workers = 0)
    



