import os
import pdb
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile

ROOT = os.getcwd()


class BlindnessDataSet(Dataset):
    def __init__(self, csv_path=None, transform=None, mode = 'train'):
        self.mode = mode
        self.csv_path = csv_path
        if csv_path == None:
            self.csv_path = os.path.join(ROOT, '..', '..', 'data', 'dataset.txt')
        self.data = pd.read_csv(self.csv_path)
        self.transform = transform

    def __getitem__(self, index):
        if self.mode == 'train':
            imgpath = os.path.join(os.path.dirname(self.csv_path), 'trainImage', self.data.loc[index, 'id_code'] + '.png')
            img = Image.open(imgpath)
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
            transforms.ToTensor()])
    dataset = BlindnessDataSet(csv_path, transform)
    train_data_loader = DataLoader(dataset, batch_size = 32, shuffle = True, num_workers = 0)
    



