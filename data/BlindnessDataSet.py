import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile

ROOT = os.getcwd()


class BlindnessDataSet(Dataset):
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
    



