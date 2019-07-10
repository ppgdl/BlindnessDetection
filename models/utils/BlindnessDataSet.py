import os
import pdb
from torch.utils.data import Dataset

ROOT = os.getcwd()

class BlindnessDataSet(Dataset):
    def __init__(self, path=None, transform=None, target_transform=None):
        if path == None:
            path = os.path.join(ROOT, '..', '..', 'data', 'dataset.txt')
        pdb.set_trace()

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

if __name__ == '__main__':
    """
    Test
    """
    dataset = BlindnessDataSet()
