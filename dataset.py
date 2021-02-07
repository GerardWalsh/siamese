import torch
from torch.utils.data import Dataset, DataLoader
import os
from numpy import array, float32
from numpy.random import seed
import time
from random import randint, choice
import torchvision.datasets as dset
from PIL.Image import open as iopen

from os.path import join
from os import listdir


class OmniglotTrain(Dataset):

    def __init__(self, data_path, transform=None):
        super(OmniglotTrain, self).__init__()
        seed(0)
        self.transform = transform
        self.data, self.num_classes = self.load_to_mem(data_path)

    def load_to_mem(self, data_path):
        print('Loading training data into memory')
        data = {}
        degrees = [0, 90, 180, 270]
        idx = 0
        for degree in degrees:
            for alpha_path in listdir(data_path):
                for char_path in listdir(join(data_path, alpha_path)):
                    data[idx] = []
                    for sample_path in listdir(join(data_path, alpha_path, char_path)):
                        filepath = join(data_path, alpha_path, char_path, sample_path)
                        data[idx].append(iopen(filepath).rotate(degree).convert('L'))
                    idx += 1
        print('Dataset in memory')
        return data, idx
    
    def __len__(self):
        return  21000000
    
    def __getitem__(self, index):
        label, img1, img2 = None, None, None
        if index % 2 == 1: # from the same class
            label = 1.0
            idx1 = randint(0, self.num_classes - 1)
            img1 = choice(self.data[idx1])
            img2 = choice(self.data[idx1])
        else:
            label = 0.0
            idx1 = randint(0, self.num_classes - 1)
            idx2 = randint(0, self.num_classes - 1)
            while idx1 == idx2:
                idx2 = randint(0, self.num_classes - 1)
            img1 = choice(self.data[idx1])
            img2 = choice(self.data[idx2])
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, torch.from_numpy(array([label], dtype=float32))


class OmniglotTest(Dataset):

    def __init__(self, data_path, transform=None, times=200, way=20):
        super(OmniglotTest, self).__init__()
        seed(1)
        self.transform = transform
        self.times = times
        self.way = way
        self.img1 = None
        self.c1 = None
        self.data, self.num_classes = self.load_to_mem(data_path)

    def load_to_mem(self, data_path):
        print('Loading testing data into memory')
        data = {}
        idx = 0
        for alpha_path in listdir(data_path):
            for char_path in listdir(join(data_path, alpha_path)):
                data[idx] = []
                for sample_path in listdir(join(data_path, alpha_path, char_path)):
                    filepath = join(data_path, alpha_path, char_path, sample_path)
                    data[idx].append(iopen(filepath).convert('L'))
                idx += 1
        print('Test data in memory')
        return data, idx

    def __len__(self):
        return self.times * self.way

    def __getitem__(self, index):
        idx = index % self.way
        label = None
        if idx == 0:
            self.c1 = randint(0, self.num_classes - 1)
            self.img1 = choice(self.data[self.c1])
        else:
            c2 = randint(0, self.num_classes - 1)
            while self.c1 == c2:
                c2 = randint(0, self.num_classes - 1)
            img2 = choice(self.data[c2])

        if self.transform:
            img1 = self.transform(self.img1)
            img2 = self.transform(img2)
        
        return img1, img2


if __name__ == '__main__':
    train = OmniglotTest('./omniglot/python/images_background/', 30000*8)
    print(len(train))