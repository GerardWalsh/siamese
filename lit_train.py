import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy


import pickle
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

import time
import numpy as np
import gflags
import sys
from collections import deque
import os

from model import LitSiamese
from dataset import OmniglotTrain, OmniglotTest

Flags = gflags.FLAGS
gflags.DEFINE_bool("cuda", True, "use cuda")
gflags.DEFINE_string("train_path", "/home/gexegetic/lightning-siamese/omniglot/python/images_background", "training folder")
gflags.DEFINE_string("test_path", "/home/gexegetic/lightning-siamese/omniglot/python/images_evaluation", 'path of testing folder')
gflags.DEFINE_integer("way", 20, "how much way one-shot learning")
gflags.DEFINE_string("times", 400, "number of samples to test accuracy")
gflags.DEFINE_integer("workers", 4, "number of dataLoader workers")
gflags.DEFINE_integer("batch_size", 80, "number of batch size")
gflags.DEFINE_float("lr", 0.00006, "learning rate")
gflags.DEFINE_integer("max_iter", 50000, "number of iterations before stopping")
gflags.DEFINE_string("model_path", "/home/data/pin/model/siamese", "path to store model")
gflags.DEFINE_string("gpu_id", "0", "gpu ids used to train")

Flags(sys.argv)

data_transforms = transforms.Compose([
    transforms.RandomAffine(15),
    transforms.ToTensor()
])

trainSet = OmniglotTrain(Flags.train_path, transform=data_transforms)
testSet = OmniglotTest(Flags.test_path, transform=transforms.ToTensor(), times = Flags.times, way = Flags.way)

trainLoader = DataLoader(trainSet, batch_size=Flags.batch_size, shuffle=False, num_workers=Flags.workers)
testLoader = DataLoader(testSet, batch_size=Flags.way, shuffle=False, num_workers=Flags.workers)

net = LitSiamese()

trainer = pl.Trainer(gpus=1)
trainer.fit(net, trainLoader)
