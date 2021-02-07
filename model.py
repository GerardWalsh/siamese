import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

class LitSiamese(pl.LightningModule):

    def __init__(self, data_dir='./', hidden_size=64, learning_rate=2e-4):
        super(LitSiamese, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 10),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 7),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 4),
            nn.ReLU(), 
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 4),
            nn.ReLU()
        )
        self.linear = nn.Sequential(
            nn.Linear(9216, 4096),
            nn.Sigmoid()
            )
        self.out = nn.Linear(4096, 1)
        self.loss = torch.nn.BCEWithLogitsLoss(size_average=True)

    def encoder(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1) # flatten
        x = self.linear(x)
        return x
    
    def forward(self, x1, x2):
        ying = self.encoder(x1)
        yang = self.encoder(x2)
        l1_distance = torch.abs(ying - yang)
        return self.out(l1_distance)

    def training_step(self, batch, batch_idx):
        img1, img2, label = batch
        ying = self.encoder(img1)
        yang = self.encoder(img2)
        y_hat = self.out(torch.abs(ying - yang))
        loss = self.loss(y_hat, label)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr = 6e-5)
        return optimizer


if __name__ == '__main__':
    net = Siamese()
    print(net)
    print(list(net.parameters()))