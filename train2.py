import argparse
import os
import shutil
import time

import GPUtil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from models2 import DeepSim
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch import nn

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"



if __name__ == '__main__':

    batch_size = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter()

    imagenet_transforms = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            '/home/torenvln/git/fastdata2/ilsvrc2012/training_images',
            imagenet_transforms),
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            '/home/torenvln/git/fastdata2/ilsvrc2012/validation_images',
            imagenet_transforms),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True)

    DS = DeepSim()
    bce_logits_loss = nn.BCEWithLogitsLoss(reduction='sum').cuda()
    mse_loss = nn.MSELoss().cuda()

    optim_gen = torch.optim.SGD(
        DS.G.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    optim_discr = torch.optim.Adam(
        DS.d_00.parameters(),
        lr=0.0002,
        betas=(0.9, 0.999),
        weight_decay=1e-4)

    # Begin training.
    print('Beginning training...')

    train_g = True
    train_d = True

    verbose = True
    DS.train()
    for epoch in range(1):
        for i, (inp, _) in enumerate(train_loader):
            y, x, gxc, gxd, cgx, cy, dgx, dy, orig_d = DS(inp)
            print(y.size())

    writer.close()
