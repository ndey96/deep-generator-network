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
import models3 
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch import nn

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"


if __name__ == '__main__':

    batch_size = 128
    lambda_feat=0.01
    lambda_adv=0.001
    lambda_img=1.0
    

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(device)
    DS = models3.DeepSim()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    DS = nn.DataParallel(DS)
    DS.to(device)
    DS.module.batch_size = batch_size

    # optim_gen = torch.optim.SGD(
    #     DS.module.G.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    # optim_discr = torch.optim.Adam(
    #     DS.module.D.parameters(),
    #     lr=0.0002,
    #     betas=(0.9, 0.999),
    #     weight_decay=1e-4)

    # Begin training.
    print('Beginning training...')
    train_generator = True
    train_discrimin = True
    verbose = True
    DS.module.train()
    t_ones = torch.ones([batch_size]).to(device)
    t_zeros = torch.zeros([batch_size]).to(device)

    for epoch in range(1):
        for i, (inp, _) in enumerate(train_loader):
            input_var = inp.to(device)
            # input_var = torch.autograd.Variable(inp)
            # input_var.to(device)
            y, x, gx, egx, cgx, cy, dgx, dy = DS.module(input_var)
            # loss_feat = DS.module.mse_loss(cgx, cy)
            # loss_img = DS.module.mse_loss(gx, y)
            # real_d = torch.flatten(dy)
            # gen_d = torch.flatten(dgx) 
            # loss_adv = DS.module.bce_logits_loss(gen_d, t_ones)
            # loss_discr = DS.module.bce_logits_loss(real_d, t_ones)  \
            #                 + DS.module.bce_logits_loss(gen_d, t_zeros)
            # loss_gen = lambda_feat * loss_feat + lambda_adv * loss_adv + lambda_img * loss_img

            # if train_generator:
            #     optim_gen.zero_grad()
            #     loss_gen.backward(retain_graph=True)
            #     optim_gen.step()

            # if train_discrimin:
            #     optim_discr.zero_grad()
            #     loss_discr.backward(retain_graph=True)
            #     optim_discr.step()
            print("Up with the Gugenheim!!")

    

    
