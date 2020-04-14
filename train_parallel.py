import data_stub
import optimizer_stub
import loss_stub
import numpy as np
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models_parallel
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# Parameters
lambda_feat=0.01
lambda_adv=0.001
lambda_img=1.0
batch_size = 128
epochs = 100

# CUDA - need to tweak this to run on a CPU. 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DS = models_parallel.DeepSim()
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  DS = nn.DataParallel(DS)
DS.to(device)
DS.module.batch_size = batch_size

# DataLoaders
imagenet_transforms, train_loader, val_loader = data_stub.get_data_tools(batch_size)

# Google, you're not evil... are you?
writer = SummaryWriter()
# dummy_input = torch.rand(1, 3, 227, 227)
# writer.add_graph(DS, dummy_input)

# Set up the optimizers.
optim_gen, optim_discr = optimizer_stub.get_optimizers(DS)
bce = nn.BCEWithLogitsLoss(reduction='sum').to(device)
mse = nn.MSELoss().to(device)
t_ones = torch.ones([batch_size]).to(device)
t_zeros = torch.zeros([batch_size]).to(device)
train_generator = True
train_discrimin = True

verbose = True
training_batches = 0 
validation_batches = 0 
for i in range(epochs):
    DS.module.G.train()
    DS.module.D.train()
    DS.module.C.eval()
    DS.module.E.eval()
    for j, (inp, _) in enumerate(train_loader):
        # data dates...
        input_var = inp.to(device)

        # forward pass, passes...
        y, x, gx, egx, cgx, cy, dgx, dy = DS(input_var)

        # calculate loss terms
        if dgx.size()[0] == batch_size:
            training_batches += 1
            loss_feat, loss_img, loss_adv, loss_discr, loss_gen = \
                loss_stub.compute_loss(y, x, gx, egx, cgx, cy, dgx, dy, t_ones, t_zeros, bce, mse, lambda_feat, lambda_adv, lambda_img)

            # apply backprop on the optimizers
            if train_generator:
                optim_gen.zero_grad()
                loss_gen.backward(retain_graph=True)
                optim_gen.step()

            if train_discrimin:
                optim_discr.zero_grad()
                loss_discr.backward(retain_graph=True)
                optim_discr.step()

            loss_discr_ratio = torch.mean((dy + dgx) / loss_discr)

            # anti-over-fitting
            if loss_discr_ratio < 1e-1 and train_discrimin:
                train_discrimin = False
                train_generator = True

            if loss_discr_ratio > 5e-1 and not train_discrimin:
                train_discrimin = True
                train_generator = True

            if loss_discr_ratio > 1e1 and train_generator:
                train_generator = False
                train_discrimin = True
            train_generator = True
            train_discrimin = True

            # book-keeping and reporting
            n = training_batches * batch_size
            writer.add_scalar('train/loss_gen', loss_gen, n )
            writer.add_scalar('train/loss_discr', loss_discr, n)
            writer.add_scalar('train/loss_feat', loss_feat, n)
            writer.add_scalar('train/loss_adv', loss_adv, n)
            writer.add_scalar('train/loss_img', loss_img, n)
            if verbose:
                print('[TRAIN] {:3.0f} : Gen_Loss={:0.5} -- Dis_Loss={:0.5}'.
                        format(n, loss_gen, loss_discr))
            del y; del x;del gx;del egx;del cgx;del cy;del dgx;del dy
            del loss_feat;del loss_img;del loss_adv;del loss_discr;del loss_gen

    writer.add_scalar('train/training_performance', training_batches * batch_size , time.time())
    grid_images = torch.cat((input_var[:5], gx[:5]))
    grid00 = torchvision.utils.make_grid(grid_images, nrow=5, normalize=True)
    writer.add_image("train/images " + str(i), grid00, i)
    del grid_images; del grid00

    # Do some validation. 
    DS.module.G.eval()
    DS.module.D.eval()
    DS.module.C.eval()
    DS.module.E.eval()
    for k, (inp, _) in enumerate(val_loader):
        input_var = inp.to(device)
        y, x, gx, egx, cgx, cy, dgx, dy = DS(input_var)

        # calculate loss terms
        if dgx.size()[0] == batch_size:
            validation_batches += 1
            loss_feat, loss_img, loss_adv, loss_discr, loss_gen = \
                loss_stub.compute_loss(y, x, gx, egx, cgx, cy, dgx, dy, t_ones, t_zeros, bce, mse, lambda_feat, lambda_adv, lambda_img)
            nv = validation_batches * batch_size
            writer.add_scalar('val/loss_gen', loss_gen, nv)
            writer.add_scalar('val/loss_discr', loss_discr, nv)
            writer.add_scalar('val/loss_feat', loss_feat, nv)
            writer.add_scalar('val/loss_adv', loss_adv, nv)
            writer.add_scalar('val/loss_img', loss_img, nv)
            if verbose:
                print('[VALID] {:3.0f} : Gen_Loss={:0.5} -- Dis_Loss={:0.5}'.
                        format(nv, loss_gen, loss_discr))
            del y; del x;del gx;del egx;del cgx;del cy;del dgx;del dy
            del loss_feat;del loss_img;del loss_adv;del loss_discr;del loss_gen
    grid_images = torch.cat((input_var[:5], gx[:5]))
    grid00 = torchvision.utils.make_grid(grid_images, nrow=5, normalize=True)
    writer.add_image('val/images ' + str(i), grid00, i)
    del grid_images;del grid00

# Google, where did you go wrong???
writer.close()
