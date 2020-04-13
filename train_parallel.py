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
batch_size = 256
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

training_epochs = 0
training_batches = 0
validation_batches = 0
missed_validation_batch = 0
missed_training_batch = 0 
verbose = True
training_test = np.inf
validation_test = np.inf
start_time = time.time()

for i in range(epochs):
    ii = 0
    DS.train()
    training_epochs += 1
    for i, (inp, _) in enumerate(train_loader):
        if ii < training_test:
            ii += 1
 
            # data dates...
            input_var = inp.to(device)

            # forward pass, passes...
            y, x, gx, egx, cgx, cy, dgx, dy = DS(input_var)

            # calculate loss terms
            if dgx.size()[0] == batch_size:
                training_batches += 1

                loss_feat, loss_img, loss_adv, loss_discr, loss_gen = \
                    loss_stub.compute_loss(y, x, gx, egx, cgx, cy, dgx, dy, t_ones, t_zeros, bce, mse, lambda_feat, lambda_adv, lambda_img)
            else:
                missed_training_batch += 1
                print("missed training batches", missed_training_batch)

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
            writer.add_scalar('train/loss_gen', loss_gen, n)
            writer.add_scalar('train/loss_discr', loss_discr, n)
            writer.add_scalar('train/loss_feat', loss_feat, n)
            writer.add_scalar('train/loss_adv', loss_adv, n)
            writer.add_scalar('train/loss_img', loss_img, n)
            if verbose:
                print('[TRAIN] {:3.0f} : Gen_Loss={:0.5} -- Dis_Loss={:0.5}'.
                        format(n, loss_gen, loss_discr))
        #training batch done
        else:
            break

    elapsed_time = time.time() - start_time
    writer.add_scalar('train/training_performance', training_batches * batch_size / elapsed_time, elapsed_time)
    grid_images = torch.cat((input_var[:5], gx[:5]))
    grid00 = torchvision.utils.make_grid(grid_images, nrow=5, normalize=True)
    writer.add_image("train/images " + str(training_epochs), grid00, training_epochs)

    # Do some validation. 
    DS.eval()
    validation_batches = 0
    jj = 0
    for i, (inp, _) in enumerate(val_loader):
        if jj < validation_test:
            jj += 1
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
            else:
                missed_validation_batch += 1
                print("missed validation batches", missed_validation_batch)
        else:
            break
    elapsed_time = time.time() - start_time
    writer.add_scalar('val/validation_performance', nv /elapsed_time, elapsed_time)    
    grid_images = torch.cat((input_var[:5], gx[:5]))
    grid00 = torchvision.utils.make_grid(grid_images, nrow=5, normalize=True)
    writer.add_image('val/images ' + str(training_epochs), grid00, training_epochs)

print("missed validation batches", missed_validation_batch)
print("missed training batches", missed_training_batch)

# Google, where did you go wrong???
writer.close()
