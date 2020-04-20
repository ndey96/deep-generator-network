import numpy as np
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision

from checkpoint_stub import save_checkpoint, load_checkpoint
from data_stub import get_data_tools
from loss_stub import compute_loss
from models_parallel import DeepGen
from optimizer_stub import get_optimizers

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# lambda_adv= 0.125 got loss of 5.5 in 1.3M
# lambda_adv= 0.0625 got loss of 5.2 in 1.3M
# Parameters
lambda_feat = 1
lambda_adv = 0.0625
lambda_img = 3
lr = 0.0002

batch_size = 64
epochs = 100
training_batches = 0
path = "./chk/"

# CUDA - need to tweak this to run on a CPU.
DS = DeepGen()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    DS = nn.DataParallel(DS)
DS.to(device)
DS.module.batch_size = batch_size
# Set up optimizers
optim_gen, optim_discr = get_optimizers(DS, lr)

# Load checkpoint
load_model = False
if load_model == True:
    # changed lr to
    # path2 = "./chk/pre_hp_change/7_04_2020-09-02-41_120105_160.ptm"
    path2 = "./chk/17_04_2020-18-40-21_10009_128_lf0.01_la1_li0.01_lr0.0002.ptm"
    DS, optim_gen, optim_discr, epoch, training_batches, lambda_feat, lambda_adv, lambda_img, batch_size, lr = load_checkpoint(
        DS, optim_gen, optim_discr, filename=path2)

# Some required math
bce = nn.BCEWithLogitsLoss(reduction='mean').to(device)
mse = nn.MSELoss(reduction='mean').to(device)
t_ones = torch.ones([batch_size]).to(device)
t_zeros = torch.zeros([batch_size]).to(device)

# DataLoaders
imagenet_transforms, train_loader, val_loader = get_data_tools(batch_size)

# Google, you're not evil... are you?
writer = SummaryWriter()
# dummy_input = torch.rand(1, 3, 227, 227)
# writer.add_graph(DS, dummy_input)

train_generator = True
train_discrimin = True

verbose = True

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

        # calculate loss terms - if dgx.size()[0] != batch_size - does it
        # mean we are at the end of the dataset and the last batch is not
        # full sized?  At any rate, whenever this condition isn't met
        # the forward pass dies, and sometime at the end of an epoch,
        # its not met. Maybe if there are bad imagenet files selected
        # in the batch, the data_loader just forges ahead and passes
        # a batch with less channnels?  Torch utilities are evil.
        if dgx.size()[0] == batch_size:
            training_batches += 1
            loss_feat, loss_img, loss_adv, loss_discr, loss_gen = \
                compute_loss(y, x, gx, egx, cgx, cy, dgx, dy, t_ones,
                    t_zeros, bce, mse, lambda_feat, lambda_adv,
                    lambda_img)

            # Make sure gen and discr don't get too far ahead of each other
            loss_discr_ratio = loss_discr / loss_adv
            if loss_discr_ratio < 1e-1:
                train_discrimin = False
            else:
                train_discrimin = True

            # apply backprop on the optimizers
            if train_generator:
                optim_gen.zero_grad()
                loss_gen.backward(retain_graph=True)
                optim_gen.step()

            if train_discrimin:
                optim_discr.zero_grad()
                loss_discr.backward(retain_graph=True)
                optim_discr.step()

            # book-keeping and reporting
            n = training_batches * batch_size
            lf = lambda_feat * loss_feat.detach()
            la = lambda_adv * loss_adv.detach()
            li = lambda_img * loss_img.detach()

            writer.add_scalar('train/dg_loss_gen', loss_gen.detach(), n)
            writer.add_scalar('train/dg_loss_discr', loss_discr.detach(), n)
            writer.add_scalar('train/dg_loss_feat', lf, n)
            writer.add_scalar('train/dg_loss_adv', la, n)
            writer.add_scalar('train/dg_loss_img', li, n)
            writer.add_scalar('train/dg_discr_train', int(train_discrimin), n)
            writer.add_scalar('train/dg_optim_discr_lr',
                              optim_discr.param_groups[0]['lr'], n)
            writer.add_scalar('train/dg_optim_gen_lr',
                              optim_gen.param_groups[0]['lr'], n)

            if verbose:
                print('[TRAIN] {:3.0f} : Gen_Loss={:0.5} -- Dis_Loss={:0.5}'.
                      format(n, loss_gen.detach(), loss_discr.detach()))

            # In the hopes of isolating/mitigating what we think are possible
            # memory leaks.
            del y
            del x
            del gx
            del egx
            del cgx
            del cy
            del dgx
            del dy
            del loss_feat
            del loss_img
            del loss_adv
            del loss_discr
            del loss_gen

    # Save a checkpoint
    save_checkpoint("dg", path, DS, optim_gen, optim_discr, training_batches,
                    lambda_feat, lambda_adv, lambda_img, batch_size, i, lr)

    grid_images = torch.cat((input_var[:5], gx[:5]))
    grid00 = torchvision.utils.make_grid(grid_images, nrow=5, normalize=True)
    writer.add_image("train/dg_images " + str(i), grid00, i)
    del grid_images
    del grid00

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
                compute_loss(y, x, gx, egx, cgx, cy, dgx, dy, t_ones, t_zeros, bce, mse, lambda_feat, lambda_adv, lambda_img)
            nv = validation_batches * batch_size

            # book-keeping and reporting
            n = training_batches * batch_size
            lf = lambda_feat * loss_feat.detach()
            la = lambda_adv * loss_adv.detach()
            li = lambda_img * loss_img.detach()

            writer.add_scalar('val/dg_loss_gen', loss_gen.detach(), n)
            writer.add_scalar('val/dg_loss_discr', loss_discr.detach(), n)
            writer.add_scalar('val/dg_loss_feat', lf, n)
            writer.add_scalar('val/dg_loss_adv', la, n)
            writer.add_scalar('val/dg_loss_img', li, n)

            if verbose:
                print('[VALID] {:3.0f} : Gen_Loss={:0.5} -- Dis_Loss={:0.5}'.
                      format(nv, loss_gen, loss_discr))
            del y
            del x
            del gx
            del egx
            del cgx
            del cy
            del dgx
            del dy
            del loss_feat
            del loss_img
            del loss_adv
            del loss_discr
            del loss_gen
    grid_images = torch.cat((input_var[:5], gx[:5]))
    grid00 = torchvision.utils.make_grid(grid_images, nrow=5, normalize=True)
    writer.add_image('val/dg_images ' + str(i), grid00, i)
    del grid_images
    del grid00

# Google, where did you go wrong???
writer.close()
