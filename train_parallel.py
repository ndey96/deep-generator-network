import datastub
import optimizerstub
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
batch_size = 320
epochs = 100

# CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DS = models_parallel.DeepSim()
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  DS = nn.DataParallel(DS)
DS.to(device)

# DataLoaders
imagenet_transforms, train_loader, val_loader = datastub.get_data_tools(batch_size)

# Google, you're not evil
writer = SummaryWriter()
# dummy_input = torch.rand(1, 3, 227, 227)
# writer.add_graph(DS, dummy_input)

# Set up the optimizers.
optim_gen, optim_discr = optimizerstub.get_optimizers(DS)
t_ones = torch.ones([batch_size]).to(device)
t_zeros = torch.zeros([batch_size]).to(device)

train_generator = True
train_discrimin = True

loss_gen_sum = 0 
loss_discr_sum = 0 
total_batches = 0
epoch_batches = 0
verbose = True

DS.train()

for i in range(epochs):
    loss_gen_sum = 0 
    loss_discr_sum = 0 
    epoch_batches = 0

    for i, (inp, _) in enumerate(train_loader):
        epoch_batches += 1
        total_batches += 1
        # data
        input_var = inp.to(device)
        # forward pass, collect error constituents
        y, x, gx, egx, cgx, cy, dgx, dy = DS(input_var)
        # calculate loss terms
        loss_feat = DS.module.mse_loss(cgx, cy)
        loss_img = DS.module.mse_loss(gx, y)
        real_d = torch.flatten(dy)
        gen_d = torch.flatten(dgx) 
        loss_adv = DS.module.bce_logits_loss(gen_d, t_ones)
        loss_discr = DS.module.bce_logits_loss(real_d, t_ones)  \
                        + DS.module.bce_logits_loss(gen_d, t_zeros)
        loss_gen = lambda_feat * loss_feat + lambda_adv * loss_adv + lambda_img * loss_img
        
        if train_generator:
            optim_gen.zero_grad()
            loss_gen.backward(retain_graph=True)
            optim_gen.step()

        if train_discrimin:
            optim_discr.zero_grad()
            loss_discr.backward(retain_graph=True)
            optim_discr.step()

        with torch.no_grad():
            loss_gen_sum += loss_gen
            loss_discr_sum += loss_discr

        n = total_batches*batch_size

        loss_discr_ratio = torch.mean((dy + dgx) / loss_discr)

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

        # return gen_loss_sum, discr_loss_sum, train_generator, train_discrimin
        writer.add_scalar('loss_gen', loss_gen, n)
        writer.add_scalar('loss_discr', loss_discr, n)
        writer.add_scalar('loss_feat', loss_feat, n)
        writer.add_scalar('loss_adv', loss_adv, n)
        writer.add_scalar('loss_img', loss_img, n)
        if verbose:
            print('[TRAIN] {:3.0f} : Gen_Loss={:0.5} -- Dis_Loss={:0.5}'.
                    format(n, loss_gen, loss_discr))

    grid_images = torch.cat((input_var[:5], gx[:5]))
    grid00 = torchvision.utils.make_grid(grid_images, nrow=5, 
                normalize=True)
    writer.add_image('images' + str(n), grid00, 0)

# Google, where did you go wrong???
writer.close()
