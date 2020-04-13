import datastub
import optimizerstub
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models_parallel

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

# Set up the optimizers.
optim_gen, optim_discr = optimizerstub.get_optimizers(DS)
t_ones = torch.ones([batch_size]).to(device)
t_zeros = torch.zeros([batch_size]).to(device)

train_generator = True
train_discrimin = True

num_batches = 0 
loss_gen_sum = 0 
loss_discr_sum = 0 

verbose = True

DS.train()

for i in range(epochs):
    for i, (inp, _) in enumerate(train_loader):
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
        num_batches += 1

        if verbose:
            print('[TRAIN] {:3.0f} : Gen_Loss={:0.5} -- Dis_Loss={:0.5}'.
                    format(num_batches*batch_size, loss_gen, loss_discr))
    #         writer.add_scalar('gen_loss', gen_loss, generator.epochs)
    #         writer.add_scalar('discr_loss', discr_loss, generator.epochs)
    #         writer.add_scalar('loss_feat', loss_feat, generator.epochs)
    #         writer.add_scalar('loss_adv', loss_adv, generator.epochs)
    #         writer.add_scalar('loss_img', loss_img, generator.epochs)
    #         writer.flush()

    #         #
    #         # 7) Switch optimizing discriminator and generator, so that neither of them overfits too much.
    #         #
    #         discr_loss_ratio = torch.mean((real_discr + gen_discr) / discr_loss)

    #         if discr_loss_ratio < 1e-1 and train_discrimin:
    #             train_discrimin = False
    #             train_generator = True

    #         if discr_loss_ratio > 5e-1 and not train_discrimin:
    #             train_discrimin = True
    #             train_generator = True

    #         if discr_loss_ratio > 1e1 and train_generator:
    #             train_generator = False
    #             train_discrimin = True
    #         train_generator = True
    #         train_discrimin = True
    #     else:
    #         break

    # gen_loss_sum /= num_batches
    # discr_loss_sum /= num_batches

    # return gen_loss_sum, discr_loss_sum, train_generator, train_discrimin
