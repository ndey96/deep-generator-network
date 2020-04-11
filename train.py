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
from models import TransposeConvGenerator as Generator
from models import Discriminator, AlexNetComparator, AlexNetEncoder, TransposeConvGenerator
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch import nn

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def compute_loss(a,
                 x,
                 x_hat,
                 comparator,
                 discriminator,
                 bce_logits_loss,
                 lambda_feat=0.01,
                 lambda_adv=0.001,
                 lambda_img=1.0):
    '''
        Input:
         a      enc ( input )
         x      input
         x_hat  gen ( enc (input) )
    '''

    # Loss in feature space.
    loss_feat = torch.sum((comparator(x_hat) - comparator(x))**2)

    # Loss in image space.
    # print("torch.sum(x), torch.sum(x_hat)", torch.sum(x).item(), torch.sum(x_hat).item())
    loss_img = torch.sum((x_hat - x)**2)

    # Adversarial losses.
    real_discr = torch.flatten(discriminator(x, a))  # D(y) [batch_size,1]
    gen_discr = torch.flatten(discriminator(
        x_hat, a))  # D(G(x)) = z from notebook [batch_size,1]

    # stabilized sigmoid loss
    t_ones = torch.ones(gen_discr.size()).cuda()
    t_zeros = torch.zeros(gen_discr.size()).cuda()
    loss_adv = bce_logits_loss(gen_discr, t_ones)
    loss_discr = bce_logits_loss(real_discr, t_ones) + bce_logits_loss(
        gen_discr, t_zeros)

    # Combine the losses for DeePSiM.
    loss = lambda_feat * loss_feat + lambda_adv * loss_adv + lambda_img * loss_img

    return loss, loss_discr, (real_discr, gen_discr)


def train(loader, optim_gen, generator, optim_discr, discriminator, encoder,
          comparator, train_generator, train_discrimin, bce_logits_loss, device,
          verbose):

    generator.train()
    discriminator.train()
    # Set up some counters.
    gen_loss_sum = 0.0
    discr_loss_sum = 0.0
    num_batches = 0
    ii = 0
    for i, (inp, _) in enumerate(loader):
        # print(ii)
        ii += 1
        if ii == ii:
            # Prime the input.
            input_var = torch.autograd.Variable(inp)
            input_var = input_var.to(device)

            # 1) Feed forward the data into the encoder.
            #    ( a )    =    enc ( x )
            features_real = encoder(input_var)

            # 2) Feed forward the data into the generator.
            # ( x_hat )   =   gen ( a )
            generator_out = generator(features_real)

            # 4) Compute the loss of the generator.
            gen_loss, discr_loss, (real_discr, gen_discr) = compute_loss(
                a=features_real,
                x=input_var,
                x_hat=generator_out,
                comparator=comparator,
                discriminator=discriminator,
                bce_logits_loss=bce_logits_loss)

            # 5) Compute the gradient and take a step.
            if train_generator:
                optim_gen.zero_grad()
                gen_loss.backward(retain_graph=True)
                optim_gen.step()

            if train_discrimin:
                optim_discr.zero_grad()
                discr_loss.backward(retain_graph=True)
                optim_discr.step()

            # 6) Update counters.
            with torch.no_grad():
                gen_loss_sum += gen_loss
                discr_loss_sum += discr_loss
            num_batches += 1

            if verbose:
                print('[TRAIN] {:3.0f} : Gen_Loss={:0.5} -- Dis_Loss={:0.5}'.
                      format(num_batches, gen_loss, discr_loss))
            writer.add_scalar('deep-generator-network:gen_loss', gen_loss,
                              num_batches)
            writer.add_scalar('deep-generator-network:discr_loss', discr_loss,
                              num_batches)
            writer.flush()

            #
            # 7) Switch optimizing discriminator and generator, so that neither of them overfits too much.
            #
            discr_loss_ratio = torch.mean((real_discr + gen_discr) / discr_loss)

            if discr_loss_ratio < 1e-1 and train_discrimin:
                train_discrimin = False
                train_generator = True

            if discr_loss_ratio > 5e-1 and not train_discrimin:
                train_discrimin = True
                train_generator = True

            if discr_loss_ratio > 1e1 and train_generator:
                train_generator = False
                train_discrimin = True
            train_generator = True
            train_discrimin = True
        else:
            break

    gen_loss_sum /= num_batches
    discr_loss_sum /= num_batches

    return gen_loss_sum, discr_loss_sum, train_generator, train_discrimin


def validate(loader, generator, discriminator, encoder, comparator, device,
             verbose):

    # Put in evaluation mode.
    generator.eval()
    discriminator.eval()

    gen_loss_sum = 0.0
    discr_loss_sum = 0.0
    num_batches = 0
    ii = 0
    for i, (inp, _) in enumerate(loader):
        # print(ii)
        ii += 1
        if ii == ii:
            # target = target.cuda(async=True) # TODO [NICK]: Here too.

            # Prime the input.
            input_var = torch.autograd.Variable(inp)
            input_var = input_var.to(device)

            #
            # 1) Feed forward the data into the encoder.
            #
            #    ( a )    =    enc ( x )
            features_real = encoder(input_var)

            #
            # 2) Feed forward the data into the generator.
            #
            # ( x_hat )   =   gen ( a )
            generator_out = generator(features_real)

            #
            # TODO: REMOVE.
            #
            # 3) Encode the generated image for comparison of features.
            #
            # ( a_hat )    =    enc ( x_hat )
            #features_recog = encoder(generator_out)

            #
            # 4) Compute the loss of the generator.
            #
            gen_loss, discr_loss, _ = compute_loss(
                a=features_real,
                x=input_var,
                x_hat=generator_out,
                comparator=comparator,
                discriminator=discriminator,
                bce_logits_loss=bce_logits_loss)

            #
            # 6) Update counters.
            #
            with torch.no_grad():
                gen_loss_sum += gen_loss
                discr_loss_sum += discr_loss
            num_batches += 1

            if verbose:
                print('[VALID] {:3.0f} : Gen_Loss={:0.5} -- Dis_Loss={:0.5}'.
                      format(num_batches, gen_loss, discr_loss))

    gen_loss_sum /= num_batches
    discr_loss_sum /= num_batches

    return gen_loss_sum, discr_loss_sum


if __name__ == '__main__':

    batch_size = 64
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

    encoder = AlexNetEncoder().cuda()
    encoder.eval()

    comparator = AlexNetComparator().cuda()
    comparator.eval()

    generator = TransposeConvGenerator().cuda()

    discriminator = Discriminator().cuda()
    try:
        bce_logits_loss = nn.BCEWithLogitsLoss(reduction='sum').cuda()
    except:
        print("hey")

    # Set up the optimizers.
    optim_gen = torch.optim.SGD(
        generator.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    optim_discr = torch.optim.Adam(
        discriminator.parameters(),
        lr=0.0002,
        betas=(0.9, 0.999),
        weight_decay=1e-4)

    # Begin training.
    print('Beginning training...')

    train_generator = True
    train_discrimin = True

    verbose = True
    generator.train()
    discriminator.train()
    epoch = 0
    for epoch in range(100):

        train_gen_loss, train_discr_loss, train_generator, train_discrimin = train(
            loader=train_loader,
            optim_gen=optim_gen,
            generator=generator,
            optim_discr=optim_discr,
            discriminator=discriminator,
            encoder=encoder,
            comparator=comparator,
            train_generator=train_generator,
            train_discrimin=train_discrimin,
            bce_logits_loss=bce_logits_loss,
            device=device,
            verbose=verbose)

        valid_gen_loss, valid_discr_loss = validate(
            loader=val_loader,
            generator=generator,
            discriminator=discriminator,
            encoder=encoder,
            comparator=comparator,
            device=device,
            verbose=verbose)

        #TODO [NICK]: Set up torchvision/tensorboard to visualize these lists??
        writer.add_scalar('deep-generator-network:avg_train_gen_losses',
                          train_gen_loss.detach(), epoch)
        writer.add_scalar('deep-generator-network:avg_train_discr_losses',
                          train_discr_loss.detach(), epoch)
        writer.add_scalar('deep-generator-network:avg_validation_gen_losses',
                          valid_gen_loss.detach(), epoch)
        writer.add_scalar('deep-generator-network:avg_validation_discr_losses',
                          valid_discr_loss.detach(), epoch)
        writer.flush()
        epoch += 1

        #TODO [NICK]: Set up saving and loading of the gen and discr weights.
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict_gen': generator.state_dict(),
                'state_dict_discr': discriminator.state_dict(),
            },
            is_best=False)
        torch.cuda.empty_cache()

        writer.flush()
    writer.close()
