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


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


# def log_softmax(z):
#     # z is [batch_size, num_output_dims]
#     return z - torch.max(z) - torch.sum(z, dim=1) # [batch_size, num_output_dims]


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
    print("torch.sum(x), torch.sum(x_hat)", torch.sum(x).item(), torch.sum(x_hat).item())
    loss_img = torch.sum((x_hat - x)**2)

    # Adversarial losses.
    real_discr = torch.flatten(discriminator(x, a))  # D(y) [batch_size,1]
    gen_discr = torch.flatten(discriminator(x_hat,a))  # D(G(x)) = z from notebook [batch_size,1]

    # stabilized sigmoid loss
    loss_adv = bce_logits_loss(gen_discr, torch.ones(64).to(device))
    loss_discr = bce_logits_loss(real_discr, torch.ones(64).to(device)) + bce_logits_loss(
        gen_discr, torch.zeros(64).to(device))

    # g_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(
    #         logits=self.fake_score_logit,
    #         labels=tf.ones_like(self.fake_score_logit)
    #         )
    # self.gen_dis_loss = tf.reduce_mean(g_fake_loss)
    # d_real_loss = tf.nn.sigmoid_cross_entropy_with_logits(
    #         logits=self.real_score_logit,
    #         labels=tf.ones_like(self.real_score_logit)
    #         )
    # d_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(
    #         logits=self.fake_score_logit,
    #         labels=tf.zeros_like(self.fake_score_logit)
    #         )
    # self.dis_loss = tf.reduce_mean(d_real_loss + d_fake_loss)

    # stabilized loss from notebook
    # smax = nn.LogSoftmax()
    # loss_adv = -torch.sum(smax(gen_discr))
    # c = torch.max(gen_discr)
    # softmax_denom = torch.sum(torch.exp(gen_discr - c))
    # loss_discr = -torch.sum(
    #     smax(real_discr) + torch.log(softmax_denom - torch.exp(
    #         gen_discr - c)) - torch.log(softmax_denom))

    # old loss implementation
    # loss_discr = -1.0 * torch.sum(
    #     torch.log(real_discr) + torch.log(1.0 - gen_discr))
    # loss_adv = -1.0 * torch.sum(torch.log(gen_discr))

    # Combine the losses for DeePSiM.
    loss = lambda_feat * loss_feat + lambda_adv * loss_adv + lambda_img * loss_img
    # print("(loss_feat, loss_adv, loss_img)",(loss_feat.item(), loss_adv.item(), loss_img.item()))

    return loss, loss_discr, (real_discr, gen_discr)


def train(loader, optim_gen, generator, optim_discr, discriminator, encoder,
          comparator, train_generator, train_discrimin, bce_logits_loss,
          device, verbose):
    # Start tensorboard
    # Put in training mode.
    generator.train()
    discriminator.train()

    # Set up some counters.
    gen_loss_sum = 0.0
    discr_loss_sum = 0.0
    num_batches = 0

    for i, (inp, _) in enumerate(loader):

        # target = target.cuda(async=True) # TODO [NICK]: Look into this for using your 30,000 GPUs.

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
        gen_loss, discr_loss, (real_discr, gen_discr) = compute_loss(
            a=features_real,
            x=input_var,
            x_hat=generator_out,
            comparator=comparator,
            discriminator=discriminator,
            bce_logits_loss=bce_logits_loss)

        #
        # 5) Compute the gradient and take a step.
        #
        if train_generator:
            optim_gen.zero_grad()
            gen_loss.backward(retain_graph=True)
            optim_gen.step()

        if train_discrimin:
            optim_discr.zero_grad()
            discr_loss.backward(retain_graph=True)
            optim_discr.step()

        #
        # 6) Update counters.
        #
        gen_loss_sum += gen_loss
        discr_loss_sum += discr_loss
        num_batches += 1

        if verbose:
            print('[TRAIN] {:3.0f} : Gen_Loss={:0.5} -- Dis_Loss={:0.5}'.format(
                num_batches, gen_loss, discr_loss))

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

    for i, (inp, _) in enumerate(loader):

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
            discriminator=discriminator)

        #
        # 6) Update counters.
        #
        gen_loss_sum += gen_loss
        discr_loss_sum += discr_loss
        num_batches += 1

        if verbose:
            print('[VALID] {:3.0f} : Gen_Loss={:0.5} -- Dis_Loss={:0.5}'.format(
                num_batches, gen_loss, discr_loss))

    gen_loss_sum /= num_batches
    discr_loss_sum /= num_batches

    return gen_loss_sum, discr_loss_sum


if __name__ == '__main__':

    batch_size = 64
    torch.cuda.empty_cache()
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )  #TODO: Use ``device``` when initializing a variable instead of hardcoding it as ``.cuda()``.
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

    # Init the models.
    # if torch.cuda.device_count() > 1:
    #     encoder = nn.DataParallel(AlexNetEncoder())
    #     encoder.to(device)
    #     encoder.eval()

    #     comparator = nn.DataParallel(AlexNetComparator())
    #     encoder.to(device)
    #     comparator.eval()

    #     generator = nn.DataParallel(TransposeConvGenerator())
    #     encoder.to(device)

    #     discriminator = nn.DataParallel(Discriminator())
    #     encoder.to(device)

    encoder = AlexNetEncoder()
    encoder.cuda()
    encoder.eval()

    comparator = AlexNetComparator()
    comparator.cuda()
    comparator.eval()

    generator = TransposeConvGenerator()
    generator.cuda()

    discriminator = Discriminator()
    discriminator.cuda()

    bce_logits_loss = nn.BCEWithLogitsLoss(reduction='sum')
    bce_logits_loss.cuda()

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

    avg_train_gen_losses = []
    avg_train_discr_losses = []

    avg_valid_gen_losses = []
    avg_valid_discr_losses = []

    train_generator = True
    train_discrimin = True

    # dummy_input = torch.rand(64, 3, 224, 224)
    # writer.add_graph(encoder, (dummy_input, ))

    verbose = True
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
        avg_train_gen_losses.append(train_gen_loss)
        avg_train_discr_losses.append(train_discr_loss)

        avg_valid_gen_losses.append(valid_gen_loss)
        avg_valid_discr_losses.append(valid_discr_loss)

        #TODO [NICK]: Set up saving and loading of the gen and discr weights.
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict_gen': generator.state_dict(),
                'state_dict_discr': discriminator.state_dict(),
            },
            is_best=False)
        writer.flush()
    writer.close()
