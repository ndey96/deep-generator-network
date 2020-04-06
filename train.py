import argparse
import os
import shutil
import time

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
from models import Discriminator, AlexNetComparator, AlexNetEncoder


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')



def generator_loss(a, a_hat, x, x_hat, comparator, discriminator, lambda_feat=0.01, lambda_adv=0.001, lambda_img=1.0):
    '''
        Input:
         a      enc ( input )
         a_hat  enc ( x_hat )
         x      input
         x_hat  gen ( enc (input) ) 
    '''

    # deep_feature_loss = torch.sum((comparator(x) - comparator(x_hat)) ** 2)
    # reconstruction_loss = torch.sum((x - x_hat) ** 2)
    # discriminator_loss = discriminator(69) 
    # loss = discriminator_loss + 0.1 * deep_feature_loss + 0.001 * reconstruction_loss
    
    # Loss in feature space.
    loss_feat = torch.sum( (comparator(x_hat) - comparator(x))**2 )

    # Loss in image space.
    loss_img = torch.sum( (x_hat - x)**2 )

    # Adversarial losses.
    gen_discr  = discriminator(x_hat, a)
    real_discr = discriminator(x, a)

    loss_discr = -1.0 * torch.sum( torch.log(real_discr) + torch.log(1.0 - gen_discr) )
    loss_adv   = -1.0 * torch.sum( torch.log(gen_discr) )
    
    # Combine the losses for DeePSiM.
    loss = lambda_feat * loss_feat + lambda_adv * loss_adv + lambda_img * loss_img

    return loss, loss_discr



def train(loader, optimizer, generator, discriminator, encoder, comparator, train_generator, train_discrimin):
    
    #generator.train() TODO: REMOVE.
    #num_batches = 0
    loss_sum = 0

    for i, (inp, tar) in enumerate(loader):

        # target = target.cuda(async=True) # TODO: ???
        
        input_var  = torch.autograd.Variable(inp)
        target_var = torch.autograd.Variable(tar)

        # TODO: Set input_var and target_var to take in the device (cpu/cuda).


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

        # TODO: Should the generator take input_var or features_out??
        

        #
        # 3) Encode the generated image for comparison of features.
        #
        # ( a_hat )    =    enc ( x_hat )
        features_recog = encoder(generator_out)

        
        #
        # TODO: Is this necessary?? Should Feeding of discriminator happen out here?? 
        #       If so, how should it be done??
        #
        # 4) Run the discriminator on the real features to get the ___ (loss?)
        #
        # discrim_real = discriminator(input_var,     features_real)
        # discrim_fake = discriminator(generator_out, features_real)
        

        #
        # 4) Compute the loss of the generator.
        #
        gen_loss, discr_loss = generator_loss(
            a=features_real, 
            a_hat=features_recog, 
            x=input_var, 
            x_hat=generator_out, 
            comparator=comparator, 
            discriminator=discriminator
        )


        #
        # TODO: This is one possible way of turning on/off the weights for each network 
        #
        # 5) Enable/Disable weights depending on out of function switches.
        #
        # If train_x = True  : Parameters will be effected by grad-update.
        # If train_x = False : Parameters will be uneffected by grad-update.
        for gen_param in generator.parameters():
            gen_param.requires_grad = train_generator
        for dis_param in discriminator.parameters():
            dis_param.requires_grad = train_discrimin


        #
        # 6) Compute the Gradient and do a SGD step.
        #
        optimizer.zero_grad()
        gen_loss.backward()
        optimizer.step()


    #
    # TODO: Figure out what to return here.
    #

    avg_loss = loss_sum
    # print(f'Avg loss: {avg_loss}')
    #return avg_loss # ???
    
    discr_real_loss = 0.0
    discr_fake_loss = 0.0
    discr_fake_for_generator_loss = 1.0
    #return discr_real_loss, discr_fake_loss, discr_fake_for_generator_loss # ???
    
    discr_loss_ratio = (discr_real_loss + discr_fake_loss) / discr_fake_for_generator_loss
    return discr_loss_ratio # ???



# TODO: Work out validate function.
def validate(loader, generator, encoder, comparator):
    generator.eval()
    num_batches = 0
    loss_sum = 0
    for i, (input, _) in enumerate(loader):
        x = torch.autograd.Variable(input)
        x = x.to(device)

        # compute activation for image
        a = encoder(x)
        # breakpoint()
        # compute output
        x_hat = generator(a)

        loss = generator_loss(a, a_hat, x, x_hat, comparator, discriminator)
        loss_sum += loss
        num_batches += 1

    avg_loss = loss_sum / num_batches
    # print(f'Avg loss: {avg_loss}')

    return avg_loss



if __name__ == '__main__':
    
    batch_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #TODO: Use ``device``` when initializing a variable instead of hardcoding it as ``.cuda()``.

    imagenet_transforms = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('/home/torenvln/git/fastdata2/ilsvrc2012/training_images', imagenet_transforms),
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('/home/torenvln/git/fastdata2/ilsvrc2012/validation_images', imagenet_transforms),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True)

    encoder = AlexNetEncoder() #TODO: pass in ``device`` as arg?
    encoder.cuda()
    encoder.eval()

    comparator = AlexNetComparator()
    comparator.cuda()
    comparator.eval()

    generator = Generator()
    generator.cuda()

    discriminator = Discriminator()
    discriminator.cuda()

    optimizer = torch.optim.SGD(
        generator.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-4
    )

    print('Beginning training...')

    avg_train_losses = []
    avg_valid_losses = []

    train_discrimin = True
    train_generator = True

    for epoch in range(100):
    
        avg_train_loss = train(
            loader=train_loader, 
            optimizer=optimizer, 
            generator=generator,
            discriminator=discriminator,
            encoder=encoder, 
            comparator=comparator,
            train_generator=train_generator,
            train_discrimin=train_discrimin
        )
        
        avg_valid_loss = validate(
            loader=val_loader, 
            generator=generator, 
            encoder=encoder, 
            comparator=comparator
        )
        
        avg_train_losses.append( avg_train_loss )
        avg_valid_losses.append( avg_valid_loss )

        
        # Switch optimizing discriminator and generator, so that neither of them overfits too much
        if avg_train_loss < 1e-1 and train_discrimin:
            train_discrimin = False
            train_generator = True

        if avg_train_loss > 5e-1 and not train_discrimin:
            train_discrimin = True
            train_generator = True

        if avg_train_loss > 1e1 and train_generator:
            train_generator = False
            train_discrimin = True

        print(f'Epoch {epoch}: avg_train_loss={avg_train_loss:.3f}, avg_valid_loss={avg_valid_loss:.3f}')

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': generator.state_dict(),
        }, is_best=False)
