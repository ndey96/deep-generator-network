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
from models import Generator, Discriminator, AlexNetComparator, AlexNetEncoder


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def generator_loss(a, a_hat, x, x_hat, comparator, discriminator):
    deep_feature_loss = torch.sum((comparator(x) - comparator(x_hat)) ** 2)
    reconstruction_loss = torch.sum((x - x_hat) ** 2)
    discriminator_loss = discriminator(69) # TODO
    loss = discriminator_loss + 0.1 * deep_feature_loss + 0.001 * reconstruction_loss
    return loss


def train(loader, optimizer, generator, discriminator, encoder, comparator):
    generator.train()
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

        # compute activation for generated image
        a_hat = encoder(x_hat)

        loss = generator_loss(a, a_hat, x, x_hat, comparator, discriminator)

        # print(loss)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = loss_sum / num_batches
    # print(f'Avg loss: {avg_loss}')
    return avg_loss


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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    imagenet_transforms = transforms.Compose([
                                 transforms.Scale(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                             ])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('/home/shared/imagenet/train', imagenet_transforms),
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('/home/shared/imagenet/val', imagenet_transforms),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True)

    encoder = AlexNetEncoder()
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
        0.1,
        momentum=0.9,
        weight_decay=1e-4)

    print('Beginning training...')

    avg_train_losses = []
    avg_val_losses = []
    for epoch in range(100):
        avg_train_loss = train(train_loader, optimizer, generator, encoder, comparator)
        avg_train_losses.append(avg_train_loss)
        avg_val_loss = validate(val_loader, generator, encoder, comparator)
        avg_val_losses.append(avg_val_loss)\

        print(f'Epoch {epoch}: avg_train_loss={avg_train_loss:.3f}, avg_val_loss={avg_val_loss:.3f}')

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': generator.state_dict(),
        }, is_best=False)
