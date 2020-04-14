import os
import torch
from time import gmtime, strftime
from models_parallel import DeepSim
from optimizer_stub import get_optimizers

def save_checkpoint(path, ds, optG, optD, training_batches, lambda_feat, lambda_adv, lambda_img, batch_size, epoch):
    dt_string = strftime("%d_%m_%Y-%H-%M-%S")
    path = path + dt_string + "_" + str(training_batches) + "_" + str(batch_size) + ".ptm"
    print("saving model checkpoint to:", path)	
    torch.save({
                'ds_state_dict': ds.state_dict(),
                'ds_generator_optimizer': optG.state_dict(),
                'ds_discriminator_optimizer': optD.state_dict(),
                'epoch': epoch,
                'training_batches': training_batches,
                'lambda_feat' : lambda_feat,
                'lambda_adv' : lambda_adv,
                'lambda_img' : lambda_img,
                'batch_size' : batch_size
                }, path)
    return


def load_checkpoint(DS, optim_gen, optim_discr, filename='checkpoint.pth.tar'):
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        DS.load_state_dict(checkpoint['ds_state_dict'])
        optim_gen.load_state_dict(checkpoint['ds_generator_optimizer'])
        optim_discr.load_state_dict(checkpoint['ds_discriminator_optimizer'])
        epoch = checkpoint['epoch']
        training_batches = checkpoint['training_batches']
        lambda_feat = checkpoint['lambda_feat']
        lambda_adv = checkpoint['lambda_adv']
        lambda_img = checkpoint['lambda_img']
        batch_size = checkpoint['batch_size']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return DS, optim_gen, optim_discr, epoch, training_batches, lambda_feat, lambda_adv, lambda_img, batch_size

