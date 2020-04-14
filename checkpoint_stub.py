import torch
from models_parallel import DeepSim
from optimizer_stub import get_optimizers

def save_deepsim(path, epoch, ds, optG, optD):
    torch.save({
                'epoch': epoch,
                'training_batches': training_batches,
                'ds_state_dict': ds.state_dict(),
                'ds_generator_optimizer': optG.state_dict(),
                'ds_discriminator_optimizer': optD.state_dict()
                }, path)
    return

def load_deepsim(path):
    DS = DeepSim()
    optim_gen, optim_discr = get_optimizers(DS)
    checkpoint = torch.load(PATH)
    DS.load_state_dict(checkpoint['ds_state_dict'])
    optim_gen.load_state_dict(checkpoint['ds_generator_optimizer'])
    optim_discr.load_state_dict(checkpoint['ds_discriminator_optimizer'])
    epoch = checkpoint['epoch']
    training_batches = checkpoint['training_batches']
    return        



