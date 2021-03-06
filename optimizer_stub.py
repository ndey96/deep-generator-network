import torch

def get_optimizers(DS, lr):
    # optim_gen = torch.optim.SGD(
    #     DS.module.G.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    optim_gen = torch.optim.Adam(
        DS.module.G.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4)

    optim_discr = torch.optim.Adam(
        DS.module.D.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4)

    return optim_gen, optim_discr