import torch
import torch.nn as nn

def compute_loss(y, x, gx, egx, cgx, cy, dgx, dy, t_ones, t_zeros, bce, mse, lambda_feat, lambda_adv, lambda_img):
        loss_feat = mse(cgx, cy)
        loss_img = mse(gx, y)
        real_d = torch.flatten(dy)
        gen_d = torch.flatten(dgx)
        if gen_d.size() != t_ones.size():
                print("trapped") 
        loss_adv = bce(gen_d, t_ones)
        loss_adv = bce(gen_d, t_ones)
        loss_discr = bce(real_d, t_ones)  \
                        + bce(gen_d, t_zeros)
        loss_gen = lambda_feat * loss_feat + lambda_adv * loss_adv + lambda_img * loss_img
        return loss_feat, loss_img, loss_adv, loss_discr, loss_gen