from models_parallel import UpsampleConvGenerator, TransposeConvGenerator
from torchvision.models import alexnet
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt


def process_img(img):
    img = img.detach().numpy()
    # img = (img - np.min(img)) / np.ptp(img)
    img = (img + 1) / 2
    img = img.T.reshape((227, 227, 3))
    return img
    # return np.moveaxis(img[0], 0, -1)


ugenerator = UpsampleConvGenerator()
tgenerator = TransposeConvGenerator()

a = 100000 * torch.rand(1, 4096)

ux_hat = ugenerator.forward(a)
tx_hat = tgenerator.forward(a)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(process_img(ux_hat))
ax[0].set_title('Upsample Generator')
ax[0].axis('off')
ax[1].imshow(process_img(tx_hat))
ax[1].set_title('Transposed Convolution Generator')
ax[1].axis('off')

plt.tight_layout()
plt.show()
