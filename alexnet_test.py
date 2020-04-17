from models_parallel import AlexNetComparator, Discriminator
from torchvision.models import alexnet
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

alex = alexnet(pretrained=True)
C = AlexNetComparator()
D = Discriminator()
img = torch.rand(64, 3, 224, 224)
cres = C.forward(img)

img = torch.rand(64, 3, 227, 227)
dres = D.forward(img, cres)

breakpoint()