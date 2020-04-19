import torch
from models_parallel import DownsampleDiscriminator

feat = torch.rand(64, 9216)
img = torch.rand(64, 3, 224, 224)

D = DownsampleDiscriminator()
D.eval()
res = D.forward(img, feat)
print(res, res.shape)

# sum(p.numel() for p in D_new.parameters() if p.requires_grad)