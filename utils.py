from torch import nn



def center_crop(x, current_size, desired_size):
    start = int((current_size - desired_size)/2)
    return x[:,:, start:(start + desired_size), start:(start + desired_size)]


class ds_flatten(nn.Module):
    def __init__(self):
        super(ds_flatten, self).__init__()

    def forward(self, x):
        return x.view(-1) 

class View(nn.Module):
    def __init__(self, a, b, c, d):
        super(View, self).__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d
   
    def forward(self, x):
        return x.view((self.a, self.b, self.c, self.d)) 


