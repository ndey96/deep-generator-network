import numpy as np
import torch

from models_parallel import AlexNetEncoder
from data_stub import get_data_tools

batch_size = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set up the encoder.
encoder = AlexNetEncoder()
for param in encoder.parameters():
    param.require_grad = False
encoder.to(device)

# Load the data from the loader.
imgnet_tform, train_loader, val_loader = get_data_tools(batch_size)

running = []
for k, (inp, _) in enumerate(val_loader):

    # Pass this input through the network.
    inp.to(device)
    out = encoder(inp)

    # Toss out uneven sized batches.
    if out.shape[0] != batch_size:
        print("Throwing out enum {}".format(k))
        continue

    # Add the output to the running list.
    running.append( out.cpu().detach().numpy() )


# NICK: This line hopefully won't throw an error.
codes = np.asarray(running).reshape(-1, 4096)

stdev = codes.std(axis=0)
np.save('magic_numbers', stdev)
