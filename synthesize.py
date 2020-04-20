import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
import time

import torch
import torch.nn as nn
from torchvision.models import alexnet

from models_parallel import DeepSim
from models_parallel import DeepGen

from checkpoint_stub import load_checkpoint
from optimizer_stub  import get_optimizers

import argparse


def get_args():
    
    # Init a parser.
    parser = argparse.ArgumentParser (
        prog='Synthesize', 
        description='Provided a class number in Range=[0, 1000], synthesize an image through the Deep Sim model, or our ``improved`` Deep Generator model (or both!!)',
        usage='python synthesize.py [--class CLASS] [--sim] [--gen] [--save] [--cuda] [--verb]'
    )
    
    # Add arguments to parser.
    #parser.add_argument('--img',  default=None,  help='Path to an image to pass through the selected architecture(s)                 (default: {})'.format('[REQUIRED]')) #TODO: REMOVE. But maybe keep for a style transfer demo...?
    
    parser.add_argument('--class', dest='_class', type=int, default=0, help='Synthesize the highest activation image of this class. Range=[0, 1000] (default: {})'.format(0)) #TODO: Find which class is ~ C O R N ~
    parser.add_argument('--sim',   action='store_true', default=False, help='Pass the provided image through the DeepSim architecture              (default: {})'.format('DISABLED'))
    parser.add_argument('--gen',   action='store_true', default=False, help='Pass the provided image through the DeepGen architecture              (default: {})'.format('DISABLED'))
    parser.add_argument('--show',  action='store_true', default=False, help='Show the initial and generated images                                 (default: {})'.format('DISABLED'))
    parser.add_argument('--save',  action='store_true', default=False, help='Save the output generated image                                       (default: {})'.format('DISABLED'))
    parser.add_argument('--cuda',  action='store_true', default=False, help='Enable CUDA for processing                                            (default: {})'.format('DISABLED'))
    parser.add_argument('--verb',  action='store_true', default=False, help='Enable Verbose output                                                 (default: {})'.format('DISABLED'))
    args = parser.parse_args()

    if not (args.sim or args.gen):
        parser.print_help()
        print("\n Please ensure an image and an architecture are chosen. For example: \n")
        print("\t python synthesize.py --class 23 --sim --gen --show")
        print(" ")
        exit()

    return args


# TODO: Make a function to save a stack of images as a .gif


def show_image(img, name="", save=False, verbose=False):

    plt.figure()
    plt.imshow(img, aspect='equal')
    plt.tight_layout()
    plt.axis('off')
    plt.title(name)

    if save:
        plt.imsave('./img/{}.png'.format(name), img, format='png')
        if verbose: print('[INFO] Saved ``{}.png``'.format(name))

    return


def synthesize(model, classifier, neuron=0, num_steps=10, lr=0.005, wdecay = 0.001, num_class=1000, code_len=4096, device=torch.device("cpu"), keep_steps=False, keep_freq=1, verbose=True):
    #num_steps=300
    
    # Init a random code to start from.
    code = torch.rand( 1, code_len ).to(device) 

    optimizer = torch.optim.LBFGS([code.requires_grad_()], lr=lr)
    
    # Make sure we're in evaluation mode.
    model.eval()
    classifier.eval()

    step = 0
    while step < num_steps:
        step += 1

        def closure():
            
            optimizer.zero_grad()

            # Produce an image from the code.          
            y = model.module.G(code)

            # Normalize said image s.t. values are between 0 and 1.
            y = (y + 1.0 ) / 2.0
            
            # Try to classify this image
            out = classifier(y)

            #TODO: REVIST CLAMP.
            #out = torch.clamp(out, min=0.0, max=3.0*torch.std(out[0]).item())
            
            # Get the loss with L2 weight decay.
            loss = -out[0, neuron] + wdecay * torch.sum( code**2 )

            loss.backward()
            
            if verbose:
                print(step, ":", "{} ?= {}".format(torch.argmax(out), neuron), 
                    "\nloss={}".format(loss.data), 
                    "\nclass={}".format(out[0,:5].data), 
                    "\ncode={}".format(code[0,:5].data))
        
            return loss
        
        optimizer.step(closure)


    # Get the final image & guessed class label.
    y = model.module.G(code)
    y = (y + 1.0 ) / 2.0

    out = classifier(y)
    
    out_img = y.cpu().detach().numpy().T.reshape(227, 227, 3)
    out_cls = torch.argmax(out).cpu().detach().numpy()
    
    #TODO: REMOVE.
    print(out_img.max(), out_img.min())
    
    return out_img, out_cls


def main():

    # Pull some arguments from the CL.
    args = get_args()

    # Pull the time at the beginning of the program.
    now = time.ctime()

    # Init a cuda device if requested and available.
    if args.cuda and not torch.cuda.is_available():
        print('[ERROR] Requested use of CUDA when unavailable... Continuing with CPU.')
    device = torch.device("cuda:0" if (args.cuda and torch.cuda.is_available()) else "cpu")

    
    # Init the imagenet classifier AlexNet.
    AlexNet = alexnet(pretrained=True)
    AlexNet.to(device)

    # Process with the DeepSim model.
    if args.sim:

        # Init and load the model.
        model = DeepSim()
        model = nn.DataParallel(model)
        model.to(device)

        opt_g, opt_d = get_optimizers(model, lr=0.0002)
        model, *_ = load_checkpoint(model, opt_g, opt_d, filename='./chk/dg18_04_2020-22-47-34_30027_128_lf1_la0.0625_li3_lr0.0001.ptm')

        # Begin processing.
        begin = time.time()
        sim_image, alex_class = synthesize(model, AlexNet, neuron=args._class, device=device, verbose=args.verb)
        end = time.time()

        # Let me know when things have finished processing.
        if args.verb:
            print('[INFO] Completed processing SIM in {:0.4}(s)'.format(end - begin)) 

        # Save/Show the image.
        show_image(img=sim_image, name='{} - Sim_{:04d}'.format(now, args._class), save=args.save, verbose=args.verb)


    #TODO: Copy above for DEEPGEN.

    if args.show:
        # Show the images at the very end.
        plt.show() 

    return


if __name__ == "__main__":
    main()
    