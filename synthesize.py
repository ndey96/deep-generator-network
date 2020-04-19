import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
import time

import torch
import torch.nn as nn

from models_parallel import DeepSim
from models_parallel import DeepGen

from checkpoint_stub import load_checkpoint
from optimizer_stub  import get_optimizers


# TODO: optimize input to generator that will maximize the 
#        activation of a particular output neuron in AlexNet

import argparse

def get_args():
    
    # Init a parser.
    parser = argparse.ArgumentParser (
        prog='Synthesize', 
        description='Provided a class number in Range=[0, ...], synthesize an image through the Deep Sim model, or our ``improved`` Deep Generator model (or both!!)',
        usage='python synthesize.py [--class CLASS] [--sim] [--gen] [--save] [--cuda] [--verb]'
    )
    
    # Add arguments to parser.
    #parser.add_argument('--img',  default=None,  help='Path to an image to pass through the selected architecture(s)                 (default: {})'.format('[REQUIRED]')) #TODO: REMOVE. But maybe keep for a style transfer demo...?
    
    parser.add_argument('--class', dest='_class', type=int, default=0, help='Synthesize the highest activation image of this class. Range=[0, ...] (default: {})'.format(0)) #TODO: Find which class is ~ C O R N ~
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


def show_image(img, name="", save=False, verbose=False):

    plt.figure()
    plt.imshow(img, aspect='equal')
    plt.axis('off')
    plt.title(name)

    if save:
        plt.imsave(save+'.png', img, format='png')
        if verbose: print('[INFO] Saved ``{}``'.format(name))

    return


def synthesize(model, neuron=0):

    model.eval()

    print(model.get_device())

    return np.random.rand(128, 128, 3)


def main():

    # Pull some arguments from the CL.
    args = get_args()

    # Pull the time at the beginning of the program.
    now = time.ctime()

    # Init a cuda device if requested and available.
    if args.cuda and not torch.cuda.is_available():
        print('[ERROR] Requested is of CUDA when unavailable... Continuing with cpu.')
    device = torch.device("cuda:0" if (args.cuda and torch.cuda.is_available()) else "cpu")


    # Process with the DeepSim model.
    if args.sim:

        # Init and load the model.
        model = DeepSim()
        model.to(device)

        opt_g, opt_d = get_optimizers(model, lr=0.0002)
        model, *_ = load_checkpoint(model, opt_g, opt_d, filename='./models/17_04_2020-18-40-21_10009_128_lf0.01_la1_li0.01_lr0.0002.ptm')


        # Begin processing.
        begin = time.time()
        sim_image = synthesize(model, neuron=args._class)
        end = time.time()


        # Let me know when things have finished processing.
        if args.verb:
            print('[INFO] Completed processing SIM in {:0.4}(s)'.format(end - begin)) 

        # Save/Show the image.
        show_image(img=sim_image, name=now+' - Sim_{:03d}'.format(args._class), save=args.save, verbose=args.verb) #TODO: Depending on the number of classes make that 3 a 4 or even 5!!


    #TODO: Copy above for DEEPGEN.

    if args.show:
        # Show the images at the very end.
        plt.show() 

    return


if __name__ == "__main__":
    main()
    

