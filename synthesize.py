import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from matplotlib.animation import ArtistAnimation
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
    
    parser.add_argument('--class', type=int, default=0, dest='_class', help='Synthesize the highest activation image of this class. Range=[0, 1000] (default: {})'.format(0)) #TODO: Find which class is ~ C O R N ~
    parser.add_argument('--steps', type=int, default=100,              help='The number of steps that should be taken to find the code.            (default: {})'.format(100))
    parser.add_argument('--sim',   action='store_true', default=False, help='Pass the provided image through the DeepSim architecture              (default: {})'.format('DISABLED'))
    parser.add_argument('--gen',   action='store_true', default=False, help='Pass the provided image through the DeepGen architecture              (default: {})'.format('DISABLED'))
    parser.add_argument('--show',  action='store_true', default=False, help='Show the initial and generated images                                 (default: {})'.format('DISABLED'))
    parser.add_argument('--save',  action='store_true', default=False, help='Save the output generated image                                       (default: {})'.format('DISABLED'))
    parser.add_argument('--vid',   action='store_true', default=False, help='Save the morphing images as a .mp4                                    (default: {})'.format('DISABLED'))
    parser.add_argument('--freq',  type=int, default=1,                help='Frequency at which images should be saved for .mp4                    (default: {})'.format(1))
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



def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img



def show_vid(img, name="", save=False, verbose=False):

    fig = plt.figure()
    plt.tight_layout()
    plt.axis('off')
    plt.title(name)
    ima = []
    for cur in img:
        im = plt.imshow(cur, animated=True, aspect='equal')
        ima.append( [im] )
    ani = ArtistAnimation(fig, ima, interval=30, blit=True)
    
    if save:
        import imageio
        img_gif = convert(np.asarray(img), 0, 255, np.uint8)
        imageio.mimwrite('img/{}.gif'.format(name), img_gif, fps=32)
        if verbose: print('[INFO] Saved ``{}.gif``'.format(name))
        
    return
    


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



def synthesize(model, classifier, neuron=0, num_steps=300, lr=0.005, wdecay=0.0001, num_class=1000, code_len=4096, dims=(227, 227, 3), device=torch.device("cpu"), keep_steps=False, keep_freq=1, verbose=True):
    
    # Init a random code to start from.
    code = torch.rand( 1, code_len ).to(device) 

    optimizer = torch.optim.LBFGS([code.requires_grad_()], lr=lr)
    
    # Make sure we're in evaluation mode.
    model.eval()
    classifier.eval()

    step = 0
    meta = 0
    keep_imgs = []
    while step < num_steps:
        step += 1
        meta = 0

        def closure():

            optimizer.zero_grad()

            # Produce an image from the code.          
            y = model.module.G(code)

            # Normalize said image s.t. values are between 0 and 1.
            y = (y + 1.0 ) / 2.0
            
            # Try to classify this image
            out = classifier(y)
            
            # Get the loss with L2 weight decay.
            loss = -out[0, neuron] + wdecay * torch.sum( code**2 )

            #loss.backward(retain_graph=True)
            loss.backward()
            
            if verbose:
                print("[INFO] {:03d} : {} ?= {}".format(step, torch.argmax(out), neuron), 
                    "\n   loss  = {}".format(loss.data), 
                    "\n   class = {}".format(out[0,:5].data), 
                    "\n   code  = {}".format(code[0,:5].data),
                    "\n   clamp = {}".format(code[0,:5].data))

            return loss
        
        optimizer.step(closure)

        with torch.no_grad():
            code.clamp_(min=0, max=3.0*code.std().item())

        if keep_steps and ((step % keep_freq) == 0):
            y = model.module.G(code)
            y = (y + 1.0 ) / 2.0
            keep_imgs.append(y.cpu().detach().numpy().T.reshape(*dims))


    # Get the final image & guessed class label.
    y = model.module.G(code)
    y = (y + 1.0 ) / 2.0

    out = classifier(y)
    
    out_img = y.cpu().detach().numpy().T.reshape(*dims)
    out_cls = torch.argmax(out).cpu().detach().numpy()
        
    return out_img, out_cls, keep_imgs



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
        model, *_ = load_checkpoint(model, opt_g, opt_d, filename='./chk/ds19_04_2020-22-39-16_120108_64_lf1_la0.0625_li3_lr0.0002.ptm')

        # Begin processing.
        begin = time.time()
    
        sim_image, alex_class, sim_video = synthesize(
            model=model, 
            classifier=AlexNet, 
            neuron=args._class, 
            num_steps=args.steps,
            lr=0.005, 
            wdecay=0.0001,
            device=device, 
            keep_steps=args.vid, 
            keep_freq=args.freq,
            verbose=args.verb
        )
        end = time.time()

        # Let me know when things have finished processing.
        if args.verb:
            print('[INFO] Completed processing SIM in {:0.4}(s)!! Requested Class {} -- Generated Class {}'.format(end - begin, args._class, alex_class)) 

        # Save/Show the image.
        show_image(img=sim_image, name='{} - Sim_{:04d}'.format(now, args._class), save=args.save, verbose=args.verb)

        if args.vid:
            show_vid(img=sim_video, name='{} - Sim_{:04d}'.format(now, args._class), save=args.save, verbose=args.verb)


    #TODO: Copy above for DEEPGEN.


    if args.show:
        # Show the images at the very end.
        plt.show() 

    return


if __name__ == "__main__":
    main()
    