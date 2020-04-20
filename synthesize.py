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
from optimizer_stub import get_optimizers

import argparse


def get_args():

    # Init a parser.
    parser = argparse.ArgumentParser(
        prog='Synthesize',
        description=
        'Provided a class number in Range=[0, 1000], synthesize an image through the Deep Sim model, or our ``improved`` Deep Generator model (or both!!)',
        usage=
        'python synthesize.py [--class CLASS] [--sim] [--gen] [--save] [--cuda] [--verb]'
    )

    # Add arguments to parser.
    #parser.add_argument('--img',  default=None,  help='Path to an image to pass through the selected architecture(s)                 (default: {})'.format('[REQUIRED]')) #TODO: REMOVE. But maybe keep for a style transfer demo...?

    parser.add_argument(
        '--class',
        dest='_class',
        type=int,
        default=0,
        help=
        'Synthesize the highest activation image of this class. Range=[0, 1000] (default: {})'.
        format(0))  #TODO: Find which class is ~ C O R N ~
    parser.add_argument(
        '--sim',
        action='store_true',
        default=False,
        help=
        'Pass the provided image through the DeepSim architecture              (default: {})'.
        format('DISABLED'))
    parser.add_argument(
        '--gen',
        action='store_true',
        default=False,
        help=
        'Pass the provided image through the DeepGen architecture              (default: {})'.
        format('DISABLED'))
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help=
        'Show the initial and generated images                                 (default: {})'.
        format('DISABLED'))
    parser.add_argument(
        '--save',
        action='store_true',
        default=False,
        help=
        'Save the output generated image                                       (default: {})'.
        format('DISABLED'))
    parser.add_argument(
        '--cuda',
        action='store_true',
        default=False,
        help=
        'Enable CUDA for processing                                            (default: {})'.
        format('DISABLED'))
    parser.add_argument(
        '--verb',
        action='store_true',
        default=False,
        help=
        'Enable Verbose output                                                 (default: {})'.
        format('DISABLED'))
    args = parser.parse_args()

    if not (args.sim or args.gen):
        parser.print_help()
        print(
            "\n Please ensure an image and an architecture are chosen. For example: \n"
        )
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


def synthesize(model, classifier, neuron=0, device=torch.device("cpu")):

    # Make sure we're in evaluation mode.
    model.eval()
    classifier.eval()

    code = torch.rand(1, 4096).to(device)  # Optimize for this.
    x = model.G(code)  # Normalize??
    y = classifier(x)

    # TODO: Revisit.
    out_img = x.cpu().detach().numpy().reshape(227, 227, 3, order='F')
    out_cls = y.cpu().detach().numpy()

    # TODO: Might need to normalize the image!!?
    out_img += np.abs(out_img.min())

    return out_img, out_cls


def main():

    # Pull some arguments from the CL.
    args = get_args()

    # Pull the time at the beginning of the program.
    now = time.ctime()

    # Init a cuda device if requested and available.
    if args.cuda and not torch.cuda.is_available():
        print(
            '[ERROR] Requested use of CUDA when unavailable... Continuing with CPU.'
        )
    device = torch.device("cuda:0" if (
        args.cuda and torch.cuda.is_available()) else "cpu")

    # Init the imagenet classifier AlexNet.
    AlexNet = alexnet(pretrained=True)
    AlexNet.to(device)

    # Process with the DeepSim model.
    if args.sim:

        # Init and load the model.
        model = DeepSim()
        model.to(device)

        #
        #TODO: need consistency between models_parallel, optimizer_stub, & checkpoint_stub.
        #
        #opt_g, opt_d = get_optimizers(model, lr=0.0002)
        #model, *_ = load_checkpoint(model, opt_g, opt_d, filename='./chk/17_04_2020-18-40-21_10009_128_lf0.01_la1_li0.01_lr0.0002.ptm')

        #TODO: model go cuda here?
        #model.to(device)

        # Begin processing.
        begin = time.time()
        sim_image, alex_class = synthesize(
            model, AlexNet, neuron=args._class, device=device)
        end = time.time()

        # Let me know when things have finished processing.
        if args.verb:
            print('[INFO] Completed processing SIM in {:0.4}(s)'.format(
                end - begin))

        # Save/Show the image.
        show_image(
            img=sim_image,
            name='{} - Sim_{:04d}'.format(now, args._class),
            save=args.save,
            verbose=args.verb)

    #TODO: Copy above for DEEPGEN.

    if args.show:
        # Show the images at the very end.
        plt.show()

    return


if __name__ == "__main__":
    main()

    from torchvision.models import alexnet
    import torch
    a = alexnet()
    img = torch.rand(1, 3, 224, 224)
    img.requires_grad = True
    optimizer = torch.optim.LBFGS([img])
    for i in range(1000):
        optimizer.zero_grad()
        acts = a.forward(img)
        candle_act = acts[:, 0]
        loss = -candle_act
        loss.backward()
        print(img.grad.shape)  # to show that img.grad gets populated
        optimizer.step()

    plt.imshow(img.detach().numpy())