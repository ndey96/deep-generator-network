# deep-generator-network

Unofficial implementation of [Synthesizing the preferred inputs for neurons in neural networks via deep generator networks by Nguyen et al.](https://papers.nips.cc/paper/6519-synthesizing-the-preferred-inputs-for-neurons-in-neural-networks-via-deep-generator-networks.pdf).

Original:
<img width="1035" alt="Screen Shot 2020-09-18 at 1 42 16 AM" src="https://user-images.githubusercontent.com/10405248/93560565-7262d100-f950-11ea-8432-c056c0bed048.png">

Ours:
<img width="1024" alt="Screen Shot 2020-09-18 at 1 42 08 AM" src="https://user-images.githubusercontent.com/10405248/93560562-6f67e080-f950-11ea-91e0-22593734dc81.png">


Pretrained models are available for download at:
https://www.dropbox.com/sh/gdkr6n1d83jx2kg/AACet2EbEWFmlpysHNOSYemHa?dl=

To train, and generate new training weights, or synthesize using either the provided pre-trained checkpoints, or your own
ilsvrc2012 is required local to your machine.  Any hardcoded paths in train_parallel_deep*.py or synthesize.py, to ilsvrc2012 training and validation image directories should be set. 

Various parameters can be set - you'll have to experiment and read the related papers to get a feel for that. 

Python3, pytorch, pyvision, and tensorboard are required... any other requirements will error out... you can install them. 

The code has not been fully tested on cpu only.  

Training:
python train_parallel_deepgen.py
python train_parallel_deepsim.py

Synthesis:
bash synth_figs.sh
