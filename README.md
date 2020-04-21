# deep-generator-network

Pretrained models are available for download at:
https://www.dropbox.com/sh/gdkr6n1d83jx2kg/AACet2EbEWFmlpysHNOSYemHa?dl=

To train, and generate new training weights, or synthesize using either the provided pre-trained checkpoints, or your own
ilsvrc2012 is required local to your machine.  Any hardcoded paths in train_parallel_deep*.py or synthesize.py, to ilsvrc2012 
training and validation image directories should be set. 

Various parameters can be set- as per the project report 

Python3, pytorch, pyvision, and tensorboard are required... any other requirements will error out... you can install them. 

The code has not been fully tested on cpu only.  

Training:
python train_parallel_deepgen.py
python train_parallel_deepsim.py

Synthesis:
bash synth_figs.py