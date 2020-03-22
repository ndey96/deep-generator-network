from models import Generator
from torchvision.models import alexnet

net = alexnet(pretrained=True)
generator = Generator()
generator.cuda()

# TODO: optimize input to generator that will maximize the activation of a particular output neuron in AlexNet