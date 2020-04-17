from torch import nn
import torch
from torchvision.models import alexnet
from utils import center_crop


class AlexNetComparator(nn.Module):

    def __init__(self):
        super(AlexNetComparator, self).__init__()
        original_model = alexnet(pretrained=True)
        self.features = original_model.features
        self.avgpool = original_model.avgpool

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class AlexNetEncoder(nn.Module):

    def __init__(self):
        super(AlexNetEncoder, self).__init__()
        original_model = alexnet(pretrained=True)
        self.deconv_output_size = 256
        self.desired_output_size = 227
        self.features = original_model.features
        self.avgpool = original_model.avgpool
        self.classifier = original_model.classifier[:3]

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class TransposeConvGenerator(nn.Module):

    def __init__(self):
        super(TransposeConvGenerator, self).__init__()
        # https://github.com/shijx12/DeepSim/blob/master/deepSimGAN/deepSimNet.py
        negative_slope = 0.01
        self.epochs = 0
        self.batch_size = 128
        self.deconv_output_size = 256
        self.desired_output_size = 227
        self.fully_connected = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(negative_slope=negative_slope),  #defc7
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(negative_slope=negative_slope),  #defc6
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(negative_slope=negative_slope),  #defc5
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1),  # 8x8x256
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=negative_slope),  #deconv5
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1),  # 8x8x512
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=negative_slope),  #deconv5_1
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1),  # 16x16x256
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=negative_slope),  #deconv4
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1),  # 16x16x256
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=negative_slope),  #deconv4_1
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1),  # 32x32x128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=negative_slope),  #deconv3
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1),  # 32x32x128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=negative_slope),  #deconv3_1
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1),  # 64x64x64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=negative_slope),  #deconv2
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1),  # 128x128x32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=negative_slope),  #deconv1
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=3,
                kernel_size=4,
                stride=2,
                padding=1),  # 256x256x3                    #deconv0
            nn.Tanh())

    def forward(self, x):
        x = self.fully_connected(x)  # 4096
        x = x.reshape((-1, 256, 4, 4))  # 4x4x256
        x = self.deconv(x)  # 256x256x3
        x = center_crop(x, self.deconv_output_size,
                        self.desired_output_size)  # 227x227x3
        return x


class UpsampleConvGenerator(nn.Module):

    def __init__(self):
        super(UpsampleConvGenerator, self).__init__()
        negative_slope = 0.01
        self.deconv_output_size = 256
        self.desired_output_size = 227
        self.fully_connected = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.deconv = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),  # 8x8x256
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1),  # 8x8x256
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1),  # 8x8x512
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.UpsamplingNearest2d(scale_factor=2),  # 16x16x512
            nn.Conv2d(
                in_channels=512,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1),  # 16x16x256
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1),  # 16x16x256
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.UpsamplingNearest2d(scale_factor=2),  # 32x32x256
            nn.Conv2d(
                in_channels=256,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1),  # 32x32x128
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1),  # 32x32x128
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.UpsamplingNearest2d(scale_factor=2),  # 64x64x128
            nn.Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1),  # 64x64x64
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.UpsamplingNearest2d(scale_factor=2),  # 128x128x64
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1),  # 128x128x32
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.UpsamplingNearest2d(scale_factor=2),  # 256x256x32
            nn.Conv2d(
                in_channels=32,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1),  # 256x256x3
            nn.Tanh())

    def forward(self, x):
        x = self.fully_connected(x)  # 4096
        x = x.reshape((-1, 256, 4, 4))  # 4x4x256
        x = self.deconv(x)  # 256x256x3
        x = center_crop(x, self.deconv_output_size,
                        self.desired_output_size)  # 227x227x3
        return x


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        negative_slope = 0.01
        self.conv = nn.Sequential(  # input: 227x227x3
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=7,
                stride=4,
                padding=0),  # 56x56x32
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=0),  # 52x52x64
            nn.ZeroPad2d((1, 0, 1, 0)),  # 53x53x64
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=0),  # 26x26x128
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=0),  # 24x24x256
            nn.ZeroPad2d((1, 0, 1, 0)),  # 25x25x256
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=0),  # 12x12x256
            nn.AvgPool2d(kernel_size=12, stride=12))  # 1x1x256

        self.features_fc = nn.Sequential(  # input: 9216
            nn.Linear(9216, 1024),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(1024, 512),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768, 512),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
        )

    def forward(self, image, features):
        x1 = self.conv(image)  # 1x1x256
        x1 = torch.flatten(x1, 1)  # 256
        x2 = self.features_fc(features)  # 512
        x = torch.cat((x1, x2), dim=1)  # 768
        x = self.fc(x)  # 1
        return x


class DeepSim(nn.Module):

    def __init__(self):
        super(DeepSim, self).__init__()
        self.batch_size = 128
        self.E = AlexNetEncoder()
        for param in self.E.parameters():
            param.require_grad = False
        self.E.eval()
        self.G = TransposeConvGenerator()
        self.C = AlexNetComparator()
        for param in self.C.parameters():
            param.require_grad = False
        self.C.eval()
        self.D = Discriminator()

    def forward(self, y):
        x = self.E(y)
        gx = self.G(x)
        egx = self.E(gx)
        cgx = self.C(gx)
        cy = self.C(y)
        dgx = self.D(gx, cgx)
        dy = self.D(y, cy)
        return y, x, gx, egx, cgx, cy, dgx, dy


class DeepGen(nn.Module):

    def __init__(self):
        super(DeepGen, self).__init__()
        self.batch_size = 128
        self.E = AlexNetEncoder()
        for param in self.E.parameters():
            param.require_grad = False
        self.E.eval()
        self.G = UpsampleConvGenerator()
        self.C = AlexNetComparator()
        for param in self.C.parameters():
            param.require_grad = False
        self.C.eval()
        self.D = Discriminator()

    def forward(self, y):
        x = self.E(y)
        gx = self.G(x)
        egx = self.E(gx)
        cgx = self.C(gx)
        cy = self.C(y)
        dgx = self.D(gx, cgx)
        dy = self.D(y, cy)
        return y, x, gx, egx, cgx, cy, dgx, dy