from torch import nn
import torch
from torchvision.models import alexnet
from utils import center_crop
from utils import View, ds_flatten


class DeepSim(nn.Module):

    def __init__(self):
        super(DeepSim, self).__init__()
        # misc
        self.epochs = 0 
        self.batch_size = 0 
        self.negative_slope = 0.3

        # encoder
        e = alexnet(pretrained=True)
        self.e_deconv_output_size = 256
        self.e_desired_output_size = 227
        self.e_features = e.features
        self.e_avgpool = e.avgpool
        self.e_classifier = e.classifier[:5]

        # generator
        self.G = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(negative_slope=self.negative_slope),  #defc7
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(negative_slope=self.negative_slope),  #defc6
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(negative_slope=self.negative_slope),  #defc5
            View(-1, 256, 4, 4),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1),  # 8x8x256
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=self.negative_slope),  #deconv5
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1),  # 8x8x512
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=self.negative_slope),  #deconv5_1
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1),  # 16x16x256
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=self.negative_slope),  #deconv4
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1),  # 16x16x256
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=self.negative_slope),  #deconv4_1
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1),  # 32x32x128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=self.negative_slope),  #deconv3
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1),  # 32x32x128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=self.negative_slope),  #deconv3_1
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1),  # 64x64x64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=self.negative_slope),  #deconv2
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1),  # 128x128x32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=self.negative_slope),  #deconv1
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=3,
                kernel_size=4,
                stride=2,
                padding=1),  # 256x256x3                    #deconv0
            nn.Tanh()
        )

        # comparator
        c = alexnet(pretrained=True)
        self.c_features = c.features
        self.c_avgpool = c.avgpool

        # discriminator
        self.d_00 = nn.Sequential(  # input: 227x227x3
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
            nn.AvgPool2d(kernel_size=12, stride=12),  # 1x1x256
            ds_flatten()
        )
        self.d_02 = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.LeakyReLU(negative_slope=self.negative_slope),
            nn.Linear(1024, 512),
        )
        self.d_03 = nn.Sequential(
            nn.LeakyReLU(negative_slope=self.negative_slope),
            nn.Dropout(0.5),
            nn.Linear(768, 512),
            nn.LeakyReLU(negative_slope=self.negative_slope),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
        )


    def forward(self, y):
        # encoder
        x = self.e_features(y)
        x = self.e_avgpool(x)
        x = torch.flatten(x, 1)
        x = self.e_classifier(x)

        # generator
        gx = self.G(x)  # 4096
        gx = center_crop(gx, self.e_deconv_output_size,
                        self.e_desired_output_size)  # 227x227x3

        # comparator
        cgx = self.c_features(gx)
        cgx = self.c_avgpool(cgx)
        cgx = torch.flatten(cgx, 1)

        cy = self.c_features(y)
        cy = self.c_avgpool(cy)
        cy = torch.flatten(cy, 1)

        # discriminator
        egx = self.e_features(gx)
        egx = self.e_avgpool(egx)
        egx = torch.flatten(egx, 1)
        egx = self.e_classifier(egx)

        # image = y, enc(image) = x
        # image = gx, enc(image) = egx
        h0y = self.d_00(y)
        h0gx = self.d_00(gx)
        h1y = self.d_02(x)
        h1gx = self.d_02(egx)
        dy = torch.cat((h0y, h1y), dim=1)
        dgx = torch.cat((h0gx, h1gx), dim=1)
        
        return y, x, gxc, gxd, cgx, cy, dgx, dy

