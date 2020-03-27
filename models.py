from torch import nn
import torch
from torchvision.models import alexnet


class AlexNetComparator(nn.Module):

    def __init__(self):
        super(AlexNetComparator, self).__init__()
        original_model = alexnet(pretrained=True)
        self.features = torch.nn.DataParallel(original_model.features)
        self.avgpool = original_model.avgpool

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class AlexNetEncoder(nn.Module):

    def __init__(self):
        super(AlexNetEncoder, self).__init__()
        original_model = models.alexnet(pretrained=True)
        self.features = torch.nn.DataParallel(original_model.features)
        self.avgpool = original_model.avgpool
        self.classifier = original_model.classifier[:5]

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class TransposeConvGenerator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        # https://github.com/shijx12/DeepSim/blob/master/deepSimGAN/deepSimNet.py
        negative_slope = 0.3
        self.fully_connected = nn.Sequential(
            nn.Linear(input_size, 4096),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1),  # 8x8x256
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1),  # 8x8x512
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1),  # 16x16x256
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1),  # 16x16x256
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1),  # 32x32x128
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1),  # 32x32x128
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1),  # 64x64x64
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1),  # 128x128x32
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=3,
                kernel_size=4,
                stride=2,
                padding=17),  # 224x224x3
        )

    def forward(self, x):
        x = self.fully_connected(x)
        x = x.reshape((-1, 4, 4, 256))
        x = self.deconv(x)
        return x


class UpsampleConvGenerator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        negative_slope = 0.3
        padding_mode = 'same'
        self.fully_connected = nn.Sequential(
            nn.Linear(input_size, 4096),
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
                padding_mode=padding_mode),  # 8x8x256
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding_mode=padding_mode),  # 8x8x512
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.UpsamplingNearest2d(scale_factor=2),  # 16x16x512
            nn.Conv2d(
                in_channels=512,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding_mode=padding_mode),  # 16x16x256
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding_mode=padding_mode),  # 16x16x256
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.UpsamplingNearest2d(scale_factor=2),  # 32x32x256
            nn.Conv2d(
                in_channels=256,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding_mode=padding_mode),  # 32x32x128
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding_mode=padding_mode),  # 32x32x128
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.UpsamplingNearest2d(scale_factor=2),  # 64x64x128
            nn.Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding_mode=padding_mode),  # 64x64x64
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.UpsamplingNearest2d(scale_factor=2),  # 128x128x64
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding_mode=padding_mode),  # 128x128x32
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.UpsamplingNearest2d(scale_factor=1.75),  # 224x224x32
            nn.Conv2d(
                in_channels=32,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding_mode=padding_mode),  # 224x224x3
        )

    def forward(self, x):
        x = self.fully_connected(x)
        x = x.reshape((-1, 4, 4, 256))
        x = self.deconv(x)
        return x


class Discriminator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        padding_mode = 'valid'
        negative_slope = 0.3
        self.conv = nn.Sequential(
            nn.Conv(
                in_channels=3,
                out_channels=32,
                kernel_size=7,
                stride=4,
                padding_mode=padding_mode),
            nn.Conv(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding_mode=padding_mode),
            nn.Conv(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding_mode=padding_mode),
            nn.Conv(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding_mode=padding_mode),
            nn.Conv(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding_mode=padding_mode),
            nn.AvgPool2d(kernel_size=11, stride=11))
        self.features_fc = nn.Sequential(
            nn.Linear(69,
                      1024),  # TODO: not sure what the input dimension is...
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(1024, 512),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(69, 512),  # TODO: not sure what the input dimension is...
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Dropout(0.5),
            nn.Linear(69, 1),  # TODO: not sure what the input dimension is...
        )

    def forward(self, image, features):
        x1 = self.conv(image)
        x1 = torch.flatten(x1, 1)
        x2 = self.features_fc(features)

        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x