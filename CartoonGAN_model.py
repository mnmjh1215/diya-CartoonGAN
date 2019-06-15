# CartoonGAN implementation in pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as tvmodels


class ResidualBlock(nn.Module):
    def __init__(self, channels=256, use_bias=False):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.BatchNorm2d(channels)
        )

    def forward(self, input):
        residual = input
        x = self.model(input)
        # element-wise sum
        out = x + residual

        return out
        

class Generator(nn.Module):
    def __init__(self, n_res_block=8, use_bias=False):
        super().__init__()

        # down sampling, or layers before residual blocks
        self.down_sampling = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=use_bias),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # res_blocks
        res_blocks = []
        for i in range(n_res_block):
            res_blocks.append(ResidualBlock(channels=256, use_bias=use_bias))
        self.res_blocks = nn.Sequential(*res_blocks)

        # up sapling, or layers after residual blocks
        self.up_sampling = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3, bias=use_bias),
            nn.Tanh()
        )

    def forward(self, input):
        x = self.down_sampling(input)
        x = self.res_blocks(x)
        out = self.up_sampling(x)
        return out


class Discriminator(nn.Module):
    def __init__(self, leaky_relu_negative_slope=0.2, use_bias=False):
        super().__init__()

        self.negative_slope = leaky_relu_negative_slope
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.LeakyReLU(self.negative_slope, inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.LeakyReLU(self.negative_slope, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(self.negative_slope, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.LeakyReLU(self.negative_slope, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(self.negative_slope, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.negative_slope, inplace=True),

            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1, bias=use_bias)

        )

    def forward(self, input):
        output = self.layers(input)
        return output

        
class FeatureExtractor(nn.Module):
    def __init__(self, network='vgg'):
        # in original paper, authors used vgg.
        # however, there exist much better convolutional networks than vgg, and we may experiment with them
        # possible models may be vgg, resnet, etc
        super().__init__()
        assert network in ['vgg']

        if network == 'vgg':
            vgg = tvmodels.vgg19_bn(pretrained=True)
            self.feature_extractor = vgg.features[:37]
            # vgg.features[36] is conv4_4 layer, which is what original CartoonGAN used

        else:
            # TODO
            pass

        # FeatureExtractor should not be trained
        for child in self.feature_extractor.children():
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, input):
        return self.feature_extractor(input)




