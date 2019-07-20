# Default Generator and Discriminator used in CycleGAN
# reference: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
# (Official CycleGAN PyTorch implementation)

import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels=256, use_bias=True):
        super().__init__()

        self.model = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0, bias=use_bias),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0, bias=use_bias),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, input):
        # element-wise sum
        out = self.model(input) + input

        return out


class Generator(nn.Module):
    def __init__(self, n_res_block=9, use_bias=True):
        super().__init__()

        self.down_sampling = nn.Sequential(
            # initial conv layer
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=0, bias=use_bias),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),

            # 2 down-samplings
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.InstanceNorm2d(256),
            nn.ReLU(True)
        )

        res_blocks = []
        for i in range(n_res_block):
            res_blocks.append(ResidualBlock(channels=256, use_bias=use_bias))
        self.res_blocks = nn.Sequential(*res_blocks)

        self.up_sampling = nn.Sequential(
            # 2 up-sampling
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),

            # final conv layer to generate image
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=0, bias=use_bias),
            nn.Tanh()

        )

    def forward(self, input):
        x = self.down_sampling(input)
        x = self.res_blocks(x)
        output = self.up_sampling(x)

        return output


class Discriminator(nn.Module):
    def __init__(self, use_bias=False):
        super().__init__()

        self.model = nn.Sequential(
            # initial conv layer, no instance normalization
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=use_bias),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=use_bias),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=use_bias),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
            # sigmoid is not necessary
        )

    def forward(self, input):
        return self.model(input)

