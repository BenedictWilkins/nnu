#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 01-02-2021 15:36:59

    Some out of the box architectures for auto-encoders.
"""
__author__ ="Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ ="Development"

import torch
import torch.nn as nn

from ..module import ResBlock2D
from ..inverse import inverse

class AE2D(nn.Module):

    def __init__(self, shape):
        super(AE2D, self).__init__()
        self.encode = Encoder2D(shape)
        self.decode = Decoder2D(shape)

    def forward(self, x):
        return self.decode(self.encode(x))

class Encoder2D(nn.Module): 

    def __init__(self, input_shape):
        super(Encoder2D, self).__init__()
        assert len(input_shape) == 3
        self.input_shape = input_shape # CHW
    
        blocks = [
            nn.Conv2d(input_shape[0], 64 // 2, 4, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64 // 2, 64, 4, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(inplace=True),
        ]

        for i in range(3):
            blocks.append(ResBlock2D(128, 128))
            blocks.append(nn.LeakyReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)

    def inverse(self, *args, **kwargs): #TODO doesnt seem to work yet...
        return inverse(self.blocks, share_weights=False)


class Decoder2D(nn.Module):
    def __init__(self, output_shape):
        super(Decoder2D, self).__init__()

        blocks = []
        for i in range(3):
            blocks.append(ResBlock2D(128, 128))
            blocks.append(nn.LeakyReLU(inplace=True))
        
        blocks.extend([
                nn.ConvTranspose2d(128, 64, 3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.ConvTranspose2d(64, 64 // 2, 4, stride=1),
                nn.LeakyReLU(inplace=True),
                nn.ConvTranspose2d(64 // 2, output_shape[0], 4, stride=2),
            ])
        
        self.blocks = nn.Sequential(*blocks)

    def forward(self, z):
        return self.blocks(z)

    def inverse(self, *args, **kwargs): #TODO doesnt seem to work yet...
        return inverse(self.blocks, share_weights=False)