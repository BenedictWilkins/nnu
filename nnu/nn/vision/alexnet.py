#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Created on 10-05-2021 16:21:00

 [Description]
"""
__author__ ="Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ ="Development"

from .. import Sequential
from .. import shape
from ..module import View


import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(Sequential):

    def __init__(self, input_shape, output_shape, dropout=False, output_activation=nn.Identity()):
        self.input_shape = input_shape = shape.as_shape(input_shape)
        self.output_shape = output_shape = shape.as_shape(output_shape)
        assert not dropout # TODO support dropout option

        features = [
            nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
            nn.LeakyReLU(),
            #nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.LeakyReLU(),
            #nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(192, 192, kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(),
            #nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
            nn.Conv2d(256, 256, kernel_size=3, stride=2),
           ]
        # using convolutions instead of maxpooling makes creating a decoder easier 
        # (its slightly more expensive but ... oh well)
        
        avgpool = [nn.AdaptiveAvgPool2d(output_size=(6, 6))] 
        
        classifier = [
                View((256,6,6), 9216),
                #nn.Dropout(p=0.5, inplace=False),
                nn.Linear(in_features=9216, out_features=4096, bias=True),
                nn.LeakyReLU(),
                #nn.Dropout(p=0.5, inplace=False),
                nn.Linear(in_features=4096, out_features=4096, bias=True),
                nn.LeakyReLU(),
                nn.Linear(in_features=4096, out_features=output_shape[0], bias=True),
        ]
        modules = features + avgpool + classifier + [output_activation]
        super().__init__(*modules)

    def inverse(self):
        return super().inverse()