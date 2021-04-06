#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Created on 19-01-2021 10:05:11

 [Description]
"""
__author__ ="Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ ="Development"

import numpy as np
import torch
import torch.nn as nn

from .. import shape

from .sample import *

class View(nn.Module):
    """
        A view of a tensor. Can be used in a network to reshape output/inputs.
    """

    def __init__(self, input_shape, output_shape):
        super(View, self).__init__()
        def infer_shape(x, y): # x contains a -1
            assert not -1 in y
            t_y, t_x = np.prod(y), - np.prod(x)
            assert t_y % t_x == 0 # shapes are incompatible...
            x = list(x)
            
            x[x.index(-1)] = t_y // t_x

            return shape.as_shape(x)

        self.input_shape = shape.as_shape(input_shape)
        self.output_shape = shape.as_shape(output_shape)

        # infer -1 in shape
        if -1 in self.output_shape:
            self.output_shape = infer_shape(self.output_shape, self.input_shape)
        if -1 in self.input_shape:
            self.input_shape = infer_shape(self.input_shape, self.output_shape)

    def __call__(self, x):
        return x.view(x.shape[0], *self.output_shape)

    def __str__(self):
        attrbs = "{0}->{1}".format(self.input_shape, self.output_shape)
        return "{0}({1})".format(self.__class__.__name__, attrbs)

    def __repr__(self):
        return str(self)

    def shape(self, *args, **kwargs):
        return self.output_shape

    def inverse(self, **kwargs):
        return View(self.output_shape, self.input_shape)

class ResBlock2D(nn.Module):

    def __init__(self, in_channel, channel):
        super(ResBlock2D, self).__init__()
        self.in_channel = in_channel
        self.channel = channel

        self.c1 = nn.Conv2d(in_channel, channel, 3, 1, 1)
        self.r1 = nn.ReLU(inplace=True)
        self.c2 = nn.Conv2d(channel, in_channel, 1)
        
    def forward(self, x):
        x_ = self.c2(self.r1(self.c1(x)))
        x_ += x
        return x_

    def shape(self, input_shape):
        return input_shape # resnet blocks are the same input/output shape!

    def inverse(self, *args, share_weights=False, **kwargs):
        return ResBlock2D(self.in_channel, self.channel) # identity inverse...

class NormaliseLogits(nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, logits):
        return logits - logits.logsumexp(dim=-1, keepdim=True)