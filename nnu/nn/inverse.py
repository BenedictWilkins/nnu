#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 16-06-2020 11:39:40

    Build inverse PyTorch layers.
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import logging
logger = logging.getLogger("nnu_logger")
logger.setLevel(logging.WARNING)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import module
from . import shape as _shape

def conv2d_transpose(layer, share_weights=False, shape=None, **kwargs):
    """ Transpose 2D convolutional layer (see torch.nn.ConvTranspose2d).

    Args:
        layer (torch.nn.Conv2d): Convolutional layer.
        share_weights (bool, optional): should the inverse layer share weights? Defaults to False.

    Returns:
        torch.nn.ConvTranspose2d: inverse layer.
    """
    
    
    convt2d = nn.ConvTranspose2d(layer.out_channels, layer.in_channels, 
                       kernel_size=layer.kernel_size, 
                       stride=layer.stride, 
                       padding=layer.padding)

    if shape is not None:
        c_shape = _shape.conv2d_shape(layer, shape)
        ct_shape = _shape.conv2dtranspose_shape(convt2d, c_shape)
        dh, dw = shape[1] - ct_shape[1], shape[2] - ct_shape[2]
        #print(convt2d.output_padding, dh, dw)
        convt2d.output_padding = (dh, dw)
    else:
        logger.log("Ambiguous inverse of Conv2d layer without shape argument.")
    
    if share_weights:
        convt2d.weight = layer.weight
    return convt2d

def linear_transpose(layer, share_weights=False, **kwargs):
    """ Transpose linear layer.

    Args:
        layer (torch.nn.Linear): Linear layer.
        share_weights (bool, optional): should the inverse layer share weights? (bias will not be shared). Defaults to False.

    Returns:
        torch.nn.Linear: inverse layer.
    """
    lt = nn.Linear(layer.out_features, layer.in_features, layer.bias is not None)
    if share_weights:
        lt.weight = nn.Parameter(layer.weight.t())
    return lt

def sequential(layer, share_weights = False, **kwargs):
    ilayers = inverse(*[module for module in layer.modules()])
    return nn.Sequential(*ilayers)
    
def local_inverse(layer, *args, **kwargs):
    return layer.inverse(*args, **kwargs)

def identity(layer, *args, **kwargs):
    return layer

inverse_map = {
        #nn.Conv1d: not_implemented,
        nn.Conv2d: conv2d_transpose,
        #nn.Conv3d: not_implemented,
        nn.Linear: linear_transpose,
        nn.Identity: lambda *args, **kwargs: nn.Identity(),
        nn.Sequential: sequential,
        nn.LeakyReLU:   lambda *args, **kwargs: nn.LeakyReLU(),
        nn.ReLU:        lambda *args, **kwargs: nn.ReLU(),
        nn.Sigmoid:     lambda *args, **kwargs: nn.Sigmoid(),
        nn.Tanh:        lambda *args, **kwargs: nn.Tanh(),
        }

def inverse(*layers, **kwargs):
    """ Inverse a sequence of layers. The returned sequence will be in reverse order, i.e. if the input is l1,l2,l3, the output will be the li3,li2,li1.

    Args:
        share_weights (bool, optional): should the inverse layers share the weights of the original?. Defaults to True.

    Returns:
        list: inverse layers (in reverse order).
    """
    inverse_layers = []
    for layer in reversed(layers):
        if type(layer) in inverse_map:
            inverse_layers.append(inverse_map[type(layer)](layer, **kwargs))
        elif hasattr(layer, "inverse"):
            inverse_layers.append(layer.inverse(**kwargs))
        else:
            raise ValueError("Failed to get inverse for layer: {0}".format(layer))

    return inverse_layers