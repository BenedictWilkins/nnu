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
logger = logging.getLogger("nnu")
logger.setLevel(logging.WARNING)

from pprint import pprint # TODO remove
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import module
from . import shape as _shape

_AMBIGUOUS_SHAPE_MSG = "Ambiguous inverse of layer {0} without shape argument."

def Conv2d(layer, share_weights=False, shape=None, **kwargs):
    convt2d = nn.ConvTranspose2d(layer.out_channels, layer.in_channels, 
                kernel_size=layer.kernel_size, 
                stride=layer.stride, 
                padding=layer.padding)

    if shape is not None:
        c_shape = _shape.Conv2d(layer, shape)
        ct_shape = _shape.ConvTranspose2d(convt2d, c_shape)
        dh, dw = shape[1] - ct_shape[1], shape[2] - ct_shape[2]
        #print(convt2d.output_padding, dh, dw)
        convt2d.output_padding = (dh, dw)
    else:
        logger.warn(_AMBIGUOUS_SHAPE_MSG.format(layer))
    
    if share_weights:
        convt2d.weight = layer.weight
    return convt2d


def MaxUnpool1d(layer, **kwargs):
    raise NotImplementedError() #TODO

def MaxUnpool2d(layer, **kwargs):
    assert isinstance(layer, nn.MaxPool2d)
    return nn.MaxUnpool2d(layer.kernel_size, layer.stride, layer.padding)

def MaxUnpool3d(layer, **kwargs):
    raise NotImplementedError() #TODO

def Linear(layer, share_weights=False, **kwargs):
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

def BatchNorm1d(layer, **kwargs):
    return copy.deepcopy(layer)

def BatchNorm2d(layer, **kwargs):
    return copy.deepcopy(layer)

def BatchNorm3d(layer, **kwargs):
    return copy.deepcopy(layer)

def Dropout(layer, **kwargs):
    return copy.deepcopy(layer)

def AdaptiveAvgPool1d(layer, shape=None, **kwargs):
    raise NotImplementedError()

def AdaptiveAvgPool2d(layer, shape=None, **kwargs):
    assert len(shape) == 3 # C,H,W 
    if shape is None:
        raise ValueError(_AMBIGUOUS_SHAPE_MSG.format(layer))
    return nn.AdaptiveAvgPool2d(shape[1:])

def AdaptiveAvgPool3d(layer, shape=None, **kwargs):
    raise NotImplementedError()

def Sequential(sequential, shape=None, **kwargs):
    # assert isinstance(sequential, nn.Sequential)
    if shape is not None:
        shapes = _shape.Sequential(sequential, shape)
    else:
        logger.warn("`shape` parameter was not provided, module output shapes may be ambiguous.")
        raise ValueError("TOOD -- take inverse without shape? (see warning)")
    
    imodules = []
    shapes = reversed(shapes)

    pprint(list(shapes))
    children = reversed(list(sequential.children()))
    for child, shape in zip(children, shapes):

        imodules.append(inverse(child, shape=shape, **kwargs)[0])
    return nn.Sequential(*imodules)

def local_inverse(layer, *args, **kwargs):
    return layer.inverse(*args, **kwargs)

def identity(layer, *args, **kwargs):
    return copy.deepcopy(layer)

inverse_map = {
        #nn.Conv1d: not_implemented,
        nn.Conv2d: Conv2d,
        #nn.Conv3d: not_implemented,
        nn.Linear: Linear,
        nn.Identity: lambda *args, **kwargs: nn.Identity(),
        nn.Sequential: Sequential,
        nn.LeakyReLU:   lambda *args, **kwargs: nn.LeakyReLU(),
        nn.ReLU:        lambda *args, **kwargs: nn.ReLU(),
        nn.Sigmoid:     lambda *args, **kwargs: nn.Sigmoid(),
        nn.Tanh:        lambda *args, **kwargs: nn.Tanh(),
        nn.BatchNorm1d: BatchNorm1d,
        nn.BatchNorm2d: BatchNorm2d,
        nn.BatchNorm3d: BatchNorm3d,
        nn.AdaptiveAvgPool1d: AdaptiveAvgPool1d,
        nn.AdaptiveAvgPool2d: AdaptiveAvgPool2d,
        nn.AdaptiveAvgPool3d: AdaptiveAvgPool3d,

        nn.MaxPool1d: MaxUnpool1d,
        nn.MaxPool2d: MaxUnpool2d,
        nn.MaxPool3d: MaxUnpool3d,

        nn.Dropout: identity,
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