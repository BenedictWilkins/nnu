#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 16-06-2020 15:16:33

    Shape utilities.
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import math
import torch.nn as nn

def as_shape(shape):
    if isinstance(shape, int):
        return (shape,)
    else:
        return tuple(shape)

def MaxPool2d(layer, input_shape, *args, **kwargs):
    return Conv2d(layer, input_shape, *args, **kwargs)

def Conv2d(layer, input_shape, *args, **kwargs):
    """ 
        Get the output shape of a 2D convolution given the input_shape.
        
    Args:
        layer (nn.Conv2d): 2D convolutional layer.
        input_shape (tuple): expected input shape (CHW format)

    Returns:
        tuple: output shape (CHW)
    """
    input_shape = as_shape(input_shape)
    h,w = input_shape[-2:]
    pad, dilation, kernel_size, stride = layer.padding, layer.dilation, layer.kernel_size, layer.stride
    
    def tuplise(x):
        if not isinstance(x, tuple):
            return (x,x)
        return x

    kernel_size = tuplise(kernel_size)
    pad         = tuplise(pad)
    dilation    = tuplise(dilation)
    stride      = tuplise(stride)
    h = math.floor(((h + (2 * pad[0]) - (dilation[0] * (kernel_size[0] - 1) ) - 1 )/ stride[0]) + 1)
    w = math.floor(((w + (2 * pad[1]) - (dilation[1] * (kernel_size[1] - 1) ) - 1 )/ stride[1]) + 1)

    if isinstance(layer, nn.MaxPool2d):
        return input_shape[-3], h, w # CHW
    return layer.out_channels, h, w

def ConvTranspose2d(layer, input_shape, *args, **kwargs):
    input_shape = as_shape(input_shape)
    h,w = input_shape[-2:]
    pad, dilation, kernel_size, stride, output_pad = layer.padding, layer.dilation, layer.kernel_size, layer.stride, layer.output_padding
    
    def tuplise(x):
        if not isinstance(x, tuple):
            return (x,x)
        return x

    kernel_size = tuplise(kernel_size)
    pad         = tuplise(pad)
    dilation    = tuplise(dilation)
    stride      = tuplise(stride)

    h = (h - 1) * stride[0] - 2 * pad[0] + dilation[0] * (kernel_size[0] - 1) + output_pad[0] + 1
    w = (w - 1) * stride[1] - 2 * pad[1] + dilation[1] * (kernel_size[1] - 1) + output_pad[1] + 1

    return layer.out_channels, h, w

def Linear(layer, *args, **kwargs):
    return (layer.weight.shape[0],)

def identity(layer, input_shape, *args, **kwargs):
    return input_shape

def Sequential(sequential, input_shape, *args, **kwargs):
    #assert isinstance(sequential, nn.Sequential)
    modules = list(sequential.children())
    shapes = []
    _shape = input_shape
    for module in modules:
        _shape = shape(module, _shape)
        shapes.append(_shape)
        if isinstance(module, nn.Sequential):
            raise NotImplementedError("TODO if nested there are issues (see _shape)")
    return shapes

def AdaptiveAvgPool1d(layer, input_shape, *args, **kwargs):
    raise NotImplementedError()

def AdaptiveAvgPool2d(layer, input_shape, *args, **kwargs):
    output_shape = list(input_shape)
    output_shape[-2:] = layer.output_size
    return tuple(output_shape)

def AdaptiveAvgPool3d(layer, input_shape, *args, **kwargs):
    raise NotImplementedError()

shape_map = {
            nn.Sequential: Sequential,

            nn.AdaptiveAvgPool1d: AdaptiveAvgPool1d,
            nn.AdaptiveAvgPool2d: AdaptiveAvgPool2d,
            nn.AdaptiveAvgPool3d: AdaptiveAvgPool3d,

            nn.Linear: Linear,
            nn.Conv2d: Conv2d,
            nn.MaxPool2d: MaxPool2d, 
            nn.ConvTranspose2d: ConvTranspose2d,

            nn.Identity: identity, 
            nn.Dropout: identity,
            # activation functions
            nn.LeakyReLU: identity,
            nn.ReLU: identity,
            nn.Sigmoid: identity,
            nn.Tanh: identity,

            nn.BatchNorm1d: identity,
            nn.BatchNorm2d: identity,
            nn.BatchNorm3d: identity,
            
            # custom nn.Modules
            
            }

def shape(layer, input_shape, *args, **kwargs):
    if type(layer) in shape_map:
        return shape_map[type(layer)](layer, input_shape, *args, **kwargs)
    elif hasattr(layer, "shape"):
        return layer.shape(input_shape, *args, **kwargs)
    else:
        raise ValueError("Failed to compute shape for layer: {0}".format(layer))