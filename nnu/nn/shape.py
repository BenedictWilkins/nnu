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

def conv2d_shape(layer, input_shape, *args, **kwargs):
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

def conv2dtranspose_shape(layer, input_shape, *args, **kwargs):
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

def linear_shape(layer, *args, **kwargs):
    """ Get the output shape of a linear layer.

    Args:
        layer (nn.Linear): linear layer.

    Returns:
        tuple: output shape (D,)
    """
    return (layer.weight.shape[0],)

def identity(layer, input_shape, *args, **kwargs):
    return input_shape

shape_map = {nn.Linear:linear_shape,
             nn.Conv2d:conv2d_shape,
             nn.MaxPool2d:conv2d_shape, # is the same as conv2d
             nn.ConvTranspose2d:conv2dtranspose_shape,
             nn.Identity:identity, 

             # activation functions
             nn.LeakyReLU:identity,
             nn.ReLU:identity,
             nn.Sigmoid:identity,
             nn.Tanh:identity,
             
             
             # custom nn.Modules
             
             }

def shape(layer, *args, **kwargs):
    if type(layer) in shape_map:
        return shape_map[type(layer)](layer, *args, **kwargs)
    elif hasattr(layer, "shape"):
        return layer.shape(*args, **kwargs)
    else:
        raise ValueError("Failed to compute shape for layer: {0}".format(layer))