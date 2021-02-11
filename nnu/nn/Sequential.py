#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 16-06-2020 11:50:11

    Simple sequential network that adds some additional functionality over torch.nn.Sequential.
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F

from .inverse import inverse
from .shape import shape, as_shape

class LayerDict(OrderedDict):

    def __init__(self, module, layers):
        self.module = module
        super(LayerDict, self).__init__(layers)

    def __setitem__(self, k, v):
        super(LayerDict, self).__setitem__(k, v)
        if isinstance(v, nn.Module):
            self.module.add_module(k, v)
        else:
            self.__dict__[k] = v
            
class Sequential(nn.Module):

    def __init__(self, *layers, **layer_dict):
        super(Sequential, self).__init__()

        if len(layers) > 0:
            _layer_dict = {str(i):v for i,v in enumerate(layers)}
            _layer_dict.update(layer_dict)
            layer_dict = _layer_dict

        self.layers = LayerDict(self, layer_dict)

    def forward(self, *args, **kwargs):
        liter = iter(self.layers.values())
        y = next(liter)(*args, **kwargs)
        for layer in liter:
            y = layer(y)
        return y

    def inverse(self, **kwargs):
        ilayers = OrderedDict()
        for k,v in reversed(self.layers.items()):
            if isinstance(v, nn.Module):
                v = inverse(v, **kwargs)[0]
            ilayers[k] = v
        return Sequential(**ilayers).to(self.device)
        
    @property
    def device(self):
        return next(self.parameters()).device

    def shape(self, input_shape):
        input_shape = as_shape(input_shape)
        result = OrderedDict()
        for k,v in self.layers.items():
            if isinstance(v, nn.Module):
                input_shape = result[k] = shape(v, input_shape)
        return result

