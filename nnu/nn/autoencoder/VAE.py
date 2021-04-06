#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:13:34 2019

@author: ben
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import AE
from ..inverse import inverse
from ..Sequential import Sequential

class VAE(AE.AE):
    
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__(encoder, decoder)

    def reparam(self, mean, logvar):
        return torch.FloatTensor(mean.size()).normal_().to(mean.device) * torch.exp(logvar / 2.) + mean

    def forward(self, *args, **kwargs):
        mu, logvar = self.encoder(*args, **kwargs)
        z = self.reparam(mu, logvar)
        return self.decoder(z), mu, logvar
    
    def encode(self, *args, **kwargs):
        mu, logvar = self.encoder(*args, **kwargs)
        return mu, logvar

    def KL(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1)

def default2D(input_shape, latent_shape, share_weights=True):
    assert len(input_shape) == 3
    assert len(latent_shape) == 1
    
    # TODO



    



