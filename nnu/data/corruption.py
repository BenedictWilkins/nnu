#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Created on 25-01-2021 11:08:14

 torch.nn.Modules that introduce noise. Useful for constructing a Denoising auto-encoder. Noise can be turned off by calling model.eval().

"""
__author__ ="Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ ="Development"


import torch
import torch.nn as nn

class Gaussian:

    """
        Additive isotropic gaussian noise
    """

    def __init__(self, mu=0., std=1., ):
        super(Gaussian, self).__init__()
        self.mu, self.std = mu, std

    def __call__(self, x):
        n = torch.empty(x.shape, device=x.device).data.normal_(self.mu, std=self.std)
        return x + n

class SaltPepper:

    def __init__(self, p=0.2, r=0.5):
        super(SaltPepper, self).__init__()
        self.p = p # amount of salt and pepper noise
        self.r = r # salt to pepper noise ratio default 0.5
        assert r == 0.5 # TODO otherwise

    def __call__(self, x):
        x = x.clone()
        n = torch.empty(x.shape).uniform_()
        x[n >=  1 - self.p/2] = 1.
        x[n <= self.p/2] = 0.
        return x
