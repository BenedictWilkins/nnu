#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 08-03-2021 17:52:15
    Some useful objective functions

"""
__author__ ="Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ ="Development"

import torch
import torch.nn as nn
import torch.nn.functional as F

class KLBalance:
    """
        Computed as $$eta_x * KL(stop_grad(x) || y) + eta_y * KL(x || stop_grad(y))$$
        x, y should be log probabilities of shape (N,...,D)
        KL divergence will be computed over the last dimension (D) and reduced over (N,...).
        
        See original paper: https://arxiv.org/pdf/2010.02193.pdf
    """
    
    def __init__(self, balance = 0.8, free = 1.0, reduction="batchmean"):
        super().__init__()
        self.balance = balance
        self.free = free 
        self.reduction = reduction
    
    def __call__(self, x, y):
        
        assert x.shape[-1] == y.shape[-1] # distributions have different shapes?
        x = x.reshape(-1, x.shape[-1])
        y = y.reshape(-1, y.shape[-1])
        
        kl1 = F.kl_div(x.detach(), y, log_target=True, reduction=self.reduction)
        kl2 = F.kl_div(x, y.detach(), log_target=True, reduction=self.reduction)

        # this allows the distributions to be a certain distance (free) apart
        # otherwise they will immediately converge to deterministic (i.e. KL(x || y) == 0)
        # and nothing can be learned from that point onward. Even though it is not mentioned anywhere
        # in the paper, this is ABSOLUTELY ESSENTIAL.
        kl1 = torch.clamp(kl1, min=self.free) # cmax
        kl2 = torch.clamp(kl2, min=self.free) # cmax

        return self.balance * kl1 + (1 - self.balance) * kl2

class Wasserstein1:

    def __init__(self, free=1.0, reduction="mean"):
        self.reduction = dict(mean=torch.mean, sum=torch.sum)[reduction]
        self.free = free

    def __call__(self, x, y):
        # assuming x and y are vectors of probabilities of categorical distributions (N,D)
        assert len(x.shape) == 2 # (N, D)
        assert x.shape == y.shape

        d = torch.abs(torch.cumsum(x, dim=1) - torch.cumsum(y, dim=1)).sum(1)
        d = torch.clamp(kl1, min=self.free) # cmax

        return self.reduction(d)




