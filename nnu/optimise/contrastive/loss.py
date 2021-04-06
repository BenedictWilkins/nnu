#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Created on 19-02-2021 14:10:07

 [Description]
"""
__author__ ="Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ ="Development"

import torch
import torch.nn
import torch.nn.functional as F
import numpy as np

def distance(fun):
    def decorator(x1, x2=None):
        if x2 is None:
            x2 = x1
        dist = fun(x1, x2)
        assert len(dist.shape) == 2
        assert dist.shape[0] == x1.shape[0]
        assert dist.shape[1] == x2.shape[0]
        return dist
    return decorator

@distance
def L22(x1, x2): # Squared L2
    n_dif = x1.unsqueeze(1) - x2.unsqueeze(0)
    return torch.sum(n_dif * n_dif, -1)

class PairedTripletLoss:

    def __init__(self, distance=L22, margin=0.2):
        self.distance = distance
        self.margin = margin

    def __call__(self, x1, x2): 
        """
            Pairs (x1, x2)_i share the same label. All others will be treated as negative examples.
            This makes training very efficient, but at the cost of ensuring that inputs are of the correct form. 
            Best used when many labels are present - or for time series data. 
        """
        d = self.distance(x1, x2)
        xp = torch.diag(d).unsqueeze(1)
        xn = d # careful with the diagonal?

        # is doesnt matter if xp is included in xn as xp_i - xn_i = 0, the original inequality is satisfied, the loss will be 0.
        # it may be more expensive to remove these values than to just compute as below.
        xf = xp.unsqueeze(2) - xn #should only consist of only ||A - P|| - ||A - N|| [batch_size x batch_size x k]
        
        xf = F.relu(xf + self.margin)
        return xf.mean()

class TripletLoss:

    def __init__(self, distance=L22, margin=0.2):
        self.distance = distance
        self.margin = margin

    def __call__(self, x, y):
        unique = np.unique(y) # get all labels
        loss = torch.FloatTensor(np.array([0.])).to(x.device) # tensor for accumulating loss

        for u in unique:
            pi = np.nonzero(y == u)[0]
            ni = np.nonzero(y != u)[0]
            
            xp_ = x_[pi]
            xn_ = x_[ni]
            xp = self.distance(xp_, xp_)
            xn = self.distance(xp_, xn_)

            #3D tensor, (a - p) - (a - n) 
            xf = xp.unsqueeze(2) - xn
            xf = F.relu(xf + self.margin) #triplet loss
            loss += xf.sum()

        return loss
