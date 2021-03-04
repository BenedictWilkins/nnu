#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Created on 01-03-2021 11:22:48

 [Description]
"""
__author__ ="Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ ="Development"

from abc import abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from types import SimpleNamespace

from .Sequential import Sequential
from .module import View
from .shape import as_shape

class GumbelSoftmax(nn.Module):
    """
        Quantizer that uses a categorical reparameterisation trick for differentiability.
        https://arxiv.org/pdf/1611.01144.pdf
    """

    def __init__(self):
        raise NotImplemented("TODO")

class DistanceQuantize(nn.Module):
    """
        Quantizer that is used in the VQ-VAE paper. 
    """

    def __init__(self):
        super().__init__()
        raise NotImplemented("TODO")

class Quantize(nn.Module):

    """
        Quantizer that is used in the Dreamer-V2 paper. 
        https://arxiv.org/pdf/2010.02193.pdf
        Works in two modes - learnable or onehot embeddings.
    """
    
    def __init__(self, latent_shape, onehot=False):
        super().__init__()
        self.latent_shape = latent_shape = as_shape(latent_shape)
        if not onehot:
            self.embed = nn.Embedding(latent_shape[0], latent_shape[0]) # learnable embedding
        else:
            self.register_buffer("eye", torch.eye(latent_shape[0], latent_shape[0]))
            def onehot_embed(ind):   
                return self.eye[ind]
            self.embed = onehot_embed
        
    def forward(self, probs):
        # probs should be of shape n x classes
        prob_dist = torch.distributions.Categorical(probs.reshape(-1, self.latent_shape[0])) 
        sample = prob_dist.sample()
        sample = self.embed(sample).view(probs.shape)
        return sample + probs - probs.detach() # straight-through Hackz

class LinearCombine(nn.Module):
    
    def __init__(self, input_shape, output_shape, activation=nn.Identity()):
        super().__init__()
        input_shape = as_shape(input_shape)
        output_shape = as_shape(output_shape)
        self.l1 = nn.Linear(input_shape[0], output_shape[0])
        self.activation = activation
    
    def forward(self, x, y):
        xy = torch.cat((x, y), dim=1)
        return self.activation(self.l1(xy))

class GRUUnit(nn.Module):

    def __init__(self, latent_shape, action_shape):
        super().__init__()
        self.gru = nn.GRUCell(latent_shape[0] + action_shape[0], latent_shape[0])

    def forward(self, z, a, h):
        za = torch.cat((z,a), dim=1)
        return self.gru(za, h)

class DreamerBase(nn.Module):

    def __init__(self, encoder, quantize, recurrent, stoch, *heads, action_shape=None, state_shape=None, latent_shape=None):
        super().__init__()

        self.encoder = encoder
        self.quantize = quantize
        self.recurrent = recurrent
        self.stoch = stoch
        self.heads = nn.ModuleList(heads)

        assert action_shape is not None
        assert state_shape is not None

        self.action_shape = as_shape(action_shape)
        self.state_shape = as_shape(state_shape)
       

    def forward(self, s, a, h=None):
        """         Expects batched sequences of shape (L,N,...) where L is the sequence length and N is the batch size.

        Args:
            s (torch.FloatTensor): batched sequences of states (L,N,C,H,W)
            a (torch.FloatTensor): batched sequences of actions (L,N,A)
            h (torch.FloatTensor): initial hidden vector
        """
        if h is None:
            h = self.initial_hidden(s.shape[1], device=s.device)

        result = SimpleNamespace(prior_logits=[], logits=[], out=[], z=[], h=h)

        for s, a in zip(s, a):
            prior_logits, logits, out, h, z = self.forward_next(s, a, h)
            result.prior_logits.append(prior_logits)
            result.logits.append(logits)
            result.out.append(out)
            result.z.append(z)
            result.h = h # we only need the last hidden vector?

        return result
        
    def forward_next(self, s, a, h): # do one iteration of the recurrence
        batch_size = s.shape[0]

        x = self.encoder(s) # encode state x_t

        logits = self.logits(x, h) # combine x_t with h_t, output should be logits (N,...,D)
        prior_logits = self.logits(torch.zeros_like(x, device=x.device), h) # (N,...,D)

        probs = torch.softmax(logits, dim=-1)
        z = self.quantize(probs) # quantize (sample) from probs

        h = self.recurrent(z, a, h)

        out = [head(h) for head in self.heads] 

        return prior_logits, logits, out, h, z
    
    def logits(self, x, h): # ... this helps with KL issues?
        return F.log_softmax(self.stoch(x,h), dim=-1)
        #return - F.relu(self.stoch(x, h))
        #return self.stoch(x, h)

    @abstractmethod
    def initial_hidden(self, batch_size, *args, **kwargs):
        pass # abstract 

    def imagine(self, a_seq, s=None, h=None):
        result = SimpleNamespace(out=[], p=[], z=[])
        length, batch_size = a_seq.shape[0], a_seq.shape[1] # imagine given an action sequence

        with torch.no_grad():
            if h is None: # if no initial hidden vector was given, use the initial hidden vector
                h = self.initial_hidden(1, device=a_seq.device)

            if s is not None: # if a state is given, use it
                x = self.encoder(s)
                logits = self.logits(x, h) # combine x_t with h_t, output should be logits (N,...,D)
                probs = torch.softmax(logits, dim=-1)
                z = self.quantize(probs) # quantize (sample) from probs
                h = self.recurrent(z, a_seq[0], h)
                out = [head(h) for head in self.heads]
                result.out.append(out)
                result.p.append(probs)
                result.z.append(z)
                a_seq = a_seq[1:] # first action has now been used
            else:
                # the shape is needed TODO think of a better way...
                x = self.encoder(torch.zeros((1, *self.state_shape), device=a_seq.device))

            # used to get prior probs (in the absence of a state)
            zeros = torch.zeros((batch_size, *x.shape[1:]), device=x.device)

            for a in a_seq:
                prior_logits = self.logits(zeros, h) # (N,...,D)
                probs = torch.softmax(prior_logits, dim=-1)
                
                z = self.quantize(probs) # quantize (sample) from probs
                h = self.recurrent(z, a, h)
                out = [head(h) for head in self.heads]

                result.out.append(out)
                result.p.append(probs)
                result.z.append(z)
                
            return result

        

            
class DreamerSimple(DreamerBase):

    def __init__(self, state_shape, action_shape, latent_shape):
        assert len(state_shape) == 3
        assert len(action_shape) == 1
        assert len(latent_shape) == 1

        self.latent_shape = latent_shape = as_shape(latent_shape)
        action_shape = as_shape(action_shape)
        state_shape = as_shape(state_shape)

        encoder = Sequential(nn.Conv2d(state_shape[0], 32, kernel_size = 7, stride = 2), nn.LeakyReLU(inplace=True),
                             nn.Conv2d(32, 64, kernel_size = 5, stride = 1), nn.LeakyReLU(inplace=True),
                             nn.Conv2d(64, 64, kernel_size = 5, stride = 1), nn.LeakyReLU(inplace=True),
                             nn.Conv2d(64, 64, kernel_size = 5, stride = 1), nn.LeakyReLU(inplace=True),
                             nn.Conv2d(64, 128, kernel_size = 3, stride = 1), nn.LeakyReLU(inplace=True),
                             nn.Conv2d(128, 128, kernel_size = 3, stride = 1), nn.LeakyReLU(inplace=True),
                             nn.Conv2d(128, 128, kernel_size = 3, stride = 1), nn.LeakyReLU(inplace=True))
        out_shape = encoder.output_shape(state_shape)
        encoder.layers.append(View(out_shape, -1))
        encoder.layers.append(nn.Linear(np.prod(out_shape), latent_shape[0]))
        decoder = encoder.inverse(state_shape) 

        quantize = Quantize(latent_shape[0], onehot=True)
        gru = GRUUnit(latent_shape, action_shape)
        stoch = LinearCombine(latent_shape[0] * 2, latent_shape[0])
       
        super().__init__(encoder, quantize, gru, stoch, decoder, state_shape=state_shape, action_shape=action_shape)

    def initial_hidden(self, batch_size, device="cpu"):
        z = torch.zeros((batch_size, self.latent_shape[0]), device=device)
        a = torch.zeros((batch_size, self.action_shape[0]), device=device)
        return self.recurrent(z, a, z) # all zeros, z.shape == h.shape

# ======================== OBJECTIVE FUNCTIONS ========================

class KLBalance:
    """
        Computed as $$eta_x * KL(stop_grad(x) || y) + eta_y * KL(x || stop_grad(y))$$
        x, y should be log probabilities of shape (N,...,D)
        KL divergence will be computed over the last dimension (D) and reduced over (N,...).
        
        See original paper: https://arxiv.org/pdf/2010.02193.pdf
    """
    
    def __init__(self, balance = 0.8, free=1.0, reduction="batchmean"):
        super().__init__()
        self.balance = balance
        self.free = free 
        self.reduction = reduction
    
    def __call__(self, x, y):
        
        assert x.shape[-1] == y.shape[-1] # distributions have different shapes?
        x = x.reshape(-1, x.shape[-1])
        y = y.reshape(-1, y.shape[-1])
        
        kl1 = F.kl_div(x.detach(), torch.softmax(y, dim=-1), log_target=False, reduction=self.reduction)
        kl2 = F.kl_div(x, torch.softmax(y.detach(), dim=-1), log_target=False, reduction=self.reduction)

        # this is in the original paper, it seems to stop the logits from exploding
        kl1 = torch.clamp(kl1, max=self.free)
        kl2 = torch.clamp(kl2, max=self.free)

        return self.balance * kl1 + (1 - self.balance) * kl2

