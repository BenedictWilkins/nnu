#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Created on 08-03-2021 15:20:10

 [Description]
"""
__author__ ="Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ ="Development"

import numpy as np
import torch
import torch.nn as nn

from ..shape import as_shape

class CategoricalStraightThrough(nn.Module): # Straight through gradients

    def __init__(self, latent_shape, normalise=True, onehot=True, epsilon=0.0):
        """ Categorical distribution that uses the straight through gradients trick to 
            allow gradients to propagate through the sampling process.


        Args:
            latent_shape (tuple, int): shape of latent space, should be 1 dimensional
            normalise (bool, optional): whether to normalise the logits. Defaults to True.
            onehot (bool, optional): whether to use a one-hot or a learnable embedding for the samples. Defaults to True.
            epsilon (float, optional): constant probability value (epsilon/latent_shape) to shape the distribution [0,1]. Defaults to 0.0.

        Returns:
            [torch.tensor]: sample, logits, probs
        """

        super().__init__()
        self.latent_shape = latent_shape = as_shape(latent_shape)
        self.normalise = False
        self.epsilon = min(0, max(epsilon, 1))

        if not onehot:
            self.embed = nn.Embedding(latent_shape[0], latent_shape[0])
        else:
            self.register_buffer("eye", torch.eye(latent_shape[0], latent_shape[0]))
            def onehot_embed(ind):   
                return self.eye[ind]
            self.embed = onehot_embed
    
    def forward(self, logits):
        original_shape = logits.shape
        logits = logits.reshape(-1, self.latent_shape[0])
        
        if self.normalise: 
            logits = logits - logits.logsumexp(dim=-1, keepdim=True) # normalise logits

        probs = torch.softmax(logits, dim=1) 

        # shape the distribution - this can prevent the distribution from collapsing if the proper regularisation is not in place
        qprobs = probs - (probs * self.epsilon) + (self.epsilon / self.latent_shape[-1])

        sample = torch.multinomial(qprobs, 1, replacement=True)
        sample = self.embed(sample).view(probs.shape)
        return (sample + probs - probs.detach()).view(original_shape), logits, probs # straight-through Hackz

class GumbelSoftmax(nn.Module): # TODO document and refactor

    def __init__(self, latent_shape, tau=1.0, hard=False):
        super().__init__()
        self.latent_shape = latent_shape = as_shape(latent_shape)
        self.hard = hard
        self.tau = tau
    
    def forward(self, logits):
        # TODO give the option of not onehot

        original_shape = logits.shape
        
        logits = logits.reshape(-1, self.latent_shape[0])
        logits = logits - logits.logsumexp(dim=-1, keepdim=True) # normalise logits
        
        gumbels = - torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / self.tau  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(1)
        
        if self.hard: # convert to 1 hot, but allow gradients to flow
            index = y_soft.max(1, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(1, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft
        
        # TODO

        return ret.view(original_shape), logits

class DiagonalGuassian(nn.Module):

    """
        Diagonal Guassian sampler, $$\mathcal{N}(\mu, \Sigma)$$ uses the reparameterisation trick for differentiability.
        Covariance matrix $$\Sigma$$ is a diagonal matrix (provided as a vector) $$[\sigma_1, sigma_2, \cdots, \sigma n] I$$
        for a n-dimensional Guassian distribution. $$\Sigma$$ should be provided in logspace.
    """ 

    def __init__(self, latent_shape):
        super().__init__()
        self.latent_shape = latent_shape = as_shape(latent_shape)
        if len(latent_shape) != 1:
            raise ValueError("Invalid latent shape {0}, must be (d,) where d is the dimension the each Guassian distribution.".format(latent_shape))

    def forward(self, mu, logvar):
        """ Generate a sample from a diagonal Guassian distribution. 

        Args:
            mu ([torch.tensor]): mean (n, m*d) where d is the dimensionality of the distribution.
            logvar ([torch.tensor]): log variance (n, m*d) where d is the dimensionality of the distribution.

        Returns:
            [torch.tensor]: n generated samples.
        """

        assert mu.shape == logvar.shape

        original_shape = mu.shape

        # each input mulogvar can be for n guassians, 
        # if mu.shape[1] == latent_shape, then the shape is unchanged below
        mu     = mu.reshape(-1, self.latent_shape[0])
        logvar = logvar.reshape(-1, self.latent_shape[0])

        sample = torch.empty_like(mu).normal_() * torch.exp(logvar / 2.) + mu

        return sample.view(original_shape), mu, logvar

    def kl_divergence(mu1, logvar1, mu2=None, logvar2=None):
        """ Compute the KL Divergence of two diagonal Gaussian distributions KL(P || Q)
            If the second distribution is not provided, the the divergence will be computed as KL(P || N(0,1)).
            Supports batched input (n, d) n is the batch_size and d is the dimensionality of the distributions P and Q.

            $$ D_{KL}(\mathcal{N}(\mu,\Sigma) || (\mathcal{N}(0,1)) = \frac{1}{2} \big[ \mu^T\mu + tr\{\Sigma\} - d - log | \Sigma | \big]$$ 
            $$ D_{KL}(\mathcal{N}(\mu_p,\Sigma_p) || \mathcal{N}(\mu_q,\Sigma_q)) = \frac{1}{2} \big[ log \frac{|\Sigma_q|}{|\Sigma_p|} + tr\{\Sigma_q^{-1}\Sigma_p \} + (\mu_p - \mu_q)^T\Sigma_q^{-1}(\mu_p - \mu_q) - d\big]$$ 

        Args:
            mu1 ([torch.tensor]): mean of P.
            logvar1 ([torch.tensor]): log variance of P.
            mu2 ([torch.tensor], optional): mean of Q. Defaults to None.
            logvar2 ([torch.tensor], optional): log variance of Q. Defaults to None.
        """

        if mu2 is None or logvar2 is None: # kl_divergence [ N(mu1, logvar1.exp()) || N(0,1)]
            assert mu2 is None and logvar2 is None # ?? 
            return ((mu1 ** 2 + logvar1.exp() - logvar1).sum(dim=-1) - mu1.shape[-1]) / 2
        
        ivar2 = 1. / logvar2.exp()
        dmu = (mu1 - mu2)
        return ((logvar1 - logvar2 + dmu * ivar2 * dmu + ivar2 * logvar1.exp()).sum(dim=-1) - mu1.shape[-1]) / 2
        