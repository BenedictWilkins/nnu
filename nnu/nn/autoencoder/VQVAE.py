#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Created on 22-01-2021 11:50:18

 [Description]
"""
__author__ ="Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ ="Development"

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

# https://github.com/rosinality/vq-vae-2-pytorch

# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch



class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, channel, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input
        return out

class Encoder(nn.Module): # requires input image size 4n + 2 to work properly with the decoder

    def __init__(self, in_channel, channel, n_res_block, n_res_channel):
        super().__init__()

        blocks = [
            nn.Conv2d(in_channel, channel // 2, 4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 2, channel, 4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, padding=1),
        ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)

class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, channel, n_res_block, n_res_channel):
        super().__init__()

        blocks = []
        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))
        blocks.extend(
            [
                nn.ConvTranspose2d(channel, channel, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel, channel // 2, 4, stride=2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel // 2, out_channel, 4, stride=2),
            ]
        )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, z):
        return self.blocks(z)


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
    ):
        super().__init__()

        self.enc = Encoder(in_channel, channel, n_res_block, n_res_channel)
        self.quantize_conv_e = nn.Conv2d(channel, embed_dim, 1)

        self.quantize = Quantize(embed_dim, n_embed)

        self.quantize_conv_d = nn.ConvTranspose2d(embed_dim, channel, 1)

        self.dec = Decoder(embed_dim, in_channel, channel, n_res_block, n_res_channel)

    def forward(self, input):
        z, dz, _ = self.encode(input)
        x = self.decode(z)
        return x, dz

    def encode(self, x):
        z = self.enc(x) # NCHW
        z = self.quantize_conv_e(z).permute(0, 2, 3, 1) 
        # now have H x W latent vectors of dimension embed_dim

        z, dz, iz = self.quantize(z) # quantize latent vectors

        z = z.permute(0, 3, 1, 2)
        dz = dz.unsqueeze(0)

        return z, dz, iz

    def decode(self, z):
        z = self.quantize_conv_d(z)
        return self.dec(z) 

    def embed(self, i):
        return self.quantize.embed_code(i).permute(0,3,1,2)

class Visualise:

    def __init__(self, model, cmap=None):
        self.model = model
        if cmap is None:
            import matplotlib
            cmap = matplotlib.cm.get_cmap("plasma") # from [0-1]
            cmap = [cmap(i / self.model.quantize.n_embed) for i in range(self.model.quantize.n_embed)]
            cmap = np.array(cmap)[:,:-1]

        self.cmap = cmap
    
    def __call__(self, x):
        with torch.no_grad():
            z, _, i = self.model.encode(x)
            y = self.model.decode(z)
            x, i, y = x.cpu().numpy(), i.cpu().numpy(), y.cpu().numpy()
            i = self.cmap[i].transpose((0,3,1,2))

            return x, y, i









