#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Created on 13-05-2021 13:14:09

 [Description]
"""
__author__ ="Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ ="Development"

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class OneHotEmbedding(nn.Module):

    def __init__(self, size):
        super().__init__()
        self.register_buffer("eye", torch.eye(size, size))
    
    def forward(self, index):
        return self.eye[index]

        