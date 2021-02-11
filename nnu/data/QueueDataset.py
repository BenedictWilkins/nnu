#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Created on 25-01-2021 15:59:31

 [Description]
"""
__author__ ="Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ ="Development"

import torch.data.Dataset

class QueueDataset(Dataset):

    def __init__(self, shape, device=None):
        super(QueueDataset, self).__init__()
        self.tensor = torch.empty(shape, device)
        self._indx = 0 # add to the queue

    def push(self, x):
        assert len(x.shape) == len(self.tensor.shape)
        assert x.shape[1:] == self.tensor.shape[1:]

        indx  nindx = self._indx, self._indx + x.shape[0]
        self.tensor[indx:nindx] = x
        

    


    def __len__(self):
        return min(self._indx, self.tensor.shape[0])


    