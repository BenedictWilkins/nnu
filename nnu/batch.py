#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Created on 22-01-2021 13:57:11

 [Description]
"""
__author__ ="Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ ="Development"

import numpy as np
import torch

def __batch_iterate__(data, batch_size):
    _shape = int(data.shape[0]/batch_size) 
    _max = _shape * batch_size
    _data = data[:_max].reshape(_shape, batch_size, *data.shape[1:])
    for x in _data:
        yield x
    if _max < data.shape[0]:
        yield data[_max:]
    
def batch_iterator(*data, batch_size=128, shuffle=False):
    if shuffle:
        data = __shuffle__(*data)
    return zip(*[__batch_iterate__(d, batch_size) for d in data])

def collect(model, *args, batch_size=128):

    it = zip(*[__batch_iterate__(d, batch_size) for d in args])
    with torch.no_grad():
        result = [model(*b).cpu() for b in it]


    return tuple([np.array(d) for d in [i for i in zip(*modes)]])




def __shuffle__(*data): #sad that this cant be done in place...
    m = max(len(d) for d in data)
    indx = np.arange(m)
    np.random.shuffle(indx)
    data = [d[indx] for d in data]
    return data

