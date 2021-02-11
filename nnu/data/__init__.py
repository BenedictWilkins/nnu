#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Created on 25-01-2021 15:59:17

 [Description]
"""
__author__ ="Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ ="Development"

from . import corruption

__all__ = ("corruption",)


def batch_apply(fun, x):
    def chunks(x, n):
        for i in range(0, len(x), n):
            yield fun(x[i:i + n])
    return [c for c in chunks()]
    