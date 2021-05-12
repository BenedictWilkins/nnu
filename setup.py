#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 13:12:18 2019

@author: ben
"""

from setuptools import setup, find_packages

setup(name='nnu',
      version='0.0.1',
      description='',
      url='',
      author='Benedict Wilkins',
      author_email='benrjw@gmail.com',
      packages=find_packages(),
      install_requires=["numpy",
                        "torch==1.7.1"],
      dependency_links = ["https://download.pytorch.org/whl/torch_stable.html"],
      zip_safe=False)
