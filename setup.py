#! /usr/bin/env python
# -*- coding: utf-8

"""
Python implementation of Non-Stationary Gabor Transform (NSGT)
derived from MATLAB code by NUHAG, University of Vienna, Austria

Thomas Grill, 2011-2017
http://grrrr.org/nsgt

Austrian Research Institute for Artificial Intelligence (OFAI)
AudioMiner project, supported by Vienna Science and Technology Fund (WWTF)

covered by the Artistic License 2.0
http://www.perlfoundation.org/artistic_license_2_0

--

Installation:

In the console (terminal application) change to the folder containing this readme.txt file.

To build the package run the following command:
python setup.py build

To install the package (with administrator rights):
sudo python setup.py install

--

Attention: some Cython versions also need the Pyrex module installed!

"""

import warnings
import os

try:
    import setuptools
except ImportError:
    warnings.warn("setuptools not found, resorting to distutils: unit test suite can not be run from setup.py")
    setuptools = None

setup_options = {}

if setuptools is None:
    from distutils.core import setup
    from distutils.extension import Extension
else:
    from setuptools import setup
    from setuptools.extension import Extension
    setup_options['test_suite'] = 'tests'
    
   
import numpy


setup(
    name = "nsgt",
    version = "0.18",
    author = "Thomas Grill",
    author_email = "gr@grrrr.org",
    maintainer = "Thomas Grill",
    maintainer_email = "gr@grrrr.org",
    description = "Python implementation of Non-Stationary Gabor Transform (NSGT)",
    license = "Artistic License",
    keywords = "fourier gabor",
    url = "http://grrrr.org/nsgt",
    requires = ["numpy"],
    include_dirs = [numpy.get_include()],
    packages = ['nsgt'],
    cmdclass = {},
    ext_modules = [],
    classifiers = [
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: Artistic License",
        "Programming Language :: Python"
    ],
    **setup_options
)
