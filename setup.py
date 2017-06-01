#! /usr/bin/env python
# -*- coding: utf-8

"""
Python implementation of Non-Stationary Gabor Transform (NSGT)
derived from MATLAB code by NUHAG, University of Vienna, Austria

Thomas Grill, 2011-2015
http://grrrr.org/nsgt

Austrian Research Institute for Artificial Intelligence (OFAI)
AudioMiner project, supported by Vienna Science and Technology Fund (WWTF)

covered by the Artistic License 2.0
http://www.perlfoundation.org/artistic_license_2_0

--

Installation:

In the console (terminal application) change to the folder containing this README.md file.

To build the package run the following command:
python setup.py build

To install the package (with administrator rights):
sudo python setup.py install

--

Attention: some Cython versions also need the Pyrex module installed!

"""

import numpy
import warnings

try:
    from setuptools import setup, find_packages
    from setuptools.extension import Extension
except ImportError:
    warnings.warn("setuptools not found, resorting to distutils: "
                  "unit test suite can not be run from setup.py")
    from distutils.core import setup
    from distutils.extension import Extension

try:
    from Cython.Distutils import build_ext

    cmdclass = {'build_ext': build_ext}
    ext_modules = [
        Extension("nsgt._nsgtf_loop", ["nsgt/nsgtf_loop.pyx"]),
        Extension("nsgt._nsigtf_loop", ["nsgt/nsigtf_loop.pyx"])
    ]
except ImportError:
    cmdclass = {}
    ext_modules = []

setup(
    name="nsgt",
    version="0.1.7",
    author="Thomas Grill",
    author_email="gr@grrrr.org",
    maintainer="Thomas Grill",
    maintainer_email="gr@grrrr.org",
    description="Python implementation of Non-Stationary Gabor Transform (NSGT)",
    license="Artistic License",
    keywords="fourier gabor",
    url="http://grrrr.org/nsgt",
    requires=["numpy", "librosa"],
    include_dirs=[numpy.get_include()],
    packages=find_packages(exclude=("tests", "tests.*", "examples", "examples.*")),
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: Artistic License",
        "Programming Language :: Python"
    ],
    **{'test_suite': 'tests'}  # test options
)
