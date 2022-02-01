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
    version = "0.0",
    author = "Thomas Grill, Sevag Hanssian",
    author_email = "gr@grrrr.org, sevagh@protonmail.com",
    maintainer = "Thomas Grill, Sevag Hanssian",
    maintainer_email = "gr@grrrr.org, sevagh@protonmail.com",
    description = "Python and PyTorch implementations of the Nonstationary Gabor Transform (NSGT)",
    license = "Artistic License",
    keywords = "fourier gabor cqt",
    url = "https://github.com/sevagh/nsgt",
    packages = ['nsgt', 'nsgt_torch'],
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
