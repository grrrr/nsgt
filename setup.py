# -*- coding: utf-8

"""
Python implementation of Non-Stationary Gabor Transform (NSGT)
derived from MATLAB code by NUHAG, University of Vienna, Austria

Thomas Grill, 2011-2012
http://grrrr.org/nsgt

Austrian Research Institute for Artificial Intelligence (OFAI)
AudioMiner project, supported by Vienna Science and Technology Fund (WWTF)


covered by Creative Commons Attribution-NonCommercial-ShareAlike license (CC BY-NC-SA)
http://creativecommons.org/licenses/by-nc-sa/3.0/at/deed.en

--

Installation:

In the console (terminal application) change to the folder containing this readme.txt file.

To build the package run the following command:
python setup.py build

To install the package (with administrator rights):
sudo python setup.py install
"""

from setuptools import setup
from distutils.extension import Extension
try:
    from Cython.Distutils import build_ext
except:
    build_ext = None

if build_ext is None:
    cmdclass = {}
    ext_modules = []
else:
    cmdclass = {'build_ext': build_ext}
    ext_modules = [
                   Extension("nsgt._nsgtf_loop", ["nsgt/nsgtf_loop.py"]),
                   Extension("nsgt._nsigtf_loop", ["nsgt/nsigtf_loop.py"])
    ]

setup(
    name = "nsgt",
    version = "0.14",
    author = "Thomas Grill",
    author_email = "gr@grrrr.org",
    maintainer = "Thomas Grill",
    maintainer_email = "gr@grrrr.org",
    description = "Python implementation of Non-Stationary Gabor Transform (NSGT)",
    license = "Creative Commons Attribution-NonCommercial-ShareAlike",
    keywords = "fourier gabor",
    url = "http://grrrr.org/nsgt",
    requires=("numpy",),
    packages=['nsgt'],
    cmdclass = cmdclass,
    ext_modules = ext_modules,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: Creative Commons Attribution-NonCommercial-ShareAlike License (CC BY-NC-SA)",
        "Programming Language :: Python"
    ],
    test_suite="nsgt.__init__",
)
