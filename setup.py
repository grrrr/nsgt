# -*- coding: utf-8

from setuptools import setup

setup(
    name = "nsgt",
    version = "0.02",
    author = "Thomas Grill",
    author_email = "gr@grrrr.org",
    maintainer = "Thomas Grill",
    maintainer_email = "gr@grrrr.org",
    description = "Python implementation of Non Stationary Gabor Transform (NSGT)",
    license = "GPL",
    keywords = "fourier gabor",
    url = "http://grrrr.org/nsgt",
    requires=("numpy",),
    packages=['nsgt'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Programming Language :: Python"
    ],
    test_suite="nsgt.__init__",
)
