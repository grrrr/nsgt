from setuptools import setup

setup(
    name = "nsgt",
    version = "0.1",
    author = "Thomas Grill",
    author_email = "gr@grrrr.org",
    description = ("Python implementation of Nonstationary Gabor transform"),
    license = "GPL",
    keywords = "fourier gabor",
    url = "http://nuhag.eu/nsgt",
    packages=['nsgt'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Programming Language :: Python"
    ],
    requires=("numpy",)
)
