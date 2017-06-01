# -*- coding: utf-8

"""
Python implementation of Non-Stationary Gabor Transform (NSGT)
derived from MATLAB code by NUHAG, University of Vienna, Austria

Thomas Grill, 2011-2015
http://grrrr.org/nsgt

Austrian Research Institute for Artificial Intelligence (OFAI)
AudioMiner project, supported by Vienna Science and Technology Fund (WWTF)
"""

from math import ceil

import numpy as np

from nsgt.utilities.compat import izip
from nsgt.utilities.fft import fftp, ifftp
from nsgt.utilities.utils import chkM

try:
    # try to import cython version
    from nsgt.cython.nsgtf_loop import nsgtf_loop
except ImportError:
    from nsgt.nsgtf_loop import nsgtf_loop

# if False:
#     # what about theano?
#     try:
#         import theano as T
#     except ImportError:
#         T = None

try:
    import multiprocessing as mp
except ImportError:
    mp = None


# @profile
def nsgtf_sl(f_slices, g, wins, nn, M=None, real=False, reducedform=0, measurefft=False, multithreading=False):
    M = chkM(M, g)
    dtype = g[0].dtype

    fft = fftp(measure=measurefft, dtype=dtype)
    ifft = ifftp(measure=measurefft, dtype=dtype)

    if real:
        assert 0 <= reducedform <= 2
        sl = slice(reducedform, len(g) // 2 + 1 - reducedform)
    else:
        sl = slice(0, None)

    maxLg = max(int(ceil(float(len(gii)) / mii)) * mii for mii, gii in izip(M[sl], g[sl]))
    temp0 = None

    if multithreading and mp is not None:
        mmap = mp.Pool().map
    else:
        mmap = map

    loopparams = []
    for mii, gii, win_range in izip(M[sl], g[sl], wins[sl]):
        Lg = len(gii)
        col = int(ceil(float(Lg) / mii))
        assert col * mii >= Lg
        gi1 = gii[:(Lg + 1) // 2]
        gi2 = gii[-(Lg // 2):]
        p = (mii, gii, gi1, gi2, win_range, Lg, col)
        loopparams.append(p)

    # main loop over slices
    for f in f_slices:
        Ls = len(f)

        # some preparation    
        ft = fft(f)

        if temp0 is None:
            # pre-allocate buffer (delayed because of dtype)
            temp0 = np.empty(maxLg, dtype=ft.dtype)

        # A small amount of zero-padding might be needed (e.g. for scale frames)
        if nn > Ls:
            ft = np.concatenate((ft, np.zeros(nn - Ls, dtype=ft.dtype)))

        # The actual transform
        c = nsgtf_loop(loopparams, ft, temp0)

        # TODO: if matrixform, perform "2D" FFT along one axis
        # this could also be nicely parallelized
        y = mmap(ifft, c)

        yield y


# non-sliced version
def nsgtf(f, g, wins, nn, M=None, real=False, reducedform=0,
          measurefft=False, multithreading=False):
    return nsgtf_sl(f_slices=(f,),
                    g=g,
                    wins=wins,
                    nn=nn,
                    M=M,
                    real=real,
                    reducedform=reducedform,
                    measurefft=measurefft,
                    multithreading=multithreading).next()
