# -*- coding: utf-8

"""
Python implementation of Non-Stationary Gabor Transform (NSGT)
derived from MATLAB code by NUHAG, University of Vienna, Austria

Thomas Grill, 2011-2015
http://grrrr.org/nsgt

Austrian Research Institute for Artificial Intelligence (OFAI)
AudioMiner project, supported by Vienna Science and Technology Fund (WWTF)
"""

import numpy as np
import torch
from math import ceil

from .util import chkM, get_torch_device
from .fft import fftp, ifftp


#@profile
def nsgtf_sl(f_slices, g, wins, nn, M=None, real=False, reducedform=0, measurefft=False, multithreading=False):
    M = chkM(M,g)
    dtype = g[0].dtype
    
    fft = fftp(measure=measurefft, dtype=dtype)
    ifft = ifftp(measure=measurefft, dtype=dtype)
    
    if real:
        assert 0 <= reducedform <= 2
        sl = slice(reducedform,len(g)//2+1-reducedform)
    else:
        sl = slice(0,None)
    
    maxLg = max(int(ceil(float(len(gii))/mii))*mii for mii,gii in zip(M[sl],g[sl]))
    temp0 = None
    
    mmap = map

    loopparams = []
    for mii,gii,win_range in zip(M[sl],g[sl],wins[sl]):
        Lg = len(gii)
        col = int(ceil(float(Lg)/mii))
        assert col*mii >= Lg
        gi1 = gii[:(Lg+1)//2]
        gi2 = gii[-(Lg//2):]

        p = (mii,gii,gi1,gi2,win_range,Lg,col)
        loopparams.append(p)

    # main loop over slices
    for f in f_slices:
        Ls = len(f)
        
        # some preparation    
        ft = fft(f)

        if temp0 is None:
            # pre-allocate buffer (delayed because of dtype)
            temp0 = torch.empty(maxLg, dtype=ft.dtype, device=get_torch_device())
        
        # A small amount of zero-padding might be needed (e.g. for scale frames)
        if nn > Ls:
            ft = torch.concatenate((ft, torch.zeros(nn-Ls, dtype=ft.dtype)))
        
        # The actual transform
        c = [] # Initialization of the result

        # TODO: torchify it
        for mii,_,gi1,gi2,win_range,Lg,col in loopparams:
            temp = temp0[:col*mii]

            # original version
        #            t = ft[win_range]*N.fft.fftshift(N.conj(gii))
        #            temp[:(Lg+1)//2] = t[Lg//2:]  # if mii is odd, this is of length mii-mii//2
        #            temp[-(Lg//2):] = t[:Lg//2]  # if mii is odd, this is of length mii//2

            # modified version to avoid superfluous memory allocation
            t1 = temp[:(Lg+1)//2]
            t1[:] = gi1  # if mii is odd, this is of length mii-mii//2
            t2 = temp[-(Lg//2):]
            t2[:] = gi2  # if mii is odd, this is of length mii//2

            ftw = ft[win_range]
            t2 *= ftw[:Lg//2]
            t1 *= ftw[Lg//2:]
            
            temp[(Lg+1)//2:-(Lg//2)] = 0  # clear gap (if any)
            
            if col > 1:
                temp = torch.sum(temp.reshape((mii,-1)), axis=1)
            else:
                temp = temp.clone()

            c.append(temp)

        # TODO: if matrixform, perform "2D" FFT along one axis
        # this could also be nicely parallelized
        y = list(mmap(ifft,c))
        
        yield y

        
# non-sliced version
def nsgtf(f, g, wins, nn, M=None, real=False, reducedform=0, measurefft=False, multithreading=False):
    return next(nsgtf_sl((f,), g, wins, nn, M=M, real=real, reducedform=reducedform, measurefft=measurefft, multithreading=multithreading))
