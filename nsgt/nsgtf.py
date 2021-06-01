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

from .util import chkM
from .fft import fftp, ifftp


#@profile
def nsgtf_sl(f_slices, g, wins, nn, M=None, real=False, reducedform=0, measurefft=False, multithreading=False, device="cuda"):
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
        assert col == 1

        p = (mii,win_range,Lg,col)
        loopparams.append(p)

    ragged_giis = [torch.nn.functional.pad(torch.unsqueeze(gii, dim=0), (0, maxLg-gii.shape[0])) for gii in g[sl]]
    giis = torch.conj(torch.cat(ragged_giis))

    #f = torch.cat([torch.unsqueeze(f, dim=0) for f in f_slices])

    ft = fft(f_slices)

    Ls = f_slices.shape[-1]

    assert nn == Ls

    # The actual transform
    c = torch.empty(*f_slices.shape[:2], len(loopparams), maxLg, dtype=ft.dtype, device=torch.device(device))

    # TODO: torchify it
    for j, (mii,win_range,Lg,col) in enumerate(loopparams):
        t = ft[:, :, win_range]*torch.fft.fftshift(torch.conj(giis[j, :Lg]))

        c[:, :, j, :(Lg+1)//2] = t[:, :, Lg//2:]  # if mii is odd, this is of length mii-mii//2
        c[:, :, j, -(Lg//2):] = t[:, :, :Lg//2]  # if mii is odd, this is of length mii//2
        c[:, :, j, (Lg+1)//2:-(Lg//2)] = 0  # clear gap (if any)

    y = ifft(c)
    
    return y

        
# non-sliced version
def nsgtf(f, g, wins, nn, M=None, real=False, reducedform=0, measurefft=False, multithreading=False, device="cuda"):
    ret = nsgtf_sl(torch.unsqueeze(f, dim=0), g, wins, nn, M=M, real=real, reducedform=reducedform, measurefft=measurefft, multithreading=multithreading, device=device)
    return torch.squeeze(ret, dim=0)
