# -*- coding: utf-8

"""
Thomas Grill, 2011-2015
http://grrrr.org/nsgt

--
Original matlab code comments follow:

NSIGTF.N - Gino Velasco 24.02.11

fr = nsigtf(c,gd,shift,Ls)

This is a modified version of nsigt.m for the case where the resolution 
evolves over frequency.

Given the cell array 'c' of non-stationary Gabor coefficients, and a set 
of windows and frequency shifts, this function computes the corresponding 
inverse non-stationary Gabor transform.

Input: 
          c           : Cell array of non-stationary Gabor coefficients
          gd          : Cell array of Fourier transforms of the synthesis 
                        windows
          shift       : Vector of frequency shifts
          Ls          : Length of the analyzed signal

Output:
          fr          : Synthesized signal

If a non-stationary Gabor frame was used to produce the coefficients 
and 'gd' is a corresponding dual frame, this function should give perfect 
reconstruction of the analyzed signal (up to numerical errors).

The inverse transform is computed by simple 
overlap-add. For each entry of the cell array c,
the coefficients of frequencies around a certain 
position in time, the Fourier transform
is taken, giving 'frequency slices' of a signal.
These slices are added onto each other with an overlap
depending on the window lengths and positions, thus
(re-)constructing the frequency side signal. In the
last step, an inverse Fourier transform brings the signal
back to the time side.

More information can be found at:
http://www.univie.ac.at/nonstatgab/

Edited by Nicki Holighaus 01.03.11
"""

import numpy as np
import torch
from itertools import chain
from .fft import fftp, ifftp, irfftp
    

#@profile
def nsigtf_sl(cseq, gd, wins, nn, Ls=None, real=False, reducedform=0, measurefft=False, multithreading=False, device="cuda"):
    dtype = gd[0].dtype

    fft = fftp(measure=measurefft, dtype=dtype)
    ifft = irfftp(measure=measurefft, dtype=dtype) if real else ifftp(measure=measurefft, dtype=dtype)

    if real:
        ln = len(gd)//2+1-reducedform*2
        if reducedform:
            sl = lambda x: chain(x[reducedform:len(gd)//2+1-reducedform],x[len(gd)//2+reducedform:len(gd)+1-reducedform])
        else:
            sl = lambda x: x
    else:
        ln = len(gd)
        sl = lambda x: x
        
    maxLg = max(len(gdii) for gdii in sl(gd))

    ragged_gdiis = [torch.nn.functional.pad(torch.unsqueeze(gdii, dim=0), (0, maxLg-gdii.shape[0])) for gdii in sl(gd)]
    gdiis = torch.conj(torch.cat(ragged_gdiis))

    fr = torch.empty(*cseq.shape[:2], nn, dtype=cseq.dtype, device=torch.device(device))  # Allocate output
    temp0 = torch.empty(*cseq.shape[:2], maxLg, dtype=fr.dtype, device=torch.device(device))  # pre-allocation

    loopparams = []
    for gdii,win_range in zip(sl(gd), sl(wins)):
        Lg = len(gdii)
        wr1 = win_range[:(Lg)//2]
        wr2 = win_range[-((Lg+1)//2):]
        p = (wr1,wr2,Lg)
        loopparams.append(p)

    # do transforms on coefficients
    # TODO: for matrixform we could do a FFT on the whole matrix along one axis
    # this could also be nicely parallalized
    fc = fft(cseq)

    # The overlap-add procedure including multiplication with the synthesis windows
    for i in range(cseq.shape[0]):
        for j in range(cseq.shape[1]):
            fr[i, j, :] = 0.
            for k,(wr1,wr2,Lg) in enumerate(loopparams[:fc.shape[2]]):
                t = fc[i, j, k]

                r = (Lg+1)//2
                l = (Lg//2)

                t1 = temp0[i, j, :r]
                t2 = temp0[i, j, Lg-l:Lg]

                t1[:] = t[:r]
                t2[:] = t[maxLg-l:maxLg]

                temp0[i, j, :Lg] *= gdiis[k, :Lg] 
                temp0[i, j, :Lg] *= len(t)

                fr[i, j, wr1] += t2
                fr[i, j, wr2] += t1

    ftr = fr[:, :, :nn//2+1] if real else fr

    sig = ifft(ftr, outn=nn)

    sig = sig[:, :, :Ls] # Truncate the signal to original length (if given)

    return sig

# non-sliced version
def nsigtf(c, gd, wins, nn, Ls=None, real=False, reducedform=0, measurefft=False, multithreading=False):
    return next(nsigtf_sl((c,), gd, wins, nn, Ls=Ls, real=real, reducedform=reducedform, measurefft=measurefft, multithreading=multithreading))
