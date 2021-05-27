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

try:
    import multiprocessing as MP
except ImportError:
    MP = None
    

@profile
def nsigtf_sl(cseq, gd, wins, nn, Ls=None, real=False, reducedform=0, measurefft=False, multithreading=False, device="cuda"):
    print('nsigtf_sl: {0}'.format(type(cseq)))
    cseq = iter(cseq)
    dtype = gd[0].dtype

    fft = fftp(measure=measurefft, dtype=dtype)
    ifft = irfftp(measure=measurefft, dtype=dtype) if real else ifftp(measure=measurefft, dtype=dtype)
    
    if real:
        ln = len(gd)//2+1-reducedform*2

        if device == "cpu":
            fftsymm = lambda c: torch.tensor(np.hstack((c[0],c.numpy()[-1:0:-1])).conj(), device=torch.device(device))
        else:
            # round trip through cpu because i don't want to deal with the lack of negative indexing in torch
            fftsymm = lambda c: torch.tensor(np.hstack((c[0].cpu().numpy(),c.cpu().numpy()[-1:0:-1])).conj(), device=torch.device(device))

        if reducedform:
            # no coefficients for f=0 and f=fs/2
            def symm(_fc):
                fc = list(_fc)
                return chain(fc, map(fftsymm, fc[::-1]))
            sl = lambda x: chain(x[reducedform:len(gd)//2+1-reducedform],x[len(gd)//2+reducedform:len(gd)+1-reducedform])
        else:
            def symm(_fc):
                fc = list(_fc)
                return chain(fc, map(fftsymm, fc[-2:0:-1]))
            sl = lambda x: x
    else:
        ln = len(gd)
        symm = lambda fc: fc
        sl = lambda x: x
        
    maxLg = max(len(gdii) for gdii in sl(gd))

    # get first slice
    c0 = next(cseq)
    print('c0.shape: {0}'.format(c0.shape))

    fr = torch.empty(nn, dtype=c0[0].dtype, device=torch.device(device))  # Allocate output
    temp0 = torch.empty(maxLg, dtype=fr.dtype, device=torch.device(device))  # pre-allocation
    
    if multithreading and MP is not None:
        mmap = MP.Pool().map
    else:
        mmap = map

    loopparams = []
    for gdii,win_range in zip(sl(gd), sl(wins)):
        Lg = len(gdii)
        temp = temp0[:Lg]
        wr1 = win_range[:(Lg)//2]
        wr2 = win_range[-((Lg+1)//2):]
#        wr1,wr2 = win_range
        sl1 = slice(None, (Lg+1)//2)
        sl2 = slice(-(Lg//2), None)
        p = (gdii,wr1,wr2,sl1,sl2,temp)
        loopparams.append(p)

    print('loopparams: {0}'.format(len(loopparams)))
        
    # main loop over slices
    for c in chain((c0,),cseq):
        #print('len(c): {0}'.format(len(c)))

        assert len(c) == ln

        # do transforms on coefficients
        # TODO: for matrixform we could do a FFT on the whole matrix along one axis
        # this could also be nicely parallalized
        fc = mmap(fft, c)
        fc = symm(fc)
        
        # The overlap-add procedure including multiplication with the synthesis windows
        #fr = nsigtf_loop(loopparams, fr, fc)
        fr[:] = 0.
        # The overlap-add procedure including multiplication with the synthesis windows
        # TODO: stuff loop into theano
        for t,(gdii,wr1,wr2,sl1,sl2,temp) in zip(fc, loopparams):
            t1 = temp[sl1]
            t2 = temp[sl2]
            t1[:] = t[sl1]
            t2[:] = t[sl2]
            temp *= gdii
            temp *= len(t)

            fr[wr1] += t2
            fr[wr2] += t1

        #return fr

        ftr = fr[:nn//2+1] if real else fr

        sig = ifft(ftr, outn=nn)

        sig = sig[:Ls] # Truncate the signal to original length (if given)

        yield sig

# non-sliced version
def nsigtf(c, gd, wins, nn, Ls=None, real=False, reducedform=0, measurefft=False, multithreading=False):
    return next(nsigtf_sl((c,), gd, wins, nn, Ls=Ls, real=real, reducedform=reducedform, measurefft=measurefft, multithreading=multithreading))
