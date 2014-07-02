# -*- coding: utf-8

"""
Thomas Grill, 2011-2012
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

import numpy as N
from itertools import izip,chain,imap
from fft import fftp,ifftp,irfftp

try:
    # try to import cython version
    from _nsigtf_loop import nsigtf_loop
except ImportError:
    nsigtf_loop = None

if nsigtf_loop is None:
    from nsigtf_loop import nsigtf_loop

if False:
    # what about theano?
    try:
        import theano as T
    except ImportError:
        T = None
    
try:
    import multiprocessing as MP
except ImportError:
    MP = None
    

#@profile
def nsigtf_sl(cseq,gd,wins,nn,Ls=None,real=False,reducedform=0,measurefft=False,multithreading=False):
    cseq = iter(cseq)
    dtype = gd[0].dtype

    fft = fftp(measure=measurefft,dtype=dtype)
    ifft = irfftp(measure=measurefft,dtype=dtype) if real else ifftp(measure=measurefft,dtype=dtype)
    
    if real:
        ln = len(gd)//2+1-reducedform*2
        fftsymm = lambda c: N.hstack((c[0],c[-1:0:-1])).conj()
        if reducedform:
            # no coefficients for f=0 and f=fs/2
            symm = lambda fc: chain(fc,imap(fftsymm,fc[::-1]))
            sl = lambda x: chain(x[reducedform:len(gd)//2+1-reducedform],x[len(gd)//2+reducedform:len(gd)+1-reducedform])
        else:
            symm = lambda fc: chain(fc,imap(fftsymm,fc[-2:0:-1]))
            sl = lambda x: x
    else:
        ln = len(gd)
        symm = lambda fc: fc
        sl = lambda x: x
        
    maxLg = max(len(gdii) for gdii in sl(gd))

    # get first slice
    c0 = cseq.next()

    fr = N.empty(nn,dtype=c0[0].dtype)  # Allocate output
    temp0 = N.empty(maxLg,dtype=fr.dtype)  # pre-allocation
    
    if multithreading and MP is not None:
        mmap = MP.Pool().map
    else:
        mmap = map

    loopparams = []
    for gdii,win_range in izip(sl(gd),sl(wins)):
        Lg = len(gdii)
        temp = temp0[:Lg]
        wr1 = win_range[:(Lg)//2]
        wr2 = win_range[-((Lg+1)//2):]
#        wr1,wr2 = win_range
        sl1 = slice(None,(Lg+1)//2)
        sl2 = slice(-(Lg//2),None)
        p = (gdii,wr1,wr2,sl1,sl2,temp)
        loopparams.append(p)
        
    # main loop over slices
    for c in chain((c0,),cseq):
        assert len(c) == ln

        # do transforms on coefficients
        # TODO: for matrixform we could do a FFT on the whole matrix along one axis
        # this could also be nicely parallalized
        fc = mmap(fft,c)
        fc = symm(fc)
        
        # The overlap-add procedure including multiplication with the synthesis windows
        fr = nsigtf_loop(loopparams,fr,fc)

        ftr = fr[:nn//2+1] if real else fr

        sig = ifft(ftr,outn=nn)

        sig = sig[:Ls] # Truncate the signal to original length (if given)

        yield sig

# non-sliced version
def nsigtf(c,gd,wins,nn,Ls=None,real=False,reducedform=0,measurefft=False,multithreading=False):
    return nsigtf_sl((c,),gd,wins,nn,Ls=Ls,real=real,reducedform=reducedform,measurefft=measurefft,multithreading=multithreading).next()
