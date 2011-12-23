# -*- coding: utf-8

"""
Thomas Grill, 2011
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
from itertools import izip
from util import fftp,ifftp,irfftp

def nsigtf_sl(cseq,gd,wins,nn,Ls=None,real=False,reducedform=False,measurefft=False):

    fft = fftp(measure=measurefft)
    ifft = irfftp(measure=measurefft) if real else ifftp(measure=measurefft)
    
    for c in cseq:
        if real:
            if reducedform:
                assert len(c) == len(gd)//2-1
                sl = slice(1,len(gd)//2)
            else:
                assert len(c) == len(gd)//2+1
                sl = slice(0,len(gd)//2+1)
        else:
            assert len(c) == len(gd)
            sl = slice(0,None)
    
        fr = N.zeros(nn,dtype=complex)  # Initialize output

        # The overlap-add procedure including multiplication with the synthesis windows
        for cii,gdii,win_range in izip(c,gd[sl],wins[sl]):
            Lg = len(gdii)
            t = fft(cii)
            if len(t) == Lg:
                temp = t
            else:
                temp = N.empty(Lg,dtype=t.dtype)
                temp[:(Lg+1)//2] = t[:(Lg+1)//2]
                temp[-(Lg//2):] = t[-(Lg//2):]
            temp *= len(cii)
            temp *= gdii
            fr[win_range] += N.fft.fftshift(temp)
        
        if real:
            fr = fr[:nn//2+1]
        fr = ifft(fr)
        fr = fr[:Ls] # Truncate the signal to original length (if given)
        yield fr

# non-sliced version
def nsigtf(c,gd,wins,nn,Ls=None,real=False,reducedform=False,measurefft=False):
    return nsigtf_sl((c,),gd,wins,nn,Ls=Ls,real=real,reducedform=reducedform,measurefft=measurefft).next()
