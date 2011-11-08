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
from util import fft,ifft

# it's exactly the same as nsigtf
def nsigtf_sl(cseq,gd,shift,Ls = None):
    assert len(gd) == len(shift)

    timepos = N.cumsum(shift)
    nn = timepos[-1] # Length of the reconstruction before truncation
    timepos -= shift[0]-1 # Calculate positions from shift vector

    for c in cseq:
        assert len(c) == len(gd)
    
        fr = N.zeros(nn,dtype=complex)  # Initialize output
            
        # The overlap-add procedure including multiplication with the synthesis windows
        for gdii,tpii,cii in izip(gd,timepos,c):
            X = len(gdii)
    
            temp = fft(cii)
            temp *= len(cii)
            temp *= gdii
            
            win_range = N.arange(-X//2,X-X//2,dtype=int)
            win_range += tpii-1
            win_range %= nn
            fr[win_range] += N.fft.fftshift(temp)
        
        # TODO: this could probably be a rifft, if real signals (as outcome) are assumed
        fr = ifft(fr)
        fr = fr[:Ls] # Truncate the signal to original length (if given)
        yield fr
