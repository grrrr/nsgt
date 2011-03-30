"""
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
http://nuhag.eu/nonstatgab/

Edited by Nicki Holighaus 01.03.11
"""

import numpy as N
from math import ceil,floor
from scipy.fftpack import fftshift
from util import fft,ifft

def nsigtf(c,gd,shift,Ls = None):
    assert len(c) == len(gd) == len(shift)

    timepos = N.cumsum(shift)-shift[0]+1 # Calculate positions from shift vector
    
    n = len(timepos) # The number of time slices
    NN = timepos[-1]+shift[0]-1 # Length of the reconstruction before truncation
    
    fr = N.zeros(NN,dtype=complex)  # Initialize output
    
    # The overlap-add procedure including multiplication with the synthesis windows    
    for ii in range(n):
        X = len(gd[ii])
        # TODO: the following indexes can be written as two slices
        ixs = N.concatenate((N.arange(0,int(ceil(X/2.))),N.arange(X-int(floor(X/2.)),X)))

        temp = fft(c[ii])*len(c[ii])
        cii = temp[ixs]
        pos = N.arange(-floor(X/2.),ceil(X/2.),dtype=int)+timepos[ii]-1
        win_range = N.mod(pos,NN)
        fr[win_range] += fftshift(cii*gd[ii])

    # TODO: this could probably be a rifft, if real signals (as outcome) are assumed
    fr = ifft(fr)
    fr = fr[:Ls] # Truncate the signal to original length (if given)
    return fr
