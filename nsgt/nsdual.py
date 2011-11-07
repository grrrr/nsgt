# -*- coding: utf-8

"""
Thomas Grill, 2011
http://grrrr.org/nsgt

--
Original matlab code comments follow:

NSDUAL.M - Nicki Holighaus 02.02.11

Computes (for the painless case) the dual frame corresponding to a given 
non-stationary Gabor frame specified by the windows 'g' and time shifts
'shift'.

Note, the time shifts corresponding to the dual window sequence is the
same as the original shift sequence and as such already given.

This routine's output can be used to achieve reconstruction of a signal 
from its non-stationary Gabor coefficients using the inverse 
non-stationary Gabor transform 'nsigt'.

More information on Non-stationary Gabor transforms
can be found at:

http://www.univie.ac.at/nonstatgab/

minor edit by Gino Velasco 23.02.11
"""

import numpy as N
from math import floor,ceil
from itertools import izip
from util import chkM

def nsdual(g,shift,M=None):
    # Check input arguments
    assert len(shift) == len(g)
    M = chkM(M,g)

    # Setup the necessary parameters
    timepos = N.cumsum(shift)-shift[0]+1
    NN = timepos[-1]+shift[0]-1
    x = N.zeros((NN,),dtype=float)
    
    # Construct the diagonal of the frame operator matrix explicitly
    win_range = []
    for ii,gi in enumerate(g):
        X = len(gi)
        sl = N.arange(-floor(X/2.),ceil(X/2.),dtype=int)
        sl += timepos[ii]-1
        w = N.mod(sl,NN)
        x[w] += N.square(N.fft.fftshift(gi))*M[ii]
        win_range.append(w)

    # Using the frame operator and the original window sequence, compute 
    # the dual window sequence
    gd = [N.fft.ifftshift(N.fft.fftshift(gi)/x[wi]) for gi,wi in izip(g,win_range)]
    return gd
