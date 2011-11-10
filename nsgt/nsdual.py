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
from itertools import izip
from util import chkM

def nsdual(g,wins,nn,M=None):

    M = chkM(M,g)

    # Construct the diagonal of the frame operator matrix explicitly
    x = N.zeros((nn,),dtype=float)
    for gi,mii,sl in izip(g,M,wins):
        x[sl] += N.square(N.fft.fftshift(gi))*mii

    # Using the frame operator and the original window sequence, compute 
    # the dual window sequence
    gd = [gi/N.fft.ifftshift(x[wi]) for gi,wi in izip(g,wins)]
    return gd
