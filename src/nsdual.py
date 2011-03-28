'''
Created on 24.03.2011

@author: thomas
'''

import numpy as N
from scipy.fftpack import fftshift,ifftshift
from util import chkM
from math import floor,ceil
from itertools import izip

def nsdual(g,shift,M=None):
    # Check input arguments
    assert len(shift) == len(g)
    M = chkM(M,g)

    # Setup the necessary parameters
    timepos = N.cumsum(shift)-shift[0]+1
    NN = timepos[-1]+shift[0]-1
    x = N.zeros((NN,),dtype=float)
    
    # Construct the diagonal of the frame operator matrix explicitly
    win_range = [] #= N.empty((len(timepos),2),dtype=int)
    for ii,gi in enumerate(g):
        X = len(gi)
        sl = N.arange(-floor(X/2.),ceil(X/2.),dtype=int)
        sl += timepos[ii]-1
        w = N.mod(sl,NN)
        x[w] += (fftshift(gi)**2)*M[ii]
        win_range.append(w)

    # Using the frame operator and the original window sequence, compute 
    # the dual window sequence
    gd = [ifftshift(fftshift(gi)/x[wi]) for gi,wi in izip(g,win_range)]
    return gd
