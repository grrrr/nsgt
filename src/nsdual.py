'''
Created on 24.03.2011

@author: thomas
'''

import numpy as N
from scipy.fftpack import fftshift,ifftshift
from util import chkM
from math import floor,ceil

def nsdual(g,shift,M=None):
    # Check input arguments
    assert len(shift) == len(g)
    M = chkM(M,g)

    # Setup the necessary parameters
    timepos = N.cumsum(shift)-shift[0]+1
    NN = timepos[-1]+shift[0]-1
    x = N.zeros((NN,),dtype=float)
    
    # Construct the diagonal of the frame operator matrix explicitly
    win_range = N.empty((len(timepos),2),dtype=int)
    for ii,gi in enumerate(g):
        X = len(gi)
        sl = N.array((-floor(X/2),ceil(X/2)),dtype=int)
        win_range[ii] = N.mod(timepos[ii]+sl-1,NN)+1
        x[slice(*win_range[ii])] += (fftshift(gi)**2)*M[ii]   

    # Using the frame operator and the original window sequence, compute 
    # the dual window sequence
    gd = []
    for ii,gi in enumerate(g):
        gdi = ifftshift(fftshift(gi)/x[slice(*win_range[ii])])
        gd.append(gdi)
    return gd
