'''
Created on 23.03.2011

@author: thomas
'''

import numpy as N
from math import ceil,floor
from scipy.fftpack import fftshift
from util import chkM

def nsgtf(f,g,shift,M=None):
    # Check input arguments
    assert len(g) == len(shift)
    
    N = len(shift)    # The number of frequency slices
    M = chkM(M,g)
    
    Ls,col = f.shape()
    
    if min(Ls,col) > 1:
        raise RuntimeError('Right now, this routine supports only single channel signals')
    
    if col != 1 and Ls == 1:
        f = f.T
        Ls = col
    
    # some preparation    
    f = N.fft.fft(f)
    
    timepos = N.cumsum(shift)-shift[0]+1 # Calculate positions from shift vector
    
    # A small amount of zero-padding might be needed (e.g. for scale frames)
    fill = timepos[N]+shift[0]-Ls-1
    f = N.concatenate((f,N.zeros(fill,dtype=f.dtype)))
    
    c = [] # Initialisation of the result
        
    # The actual transform
    for ii in range(N):
        X = len(g[ii])
        sl = N.array((-floor(X/2),ceil(X/2)-1),dtype=int)
        win_range = N.mod(timepos[ii]+sl-1,Ls+fill)+1

        if M[ii] < X: # if the number of time channels is too small, aliasing is introduced
            col = ceil(X/M[ii])
            temp = N.zeros((M[ii],col),dtype=float)
            temp[-floor(X/2):,:ceil(X/2)] = f[win_range]*fftshift(N.conj(g[ii]))
            c.append(N.fft.ifft(N.sum(temp,axis=1)))
        else:
            temp = N.zeros((M[ii],1),dtype=float)
            temp[-floor(X/2):,:ceil(X/2)] = f[win_range]*fftshift(N.conj(g[ii]))
            c.append(N.fft.ifft(temp))
    
#    if max(M) == min(M):
#        c = c.T
    return c,Ls

def nsigtf(c,gd,shift,Ls = None):
    assert len(c) == len(gd) == len(shift)

    timepos = N.cumsum(shift)-shift[0]+1 # Calculate positions from shift vector
    
    N = len(timepos) # The number of time slices
    NN = timepos[-1]+shift[0]-1 # Length of the reconstruction before truncation
    
    fr = N.zeros(NN,dtype=float)  # Initialize output
    
    # The overlap-add procedure including multiplication with the synthesis windows    
    for ii in range(N):
        X = len(gd[ii])
        temp = N.fft.fft(c[ii])*len(c[ii])  
        c[ii] = temp[:ceil(X/2),-floor(X/2):]
        sl = N.array((-floor(X/2),ceil(X/2)-1),dtype=int)
        win_range = N.mod(timepos[ii]+sl-1,NN)+1
        fr[slice(*win_range)] += fftshift(c[ii]*gd[ii])
    
    fr = N.fft.ifft(fr)
    fr = fr[:Ls] # Truncate the signal to original length (if given)
    return fr
