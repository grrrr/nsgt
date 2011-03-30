"""
NSGTF.M - Gino Velasco 24.02.11

[c,Ls] = nsgtf(f,g,shift,M)

This is a modified version of nsgt.m for the case where the resolution 
evolves over frequency.

Given the cell array 'g' of windows and the frequency shift vector 
'shift', this function computes the corresponding frequency side version
of the non-stationary gabor transform of f. 

Input: 
          f           : The signal to be analyzed
          g           : Cell array of Fourier transforms of the analysis 
                        windows
          shift       : Vector of frequency shifts
          M           : Number of time channels (optional)
                        If M is constant, the output is converted to a
                        matrix

Output:
          c           : Transform coefficients (matrix or cell array)
          Ls          : Original signal length (in samples)

The transform produces phase-locked coefficients in the
sense that each window is considered to be centered at
0 and the signal itself is shifted accordingly.

More information can be found at:
http://nuhag.eu/nonstatgab/

Edited by Nicki Holighaus 01.02.11
"""

import numpy as N
from math import ceil,floor
from scipy.fftpack import fftshift
from util import chkM,fft,ifft

def nsgtf(f,g,shift,M=None):
    # Check input arguments
    assert len(g) == len(shift)
    
    n = len(shift)    # The number of frequency slices
    M = chkM(M,g)
    
    if len(f.shape) > 1:
        raise RuntimeError('Right now, this routine supports only single channel signals')
    
    Ls = len(f)
    
    # some preparation    
    f = fft(f)
    
    timepos = N.cumsum(shift)-shift[0]+1 # Calculate positions from shift vector
    
    # A small amount of zero-padding might be needed (e.g. for scale frames)
    fill = timepos[-1]+shift[0]-Ls-1
    f = N.concatenate((f,N.zeros(fill,dtype=f.dtype)))
    
    c = [] # Initialisation of the result
        
    # The actual transform
    for ii in range(n):
        X = len(g[ii])
        pos = N.arange(-floor(X/2.),ceil(X/2.),dtype=int)+timepos[ii]-1
        win_range = N.mod(pos,Ls+fill)
        t = f[win_range]*fftshift(N.conj(g[ii]))
        # TODO: the following indexes can be written as two slices
        ixs = N.concatenate((N.arange(M[ii]-int(floor(X/2.)),M[ii]),N.arange(0,int(ceil(X/2.)))))

        if M[ii] < X: # if the number of time channels is too small, aliasing is introduced
            # TODO: branch not tested
            col = ceil(float(X)/M[ii])
            temp = N.zeros((M[ii],col),dtype=complex)
            temp[ixs] = t
            temp = N.sum(temp,axis=1)
        else:
            temp = N.zeros(M[ii],dtype=complex)
            temp[ixs] = t

        # TODO: can FFT be padded to power of 2?
        c.append(ifft(temp))
    
#    if max(M) == min(M):
#        c = c.T
    return c,Ls
