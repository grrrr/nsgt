'''
Created on 05.11.2011

@author: thomas
'''

import numpy as N
from math import ceil
from itertools import izip
from util import chkM,fft,ifft
from warnings import warn

def nsgtf_sl(cseq,g,shift,Ls,M=None,sliced=True):
    # Check input arguments
    assert len(g) == len(shift)
    
    M = chkM(M,g)

    timepos = N.cumsum(shift)
    NN = timepos[-1]
    timepos -= shift[0] # Calculate positions from shift vector
    
    wins = []
    for gii,tpii in izip(g,timepos):
        Lg = len(gii)
        win_range = N.arange(-(Lg//2)+tpii,Lg-(Lg//2)+tpii,dtype=int)
        win_range %= NN
        wins.append(win_range)

    for f in cseq:
        # some preparation    
        f = fft(f)
        
        # A small amount of zero-padding might be needed (e.g. for scale frames)
        f = N.hstack((f,N.zeros(NN-Ls,dtype=f.dtype)))
        
        c = [] # Initialization of the result
            
        # The actual transform
        for gii,mii,win_range in izip(g,M,wins):
            Lg = len(gii)
            
            if sliced:
                # it's exactly the same as the non-sliced version below
                
                t = f[win_range]*N.fft.fftshift(N.conj(gii))
    
                if mii < Lg: # if the number of time channels is too small, aliasing is introduced
                    warn("Branch not tested")
                    # branch might have difficulties with odd mii

                    col = int(ceil(float(Lg)/mii))
                    temp = N.zeros(col*mii,dtype=complex)
                    temp[col*mii-(Lg//2):] = t[:-(Lg//2)]
                    temp[:Lg//2] = t[-(Lg//2):]
                    temp = temp.reshape((mii,col))
                    temp = N.sum(temp,axis=1)
                else:
                    temp = N.zeros(mii,dtype=complex) 
                    temp[-(Lg//2):] = t[:(Lg//2)]  # if mii is odd, this is of length mii//2
                    temp[:-(Lg//2)] = t[(Lg//2):]  # if mii is odd, this is of length mii-mii//2
                    
                X = ifft(temp)
                    
            else:  # non-sliced
                t = f[win_range]*N.fft.fftshift(N.conj(gii))
                # TODO: the following indexes can be written as two slices, see above
                ixs = N.concatenate((N.arange(mii-(Lg//2),mii,dtype=int),N.arange(0,Lg-(Lg//2),dtype=int)))
        
                if mii < Lg: # if the number of time channels is too small, aliasing is introduced
                    warn("Branch not tested")

                    col = int(ceil(float(Lg)/mii))
                    temp = N.zeros((mii,col),dtype=complex)
                    temp[ixs] = t
                    temp = N.sum(temp,axis=1)
                else:
                    temp = N.zeros(mii,dtype=complex)
                    temp[ixs] = t
    
                # TODO: can FFT be padded to power of 2?
                X = ifft(temp)
                
            c.append(X)
    
    #    if max(M) == min(M):
    #        c = c.T
        yield c
