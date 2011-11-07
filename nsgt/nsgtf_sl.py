'''
Created on 05.11.2011

@author: thomas
'''

import numpy as N
from math import ceil,floor
from itertools import izip
from util import chkM,fft,ifft

def nsgtf_sl((f,fodd),g,shift,M=None,sliced=True):
    # Check input arguments
    assert len(g) == len(shift)
    
#    n = len(shift)    # The number of frequency slices
    M = chkM(M,g)
    
    if len(f.shape) > 1:
        raise RuntimeError('Currently this routine supports only single channel signals')
    
    Ls = len(f)
    
    # some preparation    
    f = fft(f)
    
    timepos = N.cumsum(shift)
    fill = timepos[-1]-Ls
    timepos -= shift[0]-1 # Calculate positions from shift vector
    
    # A small amount of zero-padding might be needed (e.g. for scale frames)
    f = N.concatenate((f,N.zeros(fill,dtype=f.dtype)))
    
    c = [] # Initialization of the result
        
    # The actual transform
    for gii,tpii,mii in izip(g,timepos,M):
        Lg = len(gii)
        pos = N.arange(-floor(Lg/2.),ceil(Lg/2.),dtype=int)
        pos += tpii-1
        win_range = N.mod(pos,Ls+fill)  # TODO: mod in place
        
        if sliced:
            gt = N.hstack((gii[Lg/2:],gii[:Lg/2]))  # TODO: use fftshift
            t = f[win_range]*gt

            if mii < Lg: # if the number of time channels is too small, aliasing is introduced
                # branch NOT tested
                col = int(ceil(float(Lg)/mii))
                temp = N.zeros(col*mii,dtype=complex)
                temp[col*mii-Lg/2:] = t[:-Lg/2]
                temp[:Lg/2] = t[-Lg/2:]
                temp = temp.reshape((mii,col))
                temp = N.sum(temp,axis=1)
            else:
                # TODO: use fftshift for the following
                temp = N.zeros(mii,dtype=complex) 
                temp[-Lg/2:] = t[:-Lg/2]
                temp[:Lg/2] = t[-Lg/2:]
                
            Xt = ifft(temp)
            X = N.empty(Xt.shape,Xt.dtype)
            if fodd:
                X[3*mii/4:] = Xt[:-3*mii/4]
                X[:3*mii/4] = Xt[-3*mii/4:]
            else:
                X[mii/4:] = Xt[:-mii/4]
                X[:mii/4] = Xt[-mii/4:]
                
        else:  # non-sliced
            t = f[win_range]*N.fft.fftshift(N.conj(gii))
            # TODO: the following indexes can be written as two slices, see above
            ixs = N.concatenate((N.arange(mii-floor(Lg/2.),mii,dtype=int),N.arange(0,ceil(Lg/2.),dtype=int)))
    
            if mii < Lg: # if the number of time channels is too small, aliasing is introduced
                # TODO: branch not tested
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
    return c,Ls
