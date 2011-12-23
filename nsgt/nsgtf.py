'''
Created on 05.11.2011

@author: thomas
'''

import numpy as N
from math import ceil
from itertools import izip
from util import chkM,fftp,ifftp

def nsgtf_sl(f_slices,g,wins,nn,M=None,real=False,reducedform=False,measurefft=False):
    M = chkM(M,g)
    
    fft = fftp(measure=measurefft)
    ifft = ifftp(measure=measurefft)
    
    if real:
        if reducedform:
            sl = slice(1,len(g)//2)
        else:
            sl = slice(0,len(g)//2+1)
    else:
        sl = slice(0,None)
    
    for f in f_slices:
        Ls = len(f)
        
        # some preparation    
        ft = fft(f)
        
        # A small amount of zero-padding might be needed (e.g. for scale frames)
        if nn > Ls:
            ft = N.concatenate((ft,N.zeros(nn-Ls,dtype=ft.dtype)))
        
        c = [] # Initialization of the result
            
        # The actual transform
        for mii,gii,win_range in izip(M[sl],g[sl],wins[sl]):
            Lg = len(gii)
            
            t = ft[win_range]*N.fft.fftshift(N.conj(gii))

            # if the number of time channels is too small (mii < Lg), aliasing is introduced
            # wrap around and sum up in the end (below)
            col = int(ceil(float(Lg)/mii)) # normally col == 1
            
            temp = N.zeros(col*mii,dtype=t.dtype)
            temp[:(Lg+1)//2] = t[Lg//2:]  # if mii is odd, this is of length mii-mii//2
            temp[-(Lg//2):] = t[:Lg//2]  # if mii is odd, this is of length mii//2
            
            if col > 1:
                temp = N.sum(temp.reshape((mii,-1)),axis=1)
                
            ci = ifft(temp)
            c.append(ci)

        yield c
        
# non-sliced version
def nsgtf(f,g,wins,nn,M=None,real=False,reducedform=False,measurefft=False):
    return nsgtf_sl((f,),g,wins,nn,M=M,real=real,reducedform=reducedform,measurefft=measurefft).next()
