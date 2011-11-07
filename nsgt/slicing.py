'''
Created on 05.11.2011

@author: thomas
'''

import numpy as N
from math import ceil
from util import hannwin
from reblock import reblock
from itertools import chain,izip

def slicing(f,sl_len,tr_area):
    '''OK'''
    assert tr_area%2 == 0
    
#    rows = int(ceil(2.*Ls/sl_len))
    hopsize = sl_len//2
    hhop = hopsize//2

    # build window function within one slice (centered with transition areas around sl_len/4 and 3*sl_len/4    
    w = hannwin(2*tr_area)  # window is shifted
    tw = N.empty(sl_len,dtype=float)
    tw[:hhop-tr_area//2] = 0
    tw[hhop-tr_area//2:hhop+tr_area//2] = w[tr_area:]
    tw[hhop+tr_area//2:3*hhop-tr_area//2] = 1
    tw[3*hhop-tr_area//2:3*hhop+tr_area//2] = w[:tr_area]
    tw[3*hhop+tr_area//2:] = 0
    # four parts of slice with centered window function
    tw = (tw[:hhop],tw[hhop:hhop*2],tw[hhop*2:hhop*3],tw[hhop*3:])
    
    pad = N.zeros(hhop,float)
    # stream of hopsize/2 blocks with leading and trailing zero blocks 
    sseq = chain((pad,),reblock(f,hhop,dtype=float,fulllast=True,padding=0.),(pad,))

#    sseq = list(sseq)
    
    slices = (
            #even: shift +hopsize/2
              (slice(hhop*3,hhop*4),slice(hhop*0,hhop*1),slice(hhop*1,hhop*2),slice(hhop*2,hhop*3)), # even
            # odd: shift -hopsize/2
              (slice(hhop*1,hhop*2),slice(hhop*2,hhop*3),slice(hhop*3,hhop*4),slice(hhop*0,hhop*1)) # odd
    )
    
    past = []
    kk = 0
    for fi in sseq:
        past.append(fi)
        if len(past) == 4:
            f_slice = N.empty(sl_len,dtype=fi.dtype)
            for sli,pi,twi in izip(slices[kk],past,tw):
                f_slice[sli] = pi*twi
#            if kk == 0:
#                for i in xrange(4):
#                    f_slice[hhop*((i+3)%4):hhop*((i+3)%4+1)] = past[i]*tw[i]
#            else:
#                for i in xrange(4):
#                    f_slice[hhop*((i+1)%4):hhop*((i+1)%4+1)] = past[i]*tw[i]
            yield f_slice,kk
            kk = 1-kk  # change sign
            past = past[2:]  # pop the two oldest slices
    
    """
#    f = N.hstack((f,N.zeros(hopsize*rows-Ls)))
    
    tw = hannwin(2*tr_area)
    tw = N.hstack((tw[tr_area:],N.ones(sl_len/2-tr_area,dtype=tw.dtype),tw[:tr_area]))
    
    loc = (
        N.hstack((N.arange(sl_len-tr_area/2,sl_len),N.arange(0,hopsize+tr_area/2))),
        N.hstack((N.arange(hopsize-tr_area/2,sl_len),N.arange(0,tr_area/2))),
        N.arange(-tr_area/2,hopsize+tr_area/2)
    )

    f1 = f[-tr_area/2:]
    f2 = f[:hopsize+tr_area/2]
    f_slice = N.zeros(sl_len,dtype=f.dtype)
    f_slice[loc[0][:len(f1)]] = f1*tw[:len(f1)]  #%temp.*tw;
    f_slice[loc[0][-len(f2):]] = f2*tw[-len(f2):]  #%temp.*tw;
    yield f_slice,0
    
    for kk in xrange(1,rows-1):
        f_slice = N.zeros(sl_len,dtype=f.dtype)
        f_slice[loc[kk%2]] = f[loc[2]+kk*hopsize]*tw   #%temp.*tw;
        yield f_slice,kk%2
        
    f1 = f[-tr_area/2-hopsize:]
    f2 = f[:tr_area/2]        
    f_slice = N.zeros(sl_len,dtype=f.dtype)
    f_slice[loc[(rows-1)%2][:len(f1)]] = f1*tw[:len(f1)]   #%temp.*tw;
    f_slice[loc[(rows-1)%2][-len(f2):]] = f2*tw[-len(f2):]   #%temp.*tw;
    yield f_slice,(rows-1)%2
    """