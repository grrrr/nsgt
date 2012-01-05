'''
Created on 05.11.2011

@author: thomas
'''

import numpy as N
from util import hannwin
from reblock import reblock
from itertools import chain,izip,cycle

def makewnd(sl_len,tr_area):
    hhop = sl_len//4
    htr = tr_area//2
    # build window function within one slice (centered with transition areas around sl_len/4 and 3*sl_len/4    
    w = hannwin(2*tr_area)  # window is shifted
    tw = N.empty(sl_len,dtype=float)
    tw[:hhop-htr] = 0
    tw[hhop-htr:hhop+htr] = w[tr_area:]
    tw[hhop+htr:3*hhop-htr] = 1
    tw[3*hhop-htr:3*hhop+htr] = w[:tr_area]
    tw[3*hhop+htr:] = 0
    return tw

def slicing(f,sl_len,tr_area):
    assert tr_area%2 == 0
    
    hhop = sl_len//4  # half hopsize

    tw = makewnd(sl_len,tr_area)
    # four parts of slice with centered window function
    tw = [tw[o:o+hhop] for o in xrange(0,sl_len,hhop)]
    
    # stream of hopsize/2 blocks with leading and trailing zero blocks
    fseq = reblock(f,hhop,dtype=float,fulllast=True,padding=0.,multichannel=True)
    
    # get first block to deduce number of channels
    fseq0 = fseq.next()
    chns = len(fseq0)
    pad = N.zeros((chns,hhop),dtype=fseq0.dtype)
    # assemble a stream of front padding, already retrieved first block, the block stream and some tail padding
    fseq = chain((pad,pad,fseq0),fseq,(pad,pad,pad))

    slices = [[slice(hhop*((i+3-k*2)%4),hhop*((i+3-k*2)%4+1)) for i in xrange(4)] for k in xrange(2)]
    slices = cycle(slices)
    
    past = []
    for fi in fseq:
        past.append(fi)
        if len(past) == 4:
            f_slice = N.empty((chns,sl_len),dtype=fi.dtype)
            sl = slices.next()
            for sli,pi,twi in izip(sl,past,tw):
                f_slice[:,sli] = pi    # signal
                f_slice[:,sli] *= twi  # multiply with part of window function
            yield f_slice
            past = past[2:]  # pop the two oldest slices
