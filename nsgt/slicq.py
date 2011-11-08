'''
Created on 05.11.2011

@author: Thomas Grill (grrrr.org)

% Perfect reconstruction sliCQ

% right now, even slice length (sl_len) is required. Parameters are the
% same as NSGTF plus slice length, minimal required window length, 
% Q-factor variation, and test run parameters.
'''

import numpy as N
from itertools import imap,izip,cycle

from slicing import slicing
from unslicing import unslicing
from nsdual import nsdual
from nsgfwin_sl import nsgfwin_sl
from nsgtf_sl import nsgtf_sl
from nsigtf_sl import nsigtf_sl
from nsgtf import nsgtf
from nsigtf import nsigtf


def arrange(cseq,M,fwd):
    ixs = (
           [(slice(3*mkk//4,mkk),slice(0,3*mkk//4)) for mkk in M],  # odd
           [(slice(mkk//4,mkk),slice(0,mkk//4)) for mkk in M]  # even
    )
    if fwd:
        ixs = cycle(ixs)
    else:
        ixs = cycle(ixs[::-1])

    return ([N.hstack((ckk[ix0],ckk[ix1])) for ckk,(ix0,ix1) in izip(ci,ixi)] for ci,ixi in izip(cseq,ixs))



class CQ_NSGT_sliced:
    def __init__(self,fmin,fmax,bins,sl_len,tr_area,fs,min_win=16,Qvar=1):
        assert sl_len%2 == 0

        self.fmin = fmin
        self.fmax = fmax
        self.bins = bins
        self.sl_len = sl_len
        self.tr_area = tr_area
        self.fs = fs
        self.min_win = min_win
        self.Qvar = Qvar

        # This is just a slightly modified version of nsgfwin
        self.g,self.shift,self.M = nsgfwin_sl(self.fmin,self.fmax,self.bins,self.fs,self.sl_len,self.min_win,self.Qvar)
        self.gd = nsdual(self.g,self.shift,self.M)


    def forward(self,s):
        'transform - s: iterable sequence of sequences' 
        # Compute the slices (zero-padded Tukey window version)
        f_sliced = slicing(s,self.sl_len,self.tr_area)
        # Slightly modified nsgtf (perfect reconstruction version)
        cseq =  nsgtf_sl(f_sliced,self.g,self.shift,self.sl_len,self.M)
    
        return arrange(cseq,self.M,True)


    def backward(self,cseq):
        'inverse transform - c: iterable sequence of coefficients'

        cseq = arrange(cseq,self.M,False)
        
        frec_sliced = nsigtf_sl(cseq,self.gd,self.shift,self.sl_len)
        
        frec_sliced = imap(N.real,frec_sliced)
        
        # Glue the parts back together
        f_rec = unslicing(frec_sliced,self.sl_len)
        
        # discard first two blocks (padding)
        f_rec.next()
        f_rec.next()
        return f_rec
