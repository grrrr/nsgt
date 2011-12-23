'''
Created on 05.11.2011

@author: Thomas Grill (grrrr.org)

% Perfect reconstruction sliCQ

% right now, even slice length (sl_len) is required. Parameters are the
% same as NSGTF plus slice length, minimal required window length, 
% Q-factor variation, and test run parameters.
'''

import numpy as N
from itertools import izip,cycle

from slicing import slicing
from unslicing import unslicing
from nsdual import nsdual
from nsgfwin_sl import nsgfwin_sl
from nsgtf import nsgtf_sl
from nsigtf import nsigtf_sl
from util import calcwinrange


def arrange(cseq,M,fwd):
    ixs = (
           [(slice(3*mkk//4,mkk),slice(0,3*mkk//4)) for mkk in M],  # odd
           [(slice(mkk//4,mkk),slice(0,mkk//4)) for mkk in M]  # even
    )
    if fwd:
        ixs = cycle(ixs)
    else:
        ixs = cycle(ixs[::-1])

    return ([N.concatenate((ckk[ix0],ckk[ix1])) for ckk,(ix0,ix1) in izip(ci,ixi)] for ci,ixi in izip(cseq,ixs))


class CQ_NSGT_sliced:
    def __init__(self,fmin,fmax,bins,sl_len,tr_area,fs,min_win=16,Qvar=1,real=False,recwnd=False,matrixform=False,measurefft=False):
        assert fmin > 0
        assert fmax > fmin
        assert bins > 0
        assert sl_len > 0
        assert tr_area >= 0
        assert sl_len > tr_area*2
        assert fs > 0

        assert sl_len%2 == 0

        self.fmin = fmin
        self.fmax = fmax
        self.bins = bins
        self.sl_len = sl_len
        self.tr_area = tr_area
        self.fs = fs
        self.real = real
        self.measurefft = measurefft
        self.userecwnd = recwnd

        self.g,self.rfbas,self.M = nsgfwin_sl(self.fmin,self.fmax,self.bins,self.fs,self.sl_len,min_win,Qvar,matrixform=matrixform)
        
#        print "rfbas",self.rfbas/float(self.sl_len)*self.fs
        
        if matrixform:
            self.M[:] = self.M.max()

        self.wins,self.nn = calcwinrange(self.g,self.rfbas,self.sl_len)
        
        self.gd = nsdual(self.g,self.wins,self.nn,self.M)


    def forward(self,s):
        'transform - s: iterable sequence of sequences' 
        # Compute the slices (zero-padded Tukey window version)
        f_sliced = slicing(s,self.sl_len,self.tr_area)

        cseq =  nsgtf_sl(f_sliced,self.g,self.wins,self.nn,self.M,real=self.real,measurefft=self.measurefft)
    
        return arrange(cseq,self.M,True)


    def backward(self,cseq):
        'inverse transform - c: iterable sequence of coefficients'

        cseq = arrange(cseq,self.M,False)
        
        frec_sliced = nsigtf_sl(cseq,self.gd,self.wins,self.nn,self.sl_len,real=self.real,measurefft=self.measurefft)
        
        # Glue the parts back together
        f_rec = unslicing(frec_sliced,self.sl_len,self.tr_area,dtype=float if self.real else complex,usewindow=self.userecwnd)
        
        # discard first two blocks (padding)
        f_rec.next()
        f_rec.next()
        return f_rec
