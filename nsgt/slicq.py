'''
Created on 05.11.2011

@author: Thomas Grill (grrrr.org)

% Perfect reconstruction sliCQ

% right now, even slice length (sl_len) is required. Parameters are the
% same as NSGTF plus slice length, minimal required window length, 
% Q-factor variation, and test run parameters.
'''

import numpy as N
from itertools import izip,cycle,chain,tee

from slicing import slicing
from unslicing import unslicing
from nsdual import nsdual
from nsgfwin_sl import nsgfwin_sl
from nsgtf import nsgtf_sl
from nsigtf import nsigtf_sl
from util import calcwinrange
from fscale import OctScale

def arrange(cseq,M,fwd):
    ixs = (
           [(slice(3*mkk//4,mkk),slice(0,3*mkk//4)) for mkk in M],  # odd
           [(slice(mkk//4,mkk),slice(0,mkk//4)) for mkk in M]  # even
    )
    if fwd:
        ixs = cycle(ixs)
    else:
        ixs = cycle(ixs[::-1])

    return ([[N.concatenate((ckk[ix0],ckk[ix1])) for ckk,(ix0,ix1) in izip(ci,ixi)] for ci in cci] for cci,ixi in izip(cseq,ixs))

def starzip(iterables):
    def inner(itr, i):
        for t in itr:
            yield t[i]
    iterables = iter(iterables) 
    it = iterables.next()
    iterables = chain((it,),iterables)
    return [inner(itr,i) for i,itr in enumerate(tee(iterables,len(it)))]

def chnmap(gen,seq):
    chns = starzip(seq) # returns a list of generators (one for each channel)
    gens = map(gen,chns) # generators including transformation
    return izip(*gens)  # packing channels to one generator yielding channel tuples

class NSGT_sliced:
    def __init__(self,scale,sl_len,tr_area,fs,min_win=16,Qvar=1,real=False,recwnd=False,matrixform=False,reducedform=False,multichannel=False,measurefft=False):
        assert sl_len > 0
        assert tr_area >= 0
        assert sl_len > tr_area*2
        assert fs > 0

        assert sl_len%2 == 0

        self.sl_len = sl_len
        self.tr_area = tr_area
        self.fs = fs
        self.real = real
        self.measurefft = measurefft
        self.userecwnd = recwnd
        self.reducedform = reducedform

        self.scale = scale
        self.frqs,q = self.scale()

        self.g,self.rfbas,self.M = nsgfwin_sl(self.frqs,q,self.fs,self.sl_len,min_win,Qvar,matrixform=matrixform)
        
#        print "rfbas",self.rfbas/float(self.sl_len)*self.fs
        
        if matrixform:
            if self.reducedform:
                self.M[:] = self.M[1:len(self.M)//2].max()
            else:
                self.M[:] = self.M.max()
                
        if multichannel:
            self.channelize = lambda seq: seq
            self.unchannelize = lambda seq: seq
        else:
            self.channelize = lambda seq: ((it,) for it in seq)
            self.unchannelize = lambda seq: (it[0] for it in seq)

        self.wins,self.nn = calcwinrange(self.g,self.rfbas,self.sl_len)
        
        self.gd = nsdual(self.g,self.wins,self.nn,self.M)
        
        self.fwd = lambda fc: nsgtf_sl(fc,self.g,self.wins,self.nn,self.M,real=self.real,reducedform=self.reducedform,measurefft=self.measurefft)
        self.bwd = lambda cc: nsigtf_sl(cc,self.gd,self.wins,self.nn,self.sl_len,real=self.real,reducedform=self.reducedform,measurefft=self.measurefft)


    def forward(self,sig):
        'transform - s: iterable sequence of sequences' 
        
        sig = self.channelize(sig)

        # Compute the slices (zero-padded Tukey window version)
        f_sliced = slicing(sig,self.sl_len,self.tr_area)

        cseq = chnmap(self.fwd,f_sliced)
    
        cseq = arrange(cseq,self.M,True)
        
        cseq = self.unchannelize(cseq)
        
        return cseq


    def backward(self,cseq):
        'inverse transform - c: iterable sequence of coefficients'
                
        cseq = self.channelize(cseq)
        
        cseq = arrange(cseq,self.M,False)
        
        frec_sliced = chnmap(self.bwd,cseq)
        
        # Glue the parts back together
        ftype = float if self.real else complex
        sig = unslicing(frec_sliced,self.sl_len,self.tr_area,dtype=ftype,usewindow=self.userecwnd)
        
        sig = self.unchannelize(sig)
        
        # discard first two blocks (padding)
        sig.next()
        sig.next()
        return sig


class CQ_NSGT_sliced:
    def __init__(self,fmin,fmax,bins,sl_len,tr_area,fs,min_win=16,Qvar=1,real=False,recwnd=False,matrixform=False,reducedform=False,multichannel=False,measurefft=False):
        assert fmin > 0
        assert fmax > fmin
        assert bins > 0

        scale = OctScale(self.fmin,self.fmax,self.bins)
        NSGT_sliced.__init__(self,scale,sl_len,tr_area,fs,min_win,Qvar,real,recwnd,matrixform,reducedform,multichannel,measurefft)

