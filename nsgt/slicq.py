# -*- coding: utf-8

"""
Python implementation of Non-Stationary Gabor Transform (NSGT)
derived from MATLAB code by NUHAG, University of Vienna, Austria

Thomas Grill, 2011-2015
http://grrrr.org/nsgt

Austrian Research Institute for Artificial Intelligence (OFAI)
AudioMiner project, supported by Vienna Science and Technology Fund (WWTF)

--

% Perfect reconstruction sliCQ

% right now, even slice length (sl_len) is required. Parameters are the
% same as NSGTF plus slice length, minimal required window length, 
% Q-factor variation, and test run parameters.
"""

import numpy as np
from itertools import cycle, chain, tee
from math import ceil

from .slicing import slicing
from .unslicing import unslicing
from .nsdual import nsdual
from .nsgfwin_sl import nsgfwin
from .nsgtf import nsgtf_sl
from .nsigtf import nsigtf_sl
from .util import calcwinrange
from .fscale import OctScale

# one of the more expensive functions (32/400)
def arrange(cseq, M, fwd):
    cseq = iter(cseq)
    try:
        c0 = next(cseq)  # grab first stream element
    except StopIteration:
        return iter(())
    cseq = chain((c0,), cseq)  # push it back in
    M = list(map(len, c0[0]))  # read off M from the coefficients
    ixs = (
           [(slice(3*mkk//4, mkk), slice(0, 3*mkk//4)) for mkk in M],  # odd
           [(slice(mkk//4, mkk), slice(0, mkk//4)) for mkk in M]  # even
    )
    if fwd:
        ixs = cycle(ixs)
    else:
        ixs = cycle(ixs[::-1])

    return ([
                [np.concatenate((ckk[ix0],ckk[ix1]))
                   for ckk,(ix0,ix1) in zip(ci, ixi)
                ]
             for ci in cci
             ]
             for cci,ixi in zip(cseq, ixs)
            )


def starzip(iterables):
    def inner(itr, i):
        for t in itr:
            yield t[i]
    iterables = iter(iterables)
    it = next(iterables)  # we need that to determine the length of one element
    iterables = chain((it,), iterables)
    return [inner(itr, i) for i,itr in enumerate(tee(iterables, len(it)))]


def chnmap(gen, seq):
    chns = starzip(seq) # returns a list of generators (one for each channel)
    gens = list(map(gen, chns)) # generators including transformation
    return zip(*gens)  # packing channels to one generator yielding channel tuples


class NSGT_sliced:
    def __init__(self, scale, sl_len, tr_area, fs,
                 min_win=16, Qvar=1,
                 real=False, recwnd=False, matrixform=False, reducedform=0,
                 multichannel=False,
                 measurefft=False,
                 multithreading=False,
                 dtype=float):
        assert fs > 0
        assert sl_len > 0
        assert tr_area >= 0
        assert sl_len > tr_area*2
        assert min_win > 0
        assert 0 <= reducedform <= 2

        assert sl_len%4 == 0
        assert tr_area%2 == 0

        self.sl_len = sl_len
        self.tr_area = tr_area
        self.fs = fs
        self.real = real
        self.measurefft = measurefft
        self.multithreading = multithreading
        self.userecwnd = recwnd
        self.reducedform = reducedform

        self.scale = scale
        self.frqs,self.q = self.scale()

        self.g,self.rfbas,self.M = nsgfwin(self.frqs, self.q, self.fs, self.sl_len, sliced=True, min_win=min_win, Qvar=Qvar, dtype=dtype)
        
#        print "rfbas",self.rfbas/float(self.sl_len)*self.fs
        if real:
            assert 0 <= reducedform <= 2
            sl = slice(reducedform,len(self.g)//2+1-reducedform)
        else:
            sl = slice(0,None)

        # coefficients per slice
        self.ncoefs = max(int(ceil(float(len(gii))/mii))*mii for mii,gii in zip(self.M[sl],self.g[sl]))
        
        if matrixform:
            if self.reducedform:
                rm = self.M[self.reducedform:len(self.M)//2+1-self.reducedform]
                self.M[:] = rm.max()
            else:
                self.M[:] = self.M.max()
                
        if multichannel:
            self.channelize = lambda seq: seq
            self.unchannelize = lambda seq: seq
        else:
            self.channelize = lambda seq: ((it,) for it in seq)
            self.unchannelize = lambda seq: (it[0] for it in seq)

        self.wins,self.nn = calcwinrange(self.g, self.rfbas, self.sl_len)
        
        self.gd = nsdual(self.g, self.wins, self.nn, self.M)
        
        self.fwd = lambda fc: nsgtf_sl(fc, self.g, self.wins, self.nn, self.M, real=self.real, reducedform=self.reducedform, measurefft=self.measurefft, multithreading=self.multithreading)
        self.bwd = lambda cc: nsigtf_sl(cc, self.gd, self.wins, self.nn, self.sl_len ,real=self.real, reducedform=self.reducedform, measurefft=self.measurefft, multithreading=self.multithreading)

    @property
    def coef_factor(self):
        return float(self.ncoefs)/self.sl_len
    
    @property
    def slice_coefs(self):
        return self.ncoefs
    
    def forward(self, sig):
        'transform - s: iterable sequence of sequences' 
        
        sig = self.channelize(sig)

        # Compute the slices (zero-padded Tukey window version)
        f_sliced = slicing(sig, self.sl_len, self.tr_area)
        
        cseq = chnmap(self.fwd, f_sliced)
    
        cseq = arrange(cseq, self.M, True)
        
        cseq = self.unchannelize(cseq)
        
        return cseq


    def backward(self, cseq):
        'inverse transform - c: iterable sequence of coefficients'
                
        cseq = self.channelize(cseq)
        
        cseq = arrange(cseq, self.M, False)

        frec_sliced = chnmap(self.bwd, cseq)
        
        # Glue the parts back together
        ftype = float if self.real else complex
        sig = unslicing(frec_sliced, self.sl_len, self.tr_area, dtype=ftype, usewindow=self.userecwnd)
        
        sig = self.unchannelize(sig)
        
        # discard first two blocks (padding)
        next(sig)
        next(sig)
        return sig


class CQ_NSGT_sliced(NSGT_sliced):
    def __init__(self, fmin, fmax, bins, sl_len, tr_area, fs, min_win=16, Qvar=1, real=False, recwnd=False, matrixform=False, reducedform=0, multichannel=False, measurefft=False, multithreading=False):
        assert fmin > 0
        assert fmax > fmin
        assert bins > 0

        self.fmin = fmin
        self.fmax = fmax
        self.bins = bins  # bins per octave

        scale = OctScale(fmin, fmax, bins)
        NSGT_sliced.__init__(self, scale, sl_len, tr_area, fs, min_win, Qvar, real, recwnd, matrixform=matrixform, reducedform=reducedform, multichannel=multichannel, measurefft=measurefft, multithreading=multithreading)
