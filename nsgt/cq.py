# -*- coding: utf-8

"""
Python implementation of Non-Stationary Gabor Transform (NSGT)
derived from MATLAB code by NUHAG, University of Vienna, Austria

Thomas Grill, 2011
http://grrrr.org/nsgt

--
Original matlab code copyright follows:

AUTHOR(s) : Monika Dörfler, Gino Angelo Velasco, Nicki Holighaus, 2010-2011

COPYRIGHT : (c) NUHAG, Dept.Math., University of Vienna, AUSTRIA
http://nuhag.eu/
Permission is granted to modify and re-distribute this
code in any manner as long as this notice is preserved.
All standard disclaimers apply.

"""

from nsgfwin import nsgfwin
from nsdual import nsdual
from nsgtf import nsgtf
from nsigtf import nsigtf
from util import calcwinrange

class CQ_NSGT:
    def __init__(self,fmin,fmax,bins,fs,Ls,real=True,measurefft=False,matrixform=False,reducedform=False,multichannel=False):
        assert fmin > 0
        assert fmax > fmin
        assert bins > 0
        assert fs > 0
        assert Ls > 0
        
        self.fmin = fmin
        self.fmax = fmax
        self.bins = bins
        self.fs = fs
        self.Ls = Ls
        self.real = real
        self.measurefft = measurefft
        self.reducedform = reducedform
        
        # calculate transform parameters
        self.g,rfbas,self.M = nsgfwin(self.fmin,self.fmax,self.bins,self.fs,self.Ls)

        if matrixform:
            if self.reducedform:
                self.M[:] = self.M[1:len(self.M)//2].max()
            else:
                self.M[:] = self.M.max()
    
        if multichannel:
            self.channelize = lambda s: s
            self.unchannelize = lambda s: s
        else:
            self.channelize = lambda s: (s,)
            self.unchannelize = lambda s: s[0]

        # calculate shifts
        self.wins,self.nn = calcwinrange(self.g,rfbas,self.Ls)
        # calculate dual windows
        self.gd = nsdual(self.g,self.wins,self.nn,self.M)

    def forward(self,s):
        'transform'
        s = self.channelize(s)
        fwd = lambda s: nsgtf(s,self.g,self.wins,self.nn,self.M,real=self.real,reducedform=self.reducedform,measurefft=self.measurefft)
        c = map(fwd,s)
        return self.unchannelize(c)

    def backward(self,c):
        'inverse transform'
        c = self.channelize(c)
        bwd = lambda c: nsigtf(c,self.gd,self.wins,self.nn,self.Ls,real=self.real,reducedform=self.reducedform,measurefft=self.measurefft)
        s = map(bwd,c)
        return self.unchannelize(s)
