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
    def __init__(self,fmin,fmax,bins,fs,Ls,real=True,measurefft=False,matrixform=False,reducedform=False):
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
    
        # calculate shifts
        self.wins,self.nn = calcwinrange(self.g,rfbas,self.Ls)
        # calculate dual windows
        self.gd = nsdual(self.g,self.wins,self.nn,self.M)

    def forward(self,s):
        'transform' 
        return nsgtf(s,self.g,self.wins,self.nn,self.M,real=self.real,reducedform=self.reducedform,measurefft=self.measurefft)

    def backward(self,c):
        'inverse transform'
        return nsigtf(c,self.gd,self.wins,self.nn,self.Ls,real=self.real,reducedform=self.reducedform,measurefft=self.measurefft)
