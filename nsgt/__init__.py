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
from util import calcshift
from slicq import CQ_NSGT_sliced

class CQ_NSGT:
    def __init__(self,fmin,fmax,bins,fs,Ls):
        self.fmin = fmin
        self.fmax = fmax
        self.bins = bins
        self.fs = fs
        self.Ls = Ls

        # calculate transform parameters
        self.g,self.a,self.M = nsgfwin(self.fmin,self.fmax,self.bins,self.fs,self.Ls)
        # calculate shifts
        self.shift = calcshift(self.a,self.Ls)
        # calculate dual windows
        self.gd = nsdual(self.g,self.shift,self.M)

    def forward(self,s):
        'transform' 
        return nsgtf(s,self.g,self.shift,self.Ls,self.M)

    def backward(self,c):
        'inverse transform'
        return nsigtf(c,self.gd,self.shift,self.Ls)


import unittest

class Test_CQ_NSGT(unittest.TestCase):

    def test_transform(self,length=100000,fmin=50,fmax=22050,bins=12,fs=44100):
        import numpy as N
        s = N.random.random(length)
        nsgt = CQ_NSGT(fmin,fmax,bins,fs,length)
        
        # forward transform 
        c = nsgt.forward(s)
        # inverse transform 
        s_r = nsgt.backward(c)
        
        norm = lambda x: N.sqrt(N.sum(N.square(N.abs(x))))
        rec_err = norm(s-s_r)/norm(s)

        self.assertAlmostEqual(rec_err,0)

if __name__ == "__main__":
    unittest.main()
