# -*- coding: utf-8

"""
Thomas Grill, 2011-2012
http://grrrr.org/nsgt
"""

import numpy as N

class Scale:
    dbnd = 1.e-8
    def __init__(self,bnds):
        self.bnds = bnds
    def __len__(self):
        return self.bnds
    def Q(self,bnd=None):
        # numerical differentiation (if self.Q not defined by sub-class)
        if bnd is None:
            bnd = N.arange(self.bnds)
        return self.F(bnd)*self.dbnd/(self.F(bnd+self.dbnd)-self.F(bnd-self.dbnd))
    def __call__(self):
        f = N.array([self.F(b) for b in xrange(self.bnds)],dtype=float)
        q = N.array([self.Q(b) for b in xrange(self.bnds)],dtype=float)
        return f,q

class OctScale(Scale):
    def __init__(self,fmin,fmax,bpo):
        bnds = int(N.ceil(N.log2(float(fmax)/fmin)*bpo))+1
        Scale.__init__(self,bnds)
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.bpo = int(bpo)
        self.pow2n = 2**(1./self.bpo)
        self.q = N.sqrt(self.pow2n)/(self.pow2n-1.)/2.
    def F(self,bnd=None):
        return self.fmin*self.pow2n**(bnd if bnd is not None else N.arange(self.bnds))
    def Q(self,bnd=None):
        return self.q

class LogScale(Scale):
    def __init__(self,fmin,fmax,bnds):
        Scale.__init__(self,bnds)
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        odiv = N.log2(self.fmax/self.fmin)/(self.bnds-1)
        self.pow2n = 2**odiv
        self.q = N.sqrt(self.pow2n)/(self.pow2n-1.)/2.
    def F(self,bnd=None):
        return self.fmin*self.pow2n**(bnd if bnd is not None else N.arange(self.bnds))
    def Q(self,bnd=None):
        return self.q
    
class LinScale(Scale):
    def __init__(self,fmin,fmax,bnds):
        Scale.__init__(self,bnds)
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.df = (self.fmax-self.fmin)/(self.bnds-1)
    def F(self,bnd=None):
        return (bnd if bnd is not None else N.arange(self.bnds))*self.df+self.fmin
    def Q(self,bnd=None):
        return self.F(bnd)/(self.df*2)

def hz2mel(f):
    "\cite{shannon:2003}"
    return N.log10(f/700.+1.)*2595.

def mel2hz(m):
    "\cite{shannon:2003}"
    return (N.power(10.,m/2595.)-1.)*700.

class MelScale(Scale):
    def __init__(self,fmin,fmax,bnds):
        Scale.__init__(self,bnds)
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.mmin = hz2mel(self.fmin)
        self.mmax = hz2mel(self.fmax)
        self.mbnd = (self.mmax-self.mmin)/(self.bnds-1)  # mels per band
    def F(self,bnd=None):
        if bnd is None:
            bnd = N.arange(self.bnds)
        return mel2hz(bnd*self.mbnd+self.mmin)
    def Q1(self,bnd=None): # obviously not exact
        if bnd is None:
            bnd = N.arange(self.bnds)
        mel = bnd*self.mbnd+self.mmin
        odivs = (N.exp(mel/-1127.)-1.)*(-781.177/self.mbnd)
        pow2n = N.power(2,1./odivs)
        return N.sqrt(pow2n)/(pow2n-1.)/2.
        
        
if __name__ == '__main__':
    scl = LinScale(50,10000,50)
    f,q = scl()
    print f
    print q
    print [scl.Q1(b) for b in xrange(len(scl))]

    
    