'''
Created on 30.12.2011

@author: thomas
'''

import numpy as N

def hz2mel(f):
    "\cite{shannon:2003}"
    return N.log10(f/100.+1.)*2595.

def mel2hz(m):
    "\cite{shannon:2003}"
    return (N.power(10.,m/2595.)-1.)*100.


def logfrqs(fmin,fmax,bnds):
    lmin = N.log2(fmin)
    lmax = N.log2(fmax)
    odiv = bnds/(lmax-lmin)
    pow2n = 2**(1./odiv)
    f = N.power(pow2n,N.arange(bnds+1,dtype=float))*fmin
    q = N.ones(len(f),dtype=float)*(N.sqrt(pow2n)/(pow2n-1.)/2.)
    return f,q
    
def octfrqs(fmin,fmax,odiv):
    lmin = N.log2(fmin)
    lmax = N.log2(fmax)
    bnds = int(N.ceil((lmax-lmin)*odiv))
    pow2n = 2**(1./odiv)
    f = N.power(pow2n,N.arange(bnds+1,dtype=float))*fmin
    q = N.ones(len(f),dtype=float)*(N.sqrt(pow2n)/(pow2n-1.)/2.)
    return f,q

def melfrqs(fmin,fmax,bnds):
    mmin = hz2mel(float(fmin))
    mmax = hz2mel(float(fmax))
    mbnd = float(mmax-mmin)/bnds  # mels per band
    mels = (N.arange(bnds,dtype=float)+0.5)*mbnd+mmin
    f = mel2hz(mels)
    odivs = (N.exp(mels/-1127.)-1.)*(-781.177/mbnd)
    pow2n = N.power(2,1./odivs)
    q = N.sqrt(pow2n)/(pow2n-1.)/2.
    return f,q

