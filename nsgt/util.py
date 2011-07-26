# -*- coding: utf-8

"""
Thomas Grill, 2011
http://grrrr.org
"""

import numpy as N

def hannwin(l):
    r = N.arange(l,dtype=float)
    r *= N.pi*2./l
    r = N.cos(r)
    r += 1.
    r *= 0.5
    return r

def _isseq(x):
    try:
        len(x)
    except TypeError:
        return False
    return True        

def chkM(M,g):
    if M is None:
        M = N.array(map(len,g))
    elif not _isseq(M):
        M = N.ones(len(g),dtype=int)*M
    return M

def calcshift(a,Ls):
    return N.concatenate(((N.mod(-a[-1],Ls),), a[1:]-a[:-1]))

# try to use FFT3 if available, else use numpy.fftpack
try:
    import fftw3
except ImportError:
    fft = N.fft.fft
    ifft = N.fft.ifft
else:
    def fft(x):
        x = x.astype(complex)
        r = N.empty_like(x)
        ft = fftw3.Plan(x,r, direction='forward', flags=('estimate',))
        ft()
        return r
    
    def ifft(x):
        r = N.empty_like(x)
        ft = fftw3.Plan(x,r, direction='backward', flags=('estimate',))
        ft()
        r /= len(r)  # normalize
        return r
