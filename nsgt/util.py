# -*- coding: utf-8

"""
Thomas Grill, 2011
http://grrrr.org
"""

import numpy as N
from math import exp,floor,ceil,pi

def hannwin(l):
    r = N.arange(l,dtype=float)
    r *= N.pi*2./l
    r = N.cos(r)
    r += 1.
    r *= 0.5
    return r

def blackharr(n,l=None,mod=True):
    if l is None: 
        l = n
    nn = (n//2)*2
    k = N.arange(n)
    if not mod:
        bh = 0.35875 - 0.48829*N.cos(k*(2*pi/nn)) + 0.14128*N.cos(k*(4*pi/nn)) -0.01168*N.cos(k*(6*pi/nn))
    else:
        bh = 0.35872 - 0.48832*N.cos(k*(2*pi/nn)) + 0.14128*N.cos(k*(4*pi/nn)) -0.01168*N.cos(k*(6*pi/nn))
    bh = N.hstack((bh,N.zeros(l-n,dtype=bh.dtype)))
    bh = N.hstack((bh[-n//2:],bh[:-n//2]))
    return bh

def cont_tukey_win(n,sl_len,tr_area):
    g = N.arange(n)*(sl_len/float(n))
    g[N.logical_or(g < sl_len/4.-tr_area/2.,g > 3*sl_len/4.+tr_area/2.)] = 0.
    g[N.logical_and(g > sl_len/4.+tr_area/2.,g < 3*sl_len/4.-tr_area/2.)] = 1.
    #
    idxs = N.logical_and(g >= sl_len/4.-tr_area/2.,g <= sl_len/4.+tr_area/2.)
    temp = g[idxs]
    temp -= sl_len/4.+tr_area/2.
    temp *= pi/tr_area
    g[idxs] = N.cos(temp)*0.5+0.5
    #
    idxs = N.logical_and(g >= 3*sl_len/4.-tr_area/2.,g <= 3*sl_len/4.+tr_area/2.)
    temp = g[idxs]
    temp += -3*sl_len/4.+tr_area/2.
    temp *= pi/tr_area
    g[idxs] = N.cos(temp)*0.5+0.5
    #
    return g

def tgauss(ess_ln,ln=0):
    if ln < ess_ln: 
        ln = ess_ln
    #
    g = N.zeros(ln,dtype=float)
    sl1 = int(floor(ess_ln/2))
    sl2 = int(ceil(ess_ln/2))+1
    r = N.arange(-sl1,sl2) # (-floor(ess_len/2):ceil(ess_len/2)-1)
    r = N.exp((r*(3.8/ess_ln))**2*-pi)
    r -= exp(-pi*1.9**2)
    #
    g[-sl1:] = r[:sl1]
    g[:sl2] = r[-sl2:]
    return g

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
