# -*- coding: utf-8

"""
Thomas Grill, 2011-2012
http://grrrr.org
"""

import numpy as N
from math import exp,floor,ceil,pi
from itertools import izip

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

def blackharrcw(bandwidth,corr_shift):
    flip = -1 if corr_shift < 0 else 1
    corr_shift *= flip
    
    M = N.ceil(bandwidth/2+corr_shift-1)*2
    win = N.concatenate((N.arange(M//2,M),N.arange(0,M//2)))-corr_shift
    win = (0.35872 - 0.48832*N.cos(win*(2*N.pi/bandwidth))+ 0.14128*N.cos(win*(4*N.pi/bandwidth)) -0.01168*N.cos(win*(6*N.pi/bandwidth)))*(win <= bandwidth)*(win >= 0)

    return win[::flip],M


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

def calcwinrange(g,rfbas,Ls):
    shift = N.concatenate(((N.mod(-rfbas[-1],Ls),), rfbas[1:]-rfbas[:-1]))
    
    timepos = N.cumsum(shift)
    nn = timepos[-1]
    timepos -= shift[0] # Calculate positions from shift vector
    
    wins = []
    for gii,tpii in izip(g,timepos):
        Lg = len(gii)
        win_range = N.arange(-(Lg//2)+tpii,Lg-(Lg//2)+tpii,dtype=int)
        win_range %= nn

#        Lg2 = Lg//2
#        oh = tpii
#        o = oh-Lg2
#        oe = oh+Lg2
#
#        if o < 0:
#            # wraparound is in first half
#            win_range = ((slice(o+nn,nn),slice(0,oh)),(slice(oh,oe),slice(0,0)))
#        elif oe > nn:
#            # wraparound is in second half
#            win_range = ((slice(o,oh),slice(0,0)),(slice(oh,nn),slice(0,oe-nn)))
#        else:
#            # no wraparound
#            win_range = ((slice(o,oh),slice(0,0)),(slice(oh,oe),slice(0,0)))

        wins.append(win_range)
        
    return wins,nn

