# -*- coding: utf-8

"""
Thomas Grill, 2011
http://grrrr.org/nsgt

--
Original matlab code comments follow:

NSGFWIN.M
---------------------------------------------------------------
 [g,rfbas,M]=nsgfwin(fmin,bins,sr,Ls) creates a set of windows whose
 centers correspond to center frequencies to be
 used for the nonstationary Gabor transform with varying Q-factor. 
---------------------------------------------------------------

INPUT : fmin ...... Minimum frequency (in Hz)
        bins ...... Vector consisting of the number of bins per octave
        sr ........ Sampling rate (in Hz)
        Ls ........ Length of signal (in samples)

OUTPUT : g ......... Cell array of window functions.
         rfbas ..... Vector of positions of the center frequencies.
         M ......... Vector of lengths of the window functions.

AUTHOR(s) : Monika Dörfler, Gino Angelo Velasco, Nicki Holighaus, 2010

COPYRIGHT : (c) NUHAG, Dept.Math., University of Vienna, AUSTRIA
http://nuhag.eu/
Permission is granted to modify and re-distribute this
code in any manner as long as this notice is preserved.
All standard disclaimers apply.

EXTERNALS : firwin
"""

import numpy as N
from util import hannwin,blackharr,_isseq
from itertools import chain
from math import floor,ceil

def diff(x):
    return x[1:]-x[:-1]

def nsgfwin_sl(fmin,fmax,bins,sr,Ls,min_win=4,Qvar = 1,sliced=True):

    nf = sr/2.
    
    if fmax > nf:
        fmax = nf
    
    b = N.ceil(N.log2(fmax/fmin))+1

    if not _isseq(bins):
        bins = N.ones(b,dtype=int)*bins
    elif len(bins) < b:
        # TODO: test this branch!
        bins[bins <= 0] = 1
        bins = N.concatenate((bins,N.ones(b-len(bins),dtype=int)*N.min(bins)))
    
    fbas = []
    for kk,bkk in enumerate(bins):
        r = N.arange(kk*bkk,(kk+1)*bkk,dtype=float)
        # TODO: use N.logspace instead
        fbas.append(2**(r/bkk)*fmin)
    fbas = N.concatenate(fbas)

    if fbas[N.min(N.where(fbas>=fmax))] >= nf:
        fbas = fbas[:N.max(N.where(fbas<fmax))+1]
    else:
        # TODO: test this branch!
        fbas = fbas[:N.min(N.where(fbas>=fmax))+1]
    
    lbas = len(fbas)
    fbas = N.concatenate(((0.,),fbas,(nf,),sr-fbas[::-1]))
    fbas *= float(Ls)/sr
    
    # TODO: put together with array indexing
    if sliced:
        M = N.empty(fbas.shape,float)
        M[0] = 2.*fmin*Ls/sr
        M[1] = fbas[1]*(2**(1./bins[0])-2**(-1./bins[0]))
        for k in chain(xrange(2,lbas),(lbas+1,)):
            M[k] = fbas[k+1]-fbas[k-1]
        M[lbas] = fbas[lbas]*(2**(1./bins[-1])-2**(-1./bins[-1]))
        M[lbas+2:2*(lbas+1)] = M[lbas:0:-1]
        M[-1] = M[1]
        M *= Qvar/4.
        M = M.astype(int)
        M *= 4
    else:  # non-sliced
        M = N.empty(fbas.shape,int)
        M[0] = N.round(2.*fmin*Ls/sr)
        for k in xrange(1,2*lbas+1):
            M[k] = N.round(fbas[k+1]-fbas[k-1])
        M[-1] = N.round(Ls-fbas[-2])
    
    N.clip(M,min_win,N.inf,out=M)
    
    if sliced: 
        g = [blackharr(m) for m in M]
    else:
        g = [hannwin(m) for m in M]
    
    if sliced:
        for kk in (1,lbas+2):
            if M[kk-1] > M[kk]:
                g[kk-1] = N.ones(M[kk-1],dtype=g[kk-1].dtype)
                g[kk-1][M[kk-1]//2-M[kk]//2:M[kk-1]//2+ceil(M[kk]/2.)] = hannwin(M[kk])
        
        rfbas = N.round(fbas/2.).astype(int)*2
        shift = N.hstack((N.mod(-rfbas[-1],Ls),diff(rfbas)))
        return g,shift,M
    else:
        fbas[lbas] = (fbas[lbas-1]+fbas[lbas+1])/2
        fbas[lbas+2] = Ls-fbas[lbas]
        rfbas = N.round(fbas).astype(int)
        return g,rfbas,M
