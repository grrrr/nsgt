# -*- coding: utf-8

"""
Thomas Grill, 2011-2012
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
from util import hannwin,blackharr,blackharrcw
from math import ceil
from warnings import warn
from itertools import chain,izip

def nsgfwin(f,q,sr,Ls,sliced=True,min_win=4,Qvar=1,dowarn=True,dtype=N.float64):
    nf = sr/2.

    lim = N.argmax(f > 0)
    if lim != 0:
        # f partly <= 0 
        f = f[lim:]
        q = q[lim:]
            
    lim = N.argmax(f >= nf)
    if lim != 0:
        # f partly >= nf 
        f = f[:lim]
        q = q[:lim]
    
    assert len(f) == len(q)
    assert N.all((f[1:]-f[:-1]) > 0)  # frequencies must be increasing
    assert N.all(q > 0)  # all q must be > 0
    
    qneeded = f*(Ls/(8.*sr))
    if N.any(q >= qneeded) and dowarn:
        warn("Q-factor too high for frequencies %s"%",".join("%.2f"%fi for fi in f[q >= qneeded]))
    
    fbas = f
    lbas = len(fbas)
    
    frqs = N.concatenate(((0.,),fbas,(nf,)))
    
    fbas = N.concatenate((frqs,sr-frqs[-2:0:-1]))

    # at this point: fbas.... frequencies in Hz
    
    fbas *= float(Ls)/sr
    
#    print "fbas",fbas
    
    # Omega[k] in the paper
    if sliced:
        M = N.zeros(fbas.shape,float)
        M[0] = 2*fbas[1]
        M[1] = fbas[1]/q[0] #(2**(1./bins[0])-2**(-1./bins[0]))
        for k in chain(xrange(2,lbas),(lbas+1,)):
            M[k] = fbas[k+1]-fbas[k-1]
        M[lbas] = fbas[lbas]/q[lbas-1] #(2**(1./bins[-1])-2**(-1./bins[-1]))
#        M[lbas+1] = fbas[lbas]/q[lbas-1] #(2**(1./bins[-1])-2**(-1./bins[-1]))
        M[lbas+2:2*(lbas+1)] = M[lbas:0:-1]
#        M[-1] = M[1]
        M *= Qvar/4.
        M = N.round(M).astype(int)
        M *= 4        
    else:
        M = N.zeros(fbas.shape,int)
        M[0] = N.round(2*fbas[1])
        for k in xrange(1,2*lbas+1):
            M[k] = N.round(fbas[k+1]-fbas[k-1])
        M[-1] = N.round(Ls-fbas[-2])
        
    N.clip(M,min_win,N.inf,out=M)

#    print "M",list(M)
    
    if sliced: 
        g = [blackharr(m).astype(dtype) for m in M]
    else:
        g = [hannwin(m).astype(dtype) for m in M]
    
    if sliced:
        for kk in (1,lbas+2):
            if M[kk-1] > M[kk]:
                g[kk-1] = N.ones(M[kk-1],dtype=g[kk-1].dtype)
                g[kk-1][M[kk-1]//2-M[kk]//2:M[kk-1]//2+ceil(M[kk]/2.)] = hannwin(M[kk])
        
        rfbas = N.round(fbas/2.).astype(int)*2
    else:
        fbas[lbas] = (fbas[lbas-1]+fbas[lbas+1])/2
        fbas[lbas+2] = Ls-fbas[lbas]
        rfbas = N.round(fbas).astype(int)
        
#    print "rfbas",rfbas
#    print "g",g

    return g,rfbas,M





def nsgfwin_new(f,q,sr,Ls,sliced=True,min_win=4,Qvar=1,dowarn=True):
    nf = sr/2.

    lim = N.argmax(f > 0)
    if lim != 0:
        # f partly <= 0 
        f = f[lim:]
        q = q[lim:]
            
    lim = N.argmax(f >= nf)
    if lim != 0:
        # f partly >= nf 
        f = f[:lim]
        q = q[:lim]
    
    assert len(f) == len(q)
    assert N.all((f[1:]-f[:-1]) > 0)  # frequencies must be increasing
    assert N.all(q > 0)  # all q must be > 0
    
    qneeded = f*(Ls/(8.*sr))
    if N.any(q >= qneeded) and dowarn:
        warn("Q-factor too high for frequencies %s"%",".join("%.2f"%fi for fi in f[q >= qneeded]))
    
    fbas = f
    lbas = len(fbas)
    
    frqs = N.concatenate(((0.,),fbas,(nf,)))
    
    fbas = N.concatenate((frqs,sr-frqs[-2:0:-1]))

    # at this point: fbas.... frequencies in Hz
    
    fbas *= float(Ls)/sr
    
#    print "fbas",fbas
    
    # Omega[k] in the paper
    if sliced:
        M = N.zeros(fbas.shape,float)
        M[0] = 2*fbas[1]
        M[1] = fbas[1]/q[0] #(2**(1./bins[0])-2**(-1./bins[0]))
        for k in chain(xrange(2,lbas),(lbas+1,)):
            M[k] = fbas[k+1]-fbas[k-1]
        M[lbas] = fbas[lbas]/q[lbas-1] #(2**(1./bins[-1])-2**(-1./bins[-1]))
#        M[lbas+1] = fbas[lbas]/q[lbas-1] #(2**(1./bins[-1])-2**(-1./bins[-1]))
        M[lbas+2:2*(lbas+1)] = M[lbas:0:-1]
#        M[-1] = M[1]

###        M *= Qvar/4.
###        M = N.round(M).astype(int)
###        M *= 4        
    else:
        M = N.zeros(fbas.shape,int)
        M[0] = N.round(2*fbas[1])
        for k in xrange(1,2*lbas+1):
            M[k] = N.round(fbas[k+1]-fbas[k-1])
        M[-1] = N.round(Ls-fbas[-2])
        
    N.clip(M,min_win,N.inf,out=M)

    if sliced: 
###        g = [blackharr(m) for m in M]

        rfbas = N.concatenate((N.floor(fbas[:lbas+2]),N.ceil(fbas[lbas+2:])))
        corr_shift = fbas-rfbas

#        shift = N.concatenate(([(-rfbas[-1])%Ls],N.diff(rfbas)))

        g,M = N.array([blackharrcw(m*Qvar,csh) for m,csh in izip(M,corr_shift)]).T  ####
#        M = N.ceil(M/4).astype(int)*4  # bailing out for some strange reason....
        M = N.array([ceil(m/4)*4 for m in M],dtype=int)
    else:
        g = [hannwin(m) for m in M]
    
    
    
    if sliced:
        for kk in (1,lbas+2):
            if M[kk-1] > M[kk]:
                g[kk-1] = N.ones(M[kk-1],dtype=g[kk-1].dtype)
                g[kk-1][M[kk-1]//2-M[kk]//2:M[kk-1]//2+ceil(M[kk]/2.)] = hannwin(M[kk])
        
        rfbas = N.round(fbas/2.).astype(int)*2
    else:
        fbas[lbas] = (fbas[lbas-1]+fbas[lbas+1])/2
        fbas[lbas+2] = Ls-fbas[lbas]
        rfbas = N.round(fbas).astype(int)
        
#    print "rfbas",rfbas
#    print "g",g    
    
    return g,rfbas,M
