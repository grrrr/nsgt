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
from util import hannwin,blackharr
from math import ceil

def nsgfwin_sl(f,q,sr,Ls,min_win=4,Qvar=1,sliced=True,matrixform=True):
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
    
    fbas = f
    lbas = len(fbas)
    
    frqs = N.concatenate(((0.,),fbas,(nf,)))
    
    fbas = N.concatenate((frqs,sr-frqs[-2:0:-1]))

    # at this point: fbas.... frequencies in Hz
    
    fbas *= float(Ls)/sr
    
    # Omega[k] in the paper
    M = N.empty(fbas.shape,int)
    M[0] = N.round(2*fbas[1])
    M[1:lbas+1] = N.round(fbas[1:lbas+1]/q[0:lbas])
    M[lbas+1] = N.round(Ls-2*fbas[lbas])
    M[-1:lbas+1:-1] = M[1:lbas+1]
    
    if sliced:
        # multiple of 4
        M //= 4
        M *= 4
    
    N.clip(M,min_win,N.inf,out=M)

#    print "M",list(M)
    
    if sliced: 
        g = [blackharr(m) for m in M]
    else:
        g = [hannwin(m) for m in M]
    
    if sliced:
        if False:
            for kk in (1,lbas+2):
                if M[kk-1] > M[kk]:
                    g[kk-1] = N.ones(M[kk-1],dtype=g[kk-1].dtype)
                    g[kk-1][M[kk-1]//2-M[kk]//2:M[kk-1]//2+ceil(M[kk]/2.)] = hannwin(M[kk])
        
        rfbas = N.round(fbas/2.).astype(int)*2
    else:
        fbas[lbas] = (fbas[lbas-1]+fbas[lbas+1])/2
        fbas[lbas+2] = Ls-fbas[lbas]
        rfbas = N.round(fbas).astype(int)
    
#    print "rfbas",list(rfbas)
    
    return g,rfbas,M
