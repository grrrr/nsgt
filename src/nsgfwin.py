# -*- coding: utf-8

# NSGFWIN.M
#---------------------------------------------------------------
# [g,rfbas,M]=nsgfwin(fmin,bins,sr,Ls) creates a set of windows whose
# centers correspond to center frequencies to be
# used for the nonstationary Gabor transform with varying Q-factor. 
#---------------------------------------------------------------
#
# INPUT : fmin ...... Minimum frequency (in Hz)
#         bins ...... Vector consisting of the number of bins per octave
#         sr ........ Sampling rate (in Hz)
#         Ls ........ Length of signal (in samples)
#
# OUTPUT : g ......... Cell array of window functions.
#          rfbas ..... Vector of positions of the center frequencies.
#          M ......... Vector of lengths of the window functions.
#
# AUTHOR(s) : Monika Dörfler, Gino Angelo Velasco, Nicki Holighaus, 2010
#
# COPYRIGHT : (c) NUHAG, Dept.Math., University of Vienna, AUSTRIA
# http://nuhag.eu/
# Permission is granted to modify and re-distribute this
# code in any manner as long as this notice is preserved.
# All standard disclaimers apply.
#
# EXTERNALS : firwin

import numpy as N
from util import hannwin,_isseq

def nsgfwin(fmin,fmax,bins,sr,Ls):

    nf = sr/2
    
    if fmax > nf:
        fmax = nf
    
    b = N.ceil(N.log2(fmax/fmin))+1

    if not _isseq(bins) == 1:
        bins = N.ones(b,dtype=float)*bins
    elif len(bins) < b:
        bins[bins <= 0] = 1
        bins = N.concatenate((bins,N.ones(b-len(bins),dtype=int)*N.min(bins)))
    
    fbas = []
    for kk,bkk in enumerate(bins):
        r = N.arange(kk*bkk,(kk+1)*bkk)/float(bkk)
        fbas.append(2**r*fmin)
    fbas = N.concatenate(fbas)

    if fbas[N.min(N.where(fbas >= fmax))] >= nf:
        fbas = fbas[:N.max(N.where(fbas<fmax))]
    else:
        fbas = fbas[:N.min(N.where(fbas>=fmax))]
    
    lbas = len(fbas)
    fbas = N.insert(fbas,0,0)
    fbas[lbas+1] = nf
    fbas[lbas+2:2*(lbas+1)] = sr-fbas[lbas:1:-1]
    
    fbas *= Ls/sr
    
    M = N.zeros(len(fbas),dtype=float);
    M[0] = 2*fmin*(Ls/sr)
    for k in xrange(1,2*lbas+1):
        M[k] = fbas[k+1]-fbas[k-1]
    M[-1] = Ls-fbas[-1]
    M = N.round(M)
    
    g = []
    for ii in xrange(2*(lbas+1)):   
        if M[ii] < 4:
            M[ii] = 4

        g.append(hannwin(N.round(M[ii])))
    
    fbas[lbas] = (fbas[lbas-1]+fbas[lbas+1])/2
    fbas[lbas+2] = Ls-fbas[lbas]
    rfbas = N.round(fbas)
    
    return g,rfbas,M
