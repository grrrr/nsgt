import numpy as np
from .util import hannwin,_isseq


def nsgfwin(fmin, fmax ,bins, sr, Ls, min_win=4, device="cpu"):

    nf = sr/2
    
    if fmax > nf:
        fmax = nf
    
    b = np.ceil(np.log2(fmax/fmin))+1

    if not _isseq(bins):
        bins = np.ones(b,dtype=int)*bins
    elif len(bins) < b:
        # TODO: test this branch!
        bins[bins <= 0] = 1
        bins = np.concatenate((bins, np.ones(b-len(bins), dtype=int)*np.min(bins)))
    
    fbas = []
    for kk,bkk in enumerate(bins):
        r = np.arange(kk*bkk, (kk+1)*bkk, dtype=float)
        # TODO: use N.logspace instead
        fbas.append(2**(r/bkk)*fmin)
    fbas = np.concatenate(fbas)

    if fbas[np.min(np.where(fbas>=fmax))] >= nf:
        fbas = fbas[:np.max(np.where(fbas<fmax))+1]
    else:
        # TODO: test this branch!
        fbas = fbas[:np.min(np.where(fbas>=fmax))+1]
    
    lbas = len(fbas)
    fbas = np.concatenate(((0.,), fbas, (nf,), sr-fbas[::-1]))
    fbas *= float(Ls)/sr
    
    # TODO: put together with array indexing
    M = np.empty(fbas.shape, dtype=int)
    M[0] = np.round(2.*fmin*Ls/sr)
    for k in range(1, 2*lbas+1):
        M[k] = np.round(fbas[k+1]-fbas[k-1])
    M[-1] = np.round(Ls-fbas[-2])
    
    M = np.clip(M, min_win, np.inf).astype(int)
    g = [hannwin(m, device=device) for m in M]
    
    fbas[lbas] = (fbas[lbas-1]+fbas[lbas+1])/2
    fbas[lbas+2] = Ls-fbas[lbas]
    rfbas = np.round(fbas).astype(int)
    
    return g,rfbas,M
