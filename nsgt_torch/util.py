import numpy as np
import torch
from math import exp, floor, ceil, pi


def hannwin(l, device="cpu"):
    r = torch.arange(l,dtype=float, device=torch.device(device))
    r *= np.pi*2./l
    r = torch.cos(r)
    r += 1.
    r *= 0.5
    return r


def blackharr(n, l=None, mod=True, device="cpu"):
    if l is None: 
        l = n
    nn = (n//2)*2
    k = torch.arange(n, device=torch.device(device))
    if not mod:
        bh = 0.35875 - 0.48829*torch.cos(k*(2*pi/nn)) + 0.14128*torch.cos(k*(4*pi/nn)) -0.01168*torch.cos(k*(6*pi/nn))
    else:
        bh = 0.35872 - 0.48832*torch.cos(k*(2*pi/nn)) + 0.14128*torch.cos(k*(4*pi/nn)) -0.01168*torch.cos(k*(6*pi/nn))
    bh = torch.hstack((bh,torch.zeros(l-n,dtype=bh.dtype,device=torch.device(device))))
    bh = torch.hstack((bh[-n//2:],bh[:-n//2]))
    return bh


def _isseq(x):
    try:
        len(x)
    except TypeError:
        return False
    return True        


def chkM(M, g):
    if M is None:
        M = np.array(list(map(len, g)))
    elif not _isseq(M):
        M = np.ones(len(g), dtype=int)*M
    return M


def calcwinrange(g, rfbas, Ls, device="cpu"):
    shift = np.concatenate(((np.mod(-rfbas[-1],Ls),), rfbas[1:]-rfbas[:-1]))
    
    timepos = np.cumsum(shift)
    nn = timepos[-1]
    timepos -= shift[0] # Calculate positions from shift vector
    
    wins = []
    for gii,tpii in zip(g, timepos):
        Lg = len(gii)
        win_range = torch.arange(-(Lg//2)+tpii, Lg-(Lg//2)+tpii, dtype=int, device=torch.device(device))
        win_range %= nn

        wins.append(win_range)
        
    return wins,nn
