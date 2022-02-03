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


def complex_2_magphase(spec):
    if type(spec) == torch.Tensor:
        return torch.abs(torch.view_as_complex(spec)), atan2(spec[..., 1], spec[..., 0])

    ret_mag = [None]*len(spec)
    ret_phase = [None]*len(spec)

    for i, C_block in enumerate(spec):
        C_block_phase = atan2(C_block[..., 1], C_block[..., 0])
        C_block_mag = torch.abs(torch.view_as_complex(C_block))

        ret_mag[i] = C_block_mag
        ret_phase[i] = C_block_phase

    return ret_mag, ret_phase


def magphase_2_complex(C_mag, C_phase):
    if type(C_mag) == torch.Tensor:
        C_cplx = torch.empty(*(*C_mag.shape, 2), dtype=C_mag.dtype, device=C_mag.device)
        C_cplx[..., 0] = C_mag * torch.cos(C_phase)
        C_cplx[..., 1] = C_mag * torch.sin(C_phase)
        return C_cplx

    C_cplx = [None]*len(C_mag)

    for i, (C_mag_block, C_phase_block) in enumerate(zip(C_mag, C_phase)):
        C_cplx_block = torch.empty(*(*C_mag_block.shape, 2), dtype=C_mag_block.dtype, device=C_mag_block.device)

        C_cplx_block[..., 0] = C_mag_block * torch.cos(C_phase_block)
        C_cplx_block[..., 1] = C_mag_block * torch.sin(C_phase_block)

        C_cplx[i] = C_cplx_block

    return C_cplx


def atan2(y, x):
    r"""Element-wise arctangent function of y/x.
    Returns a new tensor with signed angles in radians.
    It is an alternative implementation of torch.atan2

    Args:
        y (Tensor): First input tensor
        x (Tensor): Second input tensor [shape=y.shape]

    Returns:
        Tensor: [shape=y.shape].
    """
    pi = 2 * torch.asin(torch.tensor(1.0))
    x += ((x == 0) & (y == 0)) * 1.0
    out = torch.atan(y / x)
    out += ((y >= 0) & (x < 0)) * pi
    out -= ((y < 0) & (x < 0)) * pi
    out *= 1 - ((y > 0) & (x == 0)) * 1.0
    out += ((y > 0) & (x == 0)) * (pi / 2)
    out *= 1 - ((y < 0) & (x == 0)) * 1.0
    out += ((y < 0) & (x == 0)) * (-pi / 2)
    return out
