import numpy as np
import torch
from itertools import chain
from .fft import fftp, ifftp, irfftp
    

def nsigtf_sl(cseq, gd, wins, nn, Ls=None, real=False, reducedform=0, matrixform=False, device="cpu"):
    dtype = gd[0].dtype

    fft = fftp()
    ifft = irfftp() if real else ifftp()

    if real:
        ln = len(gd)//2+1-reducedform*2
        if reducedform:
            sl = lambda x: chain(x[reducedform:len(gd)//2+1-reducedform],x[len(gd)//2+reducedform:len(gd)+1-reducedform])
        else:
            sl = lambda x: x
    else:
        ln = len(gd)
        sl = lambda x: x
        
    maxLg = max(len(gdii) for gdii in sl(gd))

    ragged_gdiis = [torch.nn.functional.pad(torch.unsqueeze(gdii, dim=0), (0, maxLg-gdii.shape[0])) for gdii in sl(gd)]
    gdiis = torch.conj(torch.cat(ragged_gdiis))

    if not matrixform:
        assert type(cseq) == list
        nfreqs = 0
        for i, cseq_tsor in enumerate(cseq):
            cseq_dtype = cseq_tsor.dtype
            cseq[i] = fft(cseq_tsor)
            nfreqs += cseq_tsor.shape[2]
        cseq_shape = (*cseq_tsor.shape[:2], nfreqs)
    else:
        assert type(cseq) == torch.Tensor
        cseq_shape = cseq.shape[:3]
        cseq_dtype = cseq.dtype
        fc = fft(cseq)

    fr = torch.zeros(*cseq_shape[:2], nn, dtype=cseq_dtype, device=torch.device(device))  # Allocate output
    temp0 = torch.empty(*cseq_shape[:2], maxLg, dtype=fr.dtype, device=torch.device(device))  # pre-allocation

    fbins = cseq_shape[2]

    loopparams = []
    for gdii,win_range in zip(sl(gd), sl(wins)):
        Lg = len(gdii)
        wr1 = win_range[:(Lg)//2]
        wr2 = win_range[-((Lg+1)//2):]
        p = (wr1,wr2,Lg)
        loopparams.append(p)

    # The overlap-add procedure including multiplication with the synthesis windows
    if matrixform:
        for i,(wr1,wr2,Lg) in enumerate(loopparams[:fbins]):
            t = fc[:, :, i]

            r = (Lg+1)//2
            l = (Lg//2)

            t1 = temp0[:, :, :r]
            t2 = temp0[:, :, Lg-l:Lg]

            t1[:, :, :] = t[:, :, :r]
            t2[:, :, :] = t[:, :, maxLg-l:maxLg]

            temp0[:, :, :Lg] *= gdiis[i, :Lg] 
            temp0[:, :, :Lg] *= maxLg

            fr[:, :, wr1] += t2
            fr[:, :, wr2] += t1
    else:
        # frequencies are bucketed by same time resolution
        fbin_ptr = 0
        for i, fc in enumerate(cseq):
            Lg_outer = fc.shape[-1]

            nb_fbins = fc.shape[2]
            for i,(wr1,wr2,Lg) in enumerate(loopparams[fbin_ptr:fbin_ptr+nb_fbins][:fbins]):
                freq_idx = fbin_ptr+i

                assert Lg == Lg_outer
                t = fc[:, :, i]

                r = (Lg+1)//2
                l = (Lg//2)

                t1 = temp0[:, :, :r]
                t2 = temp0[:, :, Lg-l:Lg]

                t1[:, :, :] = t[:, :, :r]
                t2[:, :, :] = t[:, :, Lg-l:Lg]

                temp0[:, :, :Lg] *= gdiis[freq_idx, :Lg] 
                temp0[:, :, :Lg] *= Lg

                fr[:, :, wr1] += t2
                fr[:, :, wr2] += t1
            fbin_ptr += nb_fbins

    ftr = fr[:, :, :nn//2+1] if real else fr
    sig = ifft(ftr, outn=nn)
    sig = sig[:, :, :Ls] # Truncate the signal to original length (if given)
    return sig


# non-sliced version
def nsigtf(c, gd, wins, nn, Ls=None, real=False, matrixform=False, reducedform=0, device="cpu"):
    if matrixform:
        ret = nsigtf_sl(torch.unsqueeze(c, dim=0), gd, wins, nn, Ls=Ls, real=real, reducedform=reducedform, matrixform=matrixform, device=device)
    else:
        ret = nsigtf_sl([torch.unsqueeze(c_, dim=0) for c_ in c], gd, wins, nn, Ls=Ls, real=real, reducedform=reducedform, matrixform=matrixform, device=device)
    return torch.squeeze(ret, dim=0)
