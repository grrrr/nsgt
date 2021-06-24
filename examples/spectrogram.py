#! /usr/bin/env python 
# -*- coding: utf-8

"""
Python implementation of Non-Stationary Gabor Transform (NSGT)
derived from MATLAB code by NUHAG, University of Vienna, Austria

Thomas Grill, 2011-2016
http://grrrr.org/nsgt
"""

import os
from warnings import warn
import torch
import matplotlib.pyplot as plt

from nsgt import NSGT_sliced, LogScale, LinScale, MelScale, OctScale, SndReader


def overlap_add_slicq(slicq, ncoefs):
    print(f'SLICQ SHAPE: {slicq.shape}')
    slicq = slicq.permute(0, 2, 3, 1, 4)

    nb_samples, nb_channels, nb_f_bins, nb_slices, nb_m_bins = slicq.shape

    slicq = torch.flatten(slicq, start_dim=-2, end_dim=-1)

    slicq = slicq.reshape(nb_samples, nb_channels, nb_f_bins, -1)

    # use a slightly smaller overlap to avoid discontinuities around the edges
    shh = nb_m_bins//2
    fr = 0
    sh = max(0, min(shh, ncoefs-fr))

    # store second half of first slice
    slicq[..., fr:fr+sh] += slicq[..., sh:nb_m_bins]

    for i in range(1, nb_slices, 1):
        start = i*nb_m_bins
        slicq[..., fr:fr+sh] += slicq[..., start:start+sh]
        fr += sh
        sh = max(0, min(shh, ncoefs-fr))
        slicq[..., fr:fr+sh] = slicq[..., start+sh:(start+2*sh)]

    slicq = slicq[..., : fr]

    mls = slicq.permute(0, 3, 2, 1)
    mls = 20.*torch.log10(torch.abs(mls))
    print(f'OLA RETURNING mls: {mls.shape}')
    return mls


def _slicq_wins(window):
    ws = torch.ones(window)
    wa = torch.ones(window)

    return ws, wa


def overlap_add_slicq_i(slicq):
    nb_samples, nb_slices, nb_channels, nb_f_bins, nb_m_bins = slicq.shape

    window = nb_m_bins
    hop = window//2 # 50% overlap window

    ncoefs = nb_slices*nb_m_bins//2 + hop
    out = torch.zeros((nb_samples, nb_channels, nb_f_bins, ncoefs), dtype=slicq.dtype, device=slicq.device)

    w, _ = _slicq_wins(window)

    ptr = 0

    for i in range(nb_slices):
        out[:, :, :, ptr:ptr+window] += w*slicq[:, i, :, :, :]
        ptr += hop

    mls = out.permute(0, 3, 2, 1)
    mls = 20.*torch.log10(torch.abs(mls))
    print(f'OLA-i RETURNING mls: {out.shape}')
    return mls


def inverse_ola_slicq(slicq, nb_slices, nb_m_bins):
    nb_samples, ncoefs, nb_f_bins, nb_channels = slicq.shape

    window = nb_m_bins
    hop = window//2 # 50% overlap window

    assert(ncoefs == (nb_slices*nb_m_bins//2 + hop))

    out = torch.zeros((nb_samples, nb_slices, nb_m_bins, nb_f_bins, nb_channels), dtype=slicq.dtype, device=slicq.device)

    print(f'slicq: {slicq.shape}')
    print(f'out: {out.shape}')

    ptr = 0

    ws, wa = _slicq_wins(window)

    for i in range(nb_slices):
        out[:, i, :, :, :] += ws[None, :, None, None]*slicq[:, ptr:ptr+window, :, :]
        ptr += hop

    out *= hop/torch.sum(ws*wa)
    return out.permute(0, 1, 4, 3, 2)


def assemble_coeffs(cqts, ncoefs):
    """
    Build a sequence of blocks out of incoming overlapping CQT slices
    """
    print(f'CQT SHAPE: {cqts.shape}')

    mlses = []
    for cqt in cqts:
        cqt = iter(cqt)
        cqt0 = next(cqt)
        cq0 = cqt0.clone().detach().T
        shh = cq0.shape[0]//2
        out = torch.empty((ncoefs, cq0.shape[1], cq0.shape[2]), dtype=cq0.dtype)
        
        fr = 0
        sh = max(0, min(shh, ncoefs-fr))
        out[fr:fr+sh] = cq0[sh:] # store second half
        # add up slices
        for i, cqi in enumerate(cqt):
            cqi = cqi.clone().detach().T
            out[fr:fr+sh] += cqi[:sh]
            cqi = cqi[sh:]
            fr += sh
            sh = max(0, min(shh, ncoefs-fr))
            out[fr:fr+sh] = cqi[:sh]
            
        coefs = out[:fr]

        # compute magnitude spectrum
        mls = 20.*torch.log10(torch.abs(coefs))

        mlses.append(mls)

    mls = torch.cat([torch.unsqueeze(mls_, dim=0) for mls_ in mlses], dim=0)
    print(f'OLD RETURNING mls: {mls.shape}')
    return mls


from argparse import ArgumentParser
parser = ArgumentParser()

parser.add_argument("input", type=str, help="Input file")
parser.add_argument("--sr", type=int, default=44100, help="Sample rate used for the NSGT (default=%(default)s)")
parser.add_argument("--fmin", type=float, default=50, help="Minimum frequency in Hz (default=%(default)s)")
parser.add_argument("--fmax", type=float, default=22050, help="Maximum frequency in Hz (default=%(default)s)")
parser.add_argument("--scale", choices=('oct','log','mel'), default='log', help="Frequency scale oct, log, lin, or mel (default='%(default)s')")
parser.add_argument("--bins", type=int, default=50, help="Number of frequency bins (total or per octave, default=%(default)s)")
parser.add_argument("--sllen", type=int, default=2**20, help="Slice length in samples (default=%(default)s)")
parser.add_argument("--trlen", type=int, default=2**18, help="Transition area in samples (default=%(default)s)")
parser.add_argument("--plot", action='store_true', help="Plot transform (needs installed matplotlib package)")

args = parser.parse_args()
if not os.path.exists(args.input):
    parser.error("Input file '%s' not found"%args.input)

fs = args.sr

# build transform
scales = {'log':LogScale, 'lin':LinScale, 'mel':MelScale, 'oct':OctScale}
try:
    scale = scales[args.scale]
except KeyError:
    parser.error('Scale unknown (--scale option)')

scl = scale(args.fmin, args.fmax, args.bins)

slicq = NSGT_sliced(scl, args.sllen, args.trlen, fs, 
                    real=True,
                    matrixform=True, 
                    multichannel=True,
                    device="cpu"
                    )

# Read audio data
sf = SndReader(args.input, sr=fs, chns=2)
signal = sf()

signal = [torch.tensor(sig) for sig in signal]

pad = signal[0].shape[-1]-signal[-1].shape[-1]
signal[-1] = torch.nn.functional.pad(signal[-1], (0, pad), mode='constant', value=0)
signal = torch.cat(signal, dim=-1)

# duration of signal in s
dur = sf.frames/float(fs)

# total number of coefficients to represent input signal
ncoefs = int(sf.frames*slicq.coef_factor)

# generator for forward transformation
c = slicq.forward((signal,))

# add a batch
c = torch.unsqueeze(c, dim=0)

print(f'c: {c.shape}')
nb_samples, nb_slices, nb_channels, nb_f_bins, nb_m_bins = c.shape

# add up overlapping coefficient slices
mls1 = assemble_coeffs(c, ncoefs)
mls2 = overlap_add_slicq(c, ncoefs)

mls3 = overlap_add_slicq_i(c)
print(f'mls3: {mls3.shape}')

c_recon = inverse_ola_slicq(mls3, nb_slices, nb_m_bins)
print(f'c_recon: {c_recon.shape} vs. {c.shape}')
mls4 = overlap_add_slicq_i(c_recon)
print(f'mls4: {mls4.shape}')

print(f'{torch.nn.functional.mse_loss(mls4, mls3)}')
print(f'{torch.nn.functional.mse_loss(c_recon, c)}')

if args.plot:
    fig, axs = plt.subplots(4)

    for i, mls in enumerate([mls1, mls2, mls3, mls4]):
        print(f"Plotting t*f space for mls {i}: {mls.shape} {mls.device} {mls.dtype}")
        # remove batch
        mls = torch.squeeze(mls, dim=0)
        # mix down multichannel
        mls = torch.mean(mls, dim=-1)
        fs_coef = fs*slicq.coef_factor # frame rate of coefficients
        mls_dur = len(mls)/fs_coef # final duration of MLS
        mls_max = torch.quantile(mls, 0.999)
        axs[i].imshow(mls.T, aspect=mls_dur/mls.shape[1]*0.2, interpolation='nearest', origin='lower', vmin=mls_max-60., vmax=mls_max, extent=(0,mls_dur,0,mls.shape[1]))
        axs[i].set_title(f'mls {i}')

    plt.show()
