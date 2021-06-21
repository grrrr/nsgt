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

from nsgt import NSGT_sliced, LogScale, LinScale, MelScale, OctScale, SndReader


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
        for cqi in cqt:
            cqi = cqi.clone().detach().T
            out[fr:fr+sh] += cqi[:sh]
            cqi = cqi[sh:]
            fr += sh
            sh = max(0, min(shh, ncoefs-fr))
            out[fr:fr+sh] = cqi[:sh]
            
        coefs = out[:fr]

        # compute magnitude spectrum
        mindb = -100.
        mls = torch.abs(coefs)
        mindb = torch.empty_like(mls).fill_(10**(mindb/20.))
        mls = torch.maximum(mls, mindb)
        mls = torch.log10(mls)
        mls *= 20.

        mlses.append(mls)

    mls = torch.cat([torch.unsqueeze(mls_, dim=0) for mls_ in mlses], dim=0)
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

# add up overlapping coefficient slices
mls = assemble_coeffs(c, ncoefs)

print(f'PRE-MIXDOWN: mls.shape: {mls.shape}')

if args.plot:
    print("Plotting t*f space")
    # remove batch
    mls = torch.squeeze(mls, dim=0)
    # mix down multichannel
    mls = torch.mean(mls, dim=-1)
    fs_coef = fs*slicq.coef_factor # frame rate of coefficients
    mls_dur = len(mls)/fs_coef # final duration of MLS
    import matplotlib.pyplot as pl
    mls_max = torch.quantile(mls, 0.999)
    print(f'mls.T.shape: {mls.T.shape}')
    print(f'c.shape: {c.shape}')
    pl.imshow(mls.T, aspect=mls_dur/mls.shape[1]*0.2, interpolation='nearest', origin='lower', vmin=mls_max-60., vmax=mls_max, extent=(0,mls_dur,0,mls.shape[1]))
    pl.show()
