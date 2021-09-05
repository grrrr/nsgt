#! /usr/bin/env python 
# -*- coding: utf-8

"""
Python implementation of Non-Stationary Gabor Transform (NSGT)
derived from MATLAB code by NUHAG, University of Vienna, Austria

Thomas Grill, 2011-2016
http://grrrr.org/nsgt
"""

import numpy as np
import torch
from tqdm import tqdm
import os
from warnings import warn

from nsgt import NSGT_sliced, LogScale, LinScale, MelScale, OctScale, SndReader, BarkScale, VQLogScale
from nsgt_orig import NSGT_sliced as NSGT_sliced_old
import time

from argparse import ArgumentParser
parser = ArgumentParser()

parser.add_argument("input", type=str, help="Input file")
parser.add_argument("--output", type=str, help="Output data file (.npz, .hd5, .pkl)")
parser.add_argument("--sr", type=int, default=44100, help="Sample rate used for the NSGT (default=%(default)s)")
parser.add_argument("--fmin", type=float, default=50, help="Minimum frequency in Hz (default=%(default)s)")
parser.add_argument("--fmax", type=float, default=22050, help="Maximum frequency in Hz (default=%(default)s)")
parser.add_argument("--scale", choices=('oct','cqlog','vqlog','mel','bark'), default='cqlog', help="Frequency scale oct, log, lin, or mel (default='%(default)s')")
parser.add_argument("--bins", type=int, default=50, help="Number of frequency bins (total or per octave, default=%(default)s)")
parser.add_argument("--real", action='store_true', help="Assume real signal")
parser.add_argument("--old", action='store_true', help="Use old transform")
parser.add_argument("--matrixform", action='store_true', help="Use regular time division over frequency bins (matrix form)")
parser.add_argument("--torch-device", type=str, help="Which pytorch device to use", default="cpu")
parser.add_argument("--reducedform", type=int, default=0, help="If real, omit bins for f=0 and f=fs/2 (--reducedform=1), or also the transition bands (--reducedform=2) (default=%(default)s)")
parser.add_argument("--multithreading", action='store_true', help="Use multithreading")
parser.add_argument("--bench-iter", type=int, default=1000, help="Benchmark iterations")

args = parser.parse_args()
if not os.path.exists(args.input):
    parser.error("Input file '%s' not found"%args.input)

fs = args.sr

# build transform
scales = {'cqlog':LogScale, 'lin':LinScale, 'mel':MelScale, 'oct':OctScale, 'bark':BarkScale, 'vqlog':VQLogScale}
try:
    scale = scales[args.scale]
except KeyError:
    parser.error('Scale unknown (--scale option)')

scl = scale(args.fmin, args.fmax, args.bins, beyond=int(args.reducedform == 2))
sllen, trlen = scl.suggested_sllen_trlen(fs)

# Read audio data
sf = SndReader(args.input, sr=fs, chns=2)

# store generator into a list
signal_orig = list(sf())

if args.old:
    slicq = NSGT_sliced_old(scl, sllen, trlen, fs, 
                        real=True, 
                        matrixform=args.matrixform,
                        multithreading=args.multithreading,
                        multichannel=True
                        )

    # read slices from audio file and mix down signal, if necessary at all
    signal = ((np.mean(s, axis=0),) for s in signal_orig)
else:
    slicq = NSGT_sliced(scl, sllen, trlen, fs, 
                        real=True, 
                        matrixform=args.matrixform, 
                        multichannel=True,
                        device=args.torch_device
                        )

    signal = [torch.tensor(sig, device=args.torch_device) for sig in signal_orig]

    pad = signal[0].shape[-1]-signal[-1].shape[-1]
    signal[-1] = torch.nn.functional.pad(signal[-1], (0, pad), mode='constant', value=0)
    signal = torch.cat(signal, dim=-1)

tot = 0.
for _ in tqdm(range(args.bench_iter)):
    start = time.time()
    if args.old:
        # generator for forward transformation
        c = slicq.forward(signal)

        c_list = list(c)
        sig_recon = slicq.backward(c_list)
        sig = list(sig_recon)
    else:
        # torch
        c = slicq.forward((signal,))
        sig_recon = slicq.backward(c, signal.shape[-1])

    tot += time.time() - start

    # recreate generator
    if args.old:
        signal = ((np.mean(s, axis=0),) for s in signal_orig)

tot /= float(args.bench_iter)

print(f'total time: {tot:.2f}s')
