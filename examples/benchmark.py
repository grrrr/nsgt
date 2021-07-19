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
import os
from warnings import warn

from nsgt import NSGT_sliced, LogScale, LinScale, MelScale, OctScale, SndReader, BarkScale, VQLogScale
from nsgt_orig import NSGT_sliced as NSGT_sliced_old

from argparse import ArgumentParser
parser = ArgumentParser()

parser.add_argument("input", type=str, help="Input file")
parser.add_argument("--output", type=str, help="Output data file (.npz, .hd5, .pkl)")
parser.add_argument("--sr", type=int, default=44100, help="Sample rate used for the NSGT (default=%(default)s)")
parser.add_argument("--fmin", type=float, default=50, help="Minimum frequency in Hz (default=%(default)s)")
parser.add_argument("--fmax", type=float, default=22050, help="Maximum frequency in Hz (default=%(default)s)")
parser.add_argument("--scale", choices=('oct','cqlog','vqlog','mel','bark'), default='cqlog', help="Frequency scale oct, log, lin, or mel (default='%(default)s')")
parser.add_argument("--bins", type=int, default=50, help="Number of frequency bins (total or per octave, default=%(default)s)")
parser.add_argument("--sllen", type=int, default=2**20, help="Slice length in samples (default=%(default)s)")
parser.add_argument("--trlen", type=int, default=2**18, help="Transition area in samples (default=%(default)s)")
parser.add_argument("--real", action='store_true', help="Assume real signal")
parser.add_argument("--old", action='store_true', help="Use old transform")
parser.add_argument("--matrixform", action='store_true', help="Use regular time division over frequency bins (matrix form)")
parser.add_argument("--torch-device", type=str, help="Which pytorch device to use")
parser.add_argument("--reducedform", type=int, default=0, help="If real, omit bins for f=0 and f=fs/2 (--reducedform=1), or also the transition bands (--reducedform=2) (default=%(default)s)")
parser.add_argument("--multithreading", action='store_true', help="Use multithreading")
parser.add_argument("--plot", action='store_true', help="Plot transform (needs installed matplotlib package)")

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

# Read audio data
sf = SndReader(args.input, sr=fs, chns=2)
signal = sf()

if args.old:
    slicq = NSGT_sliced_old(scl, args.sllen, args.trlen, fs, 
                        real=True, 
                        matrixform=args.matrixform,
                        multithreading=args.multithreading,
                        multichannel=True
                        )

    # read slices from audio file and mix down signal, if necessary at all
    signal = ((np.mean(s, axis=0),) for s in signal)
else:
    slicq = NSGT_sliced(scl, args.sllen, args.trlen, fs, 
                        real=True, 
                        matrixform=args.matrixform, 
                        multichannel=True,
                        device=args.torch_device
                        )

    signal = [torch.tensor(sig) for sig in signal]

    pad = signal[0].shape[-1]-signal[-1].shape[-1]
    signal[-1] = torch.nn.functional.pad(signal[-1], (0, pad), mode='constant', value=0)
    signal = torch.cat(signal, dim=-1)

import time
start = time.time()
if args.old:
    # generator for forward transformation
    c = slicq.forward(signal)

    c_list = list(c)
    slicq.backward(c_list)
else:
    # generator for forward transformation
    c = slicq.forward((signal,))
    slicq.backward(c, signal.shape[-1])

tot = time.time() - start

print(f'total time: {tot:.2f}s')
