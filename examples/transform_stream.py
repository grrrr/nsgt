#! /usr/bin/env python 
# -*- coding: utf-8

"""
Python implementation of Non-Stationary Gabor Transform (NSGT)
derived from MATLAB code by NUHAG, University of Vienna, Austria

Thomas Grill, 2011-2015
http://grrrr.org/nsgt
"""

import os
import numpy as np
from argparse import ArgumentParser
from nsgt.reblock import reblock
from nsgt.slicq import NsgtSliced
from examples.e_utils import load_audio, save_audio, cputime
from nsgt.fscale import LogScale, LinScale, MelScale, OctScale


# ------------------------------------------------------------
# Generate Args
# ------------------------------------------------------------


parser = ArgumentParser()
parser.add_argument("input", type=str, help="input file")
parser.add_argument("--output", type=str, help="output audio file")
parser.add_argument("--length", type=int, default=0, help="maximum length of signal (default=%(default)s)")
parser.add_argument("--fmin", type=float, default=50, help="minimum frequency (default=%(default)s)")
parser.add_argument("--fmax", type=float, default=22050, help="maximum frequency (default=%(default)s)")
parser.add_argument("--scale", choices=('oct', 'log', 'mel'), default='log', help="frequency scale (oct,log,lin,mel)")
parser.add_argument("--bins", type=int, default=50, help="frequency bins (total or per octave, default=%(default)s)")
parser.add_argument("--sllen", type=int, default=2 ** 16, help="slice length (default=%(default)s)")
parser.add_argument("--trlen", type=int, default=4096, help="transition area (default=%(default)s)")
parser.add_argument("--real", action='store_true', help="assume real signal")
parser.add_argument("--matrixform", action='store_true', help="use regular time division (matrix form)")
parser.add_argument("--reducedform", action='store_true',
                    help="if real==1: omit bins for f=0 and f=fs/2 (lossy=1), or also the transition bands (lossy=2)")
parser.add_argument("--recwnd", action='store_true', help="use reconstruction window")
parser.add_argument("--multithreading", action='store_true', help="use multithreading")
parser.add_argument("--plot", action='store_true',
                    help="plot transform (needs installed matplotlib and scipy packages)")

args = parser.parse_args()
if not os.path.exists(args.input):
    parser.error("Input file '%s' not found" % args.input)


# ------------------------------------------------------------
# Load Audio
# ------------------------------------------------------------


s, fs = load_audio(path=args.input)

if args.length:
    s = s[:args.length]

scales = {'log': LogScale, 'lin': LinScale, 'mel': MelScale, 'oct': OctScale}
try:
    scale = scales[args.scale]
except KeyError:
    parser.error('scale unknown')

scl = scale(args.fmin, args.fmax, args.bins)
slicq = NsgtSliced(scl, args.sllen, args.trlen, fs,
                   real=args.real, recwnd=args.recwnd,
                   matrixform=args.matrixform,
                   reducedform=args.reducedform,
                   multithreading=args.multithreading)


# ------------------------------------------------------------
# Test
# ------------------------------------------------------------


t1 = cputime()
signal = (s,)
c = slicq.forward(signal)  # generator for forward transformation
c = list(c)  # realize transform from generator

outseq = slicq.backward(c) # generator for backward transformation

# make single output array from iterator
s_r = next(reblock(outseq, len(s), fulllast=False))
s_r = s_r.real
t2 = cputime()

norm = lambda x: np.sqrt(np.sum(np.abs(np.square(np.abs(x)))))
rec_err = norm(s - s_r) / norm(s)
print("Reconstruction error: %.3e" % rec_err)
print("Calculation time: %.3fs" % (t2 - t1))

# Compare the sliced coefficients with non-sliced ones
# if False:
#     # not implemented yet!
#     test_coeff_quality(c, s, g, shift, M, options.sl_len, len(s))


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------


if args.output:
    print("Writing audio file '%s'" % args.output)
    save_audio(path=args.output, sr=fs, data=s_r)

if args.plot:
    print("Plotting t*f space")
    import matplotlib.pyplot as plt

    tr = np.array([[np.mean(np.abs(cj)) for cj in ci] for ci in c])
    plt.imshow(np.log(np.flipud(tr.T) + 1.e-10),
               aspect=float(tr.shape[0]) / tr.shape[1] * 0.5,
               interpolation='nearest')
    plt.show()
