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
from nsgt.cq import NSGT
from argparse import ArgumentParser
from nsgt.utilities.compat import imap, xrange
from examples.e_utils import load_audio, save_audio, cputime
from nsgt.fscale import LogScale, LinScale, MelScale, OctScale


# ------------------------------------------------------------
# Generate Args
# ------------------------------------------------------------


parser = ArgumentParser()
parser.add_argument("input", type=str, help="input audio file")
parser.add_argument("--output", type=str, help="output audio file")
parser.add_argument("--fmin", type=float, default=80, help="minimum frequency (default=%(default)s)")
parser.add_argument("--fmax", type=float, default=22050, help="maximum frequency (default=%(default)s)")
parser.add_argument("--scale", choices=("oct", "log", "mel"), default='oct', help="frequency scale (oct,log,mel)")
parser.add_argument("--bins", type=int, default=24, help="frequency bins (total or per octave, default=%(default)s)")
parser.add_argument("--real", action='store_true', help="assume real signal")
parser.add_argument("--matrixform", action='store_true', help="use regular time division (matrix form)")
parser.add_argument("--reducedform", action='store_true',
                    help="if real==1: omit bins for f=0 and f=fs/2 (lossy=1), or also the transition bands (lossy=2)")
parser.add_argument("--time", type=int, default=1, help="timing calculation n-fold (default=%(default)s)")
parser.add_argument("--plot", action='store_true', help="plot results (needs installed matplotlib and scipy packages)")

args = parser.parse_args()
if not os.path.exists(args.input):
    parser.error("Input file '%s' not found" % args.input)


# ------------------------------------------------------------
# Load Audio
# ------------------------------------------------------------

s, fs = load_audio(args.input)

scales = {'log': LogScale, 'lin': LinScale, 'mel': MelScale, 'oct': OctScale}
try:
    scale = scales[args.scale]
except KeyError:
    parser.error('scale unknown')

scl = scale(args.fmin, args.fmax, args.bins)


# ------------------------------------------------------------
# Test
# ------------------------------------------------------------


times = list()
for _ in xrange(args.time or 1):
    t1 = cputime()
    Ls = len(s)  # calculate transform parameters
    nsgt = NSGT(scl, fs, Ls, real=args.real, matrixform=args.matrixform, reducedform=args.reducedform)
    c = nsgt.forward(s)  # forward transform
    s_r = nsgt.backward(c)  # inverse transform
    t2 = cputime()
    times.append(t2 - t1)

norm = lambda x: np.sqrt(np.sum(np.abs(np.square(x))))
rec_err = norm(s - s_r) / norm(s)
print("Reconstruction error: %.3e" % rec_err)
print("Calculation time: %.3f±%.3fs (min=%.3f s)" % (np.mean(times), np.std(times) / 2, np.min(times)))


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------


if args.output:
    save_audio(path=args.output, sr=fs, data=s_r)

if args.plot:
    print("Preparing plot")
    import matplotlib.pyplot as plt


    class Interpolate(object):
        def __init__(self, cqt, Ls):
            from scipy.interpolate import interp1d
            self.intp = [interp1d(np.linspace(0, Ls, len(r)), r) for r in cqt]

        def __call__(self, x):
            try:
                len(x)
            except TypeError:
                return np.array([i(x) for i in self.intp])
            else:
                return np.array([[i(xi) for i in self.intp] for xi in x])

    # interpolate CQT to get a grid
    x = np.linspace(0, Ls, 2000)
    hf = -1 if args.real else len(c) // 2
    grid = Interpolate(imap(np.abs, c[2:hf]), Ls)(x)
    plt.imshow(np.log(np.flipud(grid.T)),
               aspect=float(grid.shape[0]) / grid.shape[1] * 0.5,
               interpolation='nearest')
    plt.show()
