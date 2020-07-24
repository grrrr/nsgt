#! /usr/bin/env python 
# -*- coding: utf-8

"""
Python implementation of Non-Stationary Gabor Transform (NSGT)
derived from MATLAB code by NUHAG, University of Vienna, Austria

Thomas Grill, 2011-2020
http://grrrr.org/nsgt
"""

import numpy as np
from scikits.audiolab import Sndfile, Format
import os

from nsgt import NSGT_sliced, LogScale, LinScale, MelScale, OctScale
from nsgt.reblock import reblock

def cputime():
    utime, stime, cutime, cstime, elapsed_time = os.times()
    return utime


from argparse import ArgumentParser
parser = ArgumentParser()

parser.add_argument("input", type=str, help="input file")
parser.add_argument('-o', "--output", type=str, help="output audio file")
parser.add_argument('-L', "--length", type=int, default=0, help="maximum length of signal (default=%(default)s)")
parser.add_argument('-f', "--fmin", type=float, default=50, help="minimum frequency (default=%(default)s)")
parser.add_argument('-F', "--fmax", type=float, default=22050, help="maximum frequency (default=%(default)s)")
parser.add_argument('-s', "--scale", choices=('oct','log','mel'), default='log', help="frequency scale (oct,log,lin,mel)")
parser.add_argument('-b', "--bins", type=int, default=50, help="frequency bins (total or per octave, default=%(default)s)")
parser.add_argument("--sllen", type=int, default=2**16, help="slice length (default=%(default)s)")
parser.add_argument("--trlen", type=int, default=4096, help="transition area (default=%(default)s)")
parser.add_argument('-r', "--real", action='store_true', help="assume real signal")
parser.add_argument('-m', "--matrixform", action='store_true', help="use regular time division (matrix form)")
parser.add_argument('-l', "--reducedform", action='count', default=0, help="if real==1: omit bins for f=0 and f=fs/2 (lossy=1), or also the transition bands (lossy=2)")
parser.add_argument('-w', "--recwnd", action='store_true', help="use reconstruction window")
parser.add_argument('-t', "--multithreading", action='store_true', help="use multithreading")
parser.add_argument('-p', "--plot", action='store_true', help="plot transform (needs installed matplotlib and scipy packages)")

args = parser.parse_args()
if not os.path.exists(args.input):
    parser.error("Input file '%s' not found"%args.input)

# Read audio data
sf = Sndfile(args.input)
fs = sf.samplerate
s = sf.read_frames(sf.nframes)
if sf.channels > 1: 
    s = np.mean(s, axis=1)
    
if args.length:
    s = s[:args.length]

scales = {'log':LogScale,'lin':LinScale,'mel':MelScale,'oct':OctScale}
try:
    scale = scales[args.scale]
except KeyError:
    parser.error('scale unknown')

scl = scale(args.fmin, args.fmax, args.bins)
slicq = NSGT_sliced(scl, args.sllen, args.trlen, fs, 
                    real=args.real, recwnd=args.recwnd, 
                    matrixform=args.matrixform, reducedform=args.reducedform, 
                    multithreading=args.multithreading
                    )

t1 = cputime()

signal = (s,)

# generator for forward transformation
c = slicq.forward(signal)

# realize transform from generator
c = list(c)

# generator for backward transformation
outseq = slicq.backward(c)

# make single output array from iterator
s_r = next(reblock(outseq, len(s), fulllast=False))
s_r = s_r.real

t2 = cputime()

norm = lambda x: np.sqrt(np.sum(np.abs(np.square(np.abs(x)))))
rec_err = norm(s-s_r)/norm(s)
print("Reconstruction error: %.3e"%rec_err)
print("Calculation time: %.3fs"%(t2-t1))

# Compare the sliced coefficients with non-sliced ones
if False:
    # not implemented yet!
    test_coeff_quality(c, s, g, shift, M, options.sl_len, len(s))

if args.output:
    print("Writing audio file '%s'"%args.output)
    sf = Sndfile(args.output, mode='w', format=Format('wav','pcm24'), channels=1, samplerate=fs)
    sf.write_frames(s_r)
    sf.close()
    print("Done")

if args.plot:
    print("Plotting t*f space")
    import matplotlib.pyplot as pl
    tr = np.array([[np.mean(np.abs(cj)) for cj in ci] for ci in c])
    pl.imshow(np.log(np.flipud(tr.T)+1.e-10), aspect=float(tr.shape[0])/tr.shape[1]*0.5, interpolation='nearest')
    pl.show()
