#! /usr/bin/env python 
# -*- coding: utf-8

"""
Python implementation of Non-Stationary Gabor Transform (NSGT)
derived from MATLAB code by NUHAG, University of Vienna, Austria

Thomas Grill, 2011-2015
http://grrrr.org/nsgt
"""

import numpy as np
from scikits.audiolab import Sndfile
import os
from warnings import warn

from nsgt import NSGT_sliced, LogScale, LinScale, MelScale, OctScale


def assemble_coeffs(cqt, ncoefs):
    """
    Build a sequence of blocks out of incoming overlapping CQT slices
    """
    cqt = iter(cqt)
    cqt0 = cqt.next()
    cq0 = np.asarray(cqt0).T
    shh = cq0.shape[0]//2
    out = np.empty((ncoefs,cq0.shape[1]), dtype=cq0.dtype)
    
    fr = 0
    sh = max(0, min(shh, ncoefs-fr))
    out[fr:fr+sh] = cq0[sh:] # store second half
    # add up slices
    for cqi in cqt:
        cqi = np.asarray(cqi).T
        out[fr:fr+sh] += cqi[:sh]
        cqi = cqi[sh:]
        fr += sh
        sh = max(0, min(shh, ncoefs-fr))
        out[fr:fr+sh] = cqi[:sh]
        
    return out[:fr]


from argparse import ArgumentParser
parser = ArgumentParser()

parser.add_argument("input", type=str, help="input file")
parser.add_argument("--output", type=str, help="output data file (.npz, .hd5, .pkl)")
parser.add_argument("--length", type=int, default=0, help="maximum length of signal (default=%(default)s)")
parser.add_argument("--fmin", type=float, default=50, help="minimum frequency (default=%(default)s)")
parser.add_argument("--fmax", type=float, default=22050, help="maximum frequency (default=%(default)s)")
parser.add_argument("--scale", choices=('oct','log','mel'), default='log', help="frequency scale (oct,log,lin,mel)")
parser.add_argument("--bins", type=int, default=50, help="frequency bins (total or per octave, default=%(default)s)")
parser.add_argument("--sllen", type=int, default=2**16, help="slice length (default=%(default)s)")
parser.add_argument("--trlen", type=int, default=4096, help="transition area (default=%(default)s)")
parser.add_argument("--real", action='store_true', help="assume real signal")
parser.add_argument("--matrixform", action='store_true', help="use regular time division (matrix form)")
parser.add_argument("--reducedform", type=int, help="if real==1: omit bins for f=0 and f=fs/2 (--reducedform=1), or also the transition bands (--reducedform=2)")
parser.add_argument("--recwnd", action='store_true', help="use reconstruction window")
parser.add_argument("--multithreading", action='store_true', help="use multithreading")
parser.add_argument("--plot", action='store_true', help="plot transform (needs installed matplotlib and scipy packages)")

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

scales = {'log':LogScale, 'lin':LinScale, 'mel':MelScale, 'oct':OctScale}
try:
    scale = scales[args.scale]
except KeyError:
    parser.error('Scale unknown (--scale option)')

scl = scale(args.fmin, args.fmax, args.bins)

slicq = NSGT_sliced(scl, args.sllen, args.trlen, fs, 
                    real=args.real, recwnd=args.recwnd, 
                    matrixform=args.matrixform, reducedform=args.reducedform, 
                    multithreading=args.multithreading
                    )

# total number of coefficients to represent input signal
ncoefs = int(len(s)*slicq.coef_factor)

# whole signal as sequence
signal = (s,)

# generator for forward transformation
c = slicq.forward(signal)

coefs = assemble_coeffs(c, ncoefs)
mls = np.maximum(np.log10(np.abs(coefs))*20, -100.)

dur = len(s)/float(fs)
time = np.linspace(0, dur, endpoint=False, num=ncoefs)

if args.output:
    data = dict(features=coefs, time=time)
    if args.output.endswith('.pkl') or args.output.endswith('.pck'):
        import cPickle
        with file(args.output, 'wb') as f:
            cPickle.dump(data, f, protocol=cPickle.HIGHEST_PROTOCOL)
    elif args.output.endswith('.npz'):
        np.savez(args.output, **data)
    elif args.output.endswith('.hdf5') or args.output.endswith('.h5'):
        import h5py
        with h5py.File(args.output, 'w') as f:
            for k,v in data.iteritems():
                f[k] = v
    else:
        warn("Output file format not supported, skipping output.")

if args.plot:
    print "Plotting t*f space"
    import matplotlib.pyplot as pl
    mls_max = np.percentile(mls, 99.9)
    pl.imshow(mls.T, aspect=0.2/float(dur), interpolation='nearest', origin='bottom', vmin=mls_max-60., vmax=mls_max, extent=(0,1,0,dur))
    pl.show()
