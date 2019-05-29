#! /usr/bin/env python 
# -*- coding: utf-8

"""
Python implementation of Non-Stationary Gabor Transform (NSGT)
derived from MATLAB code by NUHAG, University of Vienna, Austria

Thomas Grill, 2011-2016
http://grrrr.org/nsgt
"""

import numpy as np
import os
from warnings import warn

from nsgt import NSGT_sliced, LogScale, LinScale, MelScale, OctScale, SndReader


def assemble_coeffs(cqt, ncoefs):
    """
    Build a sequence of blocks out of incoming overlapping CQT slices
    """
    cqt = iter(cqt)
    cqt0 = next(cqt)
    cq0 = np.asarray(cqt0).T
    shh = cq0.shape[0]//2
    out = np.empty((ncoefs, cq0.shape[1], cq0.shape[2]), dtype=cq0.dtype)
    
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

parser.add_argument("input", type=str, help="Input file")
parser.add_argument("--output", type=str, help="Output data file (.npz, .hd5, .pkl)")
parser.add_argument("--sr", type=int, default=44100, help="Sample rate used for the NSGT (default=%(default)s)")
parser.add_argument("--data-times", type=str, default='times', help="Data name for times (default='%(default)s')")
parser.add_argument("--data-frqs", type=str, default='f', help="Data name for frequencies (default='%(default)s')")
parser.add_argument("--data-qs", type=str, default='q', help="Data name for q factors (default='%(default)s')")
parser.add_argument("--data-coefs", type=str, default='coefs', help="Data name for coefficients (default='%(default)s')")
parser.add_argument("--fps", type=float, default=0, help="Approx. time resolution for features in fps (default=%(default)s)")
parser.add_argument("--fps-pooling", choices=('max','mean','median'), default='max', help="Temporal pooling function for features (default='%(default)s')")
parser.add_argument("--fmin", type=float, default=50, help="Minimum frequency in Hz (default=%(default)s)")
parser.add_argument("--fmax", type=float, default=22050, help="Maximum frequency in Hz (default=%(default)s)")
parser.add_argument("--scale", choices=('oct','log','mel'), default='log', help="Frequency scale oct, log, lin, or mel (default='%(default)s')")
parser.add_argument("--bins", type=int, default=50, help="Number of frequency bins (total or per octave, default=%(default)s)")
parser.add_argument("--mag-scale", choices=('dB','log'), default='dB', help="Magnitude scale dB or log (default='%(default)s')")
parser.add_argument("--sllen", type=int, default=2**20, help="Slice length in samples (default=%(default)s)")
parser.add_argument("--trlen", type=int, default=2**18, help="Transition area in samples (default=%(default)s)")
parser.add_argument("--real", action='store_true', help="Assume real signal")
parser.add_argument("--matrixform", action='store_true', help="Use regular time division over frequency bins (matrix form)")
parser.add_argument("--reducedform", type=int, default=0, help="If real, omit bins for f=0 and f=fs/2 (--reducedform=1), or also the transition bands (--reducedform=2) (default=%(default)s)")
parser.add_argument("--recwnd", action='store_true', help="Use reconstruction window")
parser.add_argument("--multithreading", action='store_true', help="Use multithreading")
parser.add_argument("--downmix-after", action='store_true', help="Downmix signal after spectrogram generation")
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

scl = scale(args.fmin, args.fmax, args.bins, beyond=int(args.reducedform == 2))

slicq = NSGT_sliced(scl, args.sllen, args.trlen, fs, 
                    real=args.real, recwnd=args.recwnd, 
                    matrixform=args.matrixform, reducedform=args.reducedform, 
                    multithreading=args.multithreading,
                    multichannel=True
                    )

# Read audio data
sf = SndReader(args.input, sr=fs, chns=2)
signal = sf()

# duration of signal in s
dur = sf.frames/float(fs)

# total number of coefficients to represent input signal
ncoefs = int(sf.frames*slicq.coef_factor)

# read slices from audio file and mix down signal, if necessary at all
if not args.downmix_after:
    signal = ((np.mean(s, axis=0),) for s in signal)

# generator for forward transformation
c = slicq.forward(signal)

# add up overlapping coefficient slices
coefs = assemble_coeffs(c, ncoefs)

del sf # not needed any more

# compute magnitude spectrum
mindb = -100.
mls = np.abs(coefs)
# mix down multichannel
mls = np.mean(mls, axis=-1)
np.maximum(mls, 10**(mindb/20.), out=mls)

if args.mag_scale == 'dB':
    np.log10(mls, out=mls)
    mls *= 20.
elif args.mag_scale == 'log':
    np.log(mls, out=mls)
else:
    raise NotImplementedError("Magnitude scale '%s' not implemented."%args.magscale)

fs_coef = fs*slicq.coef_factor # frame rate of coefficients
mls_dur = len(mls)/fs_coef # final duration of MLS

if args.fps:
    # pool features in time
    coefs_per_sec = fs*slicq.coef_factor
    poolf = int(coefs_per_sec/args.fps+0.5) # pooling factor
    pooled_len = mls.shape[0]//poolf
    mls_dur *= (pooled_len*poolf)/float(len(mls))
    mls = mls[:pooled_len*poolf]
    poolfun = np.__dict__[args.fps_pooling]
    mls = poolfun(mls.reshape((pooled_len,poolf,)+mls.shape[1:]), axis=1)

times = np.linspace(0, mls_dur, endpoint=True, num=len(mls)+1)

# save file
if args.output:
    if args.reducedform == 2:
        frqs = slicq.frqs[1:-1]
        qs = slicq.q[1:-1]
    else:
        frqs = slicq.frqs
        qs = slicq.q
    
    data = {args.data_coefs: mls, args.data_times: times, args.data_frqs: frqs, args.data_qs: qs}
    if args.output.endswith('.pkl') or args.output.endswith('.pck'):
        import pickle
        with file(args.output, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif args.output.endswith('.npz'):
        np.savez(args.output, **data)
    elif args.output.endswith('.hdf5') or args.output.endswith('.h5'):
        import h5py
        with h5py.File(args.output, 'w') as f:
            for k,v in data.items():
                f[k] = v
    else:
        warn("Output file format not supported, skipping output.")

if args.plot:
    print("Plotting t*f space")
    import matplotlib.pyplot as pl
    mls_max = np.percentile(mls, 99.9)
    pl.imshow(mls.T, aspect=mls_dur/mls.shape[1]*0.2, interpolation='nearest', origin='bottom', vmin=mls_max-60., vmax=mls_max, extent=(0,mls_dur,0,mls.shape[1]))
    pl.show()
