#! /usr/bin/env python
# -*- coding: utf-8

"""
Python implementation of Non-Stationary Gabor Transform (NSGT)
derived from MATLAB code by NUHAG, University of Vienna, Austria

Thomas Grill, 2011-2020
http://grrrr.org/nsgt
"""

import numpy as np
from nsgt import NSGT, LogScale, LinScale, MelScale, OctScale

try:
    from scikits.audiolab import Sndfile, Format
except:
    Sndfile = None
    
if Sndfile is None:
    from pysndfile import PySndfile, construct_format
    
import os


def cputime():
    utime, stime, cutime, cstime, elapsed_time = os.times()
    return utime

class interpolate:
    def __init__(self,cqt,Ls):
        from scipy.interpolate import interp1d
        self.intp = [interp1d(np.linspace(0, Ls, len(r)), r) for r in cqt]
    def __call__(self,x):
        try:
            len(x)
        except:
            return np.array([i(x) for i in self.intp])
        else:
            return np.array([[i(xi) for i in self.intp] for xi in x])


from argparse import ArgumentParser
parser = ArgumentParser()

parser.add_argument("input", type=str, help="input audio file")
parser.add_argument('-o', "--output", type=str, help="output audio file")
parser.add_argument('-f', "--fmin", type=float, default=80, help="minimum frequency (default=%(default)s)")
parser.add_argument('-F', "--fmax", type=float, default=22050, help="maximum frequency (default=%(default)s)")
parser.add_argument('-s', "--scale", choices=("oct","log","mel"), default='oct', help="frequency scale (oct,log,mel)")
parser.add_argument('-b', "--bins", type=int, default=24, help="frequency bins (total or per octave, default=%(default)s)")
parser.add_argument('-r', "--real", action='store_true', help="assume real signal")
parser.add_argument('-m', "--matrixform", action='store_true', help="use regular time division (matrix form)")
parser.add_argument('-l', "--reducedform", action='count', default=0, help="if real==1: omit bins for f=0 and f=fs/2 (lossy=1), or also the transition bands (lossy=2)")
parser.add_argument('-t', "--time", type=int, default=1, help="timing calculation n-fold (default=%(default)s)")
parser.add_argument('-p', "--plot", action='store_true', help="plot results (needs installed matplotlib and scipy packages)")

args = parser.parse_args()
if not os.path.exists(args.input):
    parser.error("Input file '%s' not found"%args.input)

# Read audio data
if Sndfile is not None:
    sf = Sndfile(args.input)
    fs = sf.samplerate
    samples = sf.nframes
else:
    sf = PySndfile(args.input)
    fs = sf.samplerate()
    samples = sf.frames()
s = sf.read_frames(samples)

if s.ndim > 1: 
    s = np.mean(s, axis=1)
    
scales = {'log':LogScale, 'lin':LinScale, 'mel':MelScale, 'oct':OctScale}
try:
    scale = scales[args.scale]
except KeyError:
    parser.error('scale unknown')

scl = scale(args.fmin, args.fmax, args.bins)

times = []

for _ in range(args.time or 1):
    t1 = cputime()
    
    # calculate transform parameters
    Ls = len(s)
    
    nsgt = NSGT(scl, fs, Ls, real=args.real, matrixform=args.matrixform, reducedform=args.reducedform)
    
    # forward transform 
    c = nsgt.forward(s)

    c = np.asarray(c)
    print("Coefficients:", c.shape)
#        print "c",len(c),N.array(map(len,c))

    # inverse transform 
    s_r = nsgt.backward(c)

    t2 = cputime()
    times.append(t2-t1)

norm = lambda x: np.sqrt(np.sum(np.abs(np.square(x))))
rec_err = norm(s-s_r)/norm(s)
print("Reconstruction error: %.3e"%rec_err)
print("Calculation time: %.3f±%.3fs (min=%.3f s)"%(np.mean(times), np.std(times)/2, np.min(times)))

if args.output:
    print("Writing audio file '%s'"%args.output)
    if Sndfile is not None:
        sf = Sndfile(args.output, mode='w', format=Format('wav','pcm24'), channels=1, samplerate=fs)
    else:
        sf = PySndfile(args.output, mode='w', format=construct_format('wav','pcm24'), channels=1, samplerate=fs)
    sf.write_frames(s_r)
    sf.close()
    print("Done")

if args.plot:
    print("Preparing plot")
    import matplotlib.pyplot as pl
    # interpolate CQT to get a grid
    hf = -1 if args.real else len(c)//2
    if not args.matrixform:
        x = np.linspace(0, Ls, 2000)
        grid = interpolate(map(np.abs, c[2:hf]), Ls)(x).T
    else:
        grid = np.abs(c[2:hf])
    np.log10(grid, out=grid)
    grid *= 20
    pmax = np.percentile(grid, 99.99)
    pl.imshow(grid, aspect='auto', origin='lower', vmin=pmax-80, vmax=pmax)
    pl.colorbar()
    pl.show()
