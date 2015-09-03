#! /usr/bin/env python 
# -*- coding: utf-8

"""
Python implementation of Non-Stationary Gabor Transform (NSGT)
derived from MATLAB code by NUHAG, University of Vienna, Austria

Thomas Grill, 2011-2015
http://grrrr.org/nsgt
"""

import numpy as np
from nsgt import NSGT,LogScale,LinScale,MelScale,OctScale
from scikits.audiolab import Sndfile,Format
import os
from itertools import imap

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
parser.add_argument("--output", type=str, help="output audio file")
parser.add_argument("--fmin", type=float, default=80, help="minimum frequency (default=%(default)s)")
parser.add_argument("--fmax", type=float, default=22050, help="maximum frequency (default=%(default)s)")
parser.add_argument("--scale", choices=("oct","log","mel"), default='oct', help="frequency scale (oct,log,mel)")
parser.add_argument("--bins", type=int, default=24, help="frequency bins (total or per octave, default=%(default)s)")
parser.add_argument("--real", action='store_true', help="assume real signal")
parser.add_argument("--matrixform", action='store_true', help="use regular time division (matrix form)")
parser.add_argument("--reducedform", action='store_true', help="if real==1: omit bins for f=0 and f=fs/2 (lossy=1), or also the transition bands (lossy=2)")
parser.add_argument("--time", type=int, default=1, help="timing calculation n-fold (default=%(default)s)")
parser.add_argument("--plot", action='store_true', help="plot results (needs installed matplotlib and scipy packages)")

args = parser.parse_args()
if not os.path.exists(args.input):
    parser.error("Input file '%s' not found"%args.input)

# Read audio data
sf = Sndfile(args.input)
fs = sf.samplerate
s = sf.read_frames(sf.nframes)
if len(s.shape) > 1: 
    s = np.mean(s, axis=1)
    
scales = {'log':LogScale,'lin':LinScale,'mel':MelScale,'oct':OctScale}
try:
    scale = scales[args.scale]
except KeyError:
    parser.error('scale unknown')

scl = scale(args.fmin, args.fmax, args.bins)

times = []

for _ in xrange(args.time or 1):
    t1 = cputime()
    
    # calculate transform parameters
    Ls = len(s)
    
    nsgt = NSGT(scl, fs, Ls, real=args.real, matrixform=args.matrixform, reducedform=args.reducedform)
    
    # forward transform 
    c = nsgt.forward(s)

#        c = N.array(c)
#        print "c",len(c),N.array(map(len,c))

    # inverse transform 
    s_r = nsgt.backward(c)

    t2 = cputime()
    times.append(t2-t1)

norm = lambda x: np.sqrt(np.sum(np.abs(np.square(x))))
rec_err = norm(s-s_r)/norm(s)
print "Reconstruction error: %.3e"%rec_err
print "Calculation time: %.3f±%.3fs (min=%.3f s)"%(np.mean(times),np.std(times)/2,np.min(times))

if args.output:
    print "Writing audio file '%s'"%args.output
    sf = Sndfile(args.output, mode='w', format=Format('wav','pcm24'), channels=1, samplerate=fs)
    sf.write_frames(s_r)
    sf.close()
    print "Done"

if args.plot:
    print "Preparing plot"
    import matplotlib.pyplot as pl
    # interpolate CQT to get a grid
    x = np.linspace(0, Ls, 2000)
    hf = -1 if args.real else len(c)//2
    grid = interpolate(imap(np.abs, c[2:hf]), Ls)(x)
    # display grid
    pl.imshow(np.log(np.flipud(grid.T)), aspect=float(grid.shape[0])/grid.shape[1]*0.5, interpolation='nearest')
    print "Plotting"
    pl.show()
