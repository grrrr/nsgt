# -*- coding: utf-8

"""
Python implementation of Non-Stationary Gabor Transform (NSGT)
derived from MATLAB code by NUHAG, University of Vienna, Austria

Thomas Grill, 2011
http://grrrr.org/nsgt
"""

import numpy as N
from nsgt import CQ_NSGT
from scikits.audiolab import Sndfile
from time import time
import os.path
from itertools import imap

class interpolate:
    def __init__(self,cqt,Ls):
        from scipy.interpolate import interp1d
        self.intp = [interp1d(N.linspace(0,Ls,len(r)),r) for r in cqt]
    def __call__(self,x):
        try:
            len(x)
        except:
            return N.array([i(x) for i in self.intp])
        else:
            return N.array([[i(xi) for i in self.intp] for xi in x])

if __name__ == "__main__":    
    from optparse import OptionParser
    parser = OptionParser()
    
    parser.add_option("--fmin",dest="fmin",type="float",default=80,help="minimum frequency")
    parser.add_option("--fmax",dest="fmax",type="float",default=22050,help="maximum frequency")
    parser.add_option("--bins",dest="bins",type="int",default=12,help="bins per octave")
    parser.add_option("--real",dest="real",type="int",default=0,help="assume real signal")
    parser.add_option("--matrixform",dest="matrixform",type="int",default=0,help="use regular time division (matrix form)")
    parser.add_option("--plot",dest="plot",type="int",default=0,help="plot transform (needs installed matplotlib and scipy packages)")
    
    (options, args) = parser.parse_args()
    if not len(args):
        parser.error("missing filename")
    elif not os.path.exists(args[0]):
        parser.error("file not found")  

    # Read audio data
    sf = Sndfile(args[0])
    fs = sf.samplerate
    s = sf.read_frames(sf.nframes)
    if len(s.shape) > 1: 
        s = N.mean(s,axis=1)
        
    t1 = time()
    
    # calculate transform parameters
    Ls = len(s)
    
    nsgt = CQ_NSGT(options.fmin,options.fmax,options.bins,fs,Ls,real=options.real,matrixform=options.matrixform)
    
    # forward transform 
    c = nsgt.forward(s)

    # inverse transform 
    s_r = nsgt.backward(c)
    
    t2 = time()

    norm = lambda x: N.sqrt(N.sum(N.abs(N.square(x))))
    rec_err = norm(s-s_r)/norm(s)
    print "Reconstruction error: %.3e"%rec_err
    print "Calculation time: %.3f s"%(t2-t1)

    if options.plot:
        import pylab as P
        # interpolate CQT to get a grid
        x = N.linspace(0,Ls,1000)
        grid = interpolate(imap(N.abs,c[2:len(c)/2]),Ls)(x)
        # display grid
        P.imshow(N.log(N.flipud(grid.T)),aspect=2)
        print "Plotting"
        P.show()
