# -*- coding: utf-8

"""
Python implementation of Non-Stationary Gabor Transform (NSGT)
derived from MATLAB code by NUHAG, University of Vienna, Austria

Thomas Grill, 2011
http://grrrr.org/nsgt
"""

import numpy as N
from scikits.audiolab import Sndfile
from time import time
import os.path
from itertools import imap

from nsgt import CQ_NSGT_sliced
from reblock import reblock

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
    parser.add_option("--slice",dest="sl_len",type="int",default=2**16,help="slice length")
    parser.add_option("--trans",dest="tr_area",type="int",default=4096,help="transition area")
    parser.add_option("--real",dest="real",type="int",default=0,help="assume real signal")
    parser.add_option("--matrixform",dest="matrixform",type="int",default=0,help="use regular time division (matrix form)")
    parser.add_option("--recwnd",dest="recwnd",type="int",default=0,help="use reconstruction window")
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
    if sf.channels > 1: 
        s = N.mean(s,axis=1)

    slicq = CQ_NSGT_sliced(options.fmin,options.fmax,options.bins,options.sl_len,options.tr_area,fs,real=options.real,recwnd=options.recwnd,matrixform=options.matrixform)

    t1 = time()
    
    signal = (s,)

    # generator for forward transformation
    c = slicq.forward(signal)

    # realize transform from iterator
    c = list(c)
    

    # generator for backward transformation
    outseq = slicq.backward(c)

    # make single output array from iterator
    s_r = reblock(outseq,len(s),fulllast=False).next() 
    
    t2 = time()

    norm = lambda x: N.sqrt(N.sum(N.abs(N.square(N.abs(x)))))
    rec_err = norm(s-s_r)/norm(s)
    print "Reconstruction error: %.3e"%rec_err
    print "Calculation time: %.3f s"%(t2-t1)

    # Compare the sliced coefficients with non-sliced ones
    if False:
        test_coeff_quality(c,s,g,shift,M,options.sl_len,len(s))

    if options.plot:
        import pylab as P
        if options.matrixform:
            tr = N.abs(N.hstack(c))
            P.imshow(tr,aspect=100,interpolation='nearest')
            P.show()
    
        else:
            Ls = sf.nframes
            hf = len(c)/2 if options.real else -1
    
            # interpolate CQT to get a grid
            x = N.linspace(0,Ls,1000)
            grid = interpolate(imap(N.abs,c[2:hf]),Ls)(x)
            print "grid",grid.shape
            # display grid
            P.imshow(N.log(N.flipud(grid.T)),aspect=2)
            print "Plotting"
            P.show()
