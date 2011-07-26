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

if __name__ == "__main__":    
    from optparse import OptionParser
    parser = OptionParser()
    
    parser.add_option("--fmin",dest="fmin",type="float",default=80,help="minimum frequency")
    parser.add_option("--fmax",dest="fmax",type="float",default=22050,help="maximum frequency")
    parser.add_option("--bins",dest="bins",type="int",default=12,help="bins per octave")
    
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
    nsgt = CQ_NSGT(options.fmin,options.fmax,options.bins,fs,Ls)
    
    # forward transform 
    c = nsgt.forward(s)
    
    # inverse transform 
    s_r = nsgt.backward(c)
    
    t2 = time()

    norm = lambda x: N.sqrt(N.sum(N.square(N.abs(x))))
    rec_err = norm(s-s_r)/norm(s)
    print "Reconstruction error: %.3e"%rec_err
    print "Calculation time: %.3f s"%(t2-t1)
