'''
Missing some copyright statement, site links, references etc.
'''

from nsgfwin import nsgfwin
from nsdual import nsdual
from nsgtf import nsgtf
from nsigtf import nsigtf
from util import calcshift

class NSGT:
    def __init__(self,fmin,fmax,bins,fs,Ls):
        self.fmin = fmin
        self.fmax = fmax
        self.bins = bins
        self.fs = fs
        self.Ls = Ls

        # calculate transform parameters
        self.g,self.a,self.M = nsgfwin(self.fmin,self.fmax,self.bins,self.fs,self.Ls)
        # calculate shifts
        self.shift = calcshift(self.a,self.Ls)
        # calculate dual windows
        self.gd = nsdual(self.g,self.shift,self.M)

    def forward(self,s):
        'transform' 
        c,_ = nsgtf(s,self.g,self.shift,self.M)
        return c

    def backward(self,c):
        'inverse transform'
        return nsigtf(c,self.gd,self.shift,self.Ls)

import unittest

class TestNSGT(unittest.TestCase):

    def test_transform(self,length=100000,fmin=50,fmax=22050,bins=12,fs=44100):
        import numpy as N
        s = N.random.random(length)
        nsgt = NSGT(fmin,fmax,bins,fs,length)
        
        # forward transform 
        c = nsgt.forward(s)
        # inverse transform 
        s_r = nsgt.backward(c)
        
        norm = lambda x: N.sqrt(N.sum(N.square(N.abs(x))))
        rec_err = norm(s-s_r)/norm(s)

        self.assertAlmostEqual(rec_err,0)

if __name__ == "__main__":
    import numpy as N
    from scikits.audiolab import Sndfile
    from time import time

    import os.path
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

    # Testing
    sf = Sndfile(args[0])
    fs = sf.samplerate
    s = sf.read_frames(sf.nframes)
    if len(s.shape) > 1: 
        s = N.mean(s,axis=1)

    t1 = time()
    
    Ls = len(s)

    # calculate transform parameters
    nsgt = NSGT(options.fmin,options.fmax,options.bins,fs,Ls)
    
    c = nsgt.forward(s)
    
    # inverse transform 
    s_r = nsgt.backward(c)
    
    t2 = time()

    norm = lambda x: N.sqrt(N.sum(N.square(N.abs(x))))
    rec_err = norm(s-s_r)/norm(s)
    print "Reconstruction error",rec_err
    print "Calculation time",t2-t1
