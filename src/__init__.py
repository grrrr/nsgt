'''
Missing some copyright statement, site links, references etc.
'''

from nsgfwin import nsgfwin
from nsdual import nsdual
from nsgtf import nsgtf
from nsigtf import nsigtf

def calcshift(a,Ls):
    return N.concatenate(((N.mod(-a[-1],Ls),), a[1:]-a[:-1]))

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
    if len(s.shape) > 1: s = N.mean(s,axis=1)

    t1 = time()
    
    Ls = len(s)

    g,a,M = nsgfwin(options.fmin,options.fmax,options.bins,fs,Ls)
    
    shift = calcshift(a,Ls)

    gd = nsdual(g,shift,M)
    
    c,_ = nsgtf(s,g,shift,M)
    
    s_r = nsigtf(c,gd,shift,Ls)
    
    t2 = time()

    norm = lambda x: N.sqrt(N.sum(N.square(N.abs(x))))
    rec_err = norm(s-s_r)/norm(s)
    print "Reconstruction error",rec_err
    print "Calculation time",t2-t1
