# -*- coding: utf-8

"""
Python implementation of Non-Stationary Gabor Transform (NSGT)
derived from MATLAB code by NUHAG, University of Vienna, Austria

Thomas Grill, 2011
http://grrrr.org/nsgt
"""

import numpy as N
from scipy.interpolate import interp1d
from scikits.audiolab import Sndfile,Format
from time import time
import os.path

from nsgt import CQ_NSGT_sliced

class interpolate:
    def __init__(self,cqt,Ls):
        self.intp = [interp1d(N.linspace(0,Ls,len(r)),r) for r in cqt]
    def __call__(self,x):
        try:
            len(x)
        except:
            return N.array([i(x) for i in self.intp])
        else:
            return N.array([[i(xi) for i in self.intp] for xi in x])

def sndreader(sf,blksz=2**16):
    if sf.channels > 1: 
        mixdown = lambda s: N.mean(s,axis=1)
    else:
        mixdown = lambda s: s
    for offs in xrange(0,sf.nframes,blksz):
        yield mixdown(sf.read_frames(min(sf.nframes-offs,blksz)))

def sndwriter(sf,blkseq,maxframes=None):
    written = 0
    for b in blkseq:
        if maxframes is not None: 
            b = b[:maxframes-written]
        sf.write_frames(b)
        written += len(b)
        
def db2lin(x):
    return 10**(x/20.)

def cfilter(cseq,framedur):
    amp = ((0,0),(3,0),(10,-50),(20,0),(1000,0))  # dB-curve over time
    famp = interp1d(*(N.array(amp).T)) 
    for f,c in enumerate(cseq):
        tm = f*framedur
#        amp = db2lin(famp(tm))
        amp = (f//100)%2
        yield [ci*amp for ci in c]

if __name__ == "__main__":    
    from optparse import OptionParser
    parser = OptionParser()
    
    parser.add_option("--input",dest="input",type="str",help="input file name")
    parser.add_option("--output",dest="output",type="str",help="output file name")
    parser.add_option("--fmin",dest="fmin",type="float",default=80,help="minimum frequency")
    parser.add_option("--fmax",dest="fmax",type="float",default=22050,help="maximum frequency")
    parser.add_option("--bins",dest="bins",type="int",default=12,help="bins per octave")
    parser.add_option("--slice",dest="sl_len",type="int",default=2**16,help="slice length")
    parser.add_option("--trans",dest="tr_area",type="int",default=4096,help="transition area")
    parser.add_option("--userecwnd",dest="userecwnd",type="int",default=0,help="use reconstruction window")

    (options, args) = parser.parse_args()
    if not options.input:
        parser.error("missing input filename")
    elif not os.path.exists(options.input):
        parser.error("input file not found")  
    if not options.output:
        parser.error("missing output filename")

    # Read audio data
    sfi = Sndfile(options.input,mode='r')
    fs = sfi.samplerate
    frames = sfi.nframes

    sfo = Sndfile(options.output,mode='w',format=Format('wav','pcm16'),channels=1,samplerate=fs)

    slicq = CQ_NSGT_sliced(options.fmin,options.fmax,options.bins,options.sl_len,options.tr_area,fs,userecwnd=options.userecwnd)

    t1 = time()

    # read source audio
    signal = sndreader(sfi)
    
    # generator for forward transformation
    c = slicq.forward(signal)

    # filter coefficients
    c = cfilter(c,float(options.sl_len)/fs)

    # generator for backward transformation
    signal = slicq.backward(c)

    # output resulting signal
    sndwriter(sfo,signal,maxframes=frames)

    t2 = time()

    print "IO time: %.3f s"%(t2-t1)
