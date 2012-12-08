# -*- coding: utf-8

"""
Python implementation of Non-Stationary Gabor Transform (NSGT)
derived from MATLAB code by NUHAG, University of Vienna, Austria

Thomas Grill, 2011-2012
http://grrrr.org/nsgt
"""

import numpy as N
from scikits.audiolab import Sndfile,Format
from time import time
import os.path
from itertools import imap

from nsgt import NSGT_sliced,LogScale,LinScale,MelScale,OctScale
from nsgt.reblock import reblock


if __name__ == "__main__":    
    from optparse import OptionParser
    parser = OptionParser()
    
    parser.add_option("--input",dest="input",type="str",help="input file")
    parser.add_option("--output",dest="output",type="str",help="output audio file")
    parser.add_option("--length",dest="length",type="int",default=0,help="maximum length of signal")
    parser.add_option("--fmin",dest="fmin",type="float",default=50,help="minimum frequency")
    parser.add_option("--fmax",dest="fmax",type="float",default=22050,help="maximum frequency")
    parser.add_option("--scale",dest="scale",type="str",default='log',help="frequency scale (oct,log,lin,mel)")
    parser.add_option("--bins",dest="bins",type="int",default=50,help="frequency bins (total or per octave)")
    parser.add_option("--sllen",dest="sl_len",type="int",default=2**16,help="slice length")
    parser.add_option("--trlen",dest="tr_area",type="int",default=4096,help="transition area")
    parser.add_option("--real",dest="real",type="int",default=0,help="assume real signal")
    parser.add_option("--matrixform",dest="matrixform",type="int",default=0,help="use regular time division (matrix form)")
    parser.add_option("--reducedform",dest="reducedform",type="int",default=0,help="if real==1: omit bins for f=0 and f=fs/2 (lossy=1), or also the transition bands (lossy=2)")
    parser.add_option("--recwnd",dest="recwnd",type="int",default=0,help="use reconstruction window")
    parser.add_option("--multithreading",dest="multithreading",type="int",default=0,help="use multithreading")
    parser.add_option("--plot",dest="plot",type="int",default=0,help="plot transform (needs installed matplotlib and scipy packages)")

    (options, args) = parser.parse_args()
    if not os.path.exists(options.input):
        parser.error("file not found")  

    # Read audio data
    sf = Sndfile(options.input)
    fs = sf.samplerate
    s = sf.read_frames(sf.nframes)
    if sf.channels > 1: 
        s = N.mean(s,axis=1)
        
    if options.length:
        s = s[:options.length]

    scales = {'log':LogScale,'lin':LinScale,'mel':MelScale,'oct':OctScale}
    try:
        scale = scales[options.scale]
    except KeyError:
        parser.error('scale unknown')

    scl = scale(options.fmin,options.fmax,options.bins)
    slicq = NSGT_sliced(scl,options.sl_len,options.tr_area,fs,real=options.real,recwnd=options.recwnd,matrixform=options.matrixform,reducedform=options.reducedform,multithreading=options.multithreading)

    t1 = time()
    
    signal = (s,)

    # generator for forward transformation
    c = slicq.forward(signal)

    # realize transform from generator
    c = list(c)
    
#    cl = map(len,c[0])
#    print "c",len(cl),cl
    
    # generator for backward transformation
    outseq = slicq.backward(c)

    # make single output array from iterator
    s_r = reblock(outseq,len(s),fulllast=False).next()
    s_r = s_r.real
    
    t2 = time()

    norm = lambda x: N.sqrt(N.sum(N.abs(N.square(N.abs(x)))))
    rec_err = norm(s-s_r)/norm(s)
    print "Reconstruction error: %.3e"%rec_err
    print "Calculation time: %.3f s"%(t2-t1)

    # Compare the sliced coefficients with non-sliced ones
    if False:
        # not implemented yet!
        test_coeff_quality(c,s,g,shift,M,options.sl_len,len(s))

    if options.output:
        print "Written audio file",options.output
        sf = Sndfile(options.output,mode='w',format=Format('wav','pcm24'),channels=1,samplerate=fs)
        sf.write_frames(s_r)
        sf.close()
        print "Done"

    if options.plot:
        print "Plotting t*f space"
        import pylab as P
        tr = N.array([[N.mean(N.abs(cj)) for cj in ci] for ci in c])
        P.imshow(N.log(N.flipud(tr.T)+1.e-10),aspect=float(tr.shape[0])/tr.shape[1]*0.5,interpolation='nearest')
        P.show()
