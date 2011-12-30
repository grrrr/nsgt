'''
Created on 30.12.2011

@author: thomas
'''

import numpy as N
from nsgt import CQ_NSGT_sliced
from scipy.interpolate import interp1d
from itertools import izip,count

class Transpose:
    def __init__(self,sh,octbins):
        # sh = ((0,0),(10,10),(20,0))  # time/shift pairs
        sha = N.array(sh).T
        sha[1] *= octbins/12.
        self.fsh = interp1d(*sha)
    def __call__(self,t,c,reducedform=False):
        ca = N.array(c,copy=True)
        cn = ca if reducedform else ca[1:-1]
        # manipulate here
        sh = round(self.fsh(t))
        if sh > 0:
            cn[sh:] = cn[:-sh]
            cn[:sh] = 0
        elif sh < 0:
            cn[:sh] = cn[-sh:]
            cn[sh:] = 0
        return ca

class Gain:
    def __init__(self,gain):
        # sh = ((0,0),(10,10),(20,0))  # time/dB pairs
        self.fgain = interp1d(*(N.array(gain).T))
    def __call__(self,t,c,reducedform=False):
        ca = N.array(c,copy=True)
        cn = ca if reducedform else ca[1:-1]
        # manipulate here
        cn *= 10**(self.fgain(t)/20.)
        return ca

class Transform:
    def __init__(self,fs,options):
        self.fs = fs
        self.options = options
        self.nsgt = CQ_NSGT_sliced(options.fmin,options.fmax,options.bins,options.sllen,options.trlen,fs,reducedform=options.lossy,real=True,recwnd=True,matrixform=True,multichannel=True)
    
    def __call__(self,s,actions):
        fs = self.fs
        options = self.options
        
        # forward transform 
        c = self.nsgt.forward(s)
        
        times = (options.sllen/float(fs)*(i-1)/2. for i in count())
        tc = izip(times,c)
    
        c = self.process(tc,actions,reducedform=options.lossy)
    
        # inverse transform 
        s_r = self.nsgt.backward(c)
        
        return s_r
    
    @staticmethod
    def process(tcseq,actions,reducedform=False):
        for t,ci in tcseq:
            chns = []
            for cc in ci:
                for action in actions:
                    cc = action(t,cc,reducedform=reducedform)
                chns.append(cc)
            yield chns
