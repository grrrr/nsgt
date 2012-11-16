'''
Created on 05.11.2011

@author: Thomas Grill (grrrr.org)

% Perfect reconstruction sliCQ

% right now, even slice length (sl_len) is required. Parameters are the
% same as NSGTF plus slice length, minimal required window length, 
% Q-factor variation, and test run parameters.
'''

import numpy as N
from itertools import izip,cycle,chain,tee

from slicing import slicing
from unslicing import unslicing
from nsdual import nsdual
from nsgfwin_sl import nsgfwin
from nsgtf import nsgtf_sl
from nsigtf import nsigtf_sl
from util import calcwinrange
from fscale import OctScale

# one of the more expensive functions (32/400)
def arrange(cseq,M,fwd):
    c0 = cseq.next()  # grab first stream element
    cseq = chain((c0,),cseq)  # push it back in 
    assert len(c0) == 1
    M = map(len,c0[0])  # read off M from the coefficients
    ixs = (
           [(slice(3*mkk//4,mkk),slice(0,3*mkk//4)) for mkk in M],  # odd
           [(slice(mkk//4,mkk),slice(0,mkk//4)) for mkk in M]  # even
    )
    if fwd:
        ixs = cycle(ixs)
    else:
        ixs = cycle(ixs[::-1])

    return ([[N.concatenate((ckk[ix0],ckk[ix1])) for ckk,(ix0,ix1) in izip(ci,ixi)] for ci in cci] for cci,ixi in izip(cseq,ixs))

def starzip(iterables):
    def inner(itr, i):
        for t in itr:
            yield t[i]
    iterables = iter(iterables) 
    it = iterables.next()  # we need that to determine the length of one element
    iterables = chain((it,),iterables)
    return [inner(itr,i) for i,itr in enumerate(tee(iterables,len(it)))]

def chnmap(gen,seq):
    chns = starzip(seq) # returns a list of generators (one for each channel)
    gens = map(gen,chns) # generators including transformation
    return izip(*gens)  # packing channels to one generator yielding channel tuples

class NSGT_sliced:
    def __init__(self,scale,sl_len,tr_area,fs,min_win=16,Qvar=1,real=False,recwnd=False,matrixform=False,reducedform=0,multichannel=False,measurefft=False):
        assert fs > 0
        assert sl_len > 0
        assert tr_area >= 0
        assert sl_len > tr_area*2
        assert min_win > 0
        assert 0 <= reducedform <= 2

        assert sl_len%2 == 0

        self.sl_len = sl_len
        self.tr_area = tr_area
        self.fs = fs
        self.real = real
        self.measurefft = measurefft
        self.userecwnd = recwnd
        self.reducedform = reducedform

        self.scale = scale
        self.frqs,self.q = self.scale()

        self.g,self.rfbas,self.M = nsgfwin(self.frqs,self.q,self.fs,self.sl_len,sliced=True,min_win=min_win,Qvar=Qvar)
        
#        print "rfbas",self.rfbas/float(self.sl_len)*self.fs
        
        if matrixform:
            if self.reducedform:
                rm = self.M[self.reducedform:len(self.M)//2+1-self.reducedform]
                self.M[:] = rm.max()
            else:
                self.M[:] = self.M.max()
                
        if multichannel:
            self.channelize = lambda seq: seq
            self.unchannelize = lambda seq: seq
        else:
            self.channelize = lambda seq: ((it,) for it in seq)
            self.unchannelize = lambda seq: (it[0] for it in seq)

        self.wins,self.nn = calcwinrange(self.g,self.rfbas,self.sl_len)
        
        self.gd = nsdual(self.g,self.wins,self.nn,self.M)
        
        self.fwd = lambda fc: nsgtf_sl(fc,self.g,self.wins,self.nn,self.M,real=self.real,reducedform=self.reducedform,measurefft=self.measurefft)
        self.bwd = lambda cc: nsigtf_sl(cc,self.gd,self.wins,self.nn,self.sl_len,real=self.real,reducedform=self.reducedform,measurefft=self.measurefft)


    def forward(self,sig):
        'transform - s: iterable sequence of sequences' 
        
        sig = self.channelize(sig)

        # Compute the slices (zero-padded Tukey window version)
        f_sliced = slicing(sig,self.sl_len,self.tr_area)

        cseq = chnmap(self.fwd,f_sliced)
    
        cseq = arrange(cseq,self.M,True)
        
        cseq = self.unchannelize(cseq)
        
        return cseq


    def backward(self,cseq):
        'inverse transform - c: iterable sequence of coefficients'
                
        cseq = self.channelize(cseq)
        
        cseq = arrange(cseq,self.M,False)

        frec_sliced = chnmap(self.bwd,cseq)
        
        # Glue the parts back together
        ftype = float if self.real else complex
        sig = unslicing(frec_sliced,self.sl_len,self.tr_area,dtype=ftype,usewindow=self.userecwnd)
        
        sig = self.unchannelize(sig)
        
        # discard first two blocks (padding)
        sig.next()
        sig.next()
        return sig


class CQ_NSGT_sliced(NSGT_sliced):
    def __init__(self,fmin,fmax,bins,sl_len,tr_area,fs,min_win=16,Qvar=1,real=False,recwnd=False,matrixform=False,reducedform=0,multichannel=False,measurefft=False):
        assert fmin > 0
        assert fmax > fmin
        assert bins > 0

        self.fmin = fmin
        self.fmax = fmax
        self.bins = bins  # bins per octave

        scale = OctScale(fmin,fmax,bins)
        NSGT_sliced.__init__(self,scale,sl_len,tr_area,fs,min_win,Qvar,real,recwnd,matrixform,reducedform,multichannel,measurefft)

import unittest
norm = lambda x: N.sqrt(N.mean(N.abs(N.square(x))))

class TestNSGT_slices(unittest.TestCase):

    def setUp(self):
        pass

    def runit(self,siglen,fmin,fmax,obins,sllen,trlen,real):
#        print siglen,fmin,fmax,obins,sllen,trlen,real
        
#        N.random.seed(0)
        sig = N.random.random((siglen,))
        scale = OctScale(fmin,fmax,obins)
        nsgt = NSGT_sliced(scale,fs=44100,sl_len=sllen,tr_area=trlen,real=real)
        c = nsgt.forward((sig,))

        rc = nsgt.backward(c)
        
        s_r = N.concatenate(map(list,rc))[:len(sig)]
        rec_err = norm(sig-s_r)
        rec_err_n = rec_err/norm(sig)
#        print "err abs",rec_err,"norm",rec_err_n
        
        if rec_err_n > 1.e-7:
            import pylab as P
#            P.plot(N.abs(sig-s_r))
#            P.show()
        
        self.assertAlmostEqual(rec_err,0)
        
    def test_1d1(self):
        self.runit(*map(int,"100000 100 18200 2 20000 5000 1".split())) # fail
        
    def test_1d11(self):
        self.runit(*map(int,"100000 80 18200 6 20000 5000 1".split())) # success
        
    def test_1(self):
        self.runit(*map(int,"100000 99 19895 6 84348 5928 1".split()))  # fail
        
    def test_1a(self):
        self.runit(*map(int,"100000 99 19895 6 84348 5928 0".split()))  # success
        
    def test_1b(self):
        self.runit(*map(int,"100000 100 20000 6 80000 5000 1".split()))  # fail
        
    def test_1c(self):
        self.runit(*map(int,"100000 100 19000 6 80000 5000 1".split())) # fail
        
    def test_1d2(self):
        self.runit(*map(int,"100000 100 18100 6 20000 5000 1".split())) # success
        
    def test_1e(self):
        self.runit(*map(int,"100000 100 18000 6 20000 5000 1".split())) # success
        
    def gtest_oct(self):
        for _ in xrange(100):
            siglen = 100000
            fmin = N.random.randint(200)+30
            fmax = N.random.randint(22048-fmin)+fmin
            obins = N.random.randint(24)+1
            sllen = max(1,N.random.randint(50000))*2
            trlen = max(2,N.random.randint(sllen//2-2))//2*2
            real = N.random.randint(2)
            self.runit(siglen, fmin, fmax, obins, sllen, trlen, real)

if __name__ == '__main__':
    unittest.main()

