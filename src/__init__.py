'''
Created on 23.03.2011

@author: thomas
'''

from nsgtf import nsgtf,nsigtf
from nsdual import nsdual
from nsgfwin import nsgfwin

def calcshift(a,Ls):
    return N.concatenate(((N.mod(-a[-1],Ls),), a[1:]-a[:-1]))

if __name__ == "__main__":
    import numpy as N
    from scikits.audiolab import Sndfile

    # Testing
    fmin = 130
    fmax = 22050
    bins = 12
    
    fn = '/Users/thomas/Documents/annotation/cqt/glock.wav'
    sf = Sndfile(fn)
    fs = sf.samplerate
    s = sf.read_frames(sf.nframes)
    if len(s.shape) > 1: s = N.mean(s,axis=1)
    
    Ls = len(s)

    g,a,M = nsgfwin(fmin,fmax,bins,fs,Ls)
    
    shift = calcshift(a,Ls)

    gd = nsdual(g,shift,M)
    
    c = nsgtf(s,g,shift,M)
    
    s_r = nsigtf(c,gd,shift,Ls)

    norm = lambda x: N.sqrt(N.sum(N.square(x)))
    rec_err = norm(s-s_r)/norm(s)
    print "Reconstruction error",rec_err
