'''
Created on 11.01.2012

@author: thomas
'''

from scikits.audiolab import Sndfile,Format

def sndreader(sf,blksz=2**16):
    if blksz < 0:
        blksz = sf.nframes
    if sf.channels > 1: 
        channels = lambda s: s.T
    else:
        channels = lambda s: s.reshape((1,-1))
    for offs in xrange(0,sf.nframes,blksz):
        yield channels(sf.read_frames(min(sf.nframes-offs,blksz)))

def sndwriter(sf,blkseq,maxframes=None):
    written = 0
    for b in blkseq:
        b = b.T
        if maxframes is not None: 
            b = b[:maxframes-written]
        sf.write_frames(b)
        written += len(b)

class SndReader(Sndfile):
    def __init__(self,fn):
        Sndfile.__init__(self,fn)
        
    def __call__(self,blksz=2**16):
        return sndreader(self,blksz)

class SndWriter(Sndfile):
    def __init__(self,fn,samplerate,filefmt='wav',datafmt='pcm16',channels=1):
        fmt = Format(filefmt,datafmt)
        Sndfile.__init__(self,fn,mode='w',format=fmt,channels=channels,samplerate=samplerate)
        
    def __call__(self,sigblks,maxframes=None):
        sndwriter(self,sigblks,maxframes=None)

