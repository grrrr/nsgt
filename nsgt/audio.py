'''
Created on 11.01.2012

@author: thomas
'''

import numpy as N
from scikits.audiolab import Sndfile,Format
try:
    from mad import MadFile
except ImportError:
    MadFile = None

def sndreader(sf,blksz=2**16,maxsz=-1):
    if blksz < 0:
        blksz = sf.nframes
    if sf.channels > 1: 
        channels = lambda s: s.T
    else:
        channels = lambda s: s.reshape((1,-1))
    if maxsz < 0:
        maxsz = sf.nframes
    for offs in xrange(0,sf.nframes,blksz):
        yield channels(sf.read_frames(min(sf.nframes-offs,blksz)))

def mp3reader(sf):
    while True:
        b = sf.read()
        if b is None:
            break
        yield N.frombuffer(b,'2h').T
    
def sndwriter(sf,blkseq,maxframes=None):
    written = 0
    for b in blkseq:
        b = b.T
        if maxframes is not None: 
            b = b[:maxframes-written]
        sf.write_frames(b)
        written += len(b)

class SndReader:
    def __init__(self,fn,blksz=2**16):
        if MadFile is not None and fn.lower().endswith('.mp3'):
            sf = MadFile(fn)
            self.channels = 2
            self.samplerate = sf.samplerate()
            self.frames = int(sf.total_time()*0.001*self.samplerate)
            self.rdr = mp3reader(sf)
        else:
            sf = Sndfile(fn)
            self.channels = sf.channels
            self.samplerate = sf.samplerate
            self.frames = sf.nframes
            self.rdr = sndreader(sf,blksz)
        
    def __call__(self):
        return self.rdr

class SndWriter(Sndfile):
    def __init__(self,fn,samplerate,filefmt='wav',datafmt='pcm16',channels=1):
        fmt = Format(filefmt,datafmt)
        Sndfile.__init__(self,fn,mode='w',format=fmt,channels=channels,samplerate=samplerate)
        
    def __call__(self,sigblks,maxframes=None):
        sndwriter(self,sigblks,maxframes=None)

