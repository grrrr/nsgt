'''
Created on 06.11.2011

@author: thomas
'''

import numpy as N
from itertools import izip

def reblock(sseq,blocksize,dtype=None,fulllast=True,padding=0,multichannel=False):
    block = None
    dt = None
    chns = None
    
    if multichannel:
        channelize = lambda s: s
        unchannelize = lambda s: s
    else:
        channelize = lambda s: (s,)
        unchannelize = lambda s: s[0]

    for si in sseq:
        # iterate through sequence of sequences

        si = channelize(si)
        
        while True:
            if block is None:
                if dt is None:
                    # output dtype still undefined
                    if dtype is None:
                        dt = type(si[0][0]) # take type is first input element
                    else:
                        dt = dtype
                chns = len(si)

                block = N.empty((chns,blocksize),dtype=dt)
                blockrem = block
                
            sout = [sj[:blockrem.shape[1]] for sj in si]
            avail = len(sout[0])
            for blr,souti in izip(blockrem,sout):
                blr[:avail] = souti # copy data per channel
            si = [sj[avail:] for sj in si]  # move ahead in input block
            blockrem = blockrem[:,avail:]  # move ahead in output block
            
            if blockrem.shape[1] == 0:
                # output block is full
                yield unchannelize(block)
                block = None
            if len(si[0]) == 0:
                # input block is exhausted
                break
            
    if block is not None:
        if fulllast:
            blockrem[:] = padding  # zero padding
            ret = block
        else:
            # output only filled part
            ret = block[:,:-len(blockrem[0])]
        yield unchannelize(ret)


if __name__ == '__main__':
    inblk = 17
    outblk = 13
    inp = (range(i*inblk,(i+1)*inblk) for i in xrange(10))
    for o in reblock(inp,outblk,dtype=None,fulllast=True,padding=-1):
        print len(o),o
