'''
Created on 06.11.2011

@author: thomas
'''

import numpy as N

def reblock(sseq,blocksize,dtype=None,fulllast=True,padding=0):
    block = None
    dt = None
    
    for si in sseq:
        # iterate through sequence of sequences
        
        while True:
            if block is None:
                if dt is None:
                    # output dtype still undefined
                    if dtype is None:
                        dt = type(si[0]) # take type is first input element
                    else:
                        dt = dtype
                
                block = N.empty(blocksize,dtype=dt)
                blockrem = block
                
            sout = si[:len(blockrem)]
            avail = len(sout)
            blockrem[:avail] = sout # copy data
            si = si[avail:]  # move ahead in input block
            blockrem = blockrem[avail:]  # move ahead in output block
            
            if len(blockrem) == 0:
                # output block is full
                yield block
                block = None
            if len(si) == 0:
                # input block is exhausted
                break
            
    if block is not None:
        if fulllast:
            blockrem[:] = padding  # zero padding
            yield block
        else:
            # output only filled part
            yield block[:-len(blockrem)]


if __name__ == '__main__':
    inblk = 17
    outblk = 13
    inp = (range(i*inblk,(i+1)*inblk) for i in xrange(10))
    for o in reblock(inp,outblk,dtype=None,fulllast=True,padding=-1):
        print len(o),o
