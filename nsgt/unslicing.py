'''
Created on 05.11.2011

@author: thomas
'''

import numpy as N
from itertools import izip,cycle

def slicequads(frec_sliced,hhop):
    slices = [[slice(hhop*((i+3-k*2)%4),hhop*((i+3-k*2)%4+1)) for i in xrange(4)] for k in xrange(2)]
    
    for fsl,sl in izip(frec_sliced,cycle(slices)):
#        assert len(fsl) == hhop*4
        yield [fsl[sli] for sli in sl] 


def unslicing(frec_sliced,sl_len,dtype=float):
    hhop = sl_len//4    
    islices = slicequads(frec_sliced,hhop)
    
    output = [N.zeros(hhop,dtype=dtype) for _ in xrange(4)]
    
    for quad in islices:
        for osl,isl in izip(output,quad):
            # in a piecewise manner add slices to output stream 
            osl[:] += isl
        for _ in xrange(2):
            # absolutely first two should be padding (and discarded by the receiver)
            yield output.pop(0)
            output.append(N.zeros(hhop,dtype=dtype))

    for _ in xrange(2):
        # absolutely last two should be padding (and discarded by the receiver)
        yield output.pop(0)

    # two more buffers remaining (and zero)
