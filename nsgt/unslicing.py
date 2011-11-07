'''
Created on 05.11.2011

@author: thomas
'''

import numpy as N
from itertools import izip

def slicequads(frec_sliced,hhop):
    slices = (
            #even: shift +hopsize/2
              (slice(hhop*3,hhop*4),slice(hhop*0,hhop*1),slice(hhop*1,hhop*2),slice(hhop*2,hhop*3)), # even
            # odd: shift -hopsize/2
              (slice(hhop*1,hhop*2),slice(hhop*2,hhop*3),slice(hhop*3,hhop*4),slice(hhop*0,hhop*1)) # odd
    )
    for kk,fsl in enumerate(frec_sliced):
        assert len(fsl) == hhop*4
        yield [fsl[sli] for sli in slices[kk%2]]    


def unslicing(frec_sliced,sl_len,dtype=float):
    hopsize = sl_len//2
    hhop = hopsize//2
    
    frec_sliced = list(frec_sliced)

    islices = slicequads(frec_sliced,hhop)
    
    output = [N.zeros(hhop,dtype=dtype),N.zeros(hhop,dtype=dtype)]
    
    for quad in islices:
        for osl,isl in izip(output,quad):
            # in a piecewise manner add slices to output stream 
            osl[:] += isl
        for _ in xrange(2):
            # absolutely first one should be padding (and discarded by the receiver)
            yield output.pop(0)
            output.append(N.zeros(hhop,dtype=dtype))

    for _ in xrange(2):
        # absolutely last one should be padding (and discarded by the receiver)
        yield output.pop(0)
    assert not output
