# -*- coding: utf-8

"""
Python implementation of Non-Stationary Gabor Transform (NSGT)
derived from MATLAB code by NUHAG, University of Vienna, Austria

Thomas Grill, 2011-2015
http://grrrr.org/nsgt

Austrian Research Institute for Artificial Intelligence (OFAI)
AudioMiner project, supported by Vienna Science and Technology Fund (WWTF)
"""

import numpy as np
from itertools import cycle, chain
from .util import hannwin

def slicequads(frec_sliced, hhop):
    slices = [[slice(hhop*((i+3-k*2)%4),hhop*((i+3-k*2)%4+1)) for i in range(4)] for k in range(2)]
    slices = cycle(slices)
    
    for fsl,sl in zip(frec_sliced, slices):
        yield [[fslc[sli] for fslc in fsl] for sli in sl]


def unslicing(frec_sliced, sl_len, tr_area, dtype=float, usewindow=True):
    hhop = sl_len//4    
    islices = slicequads(frec_sliced, hhop)
    
    if usewindow:
        tr_area2 = min(2*hhop-tr_area, 2*tr_area)
        htr = tr_area//2
        htr2 = tr_area2//2
        hw = hannwin(tr_area2)
        tw = np.zeros(sl_len, dtype=dtype)
        tw[max(hhop-htr-htr2, 0):hhop-htr] = hw[htr2:]
        tw[hhop-htr:3*hhop+htr] = 1
        tw[3*hhop+htr:min(3*hhop+htr+htr2, sl_len)] = hw[:htr2]
        tw = [tw[o:o+hhop] for o in range(0, sl_len, hhop)]
    else:
        tw = cycle((1,))
        
    # get first slice to deduce channels
    firstquad = next(islices)
    
    chns = len(firstquad[0]) # number of channels in first quad
    
    islices = chain((firstquad,), islices)
    
    output = [np.zeros((chns,hhop), dtype=dtype) for _ in range(4)]
    
    for quad in islices:
        for osl,isl,w in zip(output, quad, tw):
            # in a piecewise manner add slices to output stream 
            osl[:] += isl*w
        for _ in range(2):
            # absolutely first two should be padding (and discarded by the receiver)
            yield output.pop(0)
            output.append(np.zeros((chns,hhop), dtype=dtype))

    for _ in range(2):
        # absolutely last two should be padding (and discarded by the receiver)
        yield output.pop(0)

    # two more buffers remaining (and zero)
