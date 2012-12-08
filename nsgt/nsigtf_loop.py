'''
Created on 22.11.2012

@author: thomas
'''

import numpy as N
from itertools import izip

def nsigtf_loop(loopparams,fr,fc):
    fr[:] = 0.
    # The overlap-add procedure including multiplication with the synthesis windows
    # TODO: stuff loop into theano
    for t,(gdii,wr1,wr2,sl1,sl2,temp) in izip(fc,loopparams):
        t1 = temp[sl1]
        t2 = temp[sl2]
        t1[:] = t[sl1]
        t2[:] = t[sl2]
        temp *= gdii
        temp *= len(t)

        fr[wr1] += t2
        fr[wr2] += t1

    return fr
