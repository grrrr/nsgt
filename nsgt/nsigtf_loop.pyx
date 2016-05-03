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
cimport numpy as np

def nsigtf_loop(loopparams, np.ndarray fr not None, fc):
    fr[:] = 0.
    # The overlap-add procedure including multiplication with the synthesis windows
    # TODO: stuff loop into theano
    cdef np.ndarray gdii,t,temp,t1,t2,wr1,wr2
    cdef slice sl1,sl2
    for t,(gdii,wr1,wr2,sl1,sl2,temp) in zip(fc, loopparams):
        t1 = temp[sl1]
        t2 = temp[sl2]
        t1[:] = t[sl1]
        t2[:] = t[sl2]
        temp *= gdii
        temp *= len(t)

        fr[wr1] += t2
        fr[wr2] += t1

    return fr
