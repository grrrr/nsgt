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

def nsgtf_loop(loopparams, ft, temp0):
    c = [] # Initialization of the result
        
    # The actual transform
    # TODO: stuff loop into theano
    for mii,_,gi1,gi2,win_range,Lg,col in loopparams:
#            Lg = len(gii)            
        # if the number of time channels is too small (mii < Lg), aliasing is introduced
        # wrap around and sum up in the end (below)
#            col = int(ceil(float(Lg)/mii)) # normally col == 1                        
#            assert col*mii >= Lg                        

        temp = temp0[:col*mii]

        # original version
#            t = ft[win_range]*N.fft.fftshift(N.conj(gii))
#            temp[:(Lg+1)//2] = t[Lg//2:]  # if mii is odd, this is of length mii-mii//2
#            temp[-(Lg//2):] = t[:Lg//2]  # if mii is odd, this is of length mii//2

        # modified version to avoid superfluous memory allocation
        t1 = temp[:(Lg+1)//2]
        t1[:] = gi1  # if mii is odd, this is of length mii-mii//2
        t2 = temp[-(Lg//2):]
        t2[:] = gi2  # if mii is odd, this is of length mii//2

        ftw = ft[win_range]
        t2 *= ftw[:Lg//2]
        t1 *= ftw[Lg//2:]

#            (wh1a,wh1b),(wh2a,wh2b) = win_range
#            t2[:wh1a.stop-wh1a.start] *= ft[wh1a]
#            t2[wh1a.stop-wh1a.start:] *= ft[wh1b]
#            t1[:wh2a.stop-wh2a.start] *= ft[wh2a]
#            t1[wh2a.stop-wh2a.start:] *= ft[wh2b]
        
        temp[(Lg+1)//2:-(Lg//2)] = 0  # clear gap (if any)
        
        if col > 1:
            temp = np.sum(temp.reshape((mii,-1)), axis=1)
        else:
            temp = temp.copy()

        c.append(temp)
    return c
