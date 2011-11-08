'''
Created on 06.11.2011

@author: thomas
'''

import numpy as N
from nsigtf_sl import nsigtf_sl
from unslicing import unslicing
from nsdual import nsdual
from itertools import imap

def isliCQ(c,g,shift,M,sl_len,tr_area,blocks=0):
    gd = nsdual(g,shift,M)
                  
    # Blockwise perfect reconstruction (no changes necessary)
    # ('real-time' approach, reconstruction perfect up to num. prec)
    
    ixs = (
           [N.hstack((N.arange(mkk/4,mkk),N.arange(mkk/4))) for mkk in M],  # even
           [N.hstack((N.arange(3*mkk/4,mkk),N.arange(3*mkk/4))) for mkk in M]  # odd
    )
    
    cseq = ([ci[kk][ixs[i%2][kk]] for kk in xrange(len(gd))] for i,(ci,_) in enumerate(c))
    
    # Slight variation of nsigtf
    frec_sliced = nsigtf_sl(cseq,gd,shift,sl_len)
    
    frec_sliced = imap(N.real,frec_sliced)
    
    # Glue the parts back together
    f_rec = unslicing(frec_sliced,sl_len)
    
    # discard first two blocks (padding)
    f_rec.next()
    f_rec.next()
    return f_rec
