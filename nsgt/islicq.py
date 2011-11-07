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
    
    """
    if blocks == 1:
        # Preparation for blockwise approximate reconstruction (For test
        # purposes)
    
        if max(M)==min(M): # Matrix verison to do!!!!

            slices = size(c,3) 
            X = []
    
            for xx = 1:slices:
                X = [X;c(:,:,xx)]

        else:
    
            slices = len(c)
            X = cell(1,size(c,2));
    
            for kk = 1:size(c,2):
                mkk = M[kk]
                temp = N.zeros(slices*mkk/2,dtype=complex)
                temp([end-mkk/4+1:end,1:3*mkk/4]) = c{1,kk};
                for xx in xrange(slices-1):
                    temp(xx*mkk/2+[-mkk/4+1:3*mkk/4]) = temp(xx*mkk/2+[-mkk/4+1:3*mkk/4]) + c{xx+1,kk}
                temp([end-3*mkk/4+1:end,1:mkk/4]) = temp([end-3*mkk/4+1:end,1:mkk/4]) + c{slices,kk};
                X{kk} = temp;
            end

    
        # 'Reconstruct' sliced coefficients
        for kk = 1:size(c,2)
            mkk = M[kk]
            temp = X{kk}
            tw = cont_tukey_win(mkk,sl_len,tr_area);
            c{1,kk} = temp([end-mkk/4+1:end,1:3*mkk/4]).*tw;
            for xx in xrange(slices-1):
                c{xx+1,kk} = temp(xx*mkk/2+[-mkk/4+1:3*mkk/4]).*tw;
            c{slices,kk} = temp([end-3*mkk/4+1:end,1:mkk/4]).*tw;
    """
    
    gd = nsdual(g,shift,M)
                  
    # Blockwise perfect reconstruction (no changes necessary)
    # ('real-time' approach, reconstruction perfect up to num. prec)
    
    ixs = (
           [N.hstack((N.arange(mkk/4,mkk),N.arange(mkk/4))) for mkk in M],  # even
           [N.hstack((N.arange(3*mkk/4,mkk),N.arange(3*mkk/4))) for mkk in M]  # odd
    )
    
    cseq = ([ci[kk][ixs[i%2][kk]] for kk in xrange(len(gd))] for i,(ci,_) in enumerate(c))
    
    # Slight variation of nsigtf
    frec_sliced = (nsigtf_sl(ci,gd,shift,sl_len) for ci in cseq)
    
    frec_sliced = imap(N.real,frec_sliced)
    
    # Glue the parts back together
    f_rec = unslicing(frec_sliced,sl_len)
    
    # discard first block (padding)
    f_rec.next()
    
    return f_rec
