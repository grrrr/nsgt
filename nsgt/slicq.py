'''
Created on 05.11.2011

@author: Thomas Grill (grrrr.org)

% Perfect reconstruction sliCQ

% right now, even slice length (sl_len) is required. Parameters are the
% same as NSGTF plus slice length, minimal required window length, 
% Q-factor variation, and test run parameters.
'''

from nsgfwin_sl import nsgfwin_sl
from slicing import slicing
from nsgtf_sl import nsgtf_sl

def sliCQ(f,fmin,fmax,bins,sl_len,tr_area,sr,min_win=16,Qvar=1,M=None):

    assert sl_len%2 == 0

    # This is just a slightly modified version of nsgfwin
    g,shift,m = nsgfwin_sl(fmin,fmax,bins,sr,sl_len,min_win,Qvar)
    if M is None:
        M = m
    
    # Compute the slices (zero-padded Tukey window version)
    f_sliced = slicing(f,sl_len,tr_area)
    
    # Slightly modified nsgtf (perfect reconstruction version)
    c = (nsgtf_sl(sl,g,shift,M) for sl in f_sliced)
    
    return c,g,shift,M,sl_len,tr_area
