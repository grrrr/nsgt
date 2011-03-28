'''
Created on 24.03.2011

@author: thomas
'''

import numpy as N
from math import pi

def hannwin(l):
    r = N.arange(l,dtype=float)
    r *= pi*2./l
    r = N.cos(r)
    r += 1.
    r *= 0.5
    return r

def _isseq(x):
    try:
        len(x)
    except TypeError:
        return False
    return True        

def chkM(M,g):
    if M is None:
        M = N.array(map(len,g))
    elif not _isseq(M):
        M = N.ones(len(g),dtype=int)*M
    return M


