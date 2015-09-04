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
from warnings import warn

realized = False
        
if not realized:
    # try to use FFT3 if available, else use numpy.fftpack
    try:
        import fftw3
    except ImportError:
        fftw3 = None
    
    try:
        import fftw3f
    except ImportError:
        fftw3f = None

    if fftw3 is not None:
        # fftw3 methods
        class fftpool:
            def __init__(self, measure, dtype=float):
                self.measure = measure
                self.dtype = np.dtype(dtype)
                dtsz = self.dtype.itemsize
                if dtsz == 4:
                    self.tpfloat = np.float32
                    self.tpcplx = np.complex64
                    self.fftw = fftw3f
                elif dtsz == 8:
                    self.tpfloat = np.float64
                    self.tpcplx = np.complex128
                    self.fftw = fftw3
                else:
                    raise TypeError("nsgt.fftpool: dtype '%s' not supported"%repr(self.dtype))
                self.pool = {}
                
            def __call__(self, x, outn=None, ref=False):
                lx = len(x)
                try:
                    transform = self.pool[lx]
                except KeyError:
                    transform = self.init(lx, measure=self.measure, outn=outn)
                    self.pool[lx] = transform
                plan,pre,post = transform
                if pre is not None:
                    x = pre(x)
                plan.inarray[:] = x
                plan()
                if not ref:
                    tx = plan.outarray.copy()
                else:
                    tx = plan.outarray
                if post is not None:
                    tx = post(tx)
                return tx
    
        class fftp(fftpool):
            def __init__(self, measure=False, dtype=float):
                fftpool.__init__(self, measure, dtype=dtype)
            def init(self, n, measure, outn):
                inp = self.fftw.create_aligned_array(n, dtype=self.tpcplx)
                outp = self.fftw.create_aligned_array(n, dtype=self.tpcplx)
                plan = self.fftw.Plan(inp, outp, direction='forward', flags=('measure' if measure else 'estimate',))
                return (plan,None,None)
    
        class rfftp(fftpool):
            def __init__(self, measure=False, dtype=float):
                fftpool.__init__(self, measure, dtype=dtype)
            def init(self, n, measure, outn):
                inp = self.fftw.create_aligned_array(n, dtype=self.tpfloat)
                outp = self.fftw.create_aligned_array(n//2+1, dtype=self.tpcplx)
                plan = self.fftw.Plan(inp, outp, direction='forward', realtypes='halfcomplex r2c',flags=('measure' if measure else 'estimate',))
                return (plan,None,None)
    
        class ifftp(fftpool):
            def __init__(self, measure=False, dtype=float):
                fftpool.__init__(self, measure, dtype=dtype)
            def init(self, n, measure, outn):
                inp = self.fftw.create_aligned_array(n, dtype=self.tpcplx)
                outp = self.fftw.create_aligned_array(n, dtype=self.tpcplx)
                plan = self.fftw.Plan(inp, outp, direction='backward', flags=('measure' if measure else 'estimate',))
                return (plan,None,lambda x: x/len(x))
    
        class irfftp(fftpool):
            def __init__(self, measure=False, dtype=float):
                fftpool.__init__(self, measure, dtype=dtype)
            def init(self, n, measure, outn):
                inp = self.fftw.create_aligned_array(n, dtype=self.tpcplx)
                outp = self.fftw.create_aligned_array(outn if outn is not None else (n-1)*2, dtype=self.tpfloat)
                plan = self.fftw.Plan(inp, outp, direction='backward', realtypes='halfcomplex c2r', flags=('measure' if measure else 'estimate',))
                return (plan,lambda x: x[:n],lambda x: x/len(x))
            
        realized = True
        

if not realized:
    # fall back to numpy methods
    warn("nsgt.fft falling back to numpy.fft")
    
    class fftp:
        def __init__(self, measure=False, dtype=float):
            pass
        def __call__(self,x, outn=None, ref=False):
            return np.fft.fft(x)
    class ifftp:
        def __init__(self, measure=False, dtype=float):
            pass
        def __call__(self,x, outn=None, n=None, ref=False):
            return np.fft.ifft(x,n=n)
    class rfftp:
        def __init__(self, measure=False, dtype=float):
            pass
        def __call__(self,x, outn=None, ref=False):
            return np.fft.rfft(x)
    class irfftp:
        def __init__(self, measure=False, dtype=float):
            pass
        def __call__(self,x,outn=None,ref=False):
            return np.fft.irfft(x,n=outn)
