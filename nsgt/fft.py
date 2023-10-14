# -*- coding: utf-8

"""
Python implementation of Non-Stationary Gabor Transform (NSGT)
derived from MATLAB code by NUHAG, University of Vienna, Austria

Thomas Grill, 2011-2015
http://grrrr.org/nsgt

Austrian Research Institute for Artificial Intelligence (OFAI)
AudioMiner project, supported by Vienna Science and Technology Fund (WWTF)
"""

import atexit
import numpy as np
import os.path
import pickle
from threading import Timer
from warnings import warn

# Try engines in order of:
# PyFFTW3
# pyFFTW
# numpy.fftpack
try:
    import fftw3, fftw3f
except ImportError:
    fftw3 = None
    fftw3f = None
    try:
        import pyfftw
    except ImportError:
        pyfftw = None


if fftw3 is not None and fftw3f is not None:
    ENGINE = "PYFFTW3"
    # Use fftw3 methods
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
elif pyfftw is not None:
    ENGINE = "PYFFTW"
    # Monkey patch in pyFFTW Numpy interface
    np.fft = pyfftw.interfaces.numpy_fft
    original_fft = np.fft
    # Enable cache to keep wisdom, etc.
    pyfftw.interfaces.cache.enable()

    # Load stored wisdom
    PYFFTW_WISDOM_FILENAME = os.path.join(os.path.expanduser("~"), ".nsgt_pyfftw_wisdom.p")
    if os.path.isfile(PYFFTW_WISDOM_FILENAME):
        with open(PYFFTW_WISDOM_FILENAME, 'rb') as f:
            pyfftw.import_wisdom(pickle.load(f))
            print("Loaded pyFFTW wisdom from %s" % PYFFTW_WISDOM_FILENAME)

    def save_wisdom():
        print("Saving pyFFTW wisdom to %s" % PYFFTW_WISDOM_FILENAME)
        with open(PYFFTW_WISDOM_FILENAME, 'wb') as f:
            pickle.dump(pyfftw.export_wisdom(), f)

    # Save wisdom on exit
    atexit.register(save_wisdom)
else:
    # fall back to numpy methods
    warn("nsgt.fft falling back to numpy.fft")
    ENGINE = "NUMPY"

if ENGINE in ["PYFFTW", "NUMPY"]:
    def get_kwargs(measure):
        return ({'planner_effort': 'FFTW_MEASURE' if measure else 'FFTW_ESTIMATE'}
                if ENGINE=="PYFFTW" else {})
    class fftp:
        def __init__(self, measure=False, dtype=float):
            self.kwargs = get_kwargs(measure)
        def __call__(self,x, outn=None, ref=False):
            return np.fft.fft(x, **self.kwargs)
    class ifftp:
        def __init__(self, measure=False, dtype=float):
            self.kwargs = get_kwargs(measure)
        def __call__(self,x, outn=None, n=None, ref=False):
            return np.fft.ifft(x,n=n,**self.kwargs)
    class rfftp:
        def __init__(self, measure=False, dtype=float):
            self.kwargs = get_kwargs(measure)
        def __call__(self,x, outn=None, ref=False):
            return np.fft.rfft(x,**self.kwargs)
    class irfftp:
        def __init__(self, measure=False, dtype=float):
            self.kwargs = get_kwargs(measure)
        def __call__(self,x,outn=None,ref=False):
            return np.fft.irfft(x,n=outn,**self.kwargs)
