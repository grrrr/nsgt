import torch
import copy
import math
from torch import Tensor
import numpy as np
from itertools import cycle, chain, tee
from math import ceil
from typing import Optional, List
from .slicing import slicing
from .unslicing import unslicing
from .nsdual import nsdual
from .nsgfwin_sl import nsgfwin
from .nsgtf import nsgtf_sl
from .nsigtf import nsigtf_sl
from .util import calcwinrange, complex_2_magphase, magphase_2_complex
from .interpolation import ALLOWED_MATRIX_FORMS, interpolate_slicqt, deinterpolate_slicqt
from .reblock import reblock
from .fscale import SCALES_BY_NAME, OctScale


def arrange(cseq, fwd, device="cpu"):
    if type(cseq) == torch.Tensor:
        M = cseq.shape[-1]

        if fwd:
            odd_mid = M//4
            even_mid = 3*M//4
        else:
            odd_mid = 3*M//4
            even_mid = M//4

        # odd indices
        cseq[1::2, :, :, :] = torch.cat((cseq[1::2, :, :, odd_mid:], cseq[1::2, :, :, :odd_mid]), dim=-1)

        # even indices
        cseq[::2, :, :, :] = torch.cat((cseq[::2, :, :, even_mid:], cseq[::2, :, :, :even_mid]), dim=-1)
    elif type(cseq) == list:
        for i, cseq_tsor in enumerate(cseq):
            cseq[i] = arrange(cseq_tsor, fwd, device)
    else:
        raise ValueError(f'unsupported type {type(cseq)}')

    return cseq


def starzip(iterables):
    def inner(itr, i):
        for t in itr:
            yield t[i]
    iterables = iter(iterables)
    it = next(iterables)  # we need that to determine the length of one element
    iterables = chain((it,), iterables)
    return [inner(itr, i) for i,itr in enumerate(tee(iterables, len(it)))]


#@profile
def chnmap_forward(gen, seq, device="cpu"):
    chns = starzip(seq) # returns a list of generators (one for each channel)

    # fuck generators, use a tensor
    chns = [list(x) for x in chns]

    f_slices = torch.empty(len(chns[0]), len(chns), len(chns[0][0]), dtype=torch.float32, device=torch.device(device))

    for i, chn in enumerate(chns):
        for j, sig in enumerate(chn):
            f_slices[j, i, :] = sig

    ret = gen(f_slices)

    return ret


class NSGT_sliced(torch.nn.Module):
    def __init__(self, scale, sl_len, tr_area, fs,
                 min_win=16, Qvar=1,
                 real=False, recwnd=False, matrixform=False, reducedform=0,
                 multichannel=False,
                 dtype=torch.float32,
                 device="cpu"):
        assert fs > 0
        assert sl_len > 0
        assert tr_area >= 0
        assert sl_len > tr_area*2
        assert min_win > 0
        assert 0 <= reducedform <= 2

        assert sl_len%4 == 0
        assert tr_area%2 == 0

        super(NSGT_sliced, self).__init__()

        self.device = torch.device(device)

        self.sl_len = sl_len
        self.tr_area = tr_area
        self.fs = fs
        self.real = real
        self.userecwnd = recwnd
        self.reducedform = reducedform
        self.multichannel = multichannel

        self.scale = scale
        self.frqs,self.q = self.scale()

        self.g,self.rfbas,self.M = nsgfwin(self.frqs, self.q, self.fs, self.sl_len, sliced=True, min_win=min_win, Qvar=Qvar, dtype=dtype, device=self.device)

        if real:
            assert 0 <= reducedform <= 2
            sl = slice(reducedform,len(self.g)//2+1-reducedform)
        else:
            sl = slice(0,None)

        self.fbins_actual = sl.stop

        # coefficients per slice
        self.ncoefs = max(int(ceil(float(len(gii))/mii))*mii for mii,gii in zip(self.M[sl],self.g[sl]))

        self.matrixform = matrixform
        
        if self.matrixform:
            if self.reducedform:
                rm = self.M[self.reducedform:len(self.M)//2+1-self.reducedform]
                self.M[:] = rm.max()
            else:
                self.M[:] = self.M.max()

        if multichannel:
            self.channelize = lambda seq: seq
            self.unchannelize = lambda seq: seq
        else:
            self.channelize = lambda seq: ((it,) for it in seq)
            self.unchannelize = lambda seq: (it[0] for it in seq)

        self.wins,self.nn = calcwinrange(self.g, self.rfbas, self.sl_len, device=self.device)
        
        self.gd = nsdual(self.g, self.wins, self.nn, self.M, device=self.device)
        self.setup_lambdas()
        
    def setup_lambdas(self):
        self.fwd = lambda fc: nsgtf_sl(fc, self.g, self.wins, self.nn, self.M, real=self.real, reducedform=self.reducedform, matrixform=self.matrixform, device=self.device)
        self.bwd = lambda cc: nsigtf_sl(cc, self.gd, self.wins, self.nn, self.sl_len ,real=self.real, reducedform=self.reducedform, matrixform=self.matrixform, device=self.device)

    def _apply(self, fn):
        super(NSGT_sliced, self)._apply(fn)
        self.wins = [fn(w) for w in self.wins]
        self.g = [fn(g) for g in self.g]
        self.device = self.g[0].device
        self.setup_lambdas()

    @property
    def coef_factor(self):
        return float(self.ncoefs)/self.sl_len
    
    @property
    def slice_coefs(self):
        return self.ncoefs
    
    #@profile
    def forward(self, sig):
        'transform - s: iterable sequence of sequences' 

        sig = self.channelize(sig)

        # Compute the slices (zero-padded Tukey window version)
        f_sliced = slicing(sig, self.sl_len, self.tr_area, device=self.device)

        cseq = chnmap_forward(self.fwd, f_sliced, device=self.device)

        cseq = arrange(cseq, True, device=self.device)
    
        cseq = self.unchannelize(cseq)

        return cseq

    #@profile
    def backward(self, cseq, length):
        'inverse transform - c: iterable sequence of coefficients'
        cseq = self.channelize(cseq)

        cseq = arrange(cseq, False, device=self.device)

        frec_sliced = self.bwd(cseq)

        # Glue the parts back together
        ftype = float if self.real else complex
        sig = unslicing(frec_sliced, self.sl_len, self.tr_area, dtype=ftype, usewindow=self.userecwnd, device=self.device)

        sig = list(self.unchannelize(sig))[2:]

        # convert to tensor
        ret = next(reblock(sig, length, fulllast=False, multichannel=self.multichannel, device=self.device))

        return ret


class CQ_NSGT_sliced(NSGT_sliced):
    def __init__(self, fmin, fmax, bins, sl_len, tr_area, fs, min_win=16, Qvar=1, real=False, recwnd=False, matrixform=False, reducedform=0, multichannel=False):
        assert fmin > 0
        assert fmax > fmin
        assert bins > 0

        self.fmin = fmin
        self.fmax = fmax
        self.bins = bins  # bins per octave

        scale = OctScale(fmin, fmax, bins)
        NSGT_sliced.__init__(self, scale, sl_len, tr_area, fs, min_win, Qvar, real, recwnd, matrixform=matrixform, reducedform=reducedform, multichannel=multichannel)


def make_slicqt_filterbanks(slicqt_base, sample_rate=44100.0):
    if sample_rate != 44100.0:
        raise ValueError('i was lazy and harcoded a lot of 44100.0, forgive me')

    encoder = TorchSliCQT(slicqt_base)
    decoder = TorchISliCQT(slicqt_base)

    return encoder, decoder


class SliCQTBase(torch.nn.Module):
    def __init__(self,
        scale, fbins, fmin, fmax=22050, gamma=25.,
        sllen=None, trlen=None,
        matrixform='ragged',
        per_slice=False,
        fs=44100, device="cpu"
    ):
        super(SliCQTBase, self).__init__()

        self.fbins = fbins
        self.fmin = fmin
        self.gamma = gamma
        self.fmax = fmax

        scl_fn = None
        self.scl = None
        scl_args = None
        try:
            scl_fn = SCALES_BY_NAME[scale]
        except KeyError:
            msg = f'unsupported frequency scale {scale}'
            if scale == 'oct':
                msg += '\n\tuse `cqlog` instead of `oct`'
            raise ValueError(msg)

        if scale == 'vqlog':
            scl_args = (self.fmin, self.fmax, self.fbins, self.gamma)
        else:
            scl_args = (self.fmin, self.fmax, self.fbins)

        self.scl = scl_fn(*scl_args)
        self.device = device

        self.sllen, self.trlen = self.scl.suggested_sllen_trlen(fs)

        if sllen is not None and trlen is not None:
            print(f'using user-supplied sllen, trlen: {sllen}, {trlen}')
            self.sllen, self.trlen = sllen, trlen
        else:
            print(f'using suggested sllen, trlen: {self.sllen}, {self.trlen}')

        self.fs = fs

        if matrixform not in ALLOWED_MATRIX_FORMS:
            raise ValueError(f'{matrixform} is not one of the allowed values: {ALLOWED_MATRIX_FORMS}')
        self.matrixform = matrixform

        self.nsgt = NSGT_sliced(self.scl, self.sllen, self.trlen, fs, real=True, multichannel=True, matrixform=(matrixform=='zeropad'), device=self.device)

        self.M = self.nsgt.ncoefs
        self.fbins_actual = self.nsgt.fbins_actual

    def max_bins(self, bandwidth): # convert hz bandwidth into bins
        if bandwidth is None:
            return None
        freqs, _ = self.scl()
        max_bin = min(np.argwhere(freqs > bandwidth))[0]
        return max_bin+1

    def _apply(self, fn):
        self.nsgt._apply(fn)
        return self


class TorchSliCQT(torch.nn.Module):
    def __init__(self, nsgt):
        super(TorchSliCQT, self).__init__()
        self.nsgt = nsgt

    def _apply(self, fn):
        self.nsgt._apply(fn)
        return self

    def forward(self, x):
        shape = x.size()
        nb_samples, nb_channels, nb_timesteps = shape

        # pack batch
        x = x.view(-1, shape[-1])

        C = self.nsgt.nsgt.forward((x,))

        for i, nsgt_f in enumerate(C):
            nsgt_f = torch.moveaxis(nsgt_f, 0, -2)
            nsgt_f = torch.view_as_real(nsgt_f)
            # unpack batch
            nsgt_f = nsgt_f.view(shape[:-1] + nsgt_f.shape[-4:])
            C[i] = nsgt_f

        if self.nsgt.matrixform == 'ragged':
            return C, None
        elif self.nsgt.matrixform == 'zeropad':
            return C[0], None
        else:
            Cmag, Cphase = complex_2_magphase(C)
            Cmag, ragged_shapes = interpolate_slicqt(Cmag)
            Cphase, ragged_shapes = interpolate_slicqt(Cphase)
            C_interp = magphase_2_complex(Cmag, Cphase)
            return C_interp, ragged_shapes

    @torch.no_grad()
    def overlap_add(self, slicq, dont=False):
        if dont:
            return slicq

        if type(slicq) == list:
            ret = [None]*len(slicq)
            for i, slicq_ in enumerate(slicq):
                ret[i] = self.overlap_add(slicq_)
            return ret

        nb_samples, nb_channels, nb_f_bins, nb_slices, nb_m_bins = slicq.shape

        nwin = nb_m_bins

        ncoefs = int(math.floor((1+nb_slices)*nb_m_bins)/2)

        hop = nwin//2 # 50% overlap window

        out = torch.zeros((nb_samples, nb_channels, nb_f_bins, ncoefs), dtype=slicq.dtype, device=slicq.device)

        ptr = 0

        for i in range(nb_slices):
            # weighted overlap-add with last `hop` samples
            # implicit rectangular window
            out[:, :, :, ptr:ptr+nwin] += slicq[:, :, :, i, :]
            ptr += hop

        return out


class TorchISliCQT(torch.nn.Module):
    def __init__(self, nsgt):
        super(TorchISliCQT, self).__init__()
        self.nsgt = nsgt

    def _apply(self, fn):
        self.nsgt._apply(fn)
        return self

    def forward(self, Xorig, length: int, ragged_shapes: Optional[List[int]] = None) -> Tensor:
        X = copy.deepcopy(Xorig)

        if self.nsgt.matrixform == 'interpolate':
            Xmag, Xphase = complex_2_magphase(X)
            Xmag = deinterpolate_slicqt(Xmag, ragged_shapes)
            Xphase = deinterpolate_slicqt(Xphase, ragged_shapes)
            X = magphase_2_complex(Xmag, Xphase)

        if type(X) == Tensor:
            X_list = [X]
        else:
            X_list = X

        X_complex = [None]*len(X_list)
        for i, X in enumerate(X_list):
            Xdims = len(X.shape)

            X = torch.view_as_complex(X)

            shape = X.shape

            if Xdims == 6:
                X = X.view(X.shape[0]*X.shape[1], *X.shape[2:])
            else:
                X = X.view(X.shape[0]*X.shape[1]*X.shape[2], *X.shape[3:])

            # moveaxis back into into T x [packed-channels] x F1 x F2
            X = torch.moveaxis(X, -2, 0)

            X_complex[i] = X

        if self.nsgt.matrixform == 'zeropad':
            y = self.nsgt.nsgt.backward(X_complex[0], length)
        else:
            y = self.nsgt.nsgt.backward(X_complex, length)

        # unpack batch
        y = y.view(*shape[:-3], -1)

        return y
