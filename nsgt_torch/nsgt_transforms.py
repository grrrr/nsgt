from typing import Optional, List, Tuple

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.functional import interpolate
import warnings

from .slicq import NSGT
from .fscale import SCALES_BY_NAME


def make_filterbanks(nsgt_base, sample_rate=44100.0):
    if sample_rate != 44100.0:
        raise ValueError('i was lazy and harcoded a lot of 44100.0, forgive me')

    encoder = TorchNSGT(nsgt_base)
    decoder = TorchINSGT(nsgt_base)

    return encoder, decoder


class NSGTBase(nn.Module):
    allowed_matrixforms = ['ragged', 'interpolated', 'zeropad']
    allowed_chunk_strategies = ['no_overlap', 'hann']

    def __init__(self,
        scale, fbins, fmin, N, fmax=22050, gamma=25.,
        matrixform='ragged',
        chunk_strategy='no_overlap', n_overlap=None,
        fs=44100, device="cpu"
    ):
        super(NSGTBase self).__init__()
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
        self.fs = fs

        if matrixform not in NSGTBase.allowed_matrixforms:
            raise ValueError(f'{matrixform} is not one of the allowed values: {NSGTBase.allowed_matrixforms}')

        self.matrixform = matrixform
        self.chunk_strategy = chunk_strategy

        self.nsgt = None
        if self.matrixform == 'zeropad':
            self.nsgt = NSGT(self.scl, self.sllen, self.trlen, fs, N, real=True, multichannel=True, matrixform=True, device=self.device)
        else:
            self.nsgt = NSGT(self.scl, self.sllen, self.trlen, fs, N, real=True, multichannel=True, matrixform=False, device=self.device)

        self.M = self.nsgt.ncoefs
        self.fbins_actual = self.nsgt.fbins_actual

    def max_bins(self, bandwidth): # convert hz bandwidth into bins
        if bandwidth is None:
            return None
        freqs, _ = self.scl()
        max_bin = min(np.argwhere(freqs > bandwidth))[0]
        return max_bin+1

    def predict_input_size(self, batch_size, nb_channels, seq_dur_s):
        fwd = TorchNSGT(self)

        x = torch.rand((batch_size, nb_channels, int(seq_dur_s*self.fs)), dtype=torch.float32)
        shape = x.size()
        nb_samples, nb_channels, nb_timesteps = shape

        nsgt_f = fwd(x)
        return nsgt_f

    def _apply(self, fn):
        self.nsgt._apply(fn)
        return self


class TorchNSGT(nn.Module):
    def __init__(self, nsgt):
        super(TorchNSGT, self).__init__()
        self.nsgt = nsgt

    def _apply(self, fn):
        self.nsgt._apply(fn)
        return self

    def forward(self, x):
        shape = x.size()
        nb_samples, nb_channels, nb_timesteps = shape

        # pack batch
        x = x.view(-1, shape[-1])

        C = self.nsgt.nsgt.forward(x)

        for i, nsgt_f in enumerate(C):
            nsgt_f = torch.view_as_real(nsgt_f)
            # unpack batch
            nsgt_f = nsgt_f.view(shape[:-1] + nsgt_f.shape[-3:])
            C[i] = nsgt_f

        if self.nsgt.matrixform == 'ragged':
            return C

        # do some interpolation here
        return C

    @torch.no_grad()
    def interpolate(self, nsgt: List[Tensor]) -> Tensor:
        nb_samples, nb_channels = slicq[0].shape[:2]
        total_f_bins = sum([slicq_.shape[-2] for slicq_ in slicq])
        max_t_bins = max([slicq_.shape[-1] for slicq_ in slicq])

        interpolated = torch.zeros((nb_samples, nb_channels, total_f_bins, max_t_bins), dtype=slicq[0].dtype, device=slicq[0].device)

        fbin_ptr = 0
        for i, slicq_ in enumerate(slicq):
            nb_samples, nb_channels, nb_f_bins, nb_t_bins = slicq_.shape

            if nb_t_bins == max_t_bins:
                # same time width, no interpolation
                interpolated[:, :, fbin_ptr:fbin_ptr+nb_f_bins, :] = slicq_
            else:
                # repeated interpolation
                interp_factor = max_t_bins//nb_t_bins
                max_assigned = nb_t_bins*interp_factor
                rem = max_t_bins - max_assigned
                interpolated[:, :, fbin_ptr:fbin_ptr+nb_f_bins, : max_assigned] = torch.repeat_interleave(slicq_, interp_factor, dim=-1)
                interpolated[:, :, fbin_ptr:fbin_ptr+nb_f_bins, max_assigned : ] = torch.unsqueeze(slicq_[..., -1], dim=-1).repeat(1, 1, 1, rem)
            fbin_ptr += nb_f_bins
        return interpolated

    @torch.no_grad()
    def deinterpolate(self, interpolated: Tensor, ragged_shapes: List[Tuple[int]]) -> List[Tensor]:
        max_t_bins = interpolated.shape[-1]
        full_slicq = []
        fbin_ptr = 0
        for i, bucket_shape in enumerate(ragged_shapes):
            curr_slicq = torch.zeros(bucket_shape, dtype=interpolated.dtype, device=interpolated.device)

            nb_t_bins = bucket_shape[-1]
            freqs = bucket_shape[-2]

            if bucket_shape[-1] == interpolated.shape[-1]:
                # same time width, no interpolation
                curr_slicq = interpolated[:, :, fbin_ptr:fbin_ptr+freqs, :]
            else:
                # inverse of repeated interpolation
                interp_factor = max_t_bins//nb_t_bins
                select = torch.arange(0, max_t_bins,interp_factor, device=interpolated.device)
                curr_slicq = torch.index_select(interpolated[:, :, fbin_ptr:fbin_ptr+freqs, :], -1, select)

            # crop just in case
            full_slicq.append(curr_slicq[..., : bucket_shape[-1]])

            fbin_ptr += freqs
        return full_slicq


class TorchINSGT(nn.Module):
    def __init__(self, nsgt):
        super(TorchINSGT, self).__init__()
        self.nsgt = nsgt

    def _apply(self, fn):
        self.nsgt._apply(fn)
        return self

    def forward(self, X_list, length: int) -> Tensor:
        X_complex = [None]*len(X_list)
        for i, X in enumerate(X_list):
            Xshape = len(X.shape)

            X = torch.view_as_complex(X)

            shape = X.shape

            if Xshape == 6:
                X = X.view(X.shape[0]*X.shape[1], *X.shape[2:])
            else:
                X = X.view(X.shape[0]*X.shape[1]*X.shape[2], *X.shape[3:])

            # moveaxis back into into T x [packed-channels] x F1 x F2
            X = torch.moveaxis(X, -2, 0)

            X_complex[i] = X

        y = self.nsgt.nsgt.backward(X_complex, length)

        # unpack batch
        y = y.view(*shape[:-3], -1)

        return y


def complex_2_magphase(spec):
    ret_mag = [None]*len(spec)
    ret_phase = [None]*len(spec)

    for i, C_block in enumerate(spec):
        C_block_phase = atan2(C_block[..., 1], C_block[..., 0])
        C_block_mag = torch.pow(torch.abs(torch.view_as_complex(C_block)), 1)

        ret_mag[i] = C_block_mag
        ret_phase[i] = C_block_phase

    return ret_mag, ret_phase


def magphase_2_complex(C_mag, C_phase):
    C_cplx = [None]*len(C_mag)

    for i, (C_mag_block, C_phase_block) in enumerate(zip(C_mag, C_phase)):
        C_cplx_block = torch.empty(*(*C_mag_block.shape, 2), dtype=C_mag_block.dtype, device=C_mag_block.device)

        C_cplx_block[..., 0] = C_mag_block * torch.cos(C_phase_block)
        C_cplx_block[..., 1] = C_mag_block * torch.sin(C_phase_block)

        C_cplx[i] = C_cplx_block

    return C_cplx


def atan2(y, x):
    r"""Element-wise arctangent function of y/x.
    Returns a new tensor with signed angles in radians.
    It is an alternative implementation of torch.atan2

    Args:
        y (Tensor): First input tensor
        x (Tensor): Second input tensor [shape=y.shape]

    Returns:
        Tensor: [shape=y.shape].
    """
    pi = 2 * torch.asin(torch.tensor(1.0))
    x += ((x == 0) & (y == 0)) * 1.0
    out = torch.atan(y / x)
    out += ((y >= 0) & (x < 0)) * pi
    out -= ((y < 0) & (x < 0)) * pi
    out *= 1 - ((y > 0) & (x == 0)) * 1.0
    out += ((y > 0) & (x == 0)) * (pi / 2)
    out *= 1 - ((y < 0) & (x == 0)) * 1.0
    out += ((y < 0) & (x == 0)) * (-pi / 2)
    return out
