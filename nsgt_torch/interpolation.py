import torch
from typing import List, Tuple
from torch import Tensor
from torch.nn.functional import interpolate as torch_interpolate


ALLOWED_MATRIX_FORMS = ['ragged', 'zeropad', 'linear', 'bilinear', 'nearest', 'area', 'repeat']


@torch.no_grad()
def interpolate(nsgt: List[Tensor], interpolation_mode: str) -> Tensor:
    nb_samples, nb_channels = nsgt[0].shape[:2]
    total_f_bins = sum([nsgt_.shape[-2] for nsgt_ in nsgt])
    max_t_bins = max([nsgt_.shape[-1] for nsgt_ in nsgt])

    interpolated = torch.zeros((nb_samples, nb_channels, total_f_bins, max_t_bins), dtype=nsgt[0].dtype, device=nsgt[0].device)

    fbin_ptr = 0
    prev_shapes = [None]*len(nsgt)
    for i, nsgt_ in enumerate(nsgt):
        nb_samples, nb_channels, nb_f_bins, nb_t_bins = nsgt_.shape
        prev_shapes[i] = nsgt_.shape

        if nb_t_bins == max_t_bins:
            # same time width, no interpolation
            interpolated[:, :, fbin_ptr:fbin_ptr+nb_f_bins, :] = nsgt_
        else:
            # interpolation
            if interpolation_mode == 'repeat':
                interp_factor = max_t_bins//nb_t_bins
                max_assigned = nb_t_bins*interp_factor
                rem = max_t_bins - max_assigned
                interpolated[:, :, fbin_ptr:fbin_ptr+nb_f_bins, : max_assigned] = torch.repeat_interleave(nsgt_, interp_factor, dim=-1)
                interpolated[:, :, fbin_ptr:fbin_ptr+nb_f_bins, max_assigned : ] = torch.unsqueeze(nsgt_[..., -1], dim=-1).repeat(1, 1, 1, rem)
            else:
                interpolated[:, :, fbin_ptr:fbin_ptr+nb_f_bins, :] = torch_interpolate(nsgt_, size=(nb_f_bins, max_t_bins), mode=interpolation_mode)
        fbin_ptr += nb_f_bins

    return interpolated, prev_shapes


@torch.no_grad()
def deinterpolate(interpolated: Tensor, ragged_shapes: List[Tuple[int]], interpolation_mode: str) -> List[Tensor]:
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
            # inverse interpolation
            if interpolation_mode == 'repeat':
                interp_factor = max_t_bins//nb_t_bins
                select = torch.arange(0, max_t_bins,interp_factor, device=interpolated.device)
                curr_slicq = torch.index_select(interpolated[:, :, fbin_ptr:fbin_ptr+freqs, :], -1, select)
            else:
                curr_slicq = interpolate(torch_interpolated[:, :, fbin_ptr:fbin_ptr+freqs, :], size=(freqs, nb_t_bins), mode=interpolation_mode)

        # crop just in case
        full_slicq.append(curr_slicq[..., : bucket_shape[-1]])

        fbin_ptr += freqs
    return full_slicq
