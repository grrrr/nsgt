import torch
from typing import List, Tuple
from torch import Tensor
from torch.nn.functional import interpolate as torch_interpolate


ALLOWED_MATRIX_FORMS = ['ragged', 'zeropad', 'interpolate']


@torch.no_grad()
def interpolate_nsgt(nsgt: List[Tensor]) -> Tensor:
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
            interpolated[:, :, fbin_ptr:fbin_ptr+nb_f_bins, :] = torch_interpolate(nsgt_, size=(nb_f_bins, max_t_bins), mode='bilinear', align_corners=True)
        fbin_ptr += nb_f_bins

    return interpolated, prev_shapes


@torch.no_grad()
def deinterpolate_nsgt(interpolated: Tensor, ragged_shapes: List[Tuple[int]]) -> List[Tensor]:
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
            curr_slicq = torch_interpolate(interpolated[:, :, fbin_ptr:fbin_ptr+freqs, :], size=(freqs, nb_t_bins), mode='bilinear', align_corners=True)

        # crop just in case
        full_slicq.append(curr_slicq[..., : bucket_shape[-1]])

        fbin_ptr += freqs
    return full_slicq


@torch.no_grad()
def interpolate_slicqt(slicqt: List[Tensor]) -> Tensor:
    nb_samples, nb_channels = slicqt[0].shape[:2]
    total_f_bins = sum([slicqt_.shape[-3] for slicqt_ in slicqt])
    nb_slices = slicqt[0].shape[-2]
    max_t_bins = max([slicqt_.shape[-1] for slicqt_ in slicqt])

    interpolated = torch.zeros((nb_samples, nb_channels, total_f_bins, nb_slices, max_t_bins), dtype=slicqt[0].dtype, device=slicqt[0].device)

    fbin_ptr = 0
    prev_shapes = [None]*len(slicqt)
    for i, slicqt_ in enumerate(slicqt):
        nb_samples, nb_channels, nb_f_bins, nb_slices, nb_t_bins = slicqt_.shape
        prev_shapes[i] = slicqt_.shape

        if nb_t_bins == max_t_bins:
            # same time width, no interpolation
            interpolated[:, :, fbin_ptr:fbin_ptr+nb_f_bins, :, :] = slicqt_
        else:
            # interpolation
            interpolated[:, :, fbin_ptr:fbin_ptr+nb_f_bins, :, :] = torch_interpolate(slicqt_, size=(nb_f_bins, nb_slices, max_t_bins), mode='trilinear', align_corners=True)
        fbin_ptr += nb_f_bins

    return interpolated, prev_shapes


@torch.no_grad()
def deinterpolate_slicqt(interpolated: Tensor, ragged_shapes: List[Tuple[int]]) -> List[Tensor]:
    max_t_bins = interpolated.shape[-1]
    full_slicq = []
    fbin_ptr = 0

    for i, bucket_shape in enumerate(ragged_shapes):
        curr_slicq = torch.zeros(bucket_shape, dtype=interpolated.dtype, device=interpolated.device)

        nb_t_bins = bucket_shape[-1]
        nb_slices = bucket_shape[-2]
        freqs = bucket_shape[-3]

        if bucket_shape[-1] == interpolated.shape[-1]:
            # same time width, no interpolation
            curr_slicq = interpolated[:, :, fbin_ptr:fbin_ptr+freqs, :, :]
        else:
            curr_slicq = torch_interpolate(interpolated[:, :, fbin_ptr:fbin_ptr+freqs, :, :], size=(freqs, nb_slices, nb_t_bins), mode='trilinear', align_corners=True)

        # crop just in case
        full_slicq.append(curr_slicq[..., : bucket_shape[-1]])

        fbin_ptr += freqs
    return full_slicq
