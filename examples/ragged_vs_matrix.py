#! /usr/bin/env python 
# -*- coding: utf-8

"""
Python implementation of Non-Stationary Gabor Transform (NSGT)
derived from MATLAB code by NUHAG, University of Vienna, Austria

Thomas Grill, 2011-2016
http://grrrr.org/nsgt
"""

import os
from warnings import warn
import torch

from nsgt import NSGT, NSGT_sliced, LogScale, LinScale, MelScale, OctScale, VQLogScale, BarkScale, SndReader
from nsgt.slicq import overlap_add_slicq

from argparse import ArgumentParser
parser = ArgumentParser()

parser.add_argument("input", type=str, help="Input file")
parser.add_argument("--sr", type=int, default=44100, help="Sample rate used for the NSGT (default=%(default)s)")
parser.add_argument("--fmin", type=float, default=50, help="Minimum frequency in Hz (default=%(default)s)")
parser.add_argument("--fmax", type=float, default=22050, help="Maximum frequency in Hz (default=%(default)s)")
parser.add_argument("--scale", choices=('oct','cqlog','mel','bark','vqlog','lin'), default='cqlog', help="Frequency scale oct, cqlog, vqlog, lin, mel, bark (default='%(default)s')")
parser.add_argument("--bins", type=int, default=50, help="Number of frequency bins (total or per octave, default=%(default)s)")
parser.add_argument("--sllen", type=int, default=None, help="Slice length in samples (default=%(default)s)")
parser.add_argument("--trlen", type=int, default=None, help="Transition area in samples (default=%(default)s)")
parser.add_argument("--matrixform", action='store_true', help="Use matrixform")

args = parser.parse_args()
if not os.path.exists(args.input):
    parser.error("Input file '%s' not found"%args.input)

fs = args.sr

# build transform
scales = {'cqlog': LogScale, 'lin': LinScale, 'mel': MelScale, 'oct': OctScale, 'vqlog': VQLogScale, 'bark': BarkScale}
try:
    scale = scales[args.scale]
except KeyError:
    parser.error('Scale unknown (--scale option)')

scl = scale(args.fmin, args.fmax, args.bins)

if args.sllen is None:
    sllen, trlen = scl.suggested_sllen_trlen(fs)
else:
    sllen = args.sllen
    trlen = args.trlen

print(f'sllen: {sllen}, trlen: {trlen}')
slicq = NSGT_sliced(scl, sllen, trlen, fs, 
                    real=True,
                    matrixform=args.matrixform,
                    multichannel=True,
                    device="cpu"
                    )

# Read audio data
sf = SndReader(args.input, sr=fs, chns=2)
signal = sf()

signal = [torch.tensor(sig) for sig in signal]

#pad = signal[0].shape[-1]-signal[-1].shape[-1]
#signal[-1] = torch.nn.functional.pad(signal[-1], (0, pad), mode='constant', value=0)
signal = torch.cat(signal, dim=-1)

# duration of signal in s
dur = sf.frames/float(fs)

# total number of coefficients to represent input signal
ncoefs = int(sf.frames*slicq.coef_factor)

# generator for forward transformation
c = slicq.forward((signal,))

if args.matrixform:
    matrix_ola = torch.squeeze(overlap_add_slicq(torch.unsqueeze(c, dim=0)), dim=0)
    matrix_flat = torch.squeeze(overlap_add_slicq(torch.unsqueeze(c, dim=0), flatten=True), dim=0)
    print(f'NSGT-sliCQ matrix shape: {c.shape}, {matrix_ola.shape}, {matrix_flat.shape}')
else:
    print(f'NSGT-sliCQ jagged shape:')
    freq_idx = 0
    for i, C_block in enumerate(c):
        freq_start = freq_idx
        freq_end = C_block.shape[2]
        freq_idx += freq_end

        C_block_ola = torch.squeeze(overlap_add_slicq(torch.unsqueeze(C_block, dim=0)), dim=0)
        C_block_flatten = torch.squeeze(overlap_add_slicq(torch.unsqueeze(C_block, dim=0), flatten=True), dim=0)
        print(f'\tblock {i}, f {freq_start}-{freq_start+freq_end-1}: {C_block.shape}, {C_block_ola.shape}, {C_block_flatten.shape}')

signal_recon = slicq.backward(c, signal.shape[-1])

print(f'recon error (mse): {torch.nn.functional.mse_loss(signal_recon, signal)}')

print(f'comparing 4096 stft for fun')

print(f'signal: {signal.shape}')
S = torch.stft(signal, n_fft=4096, hop_length=1024, return_complex=True, center=False).type(torch.complex64)

print(f'stft with 4096/1024: {S.shape}')
