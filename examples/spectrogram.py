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
import matplotlib.pyplot as plt

from nsgt import NSGT, NSGT_sliced, LogScale, LinScale, MelScale, OctScale, VQLogScale, BarkScale, SndReader


def overlap_add_slicq(slicq):
    nb_samples, nb_slices, nb_channels, nb_f_bins, nb_m_bins = slicq.shape

    window = nb_m_bins
    hop = window//2 # 50% overlap window

    ncoefs = nb_slices*nb_m_bins//2 + hop
    out = torch.zeros((nb_samples, nb_channels, nb_f_bins, ncoefs), dtype=slicq.dtype, device=slicq.device)

    ptr = 0

    for i in range(nb_slices):
        out[:, :, :, ptr:ptr+window] += slicq[:, i, :, :, :]
        ptr += hop

    return out


from argparse import ArgumentParser
parser = ArgumentParser()

parser.add_argument("input", type=str, help="Input file")
parser.add_argument("--sr", type=int, default=44100, help="Sample rate used for the NSGT (default=%(default)s)")
parser.add_argument("--fmin", type=float, default=50, help="Minimum frequency in Hz (default=%(default)s)")
parser.add_argument("--fmax", type=float, default=22050, help="Maximum frequency in Hz (default=%(default)s)")
parser.add_argument("--gamma", type=float, default=15, help="variable-q frequency offset per band")
parser.add_argument("--scale", choices=('oct','cqlog','mel','bark','vqlog'), default='cqlog', help="Frequency scale")
parser.add_argument("--bins", type=int, default=50, help="Number of frequency bins (total or per octave, default=%(default)s)")
parser.add_argument("--sllen", type=int, default=2**20, help="Slice length in samples (default=%(default)s)")
parser.add_argument("--trlen", type=int, default=2**18, help="Transition area in samples (default=%(default)s)")
parser.add_argument("--plot", action='store_true', help="Plot transform (needs installed matplotlib package)")
parser.add_argument("--mono", action='store_true', help="Audio is mono")
parser.add_argument("--nonsliced", action='store_true', help="Use the non-sliced transform")

args = parser.parse_args()
if not os.path.exists(args.input):
    parser.error("Input file '%s' not found"%args.input)

fs = args.sr

# build transform
scales = {'cqlog':LogScale, 'lin':LinScale, 'mel':MelScale, 'oct':OctScale, 'bark':BarkScale, 'vqlog':VQLogScale}
try:
    scale = scales[args.scale]
except KeyError:
    parser.error('Scale unknown (--scale option)')

if args.scale != 'vqlog':
    scl = scale(args.fmin, args.fmax, args.bins)
else:
    scl = scale(args.fmin, args.fmax, args.bins, gamma=args.gamma)


freqs, qs = scl()

print(','.join(['{0:.2f}'.format(freq) for freq in freqs[:10]]))
print()

print(','.join(['{0:.2f}'.format(freq) for freq in freqs[-10:]]))
print()

print(','.join(['{0:.2f}'.format(q) for q in qs[:10]]))
print()

print(','.join(['{0:.2f}'.format(q) for q in qs[-10:]]))
print()

# Read audio data
sf = SndReader(args.input, sr=fs, chns=2)
signal = sf()

signal = [torch.tensor(sig) for sig in signal]

pad = signal[0].shape[-1]-signal[-1].shape[-1]
signal[-1] = torch.nn.functional.pad(signal[-1], (0, pad), mode='constant', value=0)
signal = torch.cat(signal, dim=-1)

# duration of signal in s
dur = sf.frames/float(fs)

if not args.nonsliced:
    slicq = NSGT_sliced(scl, args.sllen, args.trlen, fs, 
                        real=True,
                        matrixform=True, 
                        multichannel=True,
                        device="cpu"
                        )
else:
    slicq = NSGT(scl, fs, signal.shape[-1],
                 real=True,
                 matrixform=True, 
                 multichannel=True,
                 device="cpu"
                 )

# total number of coefficients to represent input signal
#ncoefs = int(sf.frames*slicq.coef_factor)

# generator for forward transformation
if args.nonsliced:
    c = slicq.forward(signal)
else:
    c = slicq.forward((signal,))

# add a batch
c = torch.unsqueeze(c, dim=0)

if args.plot:
    # dB
    if args.nonsliced:
        mls = 20.*torch.log10(torch.abs(c))
    else:
        mls = 20.*torch.log10(torch.abs(overlap_add_slicq(c)))

    plt.rcParams.update({'font.size': 14})
    fig, axs = plt.subplots(1)

    print(f"Plotting t*f space")

    # remove batch
    mls = torch.squeeze(mls, dim=0)
    # mix down multichannel
    mls = torch.mean(mls, dim=0)

    mls = mls.T

    #fs_coef = fs*slicq.coef_factor # frame rate of coefficients
    #mls_dur = len(mls)/fs_coef # final duration of MLS
    mls_dur = dur

    mls_max = torch.quantile(mls, 0.999)
    axs.imshow(mls.T, aspect=mls_dur/mls.shape[1]*0.2, interpolation='nearest', origin='lower', vmin=mls_max-60., vmax=mls_max, extent=(0,mls_dur,0,mls.shape[1]))
    axs.set_title('Magnitude NSGT, {0} scale, {1} bins'.format(args.scale, args.bins))
    axs.set_xlabel('Time (s)')
    axs.set_ylabel('Frequency bin')

    sig = torch.mean(signal, dim=0)
    print(f'sig: {sig.shape} {fs}')

    #axs[1].specgram(sig, Fs=fs, NFFT=2048, noverlap=512, mode='magnitude', scale='dB', sides='onesided')
    #axs[1].set_title(f'|STFT|')

    plt.subplots_adjust(wspace=0.001,hspace=0.001)
    plt.show()
