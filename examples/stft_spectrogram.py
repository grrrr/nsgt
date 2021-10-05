#! /usr/bin/env python 
import os
from warnings import warn
import torch
import sys
import numpy as np
from nsgt.plot import spectrogram

from nsgt import NSGT, NSGT_sliced, LogScale, LinScale, MelScale, OctScale, VQLogScale, BarkScale, SndReader
from nsgt.fscale import Pow2Scale

from argparse import ArgumentParser
import matplotlib.pyplot as plt
import matplotlib as mpl


parser = ArgumentParser()

parser.add_argument("input", type=str, help="Input file")
parser.add_argument("--output", type=str, help="output png path", default=None)
parser.add_argument("--cmap", type=str, default='hot', help="spectrogram color map")
parser.add_argument("--fontsize", type=int, default=14, help="Plot font size, default=%(default)s)")
parser.add_argument("--sr", type=int, default=44100, help="sample rate, int (Hz)")
parser.add_argument("--window", type=int, default=4096, help="STFT window size")
parser.add_argument("--overlap", type=int, default=1024, help="STFT overlap")
parser.add_argument("--plot", action='store_true', help="Plot transform (needs installed matplotlib package)")
parser.add_argument("--mono", action='store_true', help="Audio is mono")

args = parser.parse_args()
if not os.path.exists(args.input):
    parser.error("Input file '%s' not found"%args.input)

fs = args.sr

# Read audio data
sf = SndReader(args.input, sr=fs, chns=2, blksz=2**32)
signal = sf()
signal = np.asarray(list(signal))
signal = signal.T
signal = np.squeeze(signal, axis=-1)
signal = np.mean(signal, axis=-1)

# duration of signal in s
dur = sf.frames/float(fs)

print(f'signal: {signal.shape}, dur: {dur}, frames: {sf.frames}')

if args.plot:
    fig = plt.figure()

    plt.rcParams.update({'font.size': args.fontsize})
    ax = fig.add_subplot(111)

    title = f'Magnitude STFT, window={args.window}, overlap={args.overlap}'
    ax.set_title(title)

    _, _, _, cax = ax.specgram(signal, cmap=args.cmap, Fs=args.sr, NFFT=args.window, noverlap=args.overlap)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (kHz)')

    scale = 1e3                     # KHz
    ticks = mpl.ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(int(x/scale)))
    plt.locator_params(axis='y', nbins=5)
    ax.yaxis.set_major_formatter(ticks)
    ax.yaxis.get_major_locator().set_params(integer=True)

    fig.colorbar(cax, shrink=1.0, pad=0.006, label='dB')

    plt.subplots_adjust(wspace=0.001,hspace=0.001)

    if args.output is not None:
        DPI = fig.get_dpi()
        fig.set_size_inches(2560.0/float(DPI),1440.0/float(DPI))
        fig.savefig(args.output, dpi=DPI, bbox_inches='tight')
    else:
        plt.show()
