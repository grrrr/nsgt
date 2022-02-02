import os
from warnings import warn
import torch
import numpy

from nsgt_torch.plot import spectrogram
from nsgt_torch import NSGT, NSGT_sliced
from nsgt_torch.fscale import SCALES_BY_NAME
from nsgt_torch.audio import SndReader

from argparse import ArgumentParser
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("input", type=str, help="Input file")
    parser.add_argument("--output", type=str, help="output png path", default=None)
    parser.add_argument("--sr", type=int, default=44100, help="Sample rate used for the NSGT (default=%(default)s)")
    parser.add_argument("--fmin", type=float, default=50, help="Minimum frequency in Hz (default=%(default)s)")
    parser.add_argument("--fmax", type=float, default=22050, help="Maximum frequency in Hz (default=%(default)s)")
    parser.add_argument("--gamma", type=float, default=15, help="variable-q frequency offset per band")
    parser.add_argument("--cmap", type=str, default='hot', help="spectrogram color map")
    parser.add_argument("--scale", choices=('oct','cqlog','mel','bark','Bark','vqlog','pow2'), default='cqlog', help="Frequency scale")
    parser.add_argument("--bins", type=int, default=50, help="Number of frequency bins (total or per octave, default=%(default)s)")
    parser.add_argument("--fontsize", type=int, default=14, help="Plot font size, default=%(default)s)")
    parser.add_argument("--sllen", type=int, default=None, help="Slice length in samples (default=%(default)s)")
    parser.add_argument("--trlen", type=int, default=None, help="Transition area in samples (default=%(default)s)")
    parser.add_argument("--plot", action='store_true', help="Plot transform (needs installed matplotlib package)")
    parser.add_argument("--mono", action='store_true', help="Audio is mono")
    parser.add_argument("--nonsliced", action='store_true', help="Use the non-sliced transform")
    parser.add_argument("--flatten", action='store_true', help="Flatten instead of overlap")
    parser.add_argument("--plot-stft", action='store_true', help="Plot STFT and exit")
    parser.add_argument("--stft-window", type=int, default=4096, help="STFT window to use")
    parser.add_argument("--stft-overlap", type=int, default=1024, help="STFT overlap to use")

    args = parser.parse_args()
    if not os.path.exists(args.input):
        parser.error("Input file '%s' not found"%args.input)

    fs = args.sr

    # Read audio data
    sf = SndReader(args.input, sr=fs, chns=2)
    signal = sf()

    signal = [torch.tensor(sig) for sig in signal]
    signal = torch.cat(signal, dim=-1)

    # add a batch
    signal = torch.unsqueeze(signal, dim=0)

    # duration of signal in s
    dur = sf.frames/float(fs)

    if args.plot_stft:
        fig = plt.figure()

        plt.rcParams.update({'font.size': args.fontsize})
        ax = fig.add_subplot(111)

        title = f'Magnitude STFT, window={args.stft_window}, overlap={args.stft_overlap}'
        ax.set_title(title)

        _, _, _, cax = ax.specgram(signal, cmap=args.cmap, Fs=fs, NFFT=args.stft_window, noverlap=args.stft_overlap)

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

    try:
        scale = SCALES_BY_NAME[args.scale]
    except KeyError:
        parser.error('Scale unknown (--scale option)')

    if args.scale != 'vqlog':
        scl = scale(args.fmin, args.fmax, args.bins)
    else:
        scl = scale(args.fmin, args.fmax, args.bins, gamma=args.gamma)

    freqs, qs = scl()

    if args.sllen is None:
        sllen, trlen = scl.suggested_sllen_trlen(fs)
    else:
        sllen = args.sllen
        trlen = args.trlen

    if not args.nonsliced:
    nsgt = TorchNSGT(
        scl, sllen, trlen, fs, real=True, multichannel=True, device="cpu",
        matrixform=args.matrixform,
        chunk_strategy=args.chunk_strategy,
    )

    # generator for forward transformation
    if args.nonsliced:
        C = slicq.forward(signal)
    else:
        C = slicq.forward(signal)

    transform_name = 'sliCQT' if not args.nonsliced else 'NSGT'

    if args.fmin > 0.0:
        freqs = numpy.r_[[0.], freqs]

    if args.plot:
        slicq_params = '{0} scale, {1} bins, {2:.1f}-{3:.1f} Hz'.format(args.scale, args.bins, args.fmin, args.fmax)

        spectrogram(
            C,
            fs,
            slicq.coef_factor,
            transform_name,
            freqs,
            signal.shape[1],
            sliced=not args.nonsliced,
            flatten=args.flatten,
            fontsize=args.fontsize,
            cmap=args.cmap,
            slicq_name=slicq_params,
            output_file=args.output
        )
