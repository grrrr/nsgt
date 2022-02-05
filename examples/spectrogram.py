import os
from warnings import warn
import torch
from torch import Tensor
import auraloss
import numpy
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from nsgt.audio import SndReader
from nsgt_torch.plot import spectrogram
from nsgt_torch.fscale import SCALES_BY_NAME
from nsgt_torch.cq import NSGTBase, make_nsgt_filterbanks
from nsgt_torch.slicq import SliCQTBase, make_slicqt_filterbanks
from nsgt_torch.util import complex_2_magphase, magphase_2_complex
from nsgt_torch.interpolation import ALLOWED_MATRIX_FORMS


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
    parser.add_argument("--nonsliced", action='store_true', help="Use the NSGT instead of the sliCQT")
    parser.add_argument("--plot-stft", action='store_true', help="Plot STFT and exit")
    parser.add_argument("--stft-window", type=int, default=4096, help="STFT window to use")
    parser.add_argument("--stft-overlap", type=int, default=1024, help="STFT overlap to use")
    parser.add_argument("--matrixform", choices=ALLOWED_MATRIX_FORMS, default='zeropad', help="Matrix form/interpolation strategy to use")
    parser.add_argument("--flatten", action='store_true', help="Flatten the sliCQ instead of overlap-adding")

    args = parser.parse_args()
    if not os.path.exists(args.input):
        parser.error("Input file '%s' not found"%args.input)

    fs = args.sr

    # Read audio data
    sf = SndReader(args.input, sr=fs, chns=2)
    signal = sf()

    signal = [torch.tensor(sig) for sig in signal]
    signal = torch.cat(signal, dim=-1)[..., :sf.frames]

    # add a batch
    signal = torch.unsqueeze(signal, dim=0)

    # duration of signal in s
    dur = sf.frames/float(fs)
    print(f'input signal: duration {dur:.2f}, shape {signal.shape}')

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

    if args.nonsliced:
        nsgt_base = NSGTBase(
            args.scale, args.bins, args.fmin, sf.frames, fmax=args.fmax, gamma=args.gamma,
            matrixform=args.matrixform,
            fs=fs, device="cpu",
        )
        nsgt, insgt = make_nsgt_filterbanks(nsgt_base)
    else:
        nsgt_base = SliCQTBase(
            args.scale, args.bins, args.fmin,
            sllen=args.sllen, trlen=args.trlen,
            matrixform=args.matrixform,
            fs=fs, device="cpu",
        )
        nsgt, insgt = make_slicqt_filterbanks(nsgt_base)

    # generator for forward transformation
    C, ragged_shapes_for_deinterp = nsgt(signal)

    print(f'C: {type(C)}')

    if type(C) == list:
        print(f'ragged form, C len: {len(C)}')
        print(f'\tC[0]: {C[0].shape} {C[0].dtype}')
        print(f'\tC[1]: {C[1].shape} {C[1].dtype}')
        print(f'\t...')
        print(f'\tC[-2]: {C[-2].shape} {C[-2].dtype}')
        print(f'\tC[-1]: {C[-1].shape} {C[-1].dtype}')
    elif type(C) == Tensor:
        print(f'matrix form, C: {C.shape} {C.dtype}')

    signal_recon = insgt(C, sf.frames, ragged_shapes=ragged_shapes_for_deinterp)

    print('signal reconstruction errors:')
    print(f'\tSNR: {-1*auraloss.time.SNRLoss()(signal_recon, signal)}')
    print(f'\tMSE (time-domain waveform): {torch.sqrt(torch.mean((signal_recon - signal)**2))}')
    print(f'\tSI-SDR: {-1*auraloss.time.SISDRLoss()(signal_recon, signal)}')
    print(f'\tSD-SDR: {-1*auraloss.time.SDSDRLoss()(signal_recon, signal)}')

    Cmag, Cphase = complex_2_magphase(C)

    if args.plot:
        if args.matrixform == 'ragged':
            raise ValueError('spectrogram is not supported for the ragged form')

        transform_name = 'NSGT'
        if not args.nonsliced:
            transform_name = 'sliCQT'
            if args.flatten:
                Cmag = torch.flatten(Cmag, start_dim=-2, end_dim=-1)
            else:
                Cmag = nsgt.overlap_add(Cmag)

        freqs, qs = nsgt.nsgt.scl()
        if args.fmin > 0.0:
            freqs = numpy.r_[[0.], freqs]

        slicq_params = '{0} scale, {1} bins, {2:.1f}-{3:.1f} Hz'.format(args.scale, args.bins, args.fmin, args.fmax)

        spectrogram(
            Cmag.detach(),
            fs,
            nsgt.nsgt.M,
            nsgt.nsgt.nsgt.coef_factor,
            transform_name,
            freqs,
            signal.shape[-1],
            sliced=not args.nonsliced,
            fontsize=args.fontsize,
            cmap=args.cmap,
            slicq_name=slicq_params,
            output_file=args.output,
            flattened=args.flatten,
        )
