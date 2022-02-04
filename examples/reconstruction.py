import os
from warnings import warn
import torch
import numpy
import auraloss
from argparse import ArgumentParser
from torch import Tensor
from nsgt_torch.fscale import SCALES_BY_NAME
from nsgt.audio import SndReader
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
    parser.add_argument("--scale", choices=('oct','cqlog','mel','bark','Bark','vqlog','pow2'), default='cqlog', help="Frequency scale")
    parser.add_argument("--bins", type=int, default=50, help="Number of frequency bins (total or per octave, default=%(default)s)")
    parser.add_argument("--sllen", type=int, default=None, help="Slice length in samples (default=%(default)s)")
    parser.add_argument("--trlen", type=int, default=None, help="Transition area in samples (default=%(default)s)")
    parser.add_argument("--mono", action='store_true', help="Audio is mono")
    parser.add_argument("--nonsliced", action='store_true', help="Use the NSGT instead of the sliCQT")
    parser.add_argument("--stft-window", type=int, default=4096, help="STFT window to use")
    parser.add_argument("--stft-overlap", type=int, default=1024, help="STFT overlap to use")
    parser.add_argument("--matrixform", choices=ALLOWED_MATRIX_FORMS, default='zeropad', help="Matrix form/interpolation strategy to use")
    parser.add_argument("-N", type=int, default=-1, help="chunked NSGT size (-1 = full signal length)")

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

    if args.nonsliced:
        N = args.N if args.N != -1 else sf.frames
        nsgt_base = NSGTBase(
            args.scale, args.bins, args.fmin, N, fmax=args.fmax, gamma=args.gamma,
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
    C, nb_slices, ragged_shapes_for_deinterp = nsgt(signal)
    print(f'C: {type(C)}')

    if type(C) == list:
        print(f'ragged form, C len: {len(C)}')
        print(f'\tC[0]: {C[0].shape} {C[0].dtype}')
        print(f'\t...')
        print(f'\tC[-1]: {C[-1].shape} {C[-1].dtype}')
    elif type(C) == Tensor:
        print(f'matrix form, C: {C.shape} {C.dtype}')

    signal_recon = insgt(C, sf.frames, nb_slices, ragged_shapes_for_deinterp)

    print('signal reconstruction errors:')

    print(f'\tSNR: {-1*auraloss.time.SNRLoss()(signal_recon, signal)}')
    print(f'\tMSE (time-domain waveform): {torch.sqrt(torch.mean((signal_recon - signal)**2))}')
    print(f'\tSI-SDR: {-1*auraloss.time.SISDRLoss()(signal_recon, signal)}')
    print(f'\tSD-SDR: {-1*auraloss.time.SDSDRLoss()(signal_recon, signal)}')
