import os
from warnings import warn
import torch
import numpy

from nsgt_torch.fscale import SCALES_BY_NAME
from nsgt.audio import SndReader

# non-sliced NSGT
from nsgt_torch.cq import NSGTBase, make_nsgt_filterbanks

# sliced sliCQT
from nsgt_torch.slicq import SliCQTBase, make_slicqt_filterbanks

# misc utils
from nsgt_torch.util import complex_2_magphase, magphase_2_complex
from nsgt_torch.interpolation import ALLOWED_MATRIX_FORMS

from argparse import ArgumentParser


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
    parser.add_argument("--mono", action='store_true', help="Audio is mono")
    parser.add_argument("--nonsliced", action='store_true', help="Use the NSGT instead of the sliCQT")
    parser.add_argument("--flatten", action='store_true', help="Flatten instead of overlap")
    parser.add_argument("--stft-window", type=int, default=4096, help="STFT window to use")
    parser.add_argument("--stft-overlap", type=int, default=1024, help="STFT overlap to use")
    parser.add_argument("--matrixform", choices=ALLOWED_MATRIX_FORMS, default='zeropad', help="Matrix form/interpolation strategy to use")

    args = parser.parse_args()
    if not os.path.exists(args.input):
        parser.error("Input file '%s' not found"%args.input)

    if args.matrixform == 'ragged':
        raise ValueError('spectrogram is not supported for the ragged form')

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
    C = nsgt(signal)
    Cmag, Cphase = complex_2_magphase(C)

    if not args.nonsliced:
        Cmag = nsgt.overlap_add(Cmag, flatten=args.flatten)
