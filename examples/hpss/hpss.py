import numpy
import torch
from librosa.core import stft, istft
from librosa.util import fix_length
from librosa.decompose import hpss
from nsgt_torch.cq import make_nsgt_filterbanks, NSGTBase
from nsgt_torch.slicq import make_slicqt_filterbanks, SliCQTBase
from nsgt_torch.util import complex_2_magphase, magphase_2_complex


def ihpss_stft(x, iters=1, mask='soft', harmonic_margin=2.0, harmonic_frame=4096, percussive_margin=2.0, percussive_frame=256):
    power = 1.0 if mask=='soft' else numpy.inf

    # big t-f resolution for harmonic
    S1 = stft(
        x,
        n_fft=2 * harmonic_frame,
        win_length=harmonic_frame,
        hop_length=int(harmonic_frame // 2),
    )

    S_h1, S_p1 = hpss(S1, margin=harmonic_margin, power=power)

    S_r1 = S1 - (S_h1 + S_p1)

    yh1 = fix_length(istft(S_h1, dtype=x.dtype), size=x.shape[-1])
    yp1 = fix_length(istft(S_p1, dtype=x.dtype), size=x.shape[-1])
    yr1 = fix_length(istft(S_r1, dtype=x.dtype), size=x.shape[-1])

    if iters == 1:
        return torch.cat((torch.unsqueeze(torch.tensor(yh1), dim=0), torch.unsqueeze(torch.tensor(yp1), dim=0)), dim=0)

    # small t-f resolution for percussive
    S2 = stft(
        yp1 + yr1,
        n_fft=2 * percussive_frame,
        win_length=percussive_frame,
        hop_length=int(percussive_frame // 2),
    )
    _, S_p2 = hpss(S2, margin=percussive_margin, power=power)

    yp2 = fix_length(istft(S_p2, dtype=x.dtype), size=x.shape[-1])

    return torch.cat((torch.unsqueeze(torch.tensor(yh1), dim=0), torch.unsqueeze(torch.tensor(yp2), dim=0)), dim=0)


def hpss_nsgt(x_in, bpo, matrixform, nonsliced=False):
    x = torch.unsqueeze(torch.tensor(x_in), dim=0)

    if nonsliced:
        nsgt_base = NSGTBase('oct', bpo, 20., x.shape[-1], fmax=22050., matrixform=matrixform, fs=44100., device="cpu")
        nsgt, insgt = make_nsgt_filterbanks(nsgt_base)
    else:
        nsgt_base = SliCQTBase('oct', bpo, 20., fmax=22050., matrixform=matrixform, fs=44100., device="cpu")
        nsgt, insgt = make_slicqt_filterbanks(nsgt_base)

    # generator for forward transformation
    C, ragged_shapes_for_deinterp = nsgt(x)

    Cmag, Cphase = complex_2_magphase(C)

    # soft mask hpss
    # rectangular matrix forms
    Cmag_h, Cmag_p = hpss(Cmag, margin=1.0, power=1.0)

    Ch = magphase_2_complex(Cmag_h, Cphase)
    Cp = magphase_2_complex(Cmag_p, Cphase)

    yh = insgt(Ch, x.shape[-1], ragged_shapes=ragged_shapes_for_deinterp)
    yp = insgt(Cp, x.shape[-1], ragged_shapes=ragged_shapes_for_deinterp)

    return torch.cat((yh, yp), dim=0)
