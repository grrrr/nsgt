import numpy
import itertools
from scipy.signal import butter, lfilter
from librosa.decompose import hpss
from librosa.core import stft, istft
from librosa.util import fix_length
from .params import DEFAULTS


# iterative hpss
def ihpss(
    x,
    pool,
    harmonic_margin=DEFAULTS["harmonic_margin"],
    harmonic_frame=DEFAULTS["harmonic_frame"],
    percussive_margin=DEFAULTS["percussive_margin"],
    percussive_frame=DEFAULTS["percussive_frame"],
):
    print(
        "Iteration 1 of hpss: frame = {0}, margin = {1}".format(
            harmonic_frame, harmonic_margin
        )
    )
    # big t-f resolution for harmonic
    S1 = stft(
        x,
        n_fft=2 * harmonic_frame,
        win_length=harmonic_frame,
        hop_length=int(harmonic_frame // 2),
    )
    S_h1, S_p1 = hpss(S1, margin=harmonic_margin, power=numpy.inf)  # hard mask
    S_r1 = S1 - (S_h1 + S_p1)

    yp1 = fix_length(istft(S_p1, dtype=x.dtype), len(x))
    yr1 = fix_length(istft(S_r1, dtype=x.dtype), len(x))

    print(
        "Iteration 2 of hpss: frame = {0}, margin = {1}".format(
            percussive_frame, percussive_margin
        )
    )
    # small t-f resolution for percussive
    S2 = stft(
        yp1 + yr1,
        n_fft=2 * percussive_frame,
        win_length=percussive_frame,
        hop_length=int(percussive_frame // 2),
    )
    _, S_p2 = hpss(S2, margin=percussive_margin, power=numpy.inf)  # hard mask

    yp = fix_length(istft(S_p2, dtype=x.dtype), len(x))

    yp_tshaped = yp

    return yp_tshaped, yp
