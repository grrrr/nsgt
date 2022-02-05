import os
from warnings import warn
import torch
import numpy
import librosa
from slicqt.audio import SndReader
import scipy.io.wavfile
from slicqt.torch_utils import load_deoverlapnet
import auraloss

from slicqt import torch_transforms as transforms

from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("input", type=str, help="Input file")
    parser.add_argument("--deoverlapnet-path", type=str, default="./pretrained-model")
    parser.add_argument("--output-png", type=str, help="output png path", default=None)
    parser.add_argument("--sr", type=int, default=44100, help="Sample rate used for the NSGT (default=%(default)s)")
    parser.add_argument("--cmap", type=str, default='hot', help="spectrogram color map")
    parser.add_argument("--fontsize", type=int, default=14, help="Plot font size, default=%(default)s)")

    args = parser.parse_args()
    if not os.path.exists(args.input):
        parser.error("Input file '%s' not found"%args.input)

    print('loading deoverlapnet...')
    deoverlapnet, slicqt, islicqt = load_deoverlapnet(args.deoverlapnet_path)

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

    C = slicqt(signal)

    Cmag, C_phase = transforms.complex_2_magphase(C)
    Cmag_ola = slicqt.overlap_add(Cmag)
    nb_slices = Cmag[0].shape[-2]

    ragged_ola_shapes = [Cmag_ola_.shape for Cmag_ola_ in Cmag_ola]

    Cmag_interp_ola = torch.squeeze(torch.mean(slicqt.interpolate(Cmag_ola), dim=1), dim=0)
    print(f'Cmag_interp_ola: {Cmag_interp_ola.shape}, {Cmag_interp_ola.device}, {Cmag_interp_ola.dtype}')

    H, P = librosa.decompose.hpss(Cmag_interp_ola)

    H = torch.unsqueeze(torch.cat([torch.unsqueeze(H, dim=0),torch.unsqueeze(H, dim=0)], dim=0), dim=0)
    P = torch.unsqueeze(torch.cat([torch.unsqueeze(P, dim=0),torch.unsqueeze(P, dim=0)], dim=0), dim=0)

    Hmag = deoverlapnet(slicqt.deinterpolate(H, ragged_ola_shapes), nb_slices, ragged_ola_shapes)
    Pmag = deoverlapnet(slicqt.deinterpolate(P, ragged_ola_shapes), nb_slices, ragged_ola_shapes)

    Hhat = transforms.magphase_2_complex(Hmag, C_phase)
    Phat = transforms.magphase_2_complex(Pmag, C_phase)

    h_waveform = islicqt(Hhat, signal.shape[-1]).detach().cpu().numpy()
    p_waveform = islicqt(Phat, signal.shape[-1]).detach().cpu().numpy()

    scipy.io.wavfile.write("harmonic.wav", fs, h_waveform)
    scipy.io.wavfile.write("percussive.wav", fs, p_waveform)
