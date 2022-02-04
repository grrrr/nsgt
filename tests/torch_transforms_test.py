import pytest
import numpy as np
import torch
from nsgt_torch.cq import NSGTBase, make_nsgt_filterbanks
from nsgt_torch.slicq import SliCQTBase, make_slicqt_filterbanks
import auraloss


# try some durations
@pytest.fixture(params=[4096, 44100, int(44100*2)])
def nb_timesteps(request):
    return int(request.param)


@pytest.fixture(params=[1, 2])
def nb_channels(request):
    return request.param


@pytest.fixture(params=[1, 5, 13, 56])
def nb_samples(request):
    return request.param


@pytest.fixture
def audio(request, nb_samples, nb_channels, nb_timesteps):
    return torch.rand((nb_samples, nb_channels, nb_timesteps), dtype=torch.float64, device="cpu")


def test_nsgt_cpu_fwd_inv_ragged(audio):
    audio_n_samples = audio.shape[-1]

    nsgt, insgt = make_nsgt_filterbanks(NSGTBase("mel", 200, 32.9, audio_n_samples, device="cpu"))

    X, *_ = nsgt(audio)

    out = insgt(X, audio_n_samples)

    print(auraloss.time.SNRLoss()(audio, out))
    assert np.sqrt(np.mean((audio.detach().numpy() - out.detach().numpy()) ** 2)) < 1e-6


def test_nsgt_cpu_fwd_inv_zeropad(audio):
    audio_n_samples = audio.shape[-1]

    nsgt, insgt = make_nsgt_filterbanks(NSGTBase("mel", 200, 32.9, audio_n_samples, matrixform="zeropad", device="cpu"))

    X, *_ = nsgt(audio)

    out = insgt(X, audio_n_samples)

    print(auraloss.time.SNRLoss()(audio, out))
    assert np.sqrt(np.mean((audio.detach().numpy() - out.detach().numpy()) ** 2)) < 1e-6


def test_nsgt_cpu_fwd_inv_interpolate(audio):
    audio_n_samples = audio.shape[-1]

    nsgt, insgt = make_nsgt_filterbanks(NSGTBase("mel", 200, 32.9, audio_n_samples, matrixform="interpolate", device="cpu"))

    X, ragged_shapes = nsgt(audio)

    out = insgt(X, audio_n_samples, ragged_shapes=ragged_shapes)

    print(auraloss.time.SNRLoss()(audio, out))
    assert np.sqrt(np.mean((audio.detach().numpy() - out.detach().numpy()) ** 2)) < 1e-1


def test_slicqt_cpu_fwd_inv_ragged(audio):
    audio_n_samples = audio.shape[-1]

    slicqt, islicqt = make_slicqt_filterbanks(SliCQTBase("mel", 200, 32.9, device="cpu"))

    X, *_ = slicqt(audio)

    out = islicqt(X, audio_n_samples)

    print(auraloss.time.SNRLoss()(audio, out))
    assert np.sqrt(np.mean((audio.detach().numpy() - out.detach().numpy()) ** 2)) < 1e-6


def test_slicqt_cpu_fwd_inv_zeropad(audio):
    audio_n_samples = audio.shape[-1]

    slicqt, islicqt = make_slicqt_filterbanks(SliCQTBase("mel", 200, 32.9, matrixform="zeropad", device="cpu"))

    X, *_ = slicqt(audio)

    out = islicqt(X, audio_n_samples)

    print(auraloss.time.SNRLoss()(audio, out))
    assert np.sqrt(np.mean((audio.detach().numpy() - out.detach().numpy()) ** 2)) < 1e-6


def test_slicqt_cpu_fwd_inv_interpolate(audio):
    audio_n_samples = audio.shape[-1]

    slicqt, islicqt = make_slicqt_filterbanks(SliCQTBase("mel", 200, 32.9, matrixform="interpolate", device="cpu"))

    X, ragged_shapes = slicqt(audio)

    out = islicqt(X, audio_n_samples, ragged_shapes=ragged_shapes)

    print(auraloss.time.SNRLoss()(audio, out))
    assert np.sqrt(np.mean((audio.detach().numpy() - out.detach().numpy()) ** 2)) < 1e-1


import pytest
pytest.main(["-s", "tests/torch_transforms_test.py::test_nsgt_cpu_fwd_inv_ragged"])
