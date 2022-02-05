import pytest
import numpy as np
import torch
from nsgt_torch.cq import NSGTBase, make_nsgt_filterbanks
from nsgt_torch.slicq import SliCQTBase, make_slicqt_filterbanks
from nsgt_torch.util import magphase_2_complex, complex_2_magphase
from nsgt import NSGT_sliced, NSGT
import auraloss
import sys


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


#def test_nsgt_compare_with_old_ragged(audio):
#    audio_n_samples = audio.shape[-1]
#
#    nsgt, insgt = make_nsgt_filterbanks(NSGTBase("mel", 200, 32.9, audio_n_samples, device="cpu"))
#
#    X, *_ = nsgt(audio)
#
#    out = insgt(X, audio_n_samples)
#
#    print(auraloss.time.SNRLoss()(audio, out))
#    assert np.sqrt(np.mean((audio.detach().numpy() - out.detach().numpy()) ** 2)) < 1e-6
#
#
#def test_nsgt_compare_with_old_matrixform(audio):
#    audio_n_samples = audio.shape[-1]
#
#    nsgt, insgt = make_nsgt_filterbanks(NSGTBase("mel", 200, 32.9, audio_n_samples, matrixform="zeropad", device="cpu"))
#
#    X, *_ = nsgt(audio)
#
#    out = insgt(X, audio_n_samples)
#
#    print(auraloss.time.SNRLoss()(audio, out))
#    assert np.sqrt(np.mean((audio.detach().numpy() - out.detach().numpy()) ** 2)) < 1e-6
#
#
#def test_slicqt_compare_with_old_ragged(audio):
#    audio_n_samples = audio.shape[-1]
#
#    slicqt, islicqt = make_slicqt_filterbanks(SliCQTBase("mel", 200, 32.9, device="cpu"))
#
#    X, *_ = slicqt(audio)
#
#    out = islicqt(X, audio_n_samples)
#
#    print(auraloss.time.SNRLoss()(audio, out))
#    assert np.sqrt(np.mean((audio.detach().numpy() - out.detach().numpy()) ** 2)) < 1e-6
#
#
#def test_slicqt_compare_with_old_matrixform(audio):
#    audio_n_samples = audio.shape[-1]
#
#    slicqt, islicqt = make_slicqt_filterbanks(SliCQTBase("mel", 200, 32.9, matrixform="zeropad", device="cpu"))
#
#    X, *_ = slicqt(audio)
#
#    out = islicqt(X, audio_n_samples)
#
#    print(auraloss.time.SNRLoss()(audio, out))
#    assert np.sqrt(np.mean((audio.detach().numpy() - out.detach().numpy()) ** 2)) < 1e-6


def test_nsgt_magphase_complex_roundtrip_matrixform(audio):
    audio_n_samples = audio.shape[-1]

    nsgt, insgt = make_nsgt_filterbanks(NSGTBase("mel", 200, 32.9, audio_n_samples, matrixform="zeropad", device="cpu"))

    X, ragged_shapes = nsgt(audio)

    Xmag, Xphase = complex_2_magphase(X)
    X_hat = magphase_2_complex(Xmag, Xphase)

    out = insgt(X_hat, audio_n_samples, ragged_shapes=ragged_shapes)

    err = 0.
    for i, X_hat_block in enumerate(X_hat):
        err += np.sqrt(np.mean((X_hat_block.detach().numpy() - X[i].detach().numpy()) ** 2))

    err /= len(X_hat)
    assert err < 1e-6

    print(auraloss.time.SNRLoss()(audio, out))
    assert np.sqrt(np.mean((audio.detach().numpy() - out.detach().numpy()) ** 2)) < 1e-6


def test_nsgt_magphase_complex_roundtrip_ragged(audio):
    audio_n_samples = audio.shape[-1]

    nsgt, insgt = make_nsgt_filterbanks(NSGTBase("mel", 200, 32.9, audio_n_samples, matrixform="ragged", device="cpu"))

    X, ragged_shapes = nsgt(audio)

    Xmag, Xphase = complex_2_magphase(X)
    X_hat = magphase_2_complex(Xmag, Xphase)

    out = insgt(X_hat, audio_n_samples, ragged_shapes=ragged_shapes)

    err = 0.
    for i, X_hat_block in enumerate(X_hat):
        err += np.sqrt(np.mean((X_hat_block.detach().numpy() - X[i].detach().numpy()) ** 2))

    err /= len(X_hat)
    assert err < 1e-6

    print(auraloss.time.SNRLoss()(audio, out))
    assert np.sqrt(np.mean((audio.detach().numpy() - out.detach().numpy()) ** 2)) < 1e-6


def test_slicqt_magphase_complex_roundtrip_matrixform(audio):
    audio_n_samples = audio.shape[-1]

    slicqt, islicqt = make_slicqt_filterbanks(SliCQTBase("mel", 200, 32.9, matrixform="zeropad", device="cpu"))

    X, ragged_shapes = slicqt(audio)

    Xmag, Xphase = complex_2_magphase(X)
    X_hat = magphase_2_complex(Xmag, Xphase)

    out = islicqt(X_hat, audio_n_samples, ragged_shapes=ragged_shapes)

    err = 0.
    for i, X_hat_block in enumerate(X_hat):
        err += np.sqrt(np.mean((X_hat_block.detach().numpy() - X[i].detach().numpy()) ** 2))

    err /= len(X_hat)

    print(auraloss.time.SNRLoss()(audio, out))
    assert np.sqrt(np.mean((audio.detach().numpy() - out.detach().numpy()) ** 2)) < 1e-6


def test_slicqt_magphase_complex_roundtrip_ragged(audio):
    audio_n_samples = audio.shape[-1]

    slicqt, islicqt = make_slicqt_filterbanks(SliCQTBase("mel", 200, 32.9, matrixform="ragged", device="cpu"))

    X, ragged_shapes = slicqt(audio)

    Xmag, Xphase = complex_2_magphase(X)
    X_hat = magphase_2_complex(Xmag, Xphase)

    out = islicqt(X_hat, audio_n_samples, ragged_shapes=ragged_shapes)

    err = 0.
    for i, X_hat_block in enumerate(X_hat):
        err += np.sqrt(np.mean((X_hat_block.detach().numpy() - X[i].detach().numpy()) ** 2))

    err /= len(X_hat)
    assert err < 1e-6

    print(auraloss.time.SNRLoss()(audio, out))
    assert np.sqrt(np.mean((audio.detach().numpy() - out.detach().numpy()) ** 2)) < 1e-6


import pytest
pytest.main(["-s", "tests/torch_misc_test.py::test_magphase_complex_roundtrip_matrixform"])
