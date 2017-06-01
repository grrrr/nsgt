"""

    Utilities for the Examples
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import os
import wave
import librosa


def load_audio(path, mono=True):
    with wave.open(path, "rb") as wave_file:
        fs = wave_file.getframerate()
    s, _ = librosa.load(path=path, sr=fs, mono=mono)
    return s, fs


def save_audio(path, sr, data):
    if path.shape[0] == 2:
        # convert to mono.
        to_save = data.sum(axis=1) / 2
    else:
        to_save = sr
    librosa.output.write_wav(path=path, y=to_save, sr=sr)


def cputime():
    utime, stime, cutime, cstime, elapsed_time = os.times()
    return utime
