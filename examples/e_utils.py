"""

    Utilities for the Examples
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import os
import wave
import librosa
import numpy as np


def load_audio(path, mono=True, method=-1):
    """

    Load a *.wav* audio file.

    Args:
        path : str
            The path to the audio file.
        mono : bool (optional)
            If `method=1`, instruct `librosa.mono` to
            convert the audio to mono.
        method : int
            If `-1` use `librosa` to load the audio;
            otherwise use `nsgt.utilities.audio.SndReader`.

    Returns:
        s : array
            the audio file as a numpy array.
        fs : int
            sample rate of the audio.

    WARNING
    -------
    `method=-1` will not resample the audio.
    Therefore, `method != -1` is currently preferable,
    but it requires that `scikits.audiolab` is installed.

    """
    if method == -1:
        with wave.open(path, "rb") as wave_file:
            fs = wave_file.getframerate()
        s, _ = librosa.load(path=path, sr=fs, mono=mono)
    else:
        from scikits.audiolab import Sndfile  # will raise if not installed
        sf = Sndfile(path)
        fs = sf.samplerate
        s = sf.read_frames(sf.nframes)
        if sf.channels > 1:
            s = np.mean(s, axis=1)
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
