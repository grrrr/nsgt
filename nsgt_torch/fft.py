import numpy as np
from warnings import warn
import torch


class fftp:
    def __init__(self):
        pass

    def __call__(self, x, outn=None, ref=False):
        return torch.fft.fft(x)


class ifftp:
    def __init__(self):
        pass
    def __call__(self,x, outn=None, n=None, ref=False):
        return torch.fft.ifft(x, n=n)


class rfftp:
    def __init__(self):
        pass

    def __call__(self,x, outn=None, ref=False):
        return torch.fft.rfft(x)


class irfftp:
    def __init__(self):
        pass
    def __call__(self,x,outn=None,ref=False):
        return torch.fft.irfft(x,n=outn)
