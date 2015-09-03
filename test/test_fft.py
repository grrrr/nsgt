import numpy as np
from nsgt.fft import rfftp, irfftp, fftp, ifftp
import unittest

class TestFFT(unittest.TestCase):
    def test_rfft(self, n=1000):
        seq = np.random.random(n)
        ft = rfftp()
        a = ft(seq)
        b = np.fft.rfft(seq)
        self.assertTrue(np.allclose(a, b))
        
    def test_irfft(self, n=1000):
        seq = np.random.random(n)+np.random.random(n)*1.j
        ft = irfftp()
        a = ft(seq)
        b = np.fft.irfft(seq)
        self.assertTrue(np.allclose(a, b))
        
    def test_fft(self, n=1000):
        seq = np.random.random(n)
        ft = fftp()
        a = ft(seq)
        b = np.fft.fft(seq)
        self.assertTrue(np.allclose(a, b))
        
    def test_ifft(self, n=1000):
        seq = np.random.random(n)+np.random.random(n)*1.j
        ft = ifftp()
        a = ft(seq)
        b = np.fft.ifft(seq)
        self.assertTrue(np.allclose(a, b))

if __name__ == '__main__':
    unittest.main()
