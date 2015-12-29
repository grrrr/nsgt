import numpy as np
from nsgt.fft import rfftp, irfftp, fftp, ifftp
import unittest

class TestFFT(unittest.TestCase):
    def __init__(self, methodName, n=10000):
        super(TestFFT, self).__init__(methodName)
        self.n = n

    def test_rfft(self):
        seq = np.random.random(self.n)
        ft = rfftp()
        a = ft(seq)
        b = np.fft.rfft(seq)
        self.assertTrue(np.allclose(a, b))
        
    def test_irfft(self):
        seq = np.random.random(self.n)+np.random.random(self.n)*1.j
        outn = (self.n-1)*2 + np.random.randint(0,2) # even or odd output size
        ft = irfftp()
        a = ft(seq, outn=outn)
        b = np.fft.irfft(seq, n=outn)
        self.assertTrue(np.allclose(a, b))
        
    def test_fft(self):
        seq = np.random.random(self.n)
        ft = fftp()
        a = ft(seq)
        b = np.fft.fft(seq)
        self.assertTrue(np.allclose(a, b))
        
    def test_ifft(self):
        seq = np.random.random(self.n)+np.random.random(self.n)*1.j
        ft = ifftp()
        a = ft(seq)
        b = np.fft.ifft(seq)
        self.assertTrue(np.allclose(a, b))

def load_tests(*_):
    # seed random generators for unit testing
    np.random.seed(666)

    test_cases = unittest.TestSuite()
    for _ in range(100):
        l = int(10*np.random.uniform(2, 5))
        test_cases.addTest(TestFFT('test_rfft', l))
        test_cases.addTest(TestFFT('test_irfft', l))
        test_cases.addTest(TestFFT('test_fft', l))
        test_cases.addTest(TestFFT('test_ifft', l))
    return test_cases

if __name__ == '__main__':
    unittest.main()
