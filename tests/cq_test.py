import numpy as np
from nsgt import NSGT, OctScale
import unittest

class TestNSGT(unittest.TestCase):

    def test_oct(self):
        siglen = int(10**np.random.uniform(4,6))
        sig = np.random.random(siglen)
        fmin = np.random.random()*200+20
        fmax = np.random.random()*(22048-fmin)+fmin
        obins = np.random.randint(24)+1
        scale = OctScale(fmin,fmax,obins)
        nsgt = NSGT(scale, fs=44100, Ls=len(sig))
        c = nsgt.forward(sig)
        s_r = nsgt.backward(c)
        self.assertTrue(np.allclose(sig, s_r, atol=1e-07))

def load_tests(*_):
    # seed random generators for unit testing
    np.random.seed(666)

    test_cases = unittest.TestSuite()
    for _ in range(100):
        test_cases.addTest(TestNSGT('test_oct'))
    return test_cases

if __name__ == '__main__':
    unittest.main()
