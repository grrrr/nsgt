import numpy as np
from nsgt import NSGT, OctScale
import unittest

class TestNSGT(unittest.TestCase):

    def test_oct(self):
        for _ in range(100):
            sig = np.random.random(100000)
            fmin = np.random.random()*200+1
            fmax = np.random.random()*(22048-fmin)+fmin
            obins = np.random.randint(24)+1
            scale = OctScale(fmin,fmax,obins)
            nsgt = NSGT(scale, fs=44100, Ls=len(sig))
            c = nsgt.forward(sig)
            s_r = nsgt.backward(c)
            self.assertTrue(np.allclose(sig, s_r))

if __name__ == '__main__':
    unittest.main()
