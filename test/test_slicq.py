import numpy as np
from nsgt import NSGT_sliced, OctScale
import unittest

class TestNSGT_slices(unittest.TestCase):

    def runit(self, siglen, fmin, fmax, obins, sllen, trlen, real):
        sig = np.random.random((siglen,))
        scale = OctScale(fmin, fmax, obins)
        nsgt = NSGT_sliced(scale, fs=44100, sl_len=sllen, tr_area=trlen, real=real)
        c = nsgt.forward((sig,))

        rc = nsgt.backward(c)

        s_r = np.concatenate(map(list,rc))[:len(sig)]
        
        self.assertTrue(np.allclose(sig, s_r))


    def test_1d1(self):
        self.runit(*map(int,"100000 100 18200 2 20000 5000 1".split())) # fail
        
    def test_1d11(self):
        self.runit(*map(int,"100000 80 18200 6 20000 5000 1".split())) # success
        
    def test_1(self):
        self.runit(*map(int,"100000 99 19895 6 84348 5928 1".split()))  # fail
        
    def test_1a(self):
        self.runit(*map(int,"100000 99 19895 6 84348 5928 0".split()))  # success
        
    def test_1b(self):
        self.runit(*map(int,"100000 100 20000 6 80000 5000 1".split()))  # fail
        
    def test_1c(self):
        self.runit(*map(int,"100000 100 19000 6 80000 5000 1".split())) # fail
        
    def test_1d2(self):
        self.runit(*map(int,"100000 100 18100 6 20000 5000 1".split())) # success
        
    def test_1e(self):
        self.runit(*map(int,"100000 100 18000 6 20000 5000 1".split())) # success
        
    def gtest_oct(self):
        for _ in range(100):
            siglen = 100000
            fmin = np.random.randint(200)+30
            fmax = np.random.randint(22048-fmin)+fmin
            obins = np.random.randint(24)+1
            sllen = max(1,np.random.randint(50000))*2
            trlen = max(2,np.random.randint(sllen//2-2))//2*2
            real = np.random.randint(2)
            self.runit(siglen, fmin, fmax, obins, sllen, trlen, real)

if __name__ == '__main__':
    unittest.main()
