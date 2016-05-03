import numpy as np
from nsgt import NSGT_sliced, OctScale
import unittest

# reproducible noise signal
maxlen = 50000
np.random.seed(666)
rndsig = np.random.random((maxlen,))
#assert np.allclose(rndsig[:3],[ 0.70043712, 0.84418664, 0.67651434])

class TestNSGT_slices(unittest.TestCase):

    def runit(self, siglen, fmin, fmax, obins, sllen, trlen, real):
        sig = rndsig[:siglen]

        scale = OctScale(fmin, fmax, obins)
        nsgt = NSGT_sliced(scale, fs=44100, sl_len=sllen, tr_area=trlen, real=real)

        c = nsgt.forward((sig,))

        rc = nsgt.backward(c)

        s_r = np.concatenate(list(map(list,rc)))[:len(sig)]
        
        close = np.allclose(sig, s_r, atol=1.e-3)
        if not close:
            print("Failing params:", siglen, fmin, fmax, obins, sllen, trlen, real)
            dev = np.abs(s_r-sig)
            print("Error", np.where(dev>1.e-3), np.max(dev))
        self.assertTrue(close)


    def test_1d1(self):
        self.runit(*list(map(int,"100000 100 18200 2 20000 5000 1".split())))
        
    def test_1d11(self):
        self.runit(*list(map(int,"100000 80 18200 6 20000 5000 1".split())))
        
    def test_1(self):
        self.runit(*list(map(int,"100000 99 19895 6 84348 5928 1".split())))
        
    def test_1a(self):
        self.runit(*list(map(int,"100000 99 19895 6 84348 5928 0".split())))
        
    def test_1b(self):
        self.runit(*list(map(int,"100000 100 20000 6 80000 5000 1".split())))
        
    def test_1c(self):
        self.runit(*list(map(int,"100000 100 19000 6 80000 5000 1".split())))
        
    def test_1d2(self):
        self.runit(*list(map(int,"100000 100 18100 6 20000 5000 1".split())))
        
    def test_1e(self):
        self.runit(*list(map(int,"100000 100 18000 6 20000 5000 1".split())))
        
    def test_err1(self):
        self.runit(*list(map(int,"30549 104 11970 25 7286 2030 1".split())))
        
    def test_err2(self):
        self.runit(*list(map(int,"19746 88 19991 21 12674 4030 0".split())))
        
    def test_err3(self):
        self.runit(*list(map(int,"92507 114 18387 20 29306 11848 1".split())))
        
    def test_err4(self):
        self.runit(*list(map(int,"20724 191 2843 16 22354 6590 1".split())))
        
    def test_err5(self):
        self.runit(*list(map(int,"10712 97 19363 3 10238 1876 1".split())))
        
    def test_err6(self):
        self.runit(*list(map(int,"262597 100 15786 16 2858 556 1".split())))
        
    def gtest_oct(self):
        siglen = int(10**np.random.uniform(4,6))
        fmin = np.random.randint(200)+80
        fmax = max(np.random.randint(22048-fmin)+fmin,10000)
        obins = np.random.randint(24)+1
        sllen = max(1000,np.random.randint(maxlen))//4*4
        trlen = max(2,np.random.randint(sllen//2-2))//2*2
        real = np.random.randint(2)
        self.runit(siglen, fmin, fmax, obins, sllen, trlen, real)

def load_tests(*_):
    test_cases = unittest.TestSuite()
#        for t in 'test_1d1 test_1d11 test_1 test_1a test_1b test_1c test_1d2 test_1e test_err1 test_err2 test_err3 test_err4 test_err5 test_err6'.split():
#            test_cases.addTest(TestNSGT_slices(t))
    
    for _ in range(50):
        test_cases.addTest(TestNSGT_slices('gtest_oct'))
    return test_cases

if __name__ == '__main__':
    unittest.main()
