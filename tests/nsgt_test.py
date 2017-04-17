import numpy as np
from nsgt import CQ_NSGT
import unittest

class Test_CQ_NSGT(unittest.TestCase):

    def test_transform(self, length=100000, fmin=50, fmax=22050, bins=12, fs=44100):
        s = np.random.random(length)
        nsgt = CQ_NSGT(fmin, fmax, bins, fs, length)
        
        # forward transform 
        c = nsgt.forward(s)
        # inverse transform 
        s_r = nsgt.backward(c)
        
        self.assertTrue(np.allclose(s, s_r, atol=1e-07))

def load_tests(*_):
    # seed random generators for unit testing
    np.random.seed(666)

    test_cases = unittest.TestSuite()
    for _ in range(100):
        test_cases.addTest(Test_CQ_NSGT('test_transform'))
    return test_cases

if __name__ == "__main__":
    unittest.main()
