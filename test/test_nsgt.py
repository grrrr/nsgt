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
        
        self.assertTrue(np.allclose(s, s_r))

if __name__ == "__main__":
    unittest.main()
