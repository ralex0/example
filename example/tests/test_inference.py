from inference.prior import LogUniformPrior, LogJefferysPrior

import numpy as np

import unittest
from unittest import TestCase


class TestPriors(TestCase):
    def test_uniform(self):
        assert np.allclose(np.exp(LogUniformPrior(3, 5)(4)), .5)

    def test_jefferys(self):
        assert np.allclose(np.exp(LogJefferysPrior(10, 1000)(100)),
                           0.0021714724095162588)

if __name__ == '__main__':
    unittest.main()
