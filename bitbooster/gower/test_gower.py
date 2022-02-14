from unittest import TestCase

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal, assert_array_almost_equal

from bitbooster.gower.vanilla import GowerVanillaObject


class Test(TestCase):

    def test_gower_abc(self):
        df = pd.DataFrame()
        df['a'] = list('abc')
        z = GowerVanillaObject(df)
        m = z.get_sub_distance_matrix(None, None)
        assert_array_equal(m[0], [0, 1, 1])
        assert_array_equal(m[1], [1, 0, 1])
        assert_array_equal(m[2], [1, 1, 0])

    def test_gower_123(self):
        df = pd.DataFrame()
        df['a'] = [0, 1, 2]
        z = GowerVanillaObject(df)
        m = z.get_sub_distance_matrix(None, None)
        assert_array_equal(m[0], [0, .5, 1.])
        assert_array_equal(m[1], [.5, 0, .5])
        assert_array_equal(m[2], [1., .5, 0])

    def test_gower_012ab(self):
        df = pd.DataFrame()
        df['a'] = [0, 0, 1, 1, 2, 2]
        df['b'] = list('ABABAB')

        z = GowerVanillaObject(df)
        m = z.get_sub_distance_matrix(None, None)
        assert_array_equal(m[0], [0.0, 1.0, 0.5, 1.5, 1.0, 2.0])
        assert_array_equal(m[1], [1.0, 0.0, 1.5, 0.5, 2.0, 1.0])
        assert_array_equal(m[2], [0.5, 1.5, 0.0, 1.0, 0.5, 1.5])
        assert_array_equal(m[3], [1.5, 0.5, 1.0, 0.0, 1.5, 0.5])
        assert_array_equal(m[4], [1.0, 2.0, 0.5, 1.5, 0.0, 1.0])
        assert_array_equal(m[5], [2.0, 1.0, 1.5, 0.5, 1.0, 0.0])

    def test_gower_medium(self):
        """
        Based on https://medium.com/analytics-vidhya/gowers-distance-899f9c4bd553
        """
        # Input data
        df = pd.DataFrame({'age': [14, 19, 10, 14, 21, 19, 30, 35],
                           'preTestScore': [4, 24, 31, 3, 3, 4, 31, 9],
                           'postTestScore': [25, 94, 57, 30, 70, 25, 69, 95],
                           'available_credit': [2200, 1000, 22000, 2100, 2000, 1000, 6000, 2200],
                           'gender': ['M', 'M', 'M', 'M', 'F', 'F', 'F', 'F']})

        # Gower package
        import gower
        dm1 = gower.gower_matrix(df)

        # Our implementation
        z = GowerVanillaObject(df)
        # Our implementation does not normalize for the number of dimensions
        dm2 = z.get_sub_distance_matrix(None, None) / 5

        # Compare values
        assert_array_almost_equal(dm1, dm2, decimal=6)
