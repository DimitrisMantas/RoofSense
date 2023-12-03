#          Copyright © 2023 Dimitris Mantas
#
#          This file is part of RoofSense.
#
#          This program is free software: you can redistribute it and/or modify
#          it under the terms of the GNU General Public License as published by
#          the Free Software Foundation, either version 3 of the License, or
#          (at your option) any later version.
#
#          This program is distributed in the hope that it will be useful,
#          but WITHOUT ANY WARRANTY; without even the implied warranty of
#          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#          GNU General Public License for more details.
#
#          You should have received a copy of the GNU General Public License
#          along with this program.  If not, see <https://www.gnu.org/licenses/>.


import unittest

import numpy as np

import utils


class TestMath(unittest.TestCase):

    def test_dbmean_float(self):
        self.assertAlmostEqual(utils.math.dbmean(0), 1)

    def test_dbmean_array(self):
        self.assertAlmostEqual(utils.math.dbmean(np.array([0, 10])),
                               # 1 dB ≡ 100%, 10 db ≡ 1000% => μ = 1100% / 2 = 550% = 5.5 dB
                               5.5)

    def test_dbratios_float(self):
        self.assertAlmostEqual(utils.math.dbratios(0), 1)

    def test_dbratios_array(self):
        np.testing.assert_array_almost_equal(utils.math.dbratios(np.array([0, 10])),
                                             # NOTE: See previous comment.
                                             np.array([1, 10]))

    def test_decibels_float(self):
        self.assertAlmostEqual(utils.math.decibels(0),
                               # NOTE: Logarithms are not defined in ℝ_{≤0}.
                               -np.inf)

    def test_decibels_array(self):
        np.testing.assert_array_almost_equal(utils.math.decibels(np.array([0, 1])),
                                             # NOTE: See previous comment.
                                             np.array([-np.inf, 0]))


if __name__ == '__main__':
    unittest.main()
