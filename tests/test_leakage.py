"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest


# @pytest.mark.parametrize(
#     "test, expected",
#     [
#         ([ [0, 0], [0, 0], [0, 0] ], [0, 0]),
#         ([ [1, 2], [3, 4], [5, 6] ], [3, 4]),
#     ])
# def test_speckle_mean(test, expected):
def test_speckle_mean():
    """Test mean of multiplicative speckle to be 1"""
    from src.leakage.velocity_leakage import S1DopplerLeakage
    temp_class = S1DopplerLeakage(filename = '', random_state=42) 

    speckle_mean = temp_class.speckle_noise(noise_shape=(1000, 1000)).mean()
    expected = 1.001491

    npt.assert_almost_equal(speckle_mean, expected, decimal=5)