# Licensed under a 3-clause BSD style license - see LICENSE
'''
Tests for the "get_trilegal_wrapper" module.
'''

from numpy.testing import assert_allclose

from ..get_trilegal_wrapper import get_AV_infinity


def test_get_av_infinity():
    for l, b, check_av in zip([10, 312, 48, 96], [4, 1, -30, -78], [2.793, 11.046, 0.198, 0.058]):
        assert_allclose(check_av, get_AV_infinity(l, b, frame='galactic'), atol=1e-3)
