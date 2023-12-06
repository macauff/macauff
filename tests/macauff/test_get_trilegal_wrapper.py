# Licensed under a 3-clause BSD style license - see LICENSE
'''
Tests for the "get_trilegal_wrapper" module.
'''

import pytest
from numpy.testing import assert_allclose

# pylint: disable-next=no-name-in-module,import-error
from macauff.get_trilegal_wrapper import get_av_infinity


@pytest.mark.remote_data
def test_get_av_infinity():
    for l, b, check_av in zip([10, 312, 48, 96], [4, 1, -30, -78], [2.793, 11.046, 0.198, 0.058]):
        assert_allclose(check_av, get_av_infinity(l, b, frame='galactic'), atol=1e-3)

    assert_allclose([2.793, 11.046, 0.198, 0.058],
                    get_av_infinity([10, 312, 48, 96], [4, 1, -30, -78], frame='galactic'),
                    atol=1e-3)
