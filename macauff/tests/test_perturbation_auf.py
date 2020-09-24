# Licensed under a 3-clause BSD style license - see LICENSE
'''
Tests for the "perturbation_auf" module.
'''

import pytest
import os
from numpy.testing import assert_allclose
import numpy as np

from ..matching import CrossMatch
from ..perturbation_auf_fortran import perturbation_auf_fortran as paf


def test_haversine_formula():
    a, b, c, d = 0, 0, 0, 0
    e = paf.haversine(a, b, c, d)
    assert e == 0

    a, b, c, d = 0.0, 1/3600, 0, 0
    e = paf.haversine(a, b, c, d)
    assert_allclose(e, 1/3600)

    a, b, c, d = 0.0, 1/3600, 2.45734, 2.45734
    e = paf.haversine(a, b, c, d)
    assert_allclose(e, 1/3600*np.cos(np.radians(c)))

    a, b, c, d = 25.10573, 25.30573, 15.48349, 15.48349
    e = paf.haversine(b, a, c, d)
    # Reduce tolerance with DeltaRA cos(Dec) a worse approximation at larger Dec.
    assert_allclose(e, 0.2*np.cos(np.radians(c)), rtol=1e-5)

    a, b, c, d = 51.45734, 51.45734, 0, 1/3600
    e = paf.haversine(a, b, c, d)
    assert_allclose(e, 1/3600)
    e = paf.haversine(a, b, d, c)
    assert_allclose(e, 1/3600)

    c, d = 86.47239, 85.47239
    e = paf.haversine(a, b, d, c)
    assert_allclose(e, 1)


def test_closest_auf_point():
    auf_points = np.array([[1, 1], [50, 50]])
    source_points = np.array([[3, 2], [80, 50], [40, 30]])
    inds = paf.find_nearest_auf_point(source_points[:, 0], source_points[:, 1],
                                      auf_points[:, 0], auf_points[:, 1])
    assert np.all(inds == np.array([0, 1, 1]))
