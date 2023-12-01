# Licensed under a 3-clause BSD style license - see LICENSE
'''
Tests for the "shared_library" module.
'''

from numpy.testing import assert_allclose
import numpy as np

from macauff.misc_functions_fortran import misc_functions_fortran as mff


def test_haversine_formula():
    a, b, c, d = 0, 0, 0, 0
    e = mff.haversine_wrapper(a, b, c, d)
    assert e == 0

    a, b, c, d = 0.0, 1/3600, 0, 0
    e = mff.haversine_wrapper(a, b, c, d)
    assert_allclose(e, 1/3600)

    a, b, c, d = 0.0, 1/3600, 2.45734, 2.45734
    e = mff.haversine_wrapper(a, b, c, d)
    assert_allclose(e, 1/3600*np.cos(np.radians(c)))

    a, b, c, d = 25.10573, 25.30573, 15.48349, 15.48349
    e = mff.haversine_wrapper(b, a, c, d)
    assert_allclose(e, 0.2*np.cos(np.radians(c)))

    a, b, c, d = 51.45734, 51.45734, 0, 1/3600
    e = mff.haversine_wrapper(a, b, c, d)
    assert_allclose(e, 1/3600)
    e = mff.haversine_wrapper(a, b, d, c)
    assert_allclose(e, 1/3600)

    c, d = 86.47239, 85.47239
    e = mff.haversine_wrapper(a, b, d, c)
    assert_allclose(e, 1)

    a, b, c, d = 0, 180, -45, 45
    e = mff.haversine_wrapper(a, b, c, d)
    assert_allclose(e, 180)

    a, b, c, d = 10, 250, -50, 65
    e = mff.haversine_wrapper(a, b, c, d)
    sin_half_lat = np.sin(np.radians((c - d)/2))
    cos_lat_1 = np.cos(np.radians(c))
    cos_lat_2 = np.cos(np.radians(d))
    sin_half_lon = np.sin(np.radians((a - b)/2))
    assert_allclose(e, 2 * np.degrees(np.arcsin(np.sqrt(sin_half_lat**2 +
                    cos_lat_1*cos_lat_2*sin_half_lon**2))))
