# Licensed under a 3-clause BSD style license - see LICENSE
'''
Tests for the "misc_functions" module.
'''

import os
import numpy as np
from numpy.testing import assert_allclose
import scipy.special

from ..misc_functions import (create_fourier_offsets_grid, load_small_ref_ind_fourier_grid,
                              hav_dist_constant_lat)
from ..misc_functions_fortran import misc_functions_fortran as mff
from .test_misc_functions_fortran import haversine_wrapper


def test_haversine_formula():
    a, b, c, d = 0, 0, 0, 0
    e = haversine_wrapper(a, b, c, d)
    assert e == 0

    a, b, c, d = 0.0, 1/3600, 0, 0
    e = haversine_wrapper(a, b, c, d)
    assert_allclose(e, 1/3600)

    a, b, c, d = 0.0, 1/3600, 2.45734, 2.45734
    e = haversine_wrapper(a, b, c, d)
    assert_allclose(e, 1/3600*np.cos(np.radians(c)))

    a, b, c, d = 25.10573, 25.30573, 15.48349, 15.48349
    e = haversine_wrapper(b, a, c, d)
    assert_allclose(e, 0.2*np.cos(np.radians(c)))

    a, b, c, d = 51.45734, 51.45734, 0, 1/3600
    e = haversine_wrapper(a, b, c, d)
    assert_allclose(e, 1/3600)
    e = haversine_wrapper(a, b, d, c)
    assert_allclose(e, 1/3600)

    c, d = 86.47239, 85.47239
    e = haversine_wrapper(a, b, d, c)
    assert_allclose(e, 1)

    a, b, c, d = 0, 180, -45, 45
    e = haversine_wrapper(a, b, c, d)
    assert_allclose(e, 180)

    a, b, c, d = 10, 250, -50, 65
    e = haversine_wrapper(a, b, c, d)
    sin_half_lat = np.sin(np.radians((c - d)/2))
    cos_lat_1 = np.cos(np.radians(c))
    cos_lat_2 = np.cos(np.radians(d))
    sin_half_lon = np.sin(np.radians((a - b)/2))
    assert_allclose(e, 2 * np.degrees(np.arcsin(np.sqrt(sin_half_lat**2 +
                    cos_lat_1*cos_lat_2*sin_half_lon**2))))


def test_calc_j0():
    r = np.linspace(0, 5, 5000)
    rho = np.linspace(0, 100, 5000)
    j0 = mff.calc_j0(r, rho)
    assert_allclose(j0, scipy.special.j0(2 * np.pi * r.reshape(-1, 1) * rho.reshape(1, -1)),
                    rtol=1e-5)


def test_create_fourier_offsets_grid():
    a_len = np.array([[5, 10, 5], [15, 4, 8]], order='F')
    np.save('arraylengths.npy', a_len)
    auf_pointings = np.array([[10, 20], [50, 50], [100, -40]])
    filt_names = ['W1', 'W2']
    r = np.linspace(0, 5, 10)
    for j in range(0, len(auf_pointings)):
        ax1, ax2 = auf_pointings[j]
        for i in range(0, len(filt_names)):
            filt = filt_names[i]
            os.makedirs('{}/{}/{}'.format(ax1, ax2, filt), exist_ok=True)
            np.save('{}/{}/{}/fourier.npy'.format(ax1, ax2, filt),
                    (i + len(filt_names)*j)*np.ones((len(r[:-1]), a_len[i, j]), float))

    create_fourier_offsets_grid('.', auf_pointings, filt_names, r)
    a = np.lib.format.open_memmap('{}/fourier_grid.npy'.format(
        '.'), mode='r', dtype=float, shape=(9, 15, 2, 3), fortran_order=True)
    assert np.all(a.shape == (9, 15, 2, 3))
    a_manual = -1*np.ones((9, 15, 2, 3), float, order='F')
    for j in range(0, len(auf_pointings)):
        for i in range(0, len(filt_names)):
            a_manual[:, :a_len[i, j], i, j] = i + len(filt_names)*j
    assert np.all(a == a_manual)


def test_load_small_ref_ind_fourier_grid():
    a_len = np.array([[6, 10, 7], [15, 9, 8], [7, 10, 12], [8, 8, 11]], order='F')
    auf_pointings = np.array([[10, 20], [50, 50], [100, -40]])
    filt_names = ['W1', 'W2', 'W3', 'W4']
    a = np.lib.format.open_memmap('{}/fourier_grid.npy'.format(
        '.'), mode='w+', dtype=float, shape=(9, 15, 4, 3), fortran_order=True)
    for j in range(0, len(auf_pointings)):
        for i in range(0, len(filt_names)):
            a[:, :a_len[i, j], i, j] = (i*a_len[i, j] + a_len[i, j]*len(filt_names)*j +
                                        np.arange(a_len[i, j]).reshape(1, -1))
    del a
    # Unique indices: 0, 1, 2, 5; 0, 3; 0, 1, 2
    # These map to 0, 1, 2, 3; 0, 1; 0, 1, 2
    modrefind = np.array([[0, 2, 0, 2, 1, 5], [0, 3, 3, 3, 3, 0], [0, 1, 2, 1, 2, 1]])
    a, b = load_small_ref_ind_fourier_grid(modrefind, '.')

    new_small_modrefind = np.array([[0, 2, 0, 2, 1, 3], [0, 1, 1, 1, 1, 0], [0, 1, 2, 1, 2, 1]])
    new_small_fouriergrid = np.empty((9, 4, 2, 3), float, order='F')
    for j, j_old in enumerate([0, 1, 2]):
        for i, i_old in enumerate([0, 3]):
            for k, k_old in enumerate([0, 1, 2, 5]):
                new_small_fouriergrid[:, k, i, j] = (
                    k_old + i_old*a_len[i_old, j_old] + a_len[i_old, j_old]*len(filt_names)*j_old)

    assert np.all(b.shape == (3, 6))
    assert np.all(a.shape == (9, 4, 2, 3))
    assert np.all(b == new_small_modrefind)
    assert np.all(a == new_small_fouriergrid)


def test_hav_dist_constant_lat():
    lon1s = [0, 124.1, 65.34, 180, 324, 96.34]
    lon2s = [359.1, 150.23, 165.3, 210, 10.3, 60.34]

    for lat in [-86.4, -40.3, -10.1, 0, 15.5, 45.1, 73.14, 88.54]:
        for lon1, lon2 in zip(lon1s, lon2s):
            a = haversine_wrapper(lon1, lon2, lat, lat)
            b = hav_dist_constant_lat(lon1, lat, lon2)
            assert_allclose(a, b)
