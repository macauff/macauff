# Licensed under a 3-clause BSD style license - see LICENSE
'''
Tests for the "group_sources" module.
'''

import pytest
import os
from numpy.testing import assert_allclose
import numpy as np

from ..matching import CrossMatch
from ..group_sources import make_island_groupings, _load_fourier_grid_cutouts
from ..misc_functions import create_fourier_offsets_grid
from ..group_sources_fortran import group_sources_fortran as gsf
from ..misc_functions_fortran import misc_functions_fortran as mff


def test_load_fourier_grid_cutouts():
    lena = 100000
    a = np.lib.format.open_memmap('con_cat_astro.npy', mode='w+', dtype=float, shape=(lena, 3))
    for i in range(0, lena, 10000):
        a[i:i+10000, :] = 0
    a[0, :] = [50, 50, 0.1]
    a[123, :] = [48, 59, 0.5]
    a[555, :] = [39.98, 43, 0.2]
    a[1000, :] = [45, 45, 0.2]

    del a

    grid = np.lib.format.open_memmap('fourier_grid.npy', mode='w+', dtype=float,
                                     shape=(100, 2, 3, 2),
                                     fortran_order=True)

    for k in range(2):
        for j in range(3):
            for i in range(2):
                grid[:, i, j, k] = i + j*2 + k*6

    m = np.lib.format.open_memmap('modelrefinds.npy', mode='w+', dtype=int, shape=(3, lena),
                                  fortran_order=True)
    for i in range(0, lena, 10000):
        m[:, i:i+10000] = 0
    m[:, 0] = [0, 2, 1]  # should return 0 * 2*2 + 1*6 = 10 as the single grid option selected
    m[:, 123] = [0, 2, 1]
    m[:, 555] = [0, 1, 0]  # should return 0 * 1*2 + 0*6 = 2 as its subset option
    m[:, 1000] = [0, 2, 1]
    # However, above we also get in our four-source slice the extra two combinations of:
    # 0, 1, 1 -> 0 + 2 + 6 = 9; and 0, 2, 0 -> 0 + 4 + 0 = 4. This comes from our total combination
    # of indices of 0, 1/2, and 0/1
    del grid, m

    a = np.lib.format.open_memmap('con_cat_astro.npy', mode='r', dtype=float, shape=(lena, 3))
    rect = np.array([40, 60, 40, 60])

    padding = 0.1
    _a, _b, _c, _ = _load_fourier_grid_cutouts(a, rect, '.', '.', '.', padding)
    assert np.all(_a.shape == (4, 3))
    assert np.all(_a == np.array([[50, 50, 0.1], [48, 59, 0.5], [39.98, 43, 0.2], [45, 45, 0.2]]))
    assert np.all(_b.shape == (100, 1, 2, 2))
    b_guess = np.empty((100, 1, 2, 2), float)
    b_guess[:, 0, 0, 0] = 0 + 1 * 2 + 0 * 6
    b_guess[:, 0, 1, 0] = 0 + 2 * 2 + 0 * 6
    b_guess[:, 0, 0, 1] = 0 + 1 * 2 + 1 * 6
    b_guess[:, 0, 1, 1] = 0 + 2 * 2 + 1 * 6
    assert np.all(_b == b_guess)
    assert np.all(_c.shape == (3, 4))
    c_guess = np.empty((3, 4), int)
    c_guess[:, 0] = [0, 1, 1]
    c_guess[:, 1] = [0, 1, 1]
    c_guess[:, 2] = [0, 0, 0]
    c_guess[:, 3] = [0, 1, 1]
    assert np.all(_c == c_guess)

    # This should not return source #555 above, removing its potential reference index.
    # Hence we only have one unique grid reference now.
    padding = 0
    _a, _b, _c, _ = _load_fourier_grid_cutouts(a, rect, '.', '.', '.', padding)
    assert np.all(_a.shape == (3, 3))
    assert np.all(_a == np.array([[50, 50, 0.1], [48, 59, 0.5], [45, 45, 0.2]]))
    assert np.all(_b.shape == (100, 1, 1, 1))
    b_guess = np.empty((100, 1, 1, 1), float)
    b_guess[:, 0, 0, 0] = 0 + 2 * 2 + 1 * 6
    assert np.all(_b == b_guess)
    assert np.all(_c.shape == (3, 3))
    c_guess = np.empty((3, 3), int)
    c_guess[:, 0] = [0, 0, 0]
    c_guess[:, 1] = [0, 0, 0]
    c_guess[:, 2] = [0, 0, 0]
    assert np.all(_c == c_guess)


def test_cumulative_fourier_transform():
    r = np.linspace(0, 5, 10000)
    dr = np.diff(r)
    r = r[:-1] + dr/2
    rho = np.linspace(0, 100, 10000)
    drho = np.diff(rho)
    rho = rho[:-1] + drho/2
    sigma = 0.3
    f = np.exp(-2 * np.pi**2 * rho**2 * sigma**2)

    j0s = mff.calc_j0(rho, r)
    for dist in [0, 0.1, 0.5, 1, 3]:
        p = gsf.cumulative_fourier_transform(f, r, dr, rho, drho, dist, j0s)
        assert_allclose(p, 1 - np.exp(-0.5 * dist**2 / sigma**2), rtol=1e-3, atol=1e-4)


class TestOverlap():
    def setup_class(self):
        # Create 24 sources, of which 15 are common and 5/4 are separate in each
        # catalogue.
        common_position = np.array([[10, 0], [10.3, 0], [10.5, 0], [10.7, 0], [10.9, 0],
                                    [10, 0.5], [10.3, 0.5], [10.5, 0.5], [10.7, 0.5], [10.9, 0.5],
                                    [10, 1], [10.3, 1], [10.5, 1], [10.7, 1], [10.9, 1]])
        a_off = np.array(
            [[0.04, 0.07], [-0.03, -0.06], [-0.1, -0.02], [-0.07, 0.06], [-0.01, 0.02],
             [0, 0.01], [-0.02, -0.015], [-0.1, 0.01], [0.08, -0.02], [-0.05, 0.05],
             [0.02, -0.01], [-0.01, -0.01], [0.03, 0], [0.02, 0.02], [-0.01, -0.03]])

        # Place three "a" sources definitely out of the way, and two to overlap "b"
        # sources, with 2/2 "b" sources split by no overlap and overlap respectively.
        a_separate_position = np.array([[10, 3], [10.3, 3], [10.5, 3],
                                        [10+0.04/3600, -0.02/3600], [10.5-0.03/3600, 1+0.08/3600]])

        b_separate_position = np.array([[8, 0], [9, 0], [10.5+0.05/3600, 1-0.03/3600],
                                        [10.7+0.03/3600, 0.04/3600]])

        a_position = np.append(common_position + a_off/3600, a_separate_position, axis=0)
        b_position = np.append(common_position, b_separate_position, axis=0)

        self.a_axerr = np.array([0.03]*len(a_position))
        self.b_axerr = np.array([0.03]*len(b_position))

        self.max_sep = 0.25  # 6-sigma distance is basically 100% integral for pure 2-D Gaussian
        self.max_frac = 0.99  # Slightly more than 3-sigma for 2-D Gaussian

        self.a_ax_1, self.a_ax_2 = a_position[:, 0], a_position[:, 1]
        self.b_ax_1, self.b_ax_2 = b_position[:, 0], b_position[:, 1]

        self.r = np.linspace(0, self.max_sep, 9000)
        self.dr = np.diff(self.r)
        self.r = self.r[:-1] + self.dr/2
        self.rho = np.linspace(0, 100, 10000)
        self.drho = np.diff(self.rho)
        self.rho = self.rho[:-1] + self.drho/2

        self.j0s = mff.calc_j0(self.rho, self.r)

        self.amodrefind = np.zeros((4, len(self.a_ax_1)), int)
        self.bmodrefind = np.zeros((4, len(self.b_ax_1)), int)
        self.afouriergrid = np.ones((len(self.rho), 1, 1, 1, 1), float)
        self.bfouriergrid = np.ones((len(self.rho), 1, 1, 1, 1), float)

    def test_get_max_overlap_fortran(self):
        a_num, b_num = gsf.get_max_overlap(
            self.a_ax_1, self.a_ax_2, self.b_ax_1, self.b_ax_2, self.max_sep, self.a_axerr,
            self.b_axerr, self.r, self.dr, self.rho, self.drho, self.j0s, self.afouriergrid,
            self.bfouriergrid, self.amodrefind, self.bmodrefind, self.max_frac)

        assert np.all(a_num.shape == (20,))
        assert np.all(b_num.shape == (19,))
        assert np.all(a_num ==
                      np.array([1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 0, 0, 0, 1, 1]))
        assert np.all(b_num ==
                      np.array([2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 0, 0, 1, 1]))

    def test_get_overlap_indices_fortran(self):
        a_max, b_max = 2, 2
        a_inds, b_inds, a_num, b_num = gsf.get_overlap_indices(
            self.a_ax_1, self.a_ax_2, self.b_ax_1, self.b_ax_2, self.max_sep, a_max, b_max,
            self.a_axerr, self.b_axerr, self.r, self.dr, self.rho, self.drho, self.j0s,
            self.afouriergrid, self.bfouriergrid, self.amodrefind, self.bmodrefind, self.max_frac)

        assert np.all(a_num.shape == (20,))
        assert np.all(b_num.shape == (19,))
        assert np.all(a_num ==
                      np.array([1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 0, 0, 0, 1, 1]))
        assert np.all(b_num ==
                      np.array([2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 0, 0, 1, 1]))

        a_overlaps = -1*np.ones((2, 20), int)
        for _i, _inds in enumerate([[0], [1], [2], [3, 18], [4], [5], [6], [7], [8], [9], [10],
                                    [11], [12, 17], [13], [14], [], [], [], [0], [12]]):
            a_overlaps[:len(_inds), _i] = 1+np.array(_inds)
        b_overlaps = -1*np.ones((2, 19), int)
        for _i, _inds in enumerate([[0, 18], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10],
                                    [11], [12, 19], [13], [14], [], [], [12], [3]]):
            b_overlaps[:len(_inds), _i] = 1+np.array(_inds)
        assert np.all(a_inds == a_overlaps)
        assert np.all(b_inds == b_overlaps)
