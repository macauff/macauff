# Licensed under a 3-clause BSD style license - see LICENSE
'''
Tests for the "group_sources" module.
'''

import pytest
import os
from numpy.testing import assert_allclose
import numpy as np

from ..matching import CrossMatch
from ..group_sources import make_island_groupings, _load_cumulative_grid_cutouts
from ..misc_functions import create_cumulative_offsets_grid
# from .group_sources_fortran import group_sources_fortran as gsf


def test_load_cumulative_grid_cutouts():
    lena = 100000
    a = np.lib.format.open_memmap('con_cat_astro.npy', mode='w+', dtype=float, shape=(lena, 3))
    for i in range(0, lena, 10000):
        a[i:i+10000, :] = 0
    a[0, :] = [50, 50, 0.1]
    a[123, :] = [48, 59, 0.5]
    a[555, :] = [41, 43, 0.2]
    a[1000, :] = [45, 45, 0.2]

    del a

    grid = np.lib.format.open_memmap('cumulative_grid.npy', mode='w+', dtype=float,
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

    a, b, c = _load_cumulative_grid_cutouts(a, rect, '.', '.', '.', padding)
    assert np.all(a.shape == (4, 3))
    assert np.all(a == np.array([[50, 50, 0.1], [48, 59, 0.5], [41, 43, 0.2], [45, 45, 0.2]]))
    assert np.all(b.shape == (100, 1, 2, 2))
    b_guess = np.empty((100, 1, 2, 2), float)
    b_guess[:, 0, 0, 0] = 0 + 1 * 2 + 0 * 6
    b_guess[:, 0, 1, 0] = 0 + 2 * 2 + 0 * 6
    b_guess[:, 0, 0, 1] = 0 + 1 * 2 + 1 * 6
    b_guess[:, 0, 1, 1] = 0 + 2 * 2 + 1 * 6
    assert np.all(b == b_guess)
    assert np.all(c.shape == (3, 4))
    c_guess = np.empty((3, 4), int)
    c_guess[:, 0] = [0, 1, 1]
    c_guess[:, 1] = [0, 1, 1]
    c_guess[:, 2] = [0, 0, 0]
    c_guess[:, 3] = [0, 1, 1]
    assert np.all(c == c_guess)
