# Licensed under a 3-clause BSD style license - see LICENSE
'''
Tests for the "group_sources" module.
'''

import pytest
import os
from numpy.testing import assert_allclose
import numpy as np
from scipy.special import j1

from macauff.matching import CrossMatch
from macauff.group_sources import make_island_groupings, _load_fourier_grid_cutouts, _clean_overlaps
from macauff.group_sources_fortran import group_sources_fortran as gsf
from macauff.misc_functions import create_auf_params_grid
from test_matching import _replace_line


def test_load_fourier_grid_cutouts():
    lena = 100000
    a = np.lib.format.open_memmap('con_cat_astro.npy', mode='w+', dtype=float, shape=(lena, 3))
    for i in range(0, lena, 10000):
        a[i:i+10000, :] = 0
    a[0, :] = [50, 50, 0.1]
    a[123, :] = [48, 60.02, 0.5]
    a[555, :] = [39.98, 43, 0.2]
    a[1000, :] = [45, 45, 0.2]

    del a

    grid = np.empty(dtype=float, shape=(100, 2, 3, 2), order='F')

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

    a = np.lib.format.open_memmap('con_cat_astro.npy', mode='r', dtype=float, shape=(lena, 3))
    rect = np.array([40, 60, 40, 60])

    padding = 0.1
    p_a_o = {'fourier_grid': grid}
    _a, _b, _c, _ = _load_fourier_grid_cutouts(a, rect, '.', '.', p_a_o, padding, 'check',
                                               np.array([True]*lena), modelrefinds=m)
    assert np.all(_a.shape == (4, 3))
    assert np.all(_a ==
                  np.array([[50, 50, 0.1], [48, 60.02, 0.5], [39.98, 43, 0.2], [45, 45, 0.2]]))
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

    # This should not return sources 123 and 555 above, removing a potential
    # reference index. Hence we only have one unique grid reference now.
    padding = 0
    _a, _b, _c, _ = _load_fourier_grid_cutouts(a, rect, '.', '.', p_a_o, padding, 'check',
                                               np.array([True]*lena), modelrefinds=m)
    assert np.all(_a.shape == (2, 3))
    assert np.all(_a == np.array([[50, 50, 0.1], [45, 45, 0.2]]))
    assert np.all(_b.shape == (100, 1, 1, 1))
    b_guess = np.empty((100, 1, 1, 1), float)
    b_guess[:, 0, 0, 0] = 0 + 2 * 2 + 1 * 6
    assert np.all(_b == b_guess)
    assert np.all(_c.shape == (3, 2))
    c_guess = np.empty((3, 2), int)
    c_guess[:, 0] = [0, 0, 0]
    c_guess[:, 1] = [0, 0, 0]
    assert np.all(_c == c_guess)


def test_j1s():
    rho = np.linspace(0, 100, 10000)
    drho = np.diff(rho)
    rng = np.random.default_rng(seed=1231231)
    values = rng.uniform(0, 3, size=50)
    for val in values:
        j1s = gsf.calc_j1s(rho[:-1]+drho/2, np.array([val]))
        assert_allclose(j1s[:, 0], j1(2*np.pi*(rho[:-1]+drho/2)*val), rtol=1e-5)


class TestFortranCode():
    def setup_class(self):
        self.r = np.linspace(0, 5, 10000)
        self.dr = np.diff(self.r)
        self.rho = np.linspace(0, 100, 9999)
        self.drho = np.diff(self.rho)

        self.j1s = gsf.calc_j1s(self.rho[:-1]+self.drho/2, self.r[:-1]+self.dr/2)

    def test_cumulative_fourier_transform_probability(self):
        sigma = 0.3
        f = np.exp(-2 * np.pi**2 * (self.rho[:-1]+self.drho/2)**2 * sigma**2)

        for dist in [0, 0.1, 0.5, 1, 3]:
            k = np.argmin(np.abs((self.r[:-1]+self.dr/2) - dist))
            p = gsf.cumulative_fourier_probability(f, self.drho, dist, self.j1s[:, k])
            assert_allclose(p, 1 - np.exp(-0.5 * dist**2 / sigma**2), rtol=1e-3, atol=1e-4)

    def test_cumulative_fourier_transform_distance(self):
        sigma = 0.3
        f = np.exp(-2 * np.pi**2 * (self.rho[:-1]+self.drho/2)**2 * sigma**2)

        probs = [0, 0.1, 0.5, 0.95]
        d = gsf.cumulative_fourier_distance(f, self.r[:-1]+self.dr/2, self.drho, probs, self.j1s)
        assert np.all(d.shape == (len(probs),))
        for i, prob in enumerate(probs):
            # We're forced to accept an absolute precision of half a bin width
            assert_allclose(d[i]*3600, np.sqrt(-2 * sigma**2 * np.log(1 - prob)),
                            rtol=1e-3, atol=self.dr[0]/2)

    def test_get_integral_length(self):
        rng = np.random.default_rng(112233)
        a_err = rng.uniform(0.2, 0.4, 5)
        b_err = rng.uniform(0.1, 0.3, 4)
        a_four_sig = 0.2
        a_fouriergrid = np.asfortranarray(np.exp(-2 * np.pi**2 * (self.rho[:-1]+self.drho/2)**2 *
                                                 a_four_sig**2).reshape(-1, 1, 1, 1))
        b_four_sig = 0.1
        b_fouriergrid = np.asfortranarray(np.exp(-2 * np.pi**2 * (self.rho[:-1]+self.drho/2)**2 *
                                                 b_four_sig**2).reshape(-1, 1, 1, 1))
        amodrefind = np.zeros((3, len(a_err)), int, order='F')
        bmodrefind = np.zeros((3, len(b_err)), int, order='F')
        asize = rng.choice(np.arange(1, len(b_err)+1), size=5) + 1
        ainds = np.zeros((len(b_err), len(a_err)), int, order='F')
        for i in range(len(asize)):
            ainds[:asize[i], i] = rng.choice(len(b_err), size=asize[i], replace=False)
        frac_array = np.array([0.63, 0.9])

        int_dists = gsf.get_integral_length(a_err, b_err, self.r[:-1]+self.dr/2, self.rho[:-1],
                                            self.drho, self.j1s, a_fouriergrid, b_fouriergrid,
                                            amodrefind, bmodrefind, ainds, asize, frac_array)

        assert np.all(int_dists.shape == (len(a_err), len(frac_array)))
        for i in range(len(frac_array)):
            for j in range(len(a_err)):
                _berr = np.amax(b_err[ainds[:, j]])
                assert_allclose(int_dists[j, i]*3600, np.sqrt(-2 * (a_err[j]**2 + _berr**2 +
                                a_four_sig**2 + b_four_sig**2) * np.log(1 - frac_array[i])),
                                rtol=1e-3, atol=self.dr[0]/2)


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
        self.rho = np.linspace(0, 100, 10000)
        self.drho = np.diff(self.rho)

        self.j1s = gsf.calc_j1s(self.rho[:-1]+self.drho/2, self.r[:-1]+self.dr/2)

        self.amodrefind = np.zeros((3, len(self.a_ax_1)), int)
        self.bmodrefind = np.zeros((3, len(self.b_ax_1)), int)
        self.afouriergrid = np.ones((len(self.rho) - 1, 1, 1, 1), float)
        self.bfouriergrid = np.ones((len(self.rho) - 1, 1, 1, 1), float)

    def test_get_max_overlap_fortran(self):
        a_num, b_num = gsf.get_max_overlap(
            self.a_ax_1, self.a_ax_2, self.b_ax_1, self.b_ax_2, self.max_sep/3600, self.a_axerr,
            self.b_axerr, self.r[:-1]+self.dr/2, self.rho[:-1], self.drho, self.j1s,
            self.afouriergrid, self.bfouriergrid, self.amodrefind, self.bmodrefind, self.max_frac)

        assert np.all(a_num.shape == (20,))
        assert np.all(b_num.shape == (19,))
        assert np.all(a_num ==
                      np.array([1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 0, 0, 0, 1, 1]))
        assert np.all(b_num ==
                      np.array([2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 0, 0, 1, 1]))

    def test_get_overlap_indices_fortran(self):
        a_max, b_max = 2, 2
        a_inds, b_inds, a_num, b_num = gsf.get_overlap_indices(
            self.a_ax_1, self.a_ax_2, self.b_ax_1, self.b_ax_2, self.max_sep/3600, a_max, b_max,
            self.a_axerr, self.b_axerr, self.r[:-1]+self.dr/2, self.rho[:-1], self.drho, self.j1s,
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


def test_clean_overlaps():
    maxsize, size = 5, np.array([3, 5, 3, 4, 4, 5, 4, 2, 5, 4]*3)
    inds = np.empty(dtype=int, shape=(maxsize, len(size)), order='F')
    for i in range(0, 3):
        inds[:, 0+10*i] = [0, 1, 0, -1, -1]
        inds[:, 1+10*i] = [3, 4, 1, 1, 4]
        inds[:, 2+10*i] = [2, 3, 4, -1, -1]
        inds[:, 3+10*i] = [0, 0, 0, 1, -1]
        inds[:, 4+10*i] = [0, 1, 2, 3, -1]
        inds[:, 5+10*i] = [0, 0, 1, 1, 2]
        inds[:, 6+10*i] = [3, 4, 3, 4, -1]
        inds[:, 7+10*i] = [0, 1, -1, -1, -1]
        inds[:, 8+10*i] = [2, 2, 2, 2, 2]
        inds[:, 9+10*i] = [1, 1, 2, 3, -1]

    inds2, size2 = _clean_overlaps(inds, size, 2)
    compare_inds2 = np.empty((4, 30), int)
    for i in range(0, 3):
        compare_inds2[:, 0+10*i] = [0, 1, -1, -1]
        compare_inds2[:, 1+10*i] = [1, 3, 4, -1]
        compare_inds2[:, 2+10*i] = [2, 3, 4, -1]
        compare_inds2[:, 3+10*i] = [0, 1, -1, -1]
        compare_inds2[:, 4+10*i] = [0, 1, 2, 3]
        compare_inds2[:, 5+10*i] = [0, 1, 2, -1]
        compare_inds2[:, 6+10*i] = [3, 4, -1, -1]
        compare_inds2[:, 7+10*i] = [0, 1, -1, -1]
        compare_inds2[:, 8+10*i] = [2, -1, -1, -1]
        compare_inds2[:, 9+10*i] = [1, 2, 3, -1]
    assert np.all(inds2 == compare_inds2)
    assert np.all(size2 == np.array([2, 3, 3, 2, 4, 3, 2, 2, 1, 3]*3))


class TestMakeIslandGroupings():
    def setup_class(self):
        self.include_phot_like, self.use_phot_prior = False, False
        self.max_sep, self.int_fracs = 11, [0.63, 0.9, 0.99]  # max_sep in arcseconds
        self.a_filt_names, self.b_filt_names = ['G', 'RP'], ['W1', 'W2', 'W3']
        self.a_title, self.b_title = 'gaia', 'wise'
        self.a_cat_folder_path, self.b_cat_folder_path = 'gaia_cat', 'wise_cat'
        self.a_auf_folder_path, self.b_auf_folder_path = 'gaia_auf', 'wise_auf'
        self.joint_folder_path = 'joint'
        for folder in [self.a_cat_folder_path, self.b_cat_folder_path, self.joint_folder_path,
                       self.a_auf_folder_path, self.b_auf_folder_path]:
            os.makedirs(folder, exist_ok=True)
        os.makedirs('{}/reject'.format(self.joint_folder_path), exist_ok=True)
        self.r = np.linspace(0, self.max_sep, 10000)
        self.dr = np.diff(self.r)
        self.rho = np.linspace(0, 100, 9900)
        self.drho = np.diff(self.rho)

        self.j1s = gsf.calc_j1s(self.rho[:-1]+self.drho/2, self.r[:-1]+self.dr/2)

        self.a_auf_pointings = np.array([[10, -26], [11, -25], [12, -24]])
        self.b_auf_pointings = np.array([[10.1, -25.9], [11.3, -25.1], [11.9, -24]])

        self.N_a, self.N_b = 30, 45
        self.N_com = 25

        self.ax_lims = np.array([10, 12, -26, -24])

        # 99% is slightly more than 3-sigma of a 2-D Gaussian integral, for
        # int_frac[2] = 0.99
        self.sigma = 0.1
        seed = 123456  # reproducible seed!
        self.rng = np.random.default_rng(seed)

        a_coords = np.empty((self.N_a, 3), float)
        a_coords[:, 0] = self.rng.uniform(self.ax_lims[0]+0.5, self.ax_lims[1]-0.5, self.N_a)
        a_coords[:, 1] = self.rng.uniform(self.ax_lims[2]+0.5, self.ax_lims[3]-0.5, self.N_a)
        a_coords[:, 2] = self.sigma
        # Move one source to have a forced overlap of objects
        a_coords[-1, :2] = a_coords[0, :2] + [0.01*self.sigma/3600, 0.02*self.sigma/3600]
        b_coords = np.empty((self.N_b, 3), float)
        # Make sure that the N_com=25 matches all return based on Gaussian integrals.
        b_coords[:self.N_com, 0] = a_coords[:self.N_com, 0] + self.rng.uniform(
            -2, 2, self.N_com)*self.sigma/3600
        b_coords[:self.N_com, 1] = a_coords[:self.N_com, 1] + self.rng.uniform(
            -2, 2, self.N_com)*self.sigma/3600
        # This should leave us with 4 "a" and 20 "b" singular matches.
        b_coords[self.N_com:, 0] = self.rng.uniform(self.ax_lims[0]+0.5,
                                                    self.ax_lims[1]-0.5, self.N_b-self.N_com)
        b_coords[self.N_com:, 1] = self.rng.uniform(self.ax_lims[2]+0.5,
                                                    self.ax_lims[3]-0.5, self.N_b-self.N_com)
        b_coords[:, 2] = self.sigma

        self.a_coords, self.b_coords = a_coords, b_coords

        # Set the catalogue folders now to avoid an error message in
        # CrossMatch's __init__ call.
        old_line = 'cat_folder_path = gaia_folder'
        new_line = 'cat_folder_path = {}\n'.format(self.a_cat_folder_path)
        f = open(os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt')).readlines()
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                      idx, new_line, out_file=os.path.join(os.path.dirname(__file__),
                      'data/cat_a_params_.txt'))
        old_line = 'cat_folder_path = wise_folder'
        new_line = 'cat_folder_path = {}\n'.format(self.b_cat_folder_path)
        f = open(os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt')).readlines()
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'),
                      idx, new_line, out_file=os.path.join(os.path.dirname(__file__),
                      'data/cat_b_params_.txt'))

        # Save dummy files into each catalogue folder as well.
        for folder, N in zip([self.a_cat_folder_path, self.b_cat_folder_path], [3, 4]):
            np.save('{}/con_cat_astro.npy'.format(folder), np.zeros((10, 3), float))
            np.save('{}/con_cat_photo.npy'.format(folder), np.zeros((10, N), float))
            np.save('{}/magref.npy'.format(folder), np.zeros((10,), int))

        # Also set up an instance of CrossMatch at the same time.
        self.cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'))
        self.cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                                  os.path.join(os.path.dirname(__file__), 'data/cat_a_params_.txt'),
                                  os.path.join(os.path.dirname(__file__), 'data/cat_b_params_.txt'))
        self.cm.pos_corr_dist = self.max_sep
        self.cm.cross_match_extent = self.ax_lims
        self.cm.a_filt_names = self.a_filt_names
        self.cm.b_filt_names = self.b_filt_names
        self.cm.a_cat_name = self.a_title
        self.cm.b_cat_name = self.b_title
        self.cm.r, self.cm.dr, self.cm.rho, self.cm.drho = self.r, self.dr, self.rho, self.drho
        self.cm.a_auf_folder_path = self.a_auf_folder_path
        self.cm.b_auf_folder_path = self.b_auf_folder_path
        self.cm.a_auf_region_points = self.a_auf_pointings
        self.cm.b_auf_region_points = self.b_auf_pointings
        self.cm.a_cat_folder_path = self.a_cat_folder_path
        self.cm.b_cat_folder_path = self.b_cat_folder_path
        self.cm.joint_folder_path = self.joint_folder_path
        self.cm.include_phot_like = self.include_phot_like
        self.cm.use_phot_prior = self.use_phot_prior
        self.cm.j1s = self.j1s

        self.n_pool = 5

    def _fake_fourier_grid(self, N_a, N_b):
        # Fake fourier grid, in this case under the assumption that there
        # is no extra AUF component:
        fourier_grids = []
        for auf_folder, auf_points, filters, N in zip(
                [self.a_auf_folder_path, self.b_auf_folder_path],
                [self.a_auf_pointings, self.b_auf_pointings],
                [self.a_filt_names, self.b_filt_names], [N_a, N_b]):
            np.save('{}/modelrefinds.npy'.format(auf_folder), np.zeros((3, N), int))
            if 'gaia' in auf_folder:
                self.a_modelrefinds = np.zeros((3, N), int)
            else:
                self.b_modelrefinds = np.zeros((3, N), int)
            perturb_auf = {}
            for i in range(len(auf_points)):
                ax1, ax2 = auf_points[i]
                for j in range(len(filters)):
                    name = '{}-{}-{}'.format(ax1, ax2, filters[j])
                    fourieroffset = np.ones((len(self.rho) - 1, 1), float, order='F')
                    perturb_auf[name] = {'fourier': fourieroffset}
            fourier_grids.append(create_auf_params_grid(perturb_auf, auf_points, filters, 'fourier',
                                 np.ones((len(filters), len(auf_points)), int), len(self.rho)-1))
        self.a_perturb_auf_outputs = {}
        self.b_perturb_auf_outputs = {}
        self.a_perturb_auf_outputs['fourier_grid'] = fourier_grids[0]
        self.b_perturb_auf_outputs['fourier_grid'] = fourier_grids[1]

    def _comparisons_in_islands(self, alist, blist, agrplen, bgrplen, N_a, N_b, N_c):
        # Given, say, 25 common sources from 30 'a' and 45 'b' objects, we'd
        # expect 5 + 20 + 25 = 50 islands, with zero overlap. Here we expect
        # a single extra 'a' island overlap, however.
        assert np.all(alist.shape == (2, N_a - 1 + N_b - N_c))
        assert np.all(blist.shape == (1, N_a - 1 + N_b - N_c))
        assert np.all(agrplen.shape == (N_a - 1 + N_b - N_c,))
        assert np.all(bgrplen.shape == (N_a - 1 + N_b - N_c,))
        a_list_fix = -1*np.ones((2, N_a - 1 + N_b - N_c), int)
        # N_c - 1 common, 4 singular "a" matches; N_c-1+4 in arange goes to N_c+4.
        a_list_fix[0, :N_c - 1 + 4] = np.arange(1, N_c + 4)
        # The final entry is the double "a" overlap, otherwise no "a" sources.
        a_list_fix[0, -1] = 0
        a_list_fix[1, -1] = N_a - 1
        assert np.all(agrplen == np.append(np.ones((N_c - 1 + 4), int),
                      [*np.zeros((N_b-N_c), int), 2]))
        assert np.all(bgrplen == np.append(np.ones((N_c - 1), int),
                      [0, 0, 0, 0, *np.ones((N_b-N_c), int), 1]))
        assert np.all(alist == a_list_fix)
        # Here we mapped one-to-one for "b" sources that are matched to "a" objects,
        # and by default the empty groups have the null "-1" index. Also remember that
        # the arrays should be f-ordered, and here are (1, N) shape.
        assert np.all(blist == np.append(np.arange(1, N_c),
                                         np.array([-1, -1, -1, -1,
                                                   *np.arange(N_c, N_b), 0]))).reshape(1, -1)

    def test_make_island_groupings(self):
        os.system('rm -rf {}/reject/*'.format(self.joint_folder_path))
        self._fake_fourier_grid(self.N_a, self.N_b)
        N_a, N_b, N_c = self.N_a, self.N_b, self.N_com
        np.save('{}/con_cat_astro.npy'.format(self.a_cat_folder_path), self.a_coords)
        np.save('{}/con_cat_astro.npy'.format(self.b_cat_folder_path), self.b_coords)
        # For the first, full runthrough call the CrossMatch function instead of
        # directly calling make_island_groupings to test group_sources as well.
        self.cm.chunk_id = 1
        self.cm.a_perturb_auf_outputs = self.a_perturb_auf_outputs
        self.cm.b_perturb_auf_outputs = self.b_perturb_auf_outputs
        self.cm.a_modelrefinds = self.a_modelrefinds
        self.cm.b_modelrefinds = self.b_modelrefinds
        self.cm.group_sources()

        alist, blist = self.cm.group_sources_data.alist, self.cm.group_sources_data.blist
        agrplen, bgrplen = self.cm.group_sources_data.agrplen, self.cm.group_sources_data.bgrplen
        self._comparisons_in_islands(alist, blist, agrplen, bgrplen, N_a, N_b, N_c)
        assert len(os.listdir('{}/reject'.format(self.joint_folder_path))) == 0

    @pytest.mark.filterwarnings("ignore:.*island, containing.*")
    def test_mig_extra_reject(self):
        os.system('rm -rf {}/reject/*'.format(self.joint_folder_path))
        self._fake_fourier_grid(self.N_a+10, self.N_b+11)
        N_a, N_b, N_c = self.N_a, self.N_b, self.N_com
        ax_lims = self.ax_lims
        # Fake moving some sources to within max_sep of the axlims edges, to
        # test the removal of these objects -- combined with the above sources.
        a_coords = np.empty((self.N_a+10, 3), float)
        a_coords[:self.N_a, 0] = self.a_coords[:, 0]
        a_coords[:self.N_a, 1] = self.a_coords[:, 1]
        a_coords[:self.N_a, 2] = self.a_coords[:, 2]
        b_coords = np.empty((self.N_b+11, 3), float)
        b_coords[:self.N_b, 0] = self.b_coords[:, 0]
        b_coords[:self.N_b, 1] = self.b_coords[:, 1]
        b_coords[:self.N_b, 2] = self.b_coords[:, 2]
        a_c_diff = a_coords[3:6, 0] - (ax_lims[0] + self.max_sep/3600 - 1/3600)
        a_coords[3:6, 0] = ax_lims[0] + self.max_sep/3600 - 1/3600
        b_coords[3:6, 0] -= a_c_diff
        # Add extra objects to be removed due to group length being
        # exceeded during make_set_list.
        a_coords[self.N_a:, 0] = 0.5*(ax_lims[0]+ax_lims[1])
        b_coords[self.N_b:, 0] = 0.5*(ax_lims[0]+ax_lims[1])
        a_coords[self.N_a:, 1] = 0.5*(ax_lims[2]+ax_lims[3])
        b_coords[self.N_b:, 1] = 0.5*(ax_lims[2]+ax_lims[3])
        a_coords[self.N_a:, 2] = 0.5
        b_coords[self.N_b:, 2] = 0.5
        np.save('{}/con_cat_astro.npy'.format(self.a_cat_folder_path), a_coords)
        np.save('{}/con_cat_astro.npy'.format(self.b_cat_folder_path), b_coords)

        gsd = make_island_groupings(
            self.joint_folder_path, self.a_cat_folder_path, self.b_cat_folder_path,
            self.a_auf_pointings, self.b_auf_pointings, self.a_filt_names, self.b_filt_names,
            self.a_title, self.b_title, self.a_modelrefinds, self.b_modelrefinds, self.r, self.dr,
            self.rho, self.drho, self.j1s, self.max_sep, ax_lims, self.int_fracs,
            self.include_phot_like, self.use_phot_prior, self.n_pool, self.a_perturb_auf_outputs,
            self.b_perturb_auf_outputs)

        alist, blist = gsd.alist, gsd.blist
        agrplen, bgrplen = gsd.agrplen, gsd.bgrplen
        # We removed 3 extra sources this time around, which should all be 1:1 islands.
        assert np.all(alist.shape == (2, N_a - 4 + N_b - N_c))
        assert np.all(blist.shape == (1, N_a - 4 + N_b - N_c))
        assert np.all(agrplen.shape == (N_a - 4 + N_b - N_c,))
        assert np.all(bgrplen.shape == (N_a - 4 + N_b - N_c,))
        a_list_fix = -1*np.ones((2, N_a - 4 + N_b - N_c), int)
        a_list_fix[0, :2] = [1, 2]
        # Remove 3 items from between the "2nd" and "6th" entries in alist.
        a_list_fix[0, 2:N_c - 1 + 4 - 3] = np.arange(6, N_c + 4)
        a_list_fix[0, -1] = 0
        a_list_fix[1, -1] = N_a - 1
        assert np.all(agrplen == np.append(np.ones((N_c - 1 + 4 - 3), int),
                      [*np.zeros((N_b-N_c), int), 2]))
        assert np.all(bgrplen == np.append(np.ones((N_c - 1 - 3), int),
                      [0, 0, 0, 0, *np.ones((N_b-N_c), int), 1]))
        assert np.all(alist == a_list_fix)
        assert np.all(blist == np.append(np.append([1, 2], np.arange(6, N_c)),
                                         np.array([-1, -1, -1, -1,
                                                   *np.arange(N_c, N_b), 0]))).reshape(1, -1)

        areject = np.load('joint/reject/reject_a.npy')
        breject = np.load('joint/reject/reject_b.npy')
        assert np.all(areject.shape == (10+3,))
        assert np.all(breject.shape == (11+3,))
        assert np.all(areject == np.concatenate(([3, 4, 5], np.arange(N_a, N_a+10))))
        assert np.all(breject == np.concatenate(([3, 4, 5], np.arange(N_b, N_b+11))))

        assert len(os.listdir('{}/reject'.format(self.joint_folder_path))) == 2

    @pytest.mark.filterwarnings("ignore:.*island, containing.*")
    def test_mig_no_reject_ax_lims(self):
        os.system('rm -rf {}/reject/*'.format(self.joint_folder_path))
        self._fake_fourier_grid(self.N_a+10, self.N_b+11)
        N_a, N_b, N_c = self.N_a, self.N_b, self.N_com
        ax_lims = np.array([0, 360, -90, -88])
        # Check if axlims are changed to include wrap-around 0/360, or +-90 latitude,
        # then we don't reject any sources.
        a_coords = np.empty((self.N_a+10, 3), float)
        a_coords[:self.N_a, 0] = self.a_coords[:, 0]
        # Set up -26 to -24, and we now want them -90 to -88:
        a_coords[:self.N_a, 1] = self.a_coords[:, 1] - 64
        a_coords[:self.N_a, 2] = self.a_coords[:, 2]
        b_coords = np.empty((self.N_b+11, 3), float)
        b_coords[:self.N_b, 0] = self.b_coords[:, 0]
        b_coords[:self.N_b, 1] = self.b_coords[:, 1] - 64
        b_coords[:self.N_b, 2] = self.b_coords[:, 2]
        a_c_diff = a_coords[3:6, 0] - (ax_lims[0] + self.max_sep/3600 - 1/3600)
        a_coords[3:6, 0] = ax_lims[0] + self.max_sep/3600 - 1/3600
        b_coords[3:6, 0] -= a_c_diff
        a_coords[7, :2] = [15, -90+(self.max_sep-3)/3600]
        b_coords[7, :2] = a_coords[7, :2] + [0.2*self.sigma/3600, -0.15*self.sigma/3600]
        # Add extra objects to be removed due to group length being
        # exceeded during make_set_list.
        a_coords[self.N_a:, 0] = 0.5*(ax_lims[0]+ax_lims[1])
        b_coords[self.N_b:, 0] = 0.5*(ax_lims[0]+ax_lims[1])
        a_coords[self.N_a:, 1] = 0.5*(ax_lims[2]+ax_lims[3])
        b_coords[self.N_b:, 1] = 0.5*(ax_lims[2]+ax_lims[3])
        a_coords[self.N_a:, 2] = 0.5
        b_coords[self.N_b:, 2] = 0.5

        np.save('{}/con_cat_astro.npy'.format(self.a_cat_folder_path), a_coords)
        np.save('{}/con_cat_astro.npy'.format(self.b_cat_folder_path), b_coords)

        gsd = make_island_groupings(
            self.joint_folder_path, self.a_cat_folder_path, self.b_cat_folder_path,
            self.a_auf_pointings, self.b_auf_pointings, self.a_filt_names, self.b_filt_names,
            self.a_title, self.b_title, self.a_modelrefinds, self.b_modelrefinds, self.r, self.dr,
            self.rho, self.drho, self.j1s, self.max_sep, ax_lims, self.int_fracs,
            self.include_phot_like, self.use_phot_prior, self.n_pool, self.a_perturb_auf_outputs,
            self.b_perturb_auf_outputs)

        alist, blist = gsd.alist, gsd.blist
        agrplen, bgrplen = gsd.agrplen, gsd.bgrplen
        # The same tests that were ran in make_island_groupings should pass here.
        self._comparisons_in_islands(alist, blist, agrplen, bgrplen, N_a, N_b, N_c)
        areject = np.load('joint/reject/reject_a.npy')
        breject = np.load('joint/reject/reject_b.npy')
        assert np.all(areject.shape == (10,))
        assert np.all(breject.shape == (11,))
        assert np.all(areject == np.arange(N_a, N_a+10))
        assert np.all(breject == np.arange(N_b, N_b+11))
        assert len(os.listdir('{}/reject'.format(self.joint_folder_path))) == 2

    def test_make_island_groupings_include_phot_like(self):
        os.system('rm -rf {}/reject/*'.format(self.joint_folder_path))
        self._fake_fourier_grid(self.N_a, self.N_b)
        np.save('{}/con_cat_astro.npy'.format(self.a_cat_folder_path), self.a_coords)
        np.save('{}/con_cat_astro.npy'.format(self.b_cat_folder_path), self.b_coords)
        include_phot_like = True
        gsd = make_island_groupings(
            self.joint_folder_path, self.a_cat_folder_path, self.b_cat_folder_path,
            self.a_auf_pointings, self.b_auf_pointings, self.a_filt_names, self.b_filt_names,
            self.a_title, self.b_title, self.a_modelrefinds, self.b_modelrefinds,  self.r, self.dr,
            self.rho, self.drho, self.j1s, self.max_sep, self.ax_lims, self.int_fracs,
            include_phot_like, self.use_phot_prior, self.n_pool, self.a_perturb_auf_outputs,
            self.b_perturb_auf_outputs)

        # Verify that make_island_groupings doesn't change when the extra arrays
        # are calculated, as an initial test.
        alist, blist = gsd.alist, gsd.blist
        agrplen, bgrplen = gsd.agrplen, gsd.bgrplen
        self._comparisons_in_islands(alist, blist, agrplen, bgrplen, self.N_a, self.N_b,
                                     self.N_com)

        assert len(os.listdir('{}/reject'.format(self.joint_folder_path))) == 0

        aerr = np.load('{}/con_cat_astro.npy'.format(self.a_cat_folder_path))[:, 2]
        berr = np.load('{}/con_cat_astro.npy'.format(self.b_cat_folder_path))[:, 2]

        ablen = gsd.ablen
        aflen = gsd.aflen
        bblen = gsd.bblen
        bflen = gsd.bflen

        asize = gsd.asize
        ainds = gsd.ainds
        bsize = gsd.bsize
        binds = gsd.binds

        for i in range(len(ablen)):
            d = ablen[i]*3600
            if asize[i] > 0:
                _berr = np.amax(berr[ainds[:asize[i], i]])
                real_d = np.sqrt(-2 * (aerr[i]**2 + _berr**2) * np.log(1 - self.int_fracs[0]))
                assert_allclose(d, real_d, rtol=1e-3, atol=self.dr[0]/2)
            else:
                # If there is no overlap in opposing catalogue objects, we should
                # get a zero distance error circle.
                assert d == 0
        for i in range(len(aflen)):
            d = aflen[i]*3600
            if asize[i] > 0:
                _berr = np.amax(berr[ainds[:asize[i], i]])
                real_d = np.sqrt(-2 * (aerr[i]**2 + _berr**2) * np.log(1 - self.int_fracs[1]))
                assert_allclose(d, real_d, rtol=1e-3, atol=self.dr[0]/2)
            else:
                assert d == 0

        for i in range(len(bblen)):
            d = bblen[i]*3600
            if bsize[i] > 0:
                _aerr = np.amax(aerr[binds[:bsize[i], i]])
                real_d = np.sqrt(-2 * (berr[i]**2 + _aerr**2) * np.log(1 - self.int_fracs[0]))
                assert_allclose(d, real_d, rtol=1e-3, atol=self.dr[0]/2)
            else:
                assert d == 0
        for i in range(len(bflen)):
            d = bflen[i]*3600
            if bsize[i] > 0:
                _aerr = np.amax(aerr[binds[:bsize[i], i]])
                real_d = np.sqrt(-2 * (berr[i]**2 + _aerr**2) * np.log(1 - self.int_fracs[1]))
                assert_allclose(d, real_d, rtol=1e-3, atol=self.dr[0]/2)
            else:
                assert d == 0
