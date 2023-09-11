# Licensed under a 3-clause BSD style license - see LICENSE
'''
Tests for the "photometric_likelihood" module.
'''

import os
from numpy.testing import assert_allclose
import numpy as np
import pytest

from ..matching import CrossMatch
from ..photometric_likelihood import compute_photometric_likelihoods, make_bins
from ..photometric_likelihood_fortran import photometric_likelihood_fortran as plf
from ..misc_functions import StageData
from .test_matching import _replace_line


class TestOneSidedPhotometricLikelihood:
    def setup_class(self):
        self.cf_points = np.array([[a, b] for a in [131.5, 132.5, 133.5] for b in [-0.5, 0.5]])
        self.cf_areas = np.ones((6), float)
        self.joint_folder_path = 'test_path'
        self.a_cat_folder_path = 'gaia_folder'
        self.b_cat_folder_path = 'wise_folder'
        self.group_sources_data = StageData(ablen=None, ainds=None, asize=None,
                                            bblen=None, binds=None, bsize=None)
        self.use_memmap_files = True

        self.area = (134-131)*(1--1)

        self.afilts, self.bfilts = np.array(['G', 'BP', 'RP']), np.array(['W1', 'W2', 'W3', 'W4'])

        self.mem_chunk_num = 2
        self.include_phot_like, self.use_phot_priors = False, False

        os.makedirs('{}/phot_like'.format(self.joint_folder_path), exist_ok=True)
        os.makedirs(self.a_cat_folder_path, exist_ok=True)
        os.makedirs(self.b_cat_folder_path, exist_ok=True)

        # Create a random selection of sources, and then NaN out 25% of each of
        # the filters, at random.
        seed = 98765
        rng = np.random.default_rng(seed)

        self.Na, self.Nb = 200000, 78000
        for N, folder, filts, name in zip([self.Na, self.Nb],
                                          [self.a_cat_folder_path, self.b_cat_folder_path],
                                          [self.afilts, self.bfilts], ['a', 'b']):
            a = np.empty((N, 3), float)
            a[:, 0] = rng.uniform(131, 134, N)
            a[:, 1] = rng.uniform(-1, 1, N)
            a[:, 2] = 0.1

            setattr(self, '{}_astro'.format(name), a)

            a = rng.uniform(10.01, 14.99, (N, len(filts)))
            for i in range(len(filts)):
                q = rng.choice(N, size=N // 4, replace=False)
                a[q, i] = np.nan

            setattr(self, '{}_photo'.format(name), a)

        old_line = 'mem_chunk_num = 10'
        new_line = 'mem_chunk_num = 2\n'
        f = open(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt')).readlines()
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                      idx, new_line, out_file=os.path.join(os.path.dirname(__file__),
                      'data/crossmatch_params_.txt'))
        old_line = 'cf_region_points = 131 134 4 -1 1 3'
        new_line = 'cf_region_points = 131.5 133.5 3 -0.5 0.5 2\n'
        f = open(os.path.join(os.path.dirname(__file__),
                 'data/crossmatch_params_.txt')).readlines()
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params_.txt'),
                      idx, new_line)
        old_line = 'cross_match_extent = 131 138 -3 3'
        new_line = 'cross_match_extent = 131 134 -3 3\n'
        f = open(os.path.join(os.path.dirname(__file__),
                 'data/crossmatch_params_.txt')).readlines()
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params_.txt'),
                      idx, new_line)

    def test_compute_photometric_likelihoods(self):
        os.system('rm -r {}/phot_like/*'.format(self.joint_folder_path))
        Na, Nb, area = self.Na, self.Nb, self.area
        for folder, name in zip([self.a_cat_folder_path, self.b_cat_folder_path], ['a', 'b']):
            np.save('{}/con_cat_astro.npy'.format(folder), getattr(self, '{}_astro'.format(name)))
            np.save('{}/con_cat_photo.npy'.format(folder), getattr(self, '{}_photo'.format(name)))
        compute_photometric_likelihoods(
            self.joint_folder_path, self.a_cat_folder_path, self.b_cat_folder_path, self.afilts,
            self.bfilts, self.mem_chunk_num, self.cf_points, self.cf_areas, self.include_phot_like,
            self.use_phot_priors, self.group_sources_data, self.use_memmap_files)

        for file, shape, value in zip(['c_priors', 'fa_priors', 'fb_priors'],
                                      [(4, 3, 6), (4, 3, 6), (4, 3, 6)],
                                      [0.75*Nb/area/2, 0.75/area*(Na-Nb/2), 0.75/area*(Nb-Nb/2)]):
            a = np.load('{}/phot_like/{}.npy'.format(self.joint_folder_path, file))
            assert np.all(a.shape == shape)
            # Allow 3% tolerance for counting statistics in the distribution above
            # caused by the rng.choice removal of objects in each filter.
            assert_allclose(a, value, rtol=0.03)

        abinlen = np.load('{}/phot_like/abinlengths.npy'.format(self.joint_folder_path))
        bbinlen = np.load('{}/phot_like/bbinlengths.npy'.format(self.joint_folder_path))

        assert np.all(abinlen == 51*np.ones((3, 6), int))
        assert np.all(bbinlen == 26*np.ones((4, 6), int))

        for file, shape in zip(
            ['c_array', 'fa_array', 'fb_array'], [(25, 50, 4, 3, 6), (50, 4, 3, 6),
                                                  (25, 4, 3, 6)]):
            a = np.load('{}/phot_like/{}.npy'.format(self.joint_folder_path, file))
            assert np.all(a.shape == shape)
            assert_allclose(a, np.ones(shape, float), atol=1e-30)

        abins = np.load('{}/phot_like/abinsarray.npy'.format(self.joint_folder_path))
        bbins = np.load('{}/phot_like/bbinsarray.npy'.format(self.joint_folder_path))
        for folder, filts, _bins in zip([self.a_cat_folder_path, self.b_cat_folder_path],
                                        [self.afilts, self.bfilts], [abins, bbins]):
            a = np.load('{}/con_cat_astro.npy'.format(folder))
            b = np.load('{}/con_cat_photo.npy'.format(folder))
            for i, (ax1, ax2) in enumerate(self.cf_points):
                q = ((a[:, 0] >= ax1-0.5) & (a[:, 0] <= ax1+0.5) &
                     (a[:, 1] >= ax2-0.5) & (a[:, 1] <= ax2+0.5))
                for k in range(len(filts)):
                    hist, bins = np.histogram(b[q & ~np.isnan(b[:, k]), k],
                                              bins=_bins[:, k, i])
                    assert np.all(hist >= 250)

    def test_empty_filter(self):
        os.system('rm -r {}/phot_like/*'.format(self.joint_folder_path))
        # This test simply removes all "good" flags from a single pointing-filter
        # combination, to check the robustness of empty bin derivations.
        a = np.copy(self.b_photo)
        ax1_1, ax1_2 = 132, 133
        ax2_1, ax2_2 = -1, 0
        q = ((self.b_astro[:, 0] >= ax1_1) & (self.b_astro[:, 0] <= ax1_2) &
             (self.b_astro[:, 1] >= ax2_1) & (self.b_astro[:, 1] <= ax2_2))
        a[q, 2] = np.nan
        var = [[self.a_astro, self.a_photo], [self.b_astro, a]]
        for folder, obj in zip([self.a_cat_folder_path, self.b_cat_folder_path], var):
            np.save('{}/con_cat_astro.npy'.format(folder), obj[0])
            np.save('{}/con_cat_photo.npy'.format(folder), obj[1])

        compute_photometric_likelihoods(
            self.joint_folder_path, self.a_cat_folder_path, self.b_cat_folder_path, self.afilts,
            self.bfilts, self.mem_chunk_num, self.cf_points, self.cf_areas, self.include_phot_like,
            self.use_phot_priors, self.group_sources_data, self.use_memmap_files)

        abinlen = np.load('{}/phot_like/abinlengths.npy'.format(self.joint_folder_path))
        assert np.all(abinlen == 51*np.ones((3, 6), int))

        bbinlen = np.load('{}/phot_like/bbinlengths.npy'.format(self.joint_folder_path))
        b_check = 26*np.ones((4, 6), int)
        b_check[2, 2] = 1
        assert np.all(bbinlen == b_check)

        c_array = np.load('{}/phot_like/c_array.npy'.format(self.joint_folder_path))
        c_check = np.ones((25, 50, 4, 3, 6), float)
        c_check[:, :, 2, :, 2] = 0
        assert_allclose(c_array, c_check, atol=1e-30)

    def test_small_number_bins(self):
        os.system('rm -r {}/phot_like/*'.format(self.joint_folder_path))
        # This test checks if there are <N sources in a single pointing-filter
        # combination, then everything is simply lumped into one big bin.
        a = np.copy(self.b_photo)
        ax1_1, ax1_2 = 132, 133
        ax2_1, ax2_2 = -1, 0
        q = ((self.b_astro[:, 0] >= ax1_1) & (self.b_astro[:, 0] <= ax1_2) &
             (self.b_astro[:, 1] >= ax2_1) & (self.b_astro[:, 1] <= ax2_2))
        seed = 9999
        rng = np.random.default_rng(seed)
        a[q, 2] = np.append(rng.uniform(10, 15, (100,)), (np.sum(q)-100)*[np.nan])

        var = [[self.a_astro, self.a_photo], [self.b_astro, a]]
        for folder, obj in zip([self.a_cat_folder_path, self.b_cat_folder_path], var):
            np.save('{}/con_cat_astro.npy'.format(folder), obj[0])
            np.save('{}/con_cat_photo.npy'.format(folder), obj[1])

        compute_photometric_likelihoods(
            self.joint_folder_path, self.a_cat_folder_path, self.b_cat_folder_path, self.afilts,
            self.bfilts, self.mem_chunk_num, self.cf_points, self.cf_areas, self.include_phot_like,
            self.use_phot_priors, self.group_sources_data, self.use_memmap_files)

        abinlen = np.load('{}/phot_like/abinlengths.npy'.format(self.joint_folder_path))
        assert np.all(abinlen == 51*np.ones((3, 6), int))

        bbinlen = np.load('{}/phot_like/bbinlengths.npy'.format(self.joint_folder_path))
        b_check = 26*np.ones((4, 6), int)
        b_check[2, 2] = 2
        assert np.all(bbinlen == b_check)

        c_array = np.load('{}/phot_like/c_array.npy'.format(self.joint_folder_path))
        c_check = np.ones((25, 50, 4, 3, 6), float)
        c_check[1:, :, 2, :, 2] = 0
        assert_allclose(c_array, c_check, atol=1e-30)

    def test_calculate_phot_like_input(self):
        os.system('rm -r {}/phot_like/*'.format(self.joint_folder_path))
        # Here we also have to dump a random "magref" file to placate the
        # checks on CrossMatch.
        for folder, name in zip([self.a_cat_folder_path, self.b_cat_folder_path], ['a', 'b']):
            np.save('{}/con_cat_astro.npy'.format(folder), getattr(self, '{}_astro'.format(name)))
            np.save('{}/con_cat_photo.npy'.format(folder), getattr(self, '{}_photo'.format(name)))
            np.save('{}/magref.npy'.format(folder),
                    np.zeros((len(getattr(self, '{}_astro'.format(name))))))

        self.cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'), use_memmap_files=True)
        self.cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                               'data/crossmatch_params_.txt'),
                                  os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                                  os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        self.cm.group_sources_data = self.group_sources_data
        self.cm.chunk_id = 1
        files_per_phot = 6
        self.cm.calculate_phot_like(files_per_phot)

        abinlen = np.load('{}/phot_like/abinlengths.npy'.format(self.joint_folder_path))
        bbinlen = np.load('{}/phot_like/bbinlengths.npy'.format(self.joint_folder_path))

        assert np.all(abinlen == 51*np.ones((3, 6), int))
        assert np.all(bbinlen == 26*np.ones((4, 6), int))

        for file, shape in zip(
            ['c_array', 'fa_array', 'fb_array'], [(25, 50, 4, 3, 6), (50, 4, 3, 6),
                                                  (25, 4, 3, 6)]):
            a = np.load('{}/phot_like/{}.npy'.format(self.joint_folder_path, file))
            assert np.all(a.shape == shape)
            assert_allclose(a, np.ones(shape, float), atol=1e-30)

        abins = np.load('{}/phot_like/abinsarray.npy'.format(self.joint_folder_path))
        bbins = np.load('{}/phot_like/bbinsarray.npy'.format(self.joint_folder_path))
        for folder, filts, _bins in zip([self.a_cat_folder_path, self.b_cat_folder_path],
                                        [self.afilts, self.bfilts], [abins, bbins]):
            a = np.load('{}/con_cat_astro.npy'.format(folder))
            b = np.load('{}/con_cat_photo.npy'.format(folder))
            for i, (ax1, ax2) in enumerate(self.cf_points):
                q = ((a[:, 0] >= ax1-0.5) & (a[:, 0] <= ax1+0.5) &
                     (a[:, 1] >= ax2-0.5) & (a[:, 1] <= ax2+0.5))
                for k in range(len(filts)):
                    hist, bins = np.histogram(b[q & ~np.isnan(b[:, k]), k],
                                              bins=_bins[:, k, i])
                    assert np.all(hist >= 250)

    def test_calculate_phot_like_incorrect_files(self):
        os.system('rm -r {}/phot_like/*'.format(self.joint_folder_path))
        self.cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'), use_memmap_files=True)
        self.cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                               'data/crossmatch_params_.txt'),
                                  os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                                  os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        self.cm.group_sources_data = self.group_sources_data
        self.cm.chunk_id = 1
        self.cm.run_cf = False
        files_per_phot = 6

        # Dummy set up the incorrect number of files in phot_like:
        for i in range(4):
            np.save('{}/phot_like/array_{}.npy'.format(self.joint_folder_path, i),
                    np.array([0]))

        with pytest.warns(UserWarning, match='Incorrect number of photometric likelihood files.'):
            self.cm.calculate_phot_like(files_per_phot)

        abinlen = np.load('{}/phot_like/abinlengths.npy'.format(self.joint_folder_path))
        bbinlen = np.load('{}/phot_like/bbinlengths.npy'.format(self.joint_folder_path))

        assert np.all(abinlen == 51*np.ones((3, 6), int))
        assert np.all(bbinlen == 26*np.ones((4, 6), int))

        for file, shape in zip(
            ['c_array', 'fa_array', 'fb_array'], [(25, 50, 4, 3, 6), (50, 4, 3, 6),
                                                  (25, 4, 3, 6)]):
            a = np.load('{}/phot_like/{}.npy'.format(self.joint_folder_path, file))
            assert np.all(a.shape == shape)
            assert_allclose(a, np.ones(shape, float), atol=1e-30)

        abins = np.load('{}/phot_like/abinsarray.npy'.format(self.joint_folder_path))
        bbins = np.load('{}/phot_like/bbinsarray.npy'.format(self.joint_folder_path))
        for folder, filts, _bins in zip([self.a_cat_folder_path, self.b_cat_folder_path],
                                        [self.afilts, self.bfilts], [abins, bbins]):
            a = np.load('{}/con_cat_astro.npy'.format(folder))
            b = np.load('{}/con_cat_photo.npy'.format(folder))
            for i, (ax1, ax2) in enumerate(self.cf_points):
                q = ((a[:, 0] >= ax1-0.5) & (a[:, 0] <= ax1+0.5) &
                     (a[:, 1] >= ax2-0.5) & (a[:, 1] <= ax2+0.5))
                for k in range(len(filts)):
                    hist, bins = np.histogram(b[q & ~np.isnan(b[:, k]), k],
                                              bins=_bins[:, k, i])
                    assert np.all(hist >= 250)

    def test_calculate_phot_like_load_files(self, capsys):
        os.system('rm -r {}/phot_like/*'.format(self.joint_folder_path))
        self.cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'), use_memmap_files=True)
        self.cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                               'data/crossmatch_params_.txt'),
                                  os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                                  os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        self.cm.group_sources_data = self.group_sources_data
        self.cm.chunk_id = 1
        self.cm.run_cf = False
        files_per_phot = 6

        # Dummy set up the correct number of files in phot_like:
        for i in range(2 + 2 * files_per_phot):
            np.save('{}/phot_like/array_{}.npy'.format(self.joint_folder_path, i),
                    np.array([0]))

        capsys.readouterr()
        self.cm.calculate_phot_like(files_per_phot)
        output = capsys.readouterr().out
        assert 'Loading photometric priors and likelihoods' in output


def test_get_field_dists_fortran():
    rng = np.random.default_rng(11119999)
    a = np.empty((10, 2), float)
    a[:, 0] = rng.uniform(0, 1, 10)
    a[:, 1] = rng.uniform(0, 1, 10)

    b = np.empty((15, 2), float)
    b[:, 0] = rng.uniform(0, 1, 15)
    b[:, 1] = rng.uniform(0, 1, 15)
    d = rng.rayleigh(scale=0.1/3600, size=10)
    t = rng.uniform(0, 2*np.pi, 10)
    b[:10, 0] = a[:, 0] + d * np.cos(t)
    b[:10, 1] = a[:, 1] + d * np.sin(t)
    ainds = np.arange(0, 10).reshape(1, 10, order='F')
    asize = np.array([1] * 10)
    berr = np.array([0.05/3600] * 15)
    aflags = np.ones(10, bool)
    bflags = np.ones(15, bool)
    bmag = np.array([10] * 15)
    lowmag, uppmag = -999, 999

    mask, area = plf.get_field_dists(a[:, 0], a[:, 1], b[:, 0], b[:, 1], ainds, asize, berr,
                                     aflags, bflags, bmag, lowmag, uppmag)
    test_mask = np.ones(10, bool)
    test_mask[d <= berr[:10]] = 0
    assert np.all(mask == test_mask)
    assert np.sum(mask) > 0
    assert area == np.sum(np.pi * berr[:10][d <= berr[:10]]**2)


def test_brightest_mag_fortran():
    rng = np.random.default_rng(11119998)
    a = np.empty((10, 2), float)
    a[:, 0] = rng.uniform(0, 1, 10)
    a[:, 1] = rng.uniform(0, 1, 10)

    b = np.empty((15, 2), float)
    b[:, 0] = rng.uniform(0, 1, 15)
    b[:, 1] = rng.uniform(0, 1, 15)
    d = rng.rayleigh(scale=0.1/3600, size=10)
    t = rng.uniform(0, 2*np.pi, 10)
    b[:10, 0] = a[:, 0] + d * np.cos(t)
    b[:10, 1] = a[:, 1] + d * np.sin(t)
    ainds = np.arange(0, 10).reshape(1, 10, order='F')
    asize = np.array([1] * 10)
    aerr = np.array([0.05/3600] * 10)
    aflags = np.ones(10, bool)
    bflags = np.ones(15, bool)
    amag = np.array([10] * 10)
    bmag = np.array([12] * 15)
    abin = np.array([9, 11])

    mask, area = plf.brightest_mag(a[:, 0], a[:, 1], b[:, 0], b[:, 1], amag, bmag, ainds, asize,
                                   aerr, aflags, bflags, abin)
    test_mask = np.zeros((15, 1), bool)
    test_mask[:10][d <= aerr, 0] = 1
    assert np.all(mask == test_mask)
    assert np.sum(mask) > 0
    assert_allclose(area, np.sum(np.pi * aerr[d <= aerr]**2) / np.sum(d <= aerr))


def test_make_bins():
    # Test a few combinations of input magnitude arrays to ensure make_bins
    # returns expected bins.

    dm = 0.1/251
    m = np.arange(10+1e-10, 15, dm)
    # Fudge edge magnitudes away from 'too close to bin edge' checks.
    m[m < 10.01] = 10.01
    m[m > 14.99] = 14.99
    bins = make_bins(m)
    assert_allclose(bins, np.arange(10, 15+1e-10, 0.1))
    h, _ = np.histogram(m, bins=bins)
    assert np.all(h == 251)

    dm = 0.1/2500
    m = np.arange(10+1e-10, 15, dm)
    m[m < 10.01] = 10.01
    m[m > 14.99] = 14.99
    bins = make_bins(m)
    assert_allclose(bins, np.arange(10, 15+1e-10, 0.1))
    h, _ = np.histogram(m, bins=bins)
    assert np.all(h == 2500)

    dm = 0.1/249
    m = np.arange(10+1e-10, 15, dm)
    m[m < 10.01] = 10.01
    m[m > 14.99] = 14.99
    bins = make_bins(m)
    assert_allclose(bins, np.arange(10, 15+1e-10, 0.2))
    h, _ = np.histogram(m, bins=bins)
    assert np.all(h == 2*249)

    dm = 0.1/124
    m = np.arange(10+1e-10, 15, dm)
    m[-1] = 15-1e-9
    # Don't fudge data away from bin edge this time.
    bins = make_bins(m)
    fake_bins = np.arange(10, 15.11, 0.3)
    fake_bins[0] = 10-1e-4
    fake_bins[-1] = 15+1e-4
    assert_allclose(bins, fake_bins)
    h, _ = np.histogram(m, bins=bins)
    # With this bin size not quite being exactly able to fit into
    # 10 -> 15 exactly, the final bin is only 0.2 wide, and hence
    # not a "triple" sized bin like the others.
    assert np.all(h[:-1] == 3*124) & (h[-1] == 2*124)


class TestFullPhotometricLikelihood:
    def setup_class(self):
        self.cf_points = np.array([[131.5, -0.5]])
        self.cf_areas = 0.25 * np.ones((1), float)
        self.joint_folder_path = 'test_path'
        self.a_cat_folder_path = 'gaia_folder'
        self.b_cat_folder_path = 'wise_folder'
        self.group_sources_data = StageData(ablen=None, ainds=None, asize=None,
                                            bblen=None, binds=None, bsize=None)
        self.use_memmap_files = True

        self.area = 0.25

        self.afilts, self.bfilts = np.array(['G']), np.array(['G'])

        self.mem_chunk_num = 2
        self.include_phot_like, self.use_phot_priors = True, True

        os.makedirs('{}/phot_like'.format(self.joint_folder_path), exist_ok=True)
        os.makedirs(self.a_cat_folder_path, exist_ok=True)
        os.makedirs(self.b_cat_folder_path, exist_ok=True)

        # Create a random selection of sources, and then NaN out 25% of each of
        # the filters, at random.
        seed = 98763
        rng = np.random.default_rng(seed)

        asig, bsig = 0.1, 0.15
        self.Ntot = 45000

        aa = np.empty((self.Ntot, 3), float)
        aa[:, 0] = rng.uniform(131.25, 131.75, self.Ntot)
        aa[:, 1] = rng.uniform(-0.75, -0.25, self.Ntot)
        aa[:, 2] = asig

        ba = np.empty((self.Ntot, 3), float)
        d = rng.rayleigh(scale=np.sqrt(asig**2 + bsig**2)/3600, size=self.Ntot)
        t = rng.uniform(0, 2*np.pi, size=self.Ntot)
        ba[:, 0] = aa[:, 0] + d * np.cos(t)
        ba[:, 1] = aa[:, 1] + d * np.sin(t)
        ba[:, 2] = bsig

        a_phot_sig, b_phot_sig = 0.05, 0.1

        ap = rng.uniform(10, 15, (self.Ntot, len(self.afilts)))
        bp = ap + rng.normal(loc=0, scale=b_phot_sig, size=(self.Ntot, len(self.bfilts)))
        ap = ap + rng.normal(loc=0, scale=a_phot_sig, size=(self.Ntot, len(self.afilts)))

        a_cut, b_cut = ap[:, 0] > 11, bp[:, 0] < 14.5
        self.Nc = np.sum(a_cut & b_cut)

        aa[~a_cut, 0] = rng.uniform(131.25, 131.75, np.sum(~a_cut))
        aa[~a_cut, 1] = rng.uniform(-0.75, -0.25, np.sum(~a_cut))
        ba[~b_cut, 0] = rng.uniform(131.25, 131.75, np.sum(~b_cut))
        ba[~b_cut, 1] = rng.uniform(-0.75, -0.25, np.sum(~b_cut))

        self.a_astro = aa
        self.b_astro = ba
        self.a_photo = ap
        self.b_photo = bp

        os.system('rm -r {}/group/*'.format(self.joint_folder_path))
        # Have to pre-create the various overlap arrays, and integral lengths:
        asize = np.ones(self.Ntot, int)
        asize[~a_cut] = 0
        ainds = -1*np.ones((1, self.Ntot), int, order='F')
        ainds[0, :] = np.arange(0, self.Ntot)
        ainds[0, ~a_cut] = -1
        bsize = np.ones(self.Ntot, int)
        bsize[~b_cut] = 0
        binds = -1*np.ones((1, self.Ntot), int, order='F')
        binds[0, :] = np.arange(0, self.Ntot)
        binds[0, ~b_cut] = -1
        np.save('{}/group/ainds.npy'.format(self.joint_folder_path), ainds)
        np.save('{}/group/asize.npy'.format(self.joint_folder_path), asize)
        np.save('{}/group/binds.npy'.format(self.joint_folder_path), binds)
        np.save('{}/group/bsize.npy'.format(self.joint_folder_path), bsize)

        # Integrate 2-D Gaussian to N*sigma radius gives probability Y of
        # 1 - exp(-0.5 N^2 sigma^2 / sigma^2) = 1 - exp(-0.5 N^2).
        # Rearranging for N gives N = sqrt(-2 ln(1 - Y))
        self.Y_f, self.Y_b = 0.99, 0.63
        N_b = np.sqrt(-2 * np.log(1 - self.Y_b))
        N_f = np.sqrt(-2 * np.log(1 - self.Y_f))
        ablen = N_b * np.sqrt(asig**2 + bsig**2) * np.ones(self.Ntot, float) / 3600
        ablen[~a_cut] = 0
        aflen = N_f * np.sqrt(asig**2 + bsig**2) * np.ones(self.Ntot, float) / 3600
        aflen[~a_cut] = 0
        bblen = N_b * np.sqrt(asig**2 + bsig**2) * np.ones(self.Ntot, float) / 3600
        bblen[~b_cut] = 0
        bflen = N_f * np.sqrt(asig**2 + bsig**2) * np.ones(self.Ntot, float) / 3600
        bflen[~b_cut] = 0
        np.save('{}/group/ablen.npy'.format(self.joint_folder_path), ablen)
        np.save('{}/group/aflen.npy'.format(self.joint_folder_path), aflen)
        np.save('{}/group/bblen.npy'.format(self.joint_folder_path), bblen)
        np.save('{}/group/bflen.npy'.format(self.joint_folder_path), bflen)

    def test_phot_like_prior_frac_inclusion(self):
        os.system('rm -r {}/phot_like/*'.format(self.joint_folder_path))
        for ipl, upp in zip([False, True, True], [True, False, True]):
            for bf, ff in zip([None, 0.5, None], [0.5, None, None]):
                msg = 'bright_frac' if bf is None else 'field_frac'
                with pytest.raises(ValueError, match='{} must be supplied if '.format(msg)):
                    compute_photometric_likelihoods(
                        self.joint_folder_path, self.a_cat_folder_path, self.b_cat_folder_path,
                        self.afilts, self.bfilts, self.mem_chunk_num, self.cf_points,
                        self.cf_areas, ipl, upp, self.group_sources_data, self.use_memmap_files,
                        bright_frac=bf, field_frac=ff)

    def test_compute_phot_like(self):
        os.system('rm -r {}/phot_like/*'.format(self.joint_folder_path))
        for folder, name in zip([self.a_cat_folder_path, self.b_cat_folder_path], ['a', 'b']):
            np.save('{}/con_cat_astro.npy'.format(folder), getattr(self, '{}_astro'.format(name)))
            np.save('{}/con_cat_photo.npy'.format(folder), getattr(self, '{}_photo'.format(name)))
        compute_photometric_likelihoods(
            self.joint_folder_path, self.a_cat_folder_path, self.b_cat_folder_path,
            self.afilts, self.bfilts, self.mem_chunk_num, self.cf_points,
            self.cf_areas, self.include_phot_like, self.use_phot_priors, self.group_sources_data,
            self.use_memmap_files, bright_frac=self.Y_b, field_frac=self.Y_f)

        c_p = np.load('{}/phot_like/c_priors.npy'.format(self.joint_folder_path))
        c_l = np.load('{}/phot_like/c_array.npy'.format(self.joint_folder_path))
        fa_p = np.load('{}/phot_like/fa_priors.npy'.format(self.joint_folder_path))
        fa_l = np.load('{}/phot_like/fa_array.npy'.format(self.joint_folder_path))
        fb_p = np.load('{}/phot_like/fb_priors.npy'.format(self.joint_folder_path))
        fb_l = np.load('{}/phot_like/fb_array.npy'.format(self.joint_folder_path))

        fake_c_p = self.Nc / self.area
        fake_fa_p = (self.Ntot - self.Nc) / self.area
        fake_fb_p = (self.Ntot - self.Nc) / self.area
        assert_allclose(c_p, fake_c_p, rtol=0.05)
        assert_allclose(fa_p, fake_fa_p, rtol=0.01)
        assert_allclose(fb_p, fake_fb_p, rtol=0.01)

        abins = make_bins(self.a_photo[~np.isnan(self.a_photo[:, 0]), 0])
        bbins = make_bins(self.b_photo[~np.isnan(self.b_photo[:, 0]), 0])
        a = self.a_photo[~np.isnan(self.a_photo[:, 0]), 0]
        b = self.b_photo[~np.isnan(self.b_photo[:, 0]), 0]
        q = (a <= 11) | (b >= 14.5)

        fake_fa_l = np.zeros((len(abins) - 1, 1, 1, 1), float)
        h, _ = np.histogram(a[q], bins=abins, density=True)
        fake_fa_l[:, 0, 0, 0] = h
        fake_abins = np.load('{}/phot_like/abinsarray.npy'.format(self.joint_folder_path))[:, 0, 0]
        assert np.all(abins == fake_abins)
        assert np.all(fake_fa_l.shape == fa_l.shape)
        assert_allclose(fa_l, fake_fa_l, atol=0.02)

        fake_fb_l = np.zeros((len(bbins) - 1, 1, 1, 1), float)
        h, _ = np.histogram(b[q], bins=bbins, density=True)
        fake_fb_l[:, 0, 0, 0] = h
        fake_bbins = np.load('{}/phot_like/bbinsarray.npy'.format(self.joint_folder_path))[:, 0, 0]
        assert np.all(bbins == fake_bbins)
        assert np.all(fake_fb_l.shape == fb_l.shape)
        assert_allclose(fb_l, fake_fb_l, atol=0.02)

        h, _, _ = np.histogram2d(a[~q], b[~q], bins=[abins, bbins], density=True)
        fake_c_l = np.zeros((len(bbins)-1, len(abins) - 1, 1, 1, 1), float, order='F')
        fake_c_l[:, :, 0, 0, 0] = h.T
        assert np.all(fake_c_l.shape == c_l.shape)
        assert_allclose(c_l, fake_c_l, rtol=0.1, atol=0.05)

    def test_compute_phot_like_use_priors_only(self):
        os.system('rm -r {}/phot_like/*'.format(self.joint_folder_path))
        for folder, name in zip([self.a_cat_folder_path, self.b_cat_folder_path], ['a', 'b']):
            np.save('{}/con_cat_astro.npy'.format(folder), getattr(self, '{}_astro'.format(name)))
            np.save('{}/con_cat_photo.npy'.format(folder), getattr(self, '{}_photo'.format(name)))
        include_phot_like = False
        compute_photometric_likelihoods(
            self.joint_folder_path, self.a_cat_folder_path, self.b_cat_folder_path,
            self.afilts, self.bfilts, self.mem_chunk_num, self.cf_points,
            self.cf_areas, include_phot_like, self.use_phot_priors, self.group_sources_data,
            self.use_memmap_files, bright_frac=self.Y_b, field_frac=self.Y_f)

        c_p = np.load('{}/phot_like/c_priors.npy'.format(self.joint_folder_path))
        c_l = np.load('{}/phot_like/c_array.npy'.format(self.joint_folder_path))
        fa_p = np.load('{}/phot_like/fa_priors.npy'.format(self.joint_folder_path))
        fa_l = np.load('{}/phot_like/fa_array.npy'.format(self.joint_folder_path))
        fb_p = np.load('{}/phot_like/fb_priors.npy'.format(self.joint_folder_path))
        fb_l = np.load('{}/phot_like/fb_array.npy'.format(self.joint_folder_path))

        fake_c_p = self.Nc / self.area
        fake_fa_p = (self.Ntot - self.Nc) / self.area
        fake_fb_p = (self.Ntot - self.Nc) / self.area
        assert_allclose(c_p, fake_c_p, rtol=0.05)
        assert_allclose(fa_p, fake_fa_p, rtol=0.01)
        assert_allclose(fb_p, fake_fb_p, rtol=0.01)

        abins = make_bins(self.a_photo[~np.isnan(self.a_photo[:, 0]), 0])
        bbins = make_bins(self.b_photo[~np.isnan(self.b_photo[:, 0]), 0])

        fake_fa_l = np.ones((len(abins) - 1, 1, 1, 1), float)
        fake_abins = np.load('{}/phot_like/abinsarray.npy'.format(self.joint_folder_path))[:, 0, 0]
        assert np.all(abins == fake_abins)
        assert np.all(fake_fa_l.shape == fa_l.shape)
        assert_allclose(fa_l, fake_fa_l)

        fake_fb_l = np.ones((len(bbins) - 1, 1, 1, 1), float)
        fake_bbins = np.load('{}/phot_like/bbinsarray.npy'.format(self.joint_folder_path))[:, 0, 0]
        assert np.all(bbins == fake_bbins)
        assert np.all(fake_fb_l.shape == fb_l.shape)
        assert_allclose(fb_l, fake_fb_l)

        fake_c_l = np.ones((len(bbins)-1, len(abins) - 1, 1, 1, 1), float, order='F')
        assert np.all(fake_c_l.shape == c_l.shape)
        assert_allclose(c_l, fake_c_l)
