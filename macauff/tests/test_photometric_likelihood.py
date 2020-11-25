# Licensed under a 3-clause BSD style license - see LICENSE
'''
Tests for the "photometric_likelihood" module.
'''

import os
from numpy.testing import assert_allclose
import numpy as np
import pytest

from ..photometric_likelihood import compute_photometric_likelihoods


class TestOneSidedPhotometricLikelihood:
    def setup_class(self):
        self.ax1slices = np.array([10, 11, 12, 13])
        self.ax2slices = np.array([3, 4, 5])
        self.area = (self.ax1slices[-1]-self.ax1slices[0])*(self.ax2slices[-1]-self.ax2slices[0])
        self.joint_folder_path = 'joint'
        self.a_cat_folder_path = 'a_folder'
        self.b_cat_folder_path = 'b_folder'

        self.afilts, self.bfilts = np.array(['G', 'RP']), np.array(['W1', 'W2', 'W3'])

        self.mem_chunk_num = 2
        self.include_phot_like, self.use_phot_priors = False, False

        os.makedirs('{}/phot_like'.format(self.joint_folder_path), exist_ok=True)
        os.makedirs(self.a_cat_folder_path, exist_ok=True)
        os.makedirs(self.b_cat_folder_path, exist_ok=True)

        # Create a random selection of sources, and then NaN out 25% of each of
        # the filters, at random.
        seed = 98765
        rng = np.random.default_rng(seed)

        self.Na, self.Nb = 200000, 80000
        for N, folder, filts, name in zip([self.Na, self.Nb],
                                          [self.a_cat_folder_path, self.b_cat_folder_path],
                                          [self.afilts, self.bfilts], ['a', 'b']):
            a = np.empty((N, 3), float)
            a[:, 0] = rng.uniform(self.ax1slices[0], self.ax1slices[-1], N)
            a[:, 1] = rng.uniform(self.ax2slices[0], self.ax2slices[-1], N)
            a[:, 2] = 0.1

            setattr(self, '{}_astro'.format(name), a)

            a = rng.uniform(10, 15, (N, len(filts)))
            for i in range(len(filts)):
                q = rng.choice(N, size=N // 4, replace=False)
                a[q, i] = np.nan

            setattr(self, '{}_photo'.format(name), a)

    def test_compute_photometric_likelihoods(self):
        Na, Nb, area = self.Na, self.Nb, self.area
        for folder, name in zip([self.a_cat_folder_path, self.b_cat_folder_path], ['a', 'b']):
            np.save('{}/con_cat_astro.npy'.format(folder), getattr(self, '{}_astro'.format(name)))
            np.save('{}/con_cat_photo.npy'.format(folder), getattr(self, '{}_photo'.format(name)))
        compute_photometric_likelihoods(
            self.joint_folder_path, self.a_cat_folder_path, self.b_cat_folder_path, self.afilts,
            self.bfilts, self.mem_chunk_num, self.ax1slices, self.ax2slices,
            self.include_phot_like, self.use_phot_priors)

        for file, shape, value in zip(['c_priors', 'fa_priors', 'fb_priors'],
                                      [(3, 2, 2, 3), (3, 2, 2, 3), (3, 2, 2, 3)],
                                      [0.75*Nb/area/2, 0.75/area*(Na-Nb/2), 0.75/area*(Nb-Nb/2)]):
            a = np.load('{}/phot_like/{}.npy'.format(self.joint_folder_path, file))
            assert np.all(a.shape == shape)
            # Allow 3% tolerance for counting statistics in the distribution above
            # caused by the rng.choice removal of objects in each filter.
            assert_allclose(a, value, rtol=0.03)

        abinlen = np.load('{}/phot_like/abinlengths.npy'.format(self.joint_folder_path))
        bbinlen = np.load('{}/phot_like/bbinlengths.npy'.format(self.joint_folder_path))

        assert np.all(abinlen == 51*np.ones((2, 2, 3), int))
        assert np.all(bbinlen == 26*np.ones((3, 2, 3), int))

        for file, shape in zip(
            ['c_array', 'fa_array', 'fb_array'], [(25, 50, 3, 2, 2, 3), (50, 3, 2, 2, 3),
                                                  (25, 3, 2, 2, 3)]):
            a = np.load('{}/phot_like/{}.npy'.format(self.joint_folder_path, file))
            assert np.all(a.shape == shape)
            assert np.all(a == np.ones(shape, float))

        abins = np.load('{}/phot_like/abinsarray.npy'.format(self.joint_folder_path))
        bbins = np.load('{}/phot_like/bbinsarray.npy'.format(self.joint_folder_path))
        for folder, filts, _bins in zip([self.a_cat_folder_path, self.b_cat_folder_path],
                                        [self.afilts, self.bfilts], [abins, bbins]):
            a = np.load('{}/con_cat_astro.npy'.format(folder))
            b = np.load('{}/con_cat_photo.npy'.format(folder))
            for i, (ax1_1, ax1_2) in enumerate(zip(self.ax1slices[:-1], self.ax1slices[1:])):
                for j, (ax2_1, ax2_2) in enumerate(zip(self.ax2slices[:-1], self.ax2slices[1:])):
                    q = ((a[:, 0] >= ax1_1) & (a[:, 0] <= ax1_2) &
                         (a[:, 1] >= ax2_1) & (a[:, 1] <= ax2_2))
                    for k in range(len(filts)):
                        hist, bins = np.histogram(b[q & ~np.isnan(b[:, k]), k],
                                                  bins=_bins[:, k, j, i])
                        assert np.all(hist >= 250)

    def test_raise_error_message(self):
        include_phot_like, use_phot_priors = False, True
        for folder, name in zip([self.a_cat_folder_path, self.b_cat_folder_path], ['a', 'b']):
            np.save('{}/con_cat_astro.npy'.format(folder), getattr(self, '{}_astro'.format(name)))
            np.save('{}/con_cat_photo.npy'.format(folder), getattr(self, '{}_photo'.format(name)))
        with pytest.raises(NotImplementedError, match='Only one-sided, asymmetric'):
            compute_photometric_likelihoods(
                self.joint_folder_path, self.a_cat_folder_path, self.b_cat_folder_path,
                self.afilts, self.bfilts, self.mem_chunk_num, self.ax1slices, self.ax2slices,
                include_phot_like, use_phot_priors)

        include_phot_like, use_phot_priors = True, False
        with pytest.raises(NotImplementedError, match='Photometric likelihoods not currently'):
            compute_photometric_likelihoods(
                self.joint_folder_path, self.a_cat_folder_path, self.b_cat_folder_path,
                self.afilts, self.bfilts, self.mem_chunk_num, self.ax1slices, self.ax2slices,
                include_phot_like, use_phot_priors)

        include_phot_like, use_phot_priors = True, True
        with pytest.raises(NotImplementedError, match='Photometric likelihoods not currently'):
            compute_photometric_likelihoods(
                self.joint_folder_path, self.a_cat_folder_path, self.b_cat_folder_path,
                self.afilts, self.bfilts, self.mem_chunk_num, self.ax1slices, self.ax2slices,
                include_phot_like, use_phot_priors)

    def test_empty_filter(self):
        # This test simply removes all "good" flags from a single pointing-filter
        # combination, to check the robustness of empty bin derivations.
        a = np.copy(self.b_photo)
        ax1_1, ax1_2 = self.ax1slices[2], self.ax1slices[3]
        ax2_1, ax2_2 = self.ax2slices[0], self.ax2slices[1]
        q = ((self.b_astro[:, 0] >= ax1_1) & (self.b_astro[:, 0] <= ax1_2) &
             (self.b_astro[:, 1] >= ax2_1) & (self.b_astro[:, 1] <= ax2_2))
        a[q, 2] = np.nan
        var = [[self.a_astro, self.a_photo], [self.b_astro, a]]
        for folder, obj in zip([self.a_cat_folder_path, self.b_cat_folder_path], var):
            np.save('{}/con_cat_astro.npy'.format(folder), obj[0])
            np.save('{}/con_cat_photo.npy'.format(folder), obj[1])

        compute_photometric_likelihoods(
            self.joint_folder_path, self.a_cat_folder_path, self.b_cat_folder_path, self.afilts,
            self.bfilts, self.mem_chunk_num, self.ax1slices, self.ax2slices,
            self.include_phot_like, self.use_phot_priors)

        abinlen = np.load('{}/phot_like/abinlengths.npy'.format(self.joint_folder_path))
        assert np.all(abinlen == 51*np.ones((2, 2, 3), int))

        bbinlen = np.load('{}/phot_like/bbinlengths.npy'.format(self.joint_folder_path))
        b_check = 26*np.ones((3, 2, 3), int)
        b_check[2, 0, 2] = 1
        assert np.all(bbinlen == b_check)

        c_array = np.load('{}/phot_like/c_array.npy'.format(self.joint_folder_path))
        c_check = np.ones((25, 50, 3, 2, 2, 3), float)
        c_check[:, :, 2, :, 0, 2] = 0
        assert np.all(c_array == c_check)

    def test_small_number_bins(self):
        # This test checks if there are <N sources in a single pointing-filter
        # combination, then everything is simply lumped into one big bin.
        a = np.copy(self.b_photo)
        ax1_1, ax1_2 = self.ax1slices[2], self.ax1slices[3]
        ax2_1, ax2_2 = self.ax2slices[0], self.ax2slices[1]
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
            self.bfilts, self.mem_chunk_num, self.ax1slices, self.ax2slices,
            self.include_phot_like, self.use_phot_priors)

        abinlen = np.load('{}/phot_like/abinlengths.npy'.format(self.joint_folder_path))
        assert np.all(abinlen == 51*np.ones((2, 2, 3), int))

        bbinlen = np.load('{}/phot_like/bbinlengths.npy'.format(self.joint_folder_path))
        b_check = 26*np.ones((3, 2, 3), int)
        b_check[2, 0, 2] = 2
        assert np.all(bbinlen == b_check)

        c_array = np.load('{}/phot_like/c_array.npy'.format(self.joint_folder_path))
        c_check = np.ones((25, 50, 3, 2, 2, 3), float)
        c_check[1:, :, 2, :, 0, 2] = 0
        assert np.all(c_array == c_check)
