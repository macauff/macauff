# Licensed under a 3-clause BSD style license - see LICENSE
'''
Tests for the "photometric_likelihood" module.
'''

import os
from numpy.testing import assert_allclose
import numpy as np

from ..photometric_likelihood import compute_photometric_likelihoods


def test_compute_photometric_likelihoods():
    ax1slices = np.array([10, 11, 12, 13])
    ax2slices = np.array([3, 4, 5])
    area = (ax1slices[-1] - ax1slices[0]) * (ax2slices[-1] - ax2slices[0])
    joint_folder_path = 'joint'
    a_cat_folder_path = 'a_folder'
    b_cat_folder_path = 'b_folder'

    afilts, bfilts = np.array(['G', 'RP']), np.array(['W1', 'W2', 'W3'])

    mem_chunk_num = 2
    include_phot_like, use_phot_priors = False, False

    os.makedirs('{}/phot_like'.format(joint_folder_path), exist_ok=True)
    os.makedirs(a_cat_folder_path, exist_ok=True)
    os.makedirs(b_cat_folder_path, exist_ok=True)

    # Create a random selection of sources, and then NaN out 25% of each of
    # the filters, at random.
    seed = 98765
    rng = np.random.default_rng(seed)

    Na, Nb = 200000, 80000
    for N, folder, filts in zip([Na, Nb], [a_cat_folder_path, b_cat_folder_path],
                                [afilts, bfilts]):
        a = np.empty((N, 3), float)
        a[:, 0] = rng.uniform(ax1slices[0], ax1slices[-1], N)
        a[:, 1] = rng.uniform(ax2slices[0], ax2slices[-1], N)
        a[:, 2] = 0.1

        np.save('{}/con_cat_astro.npy'.format(folder), a)

        a = rng.uniform(10, 15, (N, len(filts)))
        for i in range(len(filts)):
            q = rng.choice(N, size=N // 4, replace=False)
            a[q, i] = np.nan
        np.save('{}/con_cat_photo.npy'.format(folder), a)

    compute_photometric_likelihoods(joint_folder_path, a_cat_folder_path, b_cat_folder_path,
                                    afilts, bfilts, mem_chunk_num, ax1slices, ax2slices,
                                    include_phot_like, use_phot_priors)

    for file, shape, value in zip(['c_priors', 'fa_priors', 'fb_priors'],
                                  [(3, 2, 2, 3), (3, 2, 2, 3), (3, 2, 2, 3)],
                                  [0.75*Nb/area/2, 0.75/area*(Na-Nb/2), 0.75/area*(Nb-Nb/2)]):
        a = np.load('{}/phot_like/{}.npy'.format(joint_folder_path, file))
        assert np.all(a.shape == shape)
        # Allow 3% tolerance for counting statistics in the distribution above
        # caused by the rng.choice removal of objects in each filter.
        assert_allclose(a, value, rtol=0.03)

    abinlen = np.load('{}/phot_like/abinlengths.npy'.format(joint_folder_path))
    bbinlen = np.load('{}/phot_like/bbinlengths.npy'.format(joint_folder_path))

    assert np.all(abinlen == 51*np.ones((2, 2, 3), int))
    assert np.all(bbinlen == 26*np.ones((3, 2, 3), int))

    for file, shape in zip(
        ['c_array', 'fa_array', 'fb_array'], [(25, 50, 3, 2, 2, 3), (50, 3, 2, 2, 3),
                                              (25, 3, 2, 2, 3)]):
        a = np.load('{}/phot_like/{}.npy'.format(joint_folder_path, file))
        assert np.all(a.shape == shape)
        assert np.all(a == np.ones(shape, float))

    abins = np.load('{}/phot_like/abinsarray.npy'.format(joint_folder_path))
    bbins = np.load('{}/phot_like/bbinsarray.npy'.format(joint_folder_path))
    for folder, filts, _bins in zip([a_cat_folder_path, b_cat_folder_path], [afilts, bfilts],
                                    [abins, bbins]):
        a = np.load('{}/con_cat_astro.npy'.format(folder))
        b = np.load('{}/con_cat_photo.npy'.format(folder))
        for i, (ax1_1, ax1_2) in enumerate(zip(ax1slices[:-1], ax1slices[1:])):
            for j, (ax2_1, ax2_2) in enumerate(zip(ax2slices[:-1], ax2slices[1:])):
                q = ((a[:, 0] >= ax1_1) & (a[:, 0] <= ax1_2) &
                     (a[:, 1] >= ax2_1) & (a[:, 1] <= ax2_2))
                for k in range(len(filts)):
                    hist, bins = np.histogram(b[q & ~np.isnan(b[:, k]), k], bins=_bins[:, k, j, i])
                    assert np.all(hist >= 250)
