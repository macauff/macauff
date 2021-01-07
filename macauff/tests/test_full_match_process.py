# Licensed under a 3-clause BSD style license - see LICENSE
'''
Tests of full, end-to-end runs of the crossmatch process.
'''

import os
import numpy as np

from ..matching import CrossMatch
from .test_matching import _replace_line

__all__ = ['generate_random_data']


def generate_random_data(N_a, N_b, N_c, extent, n_a_filts, n_b_filts, a_astro_sig, b_astro_sig,
                         a_cat, b_cat, seed=None):
    '''
    Convenience function to allow for the generation of two test datasets.

    Parameters
    ----------
    N_a : integer
        The number of sources to be generated in catalogue "a".
    N_b : integer
        The number of catalogue "b" fake sources.
    N_c : integer
        The number of common, overlapping sources, in both catalogues.
    extent : list of integers
        The on-sky coordinates that mark out the rectangular limits over which
        sources should be generated. Should be of the form
        ``[lower_lon, upper_lon, lower_lat, upper_lat]``, in degrees.
    n_a_filts : integer
        The number of photometric filters catalogue "a" should have.
    n_b_filts : integer
        The number of catalogue "b" photometric filters.
    a_astro_sig : float
        The astrometric uncertainty of catalogue "a", in arcseconds.
    b_astro_sig : float
        Catalogue "b"'s astrometric uncertainty, in arcseconds.
    a_cat : string
        The folder path into which to save the binary files containing
        catalogue "a"'s astrometric and photometric data.
    b_cat : string
        Folder describing the save location of catalogue "b"'s data.
    seed : integer, optional
        Random number generator seed. If ``None``, will be passed to
        ``np.random.default_rng`` as such, and a seed will be generated
        as per ``default_rng``'s documentation.
    '''
    if N_a > N_b:
        raise ValueError("N_a must be smaller or equal to N_b.")
    if N_c > N_a:
        raise ValueError("N_c must be smaller or equal to N_a.")

    a_astro = np.empty((N_a, 3), float)
    b_astro = np.empty((N_b, 3), float)

    rng = np.random.default_rng(seed)
    a_astro[:, 0] = rng.uniform(extent[0], extent[1], size=N_a)
    a_astro[:, 1] = rng.uniform(extent[2], extent[3], size=N_a)
    if np.isscalar(a_astro_sig):
        a_astro[:, 2] = a_astro_sig
    else:
        # Here we assume that astrometric uncertainty goes quadratically
        # with magnitude
        raise ValueError("a_sig currently has to be an integer for all generated data.")

    a_pair_indices = np.arange(N_c)
    b_pair_indices = rng.choice(N_b, N_c, replace=False)
    b_astro[b_pair_indices, 0] = a_astro[a_pair_indices, 0]
    b_astro[b_pair_indices, 1] = a_astro[a_pair_indices, 1]
    inv_b_pair = np.delete(np.arange(N_b), b_pair_indices)
    b_astro[inv_b_pair, 0] = rng.uniform(extent[0], extent[1], size=N_b-N_c)
    b_astro[inv_b_pair, 1] = rng.uniform(extent[2], extent[3], size=N_b-N_c)
    if np.isscalar(b_astro_sig):
        b_astro[:, 2] = b_astro_sig
    else:
        # Here we assume that astrometric uncertainty goes quadratically
        # with magnitude
        raise ValueError("b_sig currently has to be an integer for all generated data.")

    a_astro[:, 0] = a_astro[:, 0] + rng.normal(loc=0, scale=a_astro_sig, size=N_c) / 3600
    a_astro[:, 1] = a_astro[:, 1] + rng.normal(loc=0, scale=a_astro_sig, size=N_c) / 3600
    b_astro[:, 0] = b_astro[:, 0] + rng.normal(loc=0, scale=b_astro_sig, size=N_c) / 3600
    b_astro[:, 1] = b_astro[:, 1] + rng.normal(loc=0, scale=b_astro_sig, size=N_c) / 3600

    # Currently all we do, given the only option available is a naive Bayes match,
    # is ignore the photometry -- but we still require its file to be present.
    a_photo = rng.uniform(0.9, 1.1, size=(N_a, n_a_filts))
    b_photo = rng.uniform(0.9, 1.1, size=(N_b, n_b_filts))

    # Similarly, we need magref for each catalogue, but don't care what's in it.
    amagref = rng.choice(n_a_filts, size=N_a)
    bmagref = rng.choice(n_b_filts, size=N_b)

    for f in [a_cat, b_cat]:
        os.makedirs(f, exist_ok=True)
    np.save('{}/con_cat_astro.npy'.format(a_cat), a_astro)
    np.save('{}/con_cat_astro.npy'.format(b_cat), b_astro)
    np.save('{}/con_cat_photo.npy'.format(a_cat), a_photo)
    np.save('{}/con_cat_photo.npy'.format(b_cat), b_photo)
    np.save('{}/magref.npy'.format(a_cat), amagref)
    np.save('{}/magref.npy'.format(b_cat), bmagref)

    np.save('{}/test_match_indices.npy'.format(a_cat), a_pair_indices)
    np.save('{}/test_match_indices.npy'.format(b_cat), b_pair_indices)


def test_naive_bayes_match():
    # Generate a small number of sources randomly, then run through the
    # cross-match process.
    N_a, N_b, N_c = 20, 30, 15
    n_a_filts, n_b_filts = 3, 4
    a_astro_sig, b_astro_sig = 0.3, 0.5
    r = 5 * np.sqrt(a_astro_sig**2 + b_astro_sig**2)
    extent = [131, 131 + np.sqrt(N_b * np.pi * r**2)/3600, 0, np.sqrt(N_b * np.pi * r**2)/3600]

    a_cat, b_cat = 'a_cat', 'b_cat'

    generate_random_data(N_a, N_b, N_c, extent, n_a_filts, n_b_filts, a_astro_sig, b_astro_sig,
                         a_cat, b_cat, seed=9999)

    ol, nl = 'run_auf = no', 'run_auf = yes\n'
    f = open(os.path.join(os.path.dirname(__file__),
                          'data/crossmatch_params.txt')).readlines()
    idx = np.where([ol in line for line in f])[0][0]
    _replace_line(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                  idx, nl, out_file=os.path.join(os.path.dirname(__file__),
                  'data/crossmatch_params_.txt'))

    new_ext = [extent[0] - r/3600 - 0.1/3600, extent[1] + r/3600 + 0.1/3600,
               extent[2] - r/3600 - 0.1/3600, extent[3] + r/3600 + 0.1/3600]
    for ol, nl in zip(['run_group = no', 'pos_corr_dist = 11',
                       'cross_match_extent = 131 138 -3 3', 'joint_folder_path = test_path',
                       'cf_region_points = 131 134 4 -1 1 3'],
                      ['run_group = yes\n', 'pos_corr_dist = {:.2f}\n'.format(r),
                       'cross_match_extent = {:.3f} {:.3f} {:.3f} {:.3f}\n'.format(*new_ext),
                       'joint_folder_path = new_test_path\n',
                       'cf_region_points = 131 131 1 0 0 1\n']):
        f = open(os.path.join(os.path.dirname(__file__),
                              'data/crossmatch_params.txt')).readlines()
        idx = np.where([ol in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params_.txt'),
                      idx, nl)

    ol, nl = 'auf_region_points = 131 134 4 -1 1 {}', 'auf_region_points = 131 131 1 0 0 1\n'
    for file_name in ['cat_a_params', 'cat_b_params']:
        _ol = ol.format('3' if '_a_' in file_name else '4')
        f = open(os.path.join(os.path.dirname(__file__),
                              'data/{}.txt'.format(file_name))).readlines()
        idx = np.where([_ol in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/{}.txt'.format(file_name)),
                      idx, nl, out_file=os.path.join(os.path.dirname(__file__),
                      'data/{}_.txt'.format(file_name)))

    for cat, ol, nl in zip(['cat_a_params', 'cat_b_params'], ['cat_folder_path = gaia_folder',
                           'cat_folder_path = wise_folder'], ['cat_folder_path = a_cat\n',
                           'cat_folder_path = b_cat\n']):
        f = open(os.path.join(os.path.dirname(__file__),
                              'data/{}.txt'.format(cat))).readlines()
        idx = np.where([ol in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/{}_.txt'.format(cat)),
                      idx, nl)

    cm = CrossMatch(os.path.join(os.path.dirname(__file__),
                                 'data/crossmatch_params_.txt'),
                    os.path.join(os.path.dirname(__file__), 'data/cat_a_params_.txt'),
                    os.path.join(os.path.dirname(__file__), 'data/cat_b_params_.txt'))

    cm()

    ac = np.load('{}/pairing/ac.npy'.format(cm.joint_folder_path))
    bc = np.load('{}/pairing/bc.npy'.format(cm.joint_folder_path))
    assert len(ac) == N_c
    assert len(bc) == N_c

    a_right_inds = np.load('{}/test_match_indices.npy'.format(a_cat))
    b_right_inds = np.load('{}/test_match_indices.npy'.format(b_cat))

    for i in range(0, N_c):
        assert a_right_inds[i] in ac
        assert b_right_inds[i] in bc
        q = np.where(a_right_inds[i] == ac)[0][0]
        assert np.all([a_right_inds[i], b_right_inds[i]] == [ac[q], bc[q]])
