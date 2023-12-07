# Licensed under a 3-clause BSD style license - see LICENSE
'''
This module provides testing utility functionality for the package.
'''

import os

import numpy as np


def generate_random_data(n_a, n_b, n_c, extent, n_a_filts, n_b_filts, a_astro_sig, b_astro_sig,
                         a_cat, b_cat, seed=None):
    '''
    Convenience function to allow for the generation of two test datasets.

    Parameters
    ----------
    n_a : integer
        The number of sources to be generated in catalogue "a".
    n_b : integer
        The number of catalogue "b" fake sources.
    n_c : integer
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
    if n_a > n_b:
        raise ValueError("n_a must be smaller or equal to n_b.")
    if n_c > n_a:
        raise ValueError("n_c must be smaller or equal to n_a.")

    a_astro = np.empty((n_a, 3), float)
    b_astro = np.empty((n_b, 3), float)

    rng = np.random.default_rng(seed)
    a_astro[:, 0] = rng.uniform(extent[0], extent[1], size=n_a)
    a_astro[:, 1] = rng.uniform(extent[2], extent[3], size=n_a)
    if np.isscalar(a_astro_sig):
        a_astro[:, 2] = a_astro_sig
    else:
        # Here we assume that astrometric uncertainty goes quadratically
        # with magnitude
        raise ValueError("a_sig currently has to be an integer for all generated data.")

    a_pair_indices = np.arange(n_c)
    b_pair_indices = rng.choice(n_b, n_c, replace=False)
    b_astro[b_pair_indices, 0] = a_astro[a_pair_indices, 0]
    b_astro[b_pair_indices, 1] = a_astro[a_pair_indices, 1]
    inv_b_pair = np.delete(np.arange(n_b), b_pair_indices)
    b_astro[inv_b_pair, 0] = rng.uniform(extent[0], extent[1], size=n_b - n_c)
    b_astro[inv_b_pair, 1] = rng.uniform(extent[2], extent[3], size=n_b - n_c)
    if np.isscalar(b_astro_sig):
        b_astro[:, 2] = b_astro_sig
    else:
        # Here we assume that astrometric uncertainty goes quadratically
        # with magnitude
        raise ValueError("b_sig currently has to be an integer for all generated data.")

    a_circ_dist = rng.normal(loc=0, scale=a_astro[:, 2], size=n_a) / 3600
    a_circ_angle = rng.uniform(0, 2 * np.pi, size=n_a)
    a_astro[:, 0] = a_astro[:, 0] + a_circ_dist * np.cos(a_circ_angle)
    a_astro[:, 1] = a_astro[:, 1] + a_circ_dist * np.sin(a_circ_angle)
    b_circ_dist = rng.normal(loc=0, scale=b_astro[:, 2], size=n_b) / 3600
    b_circ_angle = rng.uniform(0, 2 * np.pi, size=n_b)
    b_astro[:, 0] = b_astro[:, 0] + b_circ_dist * np.cos(b_circ_angle)
    b_astro[:, 1] = b_astro[:, 1] + b_circ_dist * np.sin(b_circ_angle)

    # Currently all we do, given the only option available is a naive Bayes match,
    # is ignore the photometry -- but we still require its file to be present.
    a_photo = rng.uniform(0.9, 1.1, size=(n_a, n_a_filts))
    b_photo = rng.uniform(0.9, 1.1, size=(n_b, n_b_filts))

    # Similarly, we need magref for each catalogue, but don't care what's in it.
    amagref = rng.choice(n_a_filts, size=n_a)
    bmagref = rng.choice(n_b_filts, size=n_b)

    for f in [a_cat, b_cat]:
        os.makedirs(f, exist_ok=True)
    np.save(f'{a_cat}/con_cat_astro.npy', a_astro)
    np.save(f'{b_cat}/con_cat_astro.npy', b_astro)
    np.save(f'{a_cat}/con_cat_photo.npy', a_photo)
    np.save(f'{b_cat}/con_cat_photo.npy', b_photo)
    np.save(f'{a_cat}/magref.npy', amagref)
    np.save(f'{b_cat}/magref.npy', bmagref)

    # Fake uninformative "overlap" flag data, where no sources are in the
    # halo of the chunk.
    np.save(f'{a_cat}/in_chunk_overlap.npy', np.zeros(n_a, bool))
    np.save(f'{b_cat}/in_chunk_overlap.npy', np.zeros(n_b, bool))

    np.save(f'{a_cat}/test_match_indices.npy', a_pair_indices)
    np.save(f'{b_cat}/test_match_indices.npy', b_pair_indices)
