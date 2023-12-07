# Licensed under a 3-clause BSD style license - see LICENSE
'''
Tests of full, end-to-end runs of the crossmatch process.
'''

import os

import numpy as np
import pytest
from test_matching import _replace_line

# pylint: disable=import-error,no-name-in-module
from macauff.matching import CrossMatch
from macauff.utils import generate_random_data

# pylint: enable=import-error,no-name-in-module
# pylint: disable=duplicate-code


@pytest.mark.parametrize("x,y", [(131, 0), (0, 0)])
# pylint: disable=too-many-locals
def test_naive_bayes_match(x, y):
    # Generate a small number of sources randomly, then run through the
    # cross-match process.
    n_a, n_b, n_c = 40, 50, 35
    n_a_filts, n_b_filts = 3, 4
    a_astro_sig, b_astro_sig = 0.3, 0.5
    r = 5 * np.sqrt(a_astro_sig**2 + b_astro_sig**2)
    dx = np.sqrt(n_b * np.pi * r**2)/3600
    extent = [x-dx/2, x+dx/2, y-dx/2, y+dx/2]

    a_cat, b_cat = 'a_cat', 'b_cat'

    generate_random_data(n_a, n_b, n_c, extent, n_a_filts, n_b_filts, a_astro_sig, b_astro_sig,
                         a_cat, b_cat, seed=9999)

    # Ensure output chunk directory exists
    os.makedirs(os.path.join(os.path.dirname(__file__), "data/chunk0"), exist_ok=True)

    ol, nl = 'pos_corr_dist = 11', f'pos_corr_dist = {r:.2f}\n'
    with open(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
              encoding='utf-8') as file:
        f = file.readlines()
    idx = np.where([ol in line for line in f])[0][0]
    _replace_line(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                  idx, nl, out_file=os.path.join(os.path.dirname(__file__),
                  'data/chunk0/crossmatch_params_.txt'))

    new_region_points = f'{x} {x} 1 {y} {y} 1'

    new_ext = [extent[0] - r/3600 - 0.1/3600, extent[1] + r/3600 + 0.1/3600,
               extent[2] - r/3600 - 0.1/3600, extent[3] + r/3600 + 0.1/3600]
    for ol, nl in zip(['cross_match_extent = 131 138 -3 3', 'joint_folder_path = test_path',
                       'cf_region_points = 131 134 4 -1 1 3'],
                      ['cross_match_extent = {:.3f} {:.3f} {:.3f} {:.3f}\n'.format(*new_ext),
                       'joint_folder_path = new_test_path\n',
                       f'cf_region_points = {new_region_points}\n']):
        with open(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        idx = np.where([ol in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/chunk0/crossmatch_params_.txt'),
                      idx, nl)

    ol = 'auf_region_points = 131 134 4 -1 1 {}'
    nl = f'auf_region_points = {new_region_points}\n'
    for file_name in ['cat_a_params', 'cat_b_params']:
        _ol = ol.format('3' if '_a_' in file_name else '4')
        with open(os.path.join(os.path.dirname(__file__), f'data/{file_name}.txt'), encoding='utf-8') as file:
            f = file.readlines()
        idx = np.where([_ol in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), f'data/{file_name}.txt'),
                      idx, nl, out_file=os.path.join(os.path.dirname(__file__),
                      f'data/chunk0/{file_name}_.txt'))

    for cat, ol, nl in zip(['cat_a_params', 'cat_b_params'], ['cat_folder_path = gaia_folder',
                           'cat_folder_path = wise_folder'], ['cat_folder_path = a_cat\n',
                           'cat_folder_path = b_cat\n']):
        with open(os.path.join(os.path.dirname(__file__), f'data/{cat}.txt'), encoding='utf-8') as file:
            f = file.readlines()
        idx = np.where([ol in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), f'data/chunk0/{cat}_.txt'), idx, nl)

    cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'))
    cm()

    ac = np.load(f'{cm.joint_folder_path}/pairing/ac.npy')
    bc = np.load(f'{cm.joint_folder_path}/pairing/bc.npy')
    assert len(ac) == n_c
    assert len(bc) == n_c

    a_right_inds = np.load(f'{a_cat}/test_match_indices.npy')
    b_right_inds = np.load(f'{b_cat}/test_match_indices.npy')

    for i in range(0, n_c):
        assert a_right_inds[i] in ac
        assert b_right_inds[i] in bc
        q = np.where(a_right_inds[i] == ac)[0][0]
        assert np.all([a_right_inds[i], b_right_inds[i]] == [ac[q], bc[q]])
