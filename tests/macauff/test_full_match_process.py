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


@pytest.mark.parametrize("shape", ['circle', 'rectangle'])
@pytest.mark.parametrize("x,y", [(131, 0), (0, 0)])
# pylint: disable-next=too-many-locals,too-many-statements,too-many-branches
def test_naive_bayes_match(shape, x, y):
    # Generate a small number of sources randomly, then run through the
    # cross-match process.
    if shape == 'rectangle':
        n_a, n_b, n_c = 40, 50, 35
    else:
        n_a, n_b, n_c = 400, 500, 370
    n_a_filts, n_b_filts = 3, 4
    a_astro_sig, b_astro_sig = 0.3, 0.5
    r = 5 * np.sqrt(a_astro_sig**2 + b_astro_sig**2)
    dx = np.sqrt(n_b * np.pi * r**2)/3600
    if shape == 'circle':
        dx *= 3
    extent = np.array([x-1.3*dx/2-r/3600, x+1.3*dx/2+r/3600, y-1.3*dx/2-r/3600, y+1.3*dx/2+r/3600])
    if shape == 'rectangle':
        forced_extent = np.array([x-0.06, x+0.06, y-0.06, y+0.06])
    else:
        t = np.linspace(0, 2*np.pi, 31)
        radius = 0.5 * (extent[3] - extent[2]) * 1.1
        mid_lon, mid_lat = 0.5 * (extent[0] + extent[1]), 0.5 * (extent[2] + extent[3])
        forced_extent = np.array([radius * np.cos(t) + mid_lon, radius * np.sin(t) + mid_lat]).T

    a_cat, b_cat = 'a_cat', 'b_cat'

    generate_random_data(n_a, n_b, n_c, extent + np.array([1.1*r/3600, -1.1*r/3600, 1.1*r/3600, -1.1*r/3600]),
                         n_a_filts, n_b_filts, a_astro_sig, b_astro_sig, a_cat, b_cat, shape=shape, seed=9999)
    a_astro = np.load(f"{a_cat}/con_cat_astro.npy")
    a_mp = np.load(f"{a_cat}/test_match_indices.npy")
    lonely_counter = 0
    for i in range(len(a_astro)):
        if i not in a_mp:
            if shape == 'rectangle':
                if lonely_counter == 0:
                    a_astro[i, [0, 1]] = [forced_extent[0], forced_extent[2]]
                if lonely_counter == 1:
                    a_astro[i, [0, 1]] = [forced_extent[1], forced_extent[2]]
                if lonely_counter == 2:
                    a_astro[i, [0, 1]] = [forced_extent[0], forced_extent[3]]
                if lonely_counter == 3:
                    a_astro[i, [0, 1]] = [forced_extent[1], forced_extent[3]]
                if lonely_counter == 4:
                    break
            else:
                a_astro[i, [0, 1]] = forced_extent[lonely_counter]
                if lonely_counter == len(t)-1:
                    break
            lonely_counter += 1
    b_astro = np.load(f"{b_cat}/con_cat_astro.npy")
    b_mp = np.load(f"{b_cat}/test_match_indices.npy")
    lonely_counter = 0
    for i in range(len(b_astro)):
        if i not in b_mp:
            if shape == 'rectangle':
                if lonely_counter == 0:
                    b_astro[i, [0, 1]] = [forced_extent[0], forced_extent[2]]
                if lonely_counter == 1:
                    b_astro[i, [0, 1]] = [forced_extent[1], forced_extent[2]]
                if lonely_counter == 2:
                    b_astro[i, [0, 1]] = [forced_extent[0], forced_extent[3]]
                if lonely_counter == 3:
                    b_astro[i, [0, 1]] = [forced_extent[1], forced_extent[3]]
                if lonely_counter == 4:
                    break
            else:
                b_astro[i, [0, 1]] = forced_extent[lonely_counter]
                if lonely_counter == len(t)-1:
                    break
            lonely_counter += 1
    np.save(f"{a_cat}/con_cat_astro.npy", a_astro)
    np.save(f"{b_cat}/con_cat_astro.npy", b_astro)

    if shape == 'circle':
        n_a, n_b, n_c = len(a_astro), len(b_astro), len(a_mp)

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

    if shape == 'rectangle':
        new_region_points = f'{x} {x} 1 {y} {y} 1'
    else:
        new_region_points = f'{x-radius/2:.2f} {x+radius/2:.2f} 2 {y} {y} 1'

    for ol, nl in zip(['joint_folder_path = test_path', 'cf_region_points = 131 134 4 -1 1 3'],
                      ['joint_folder_path = new_test_path\n', f'cf_region_points = {new_region_points}\n']):
        with open(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        idx = np.where([ol in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/chunk0/crossmatch_params_.txt'),
                      idx, nl)
    if shape == 'circle':
        wowp = "yes" if x == 131 else "no"
        for ol, nl in zip(['include_phot_like = no', 'use_phot_priors = no'],
                          [f'include_phot_like = yes\nwith_and_without_photometry = {wowp}\n',
                           'use_phot_priors = yes\n']):
            with open(os.path.join(os.path.dirname(__file__), 'data/chunk0/crossmatch_params_.txt'),
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

    os.makedirs('new_test_path', exist_ok=True)
    cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'))
    cm()

    ac = np.load(f'{cm.joint_folder_path}/ac.npy')
    bc = np.load(f'{cm.joint_folder_path}/bc.npy')
    assert len(ac) == n_c
    assert len(bc) == n_c

    a_right_inds = np.load(f'{a_cat}/test_match_indices.npy')
    b_right_inds = np.load(f'{b_cat}/test_match_indices.npy')

    for i in range(0, n_c):
        assert a_right_inds[i] in ac
        assert b_right_inds[i] in bc
        q = np.where(a_right_inds[i] == ac)[0][0]
        assert np.all([a_right_inds[i], b_right_inds[i]] == [ac[q], bc[q]])

    if shape == 'circle' and x == 131:
        ac = np.load(f'{cm.joint_folder_path}/ac_without_photometry.npy')
        bc = np.load(f'{cm.joint_folder_path}/bc_without_photometry.npy')
        assert len(ac) == n_c
        assert len(bc) == n_c
        for i in range(0, n_c):
            assert a_right_inds[i] in ac
            assert b_right_inds[i] in bc
            q = np.where(a_right_inds[i] == ac)[0][0]
            assert np.all([a_right_inds[i], b_right_inds[i]] == [ac[q], bc[q]])

    os.system('rm -r new_test_path')
