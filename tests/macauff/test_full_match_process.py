# Licensed under a 3-clause BSD style license - see LICENSE
'''
Tests of full, end-to-end runs of the crossmatch process.
'''

import os

import numpy as np
import pytest
from test_utils import mock_filename
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

    with open(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.yaml'),
              encoding='utf-8') as cm_p:
        cm_p_text = cm_p.read()
    with open(os.path.join(os.path.dirname(__file__), 'data/cat_a_params.yaml'),
              encoding='utf-8') as ca_p:
        ca_p_text = ca_p.read()
    with open(os.path.join(os.path.dirname(__file__), 'data/cat_b_params.yaml'),
              encoding='utf-8') as cb_p:
        cb_p_text = cb_p.read()
    cm_p_ = cm_p_text.replace('pos_corr_dist: 11', f'pos_corr_dist: {r:.2f}')

    if shape == 'rectangle':
        new_region_points = f'[{x}, {x}, 1, {y}, {y}, 1]'
    else:
        new_region_points = f'[{x-radius/2:.2f}, {x+radius/2:.2f}, 2, {y}, {y}, 1]'

    for ol, nl in zip([r'joint_folder_path: test_path_{}', '  - [131, 134, 4, -1, 1, 3]'],
                      [r'joint_folder_path: new_test_path_{}', f'  - {new_region_points}']):
        cm_p_ = cm_p_.replace(ol, nl)
    if shape == 'circle':
        wowp = "True" if x == 131 else "False"
        for ol, nl in zip(['include_phot_like: False', 'use_phot_priors: False'],
                          [f'include_phot_like: True\nwith_and_without_photometry: {wowp}',
                           'use_phot_priors: True']):
            cm_p_ = cm_p_.replace(ol, nl)

    ca_p_ = ca_p_text.replace('auf_region_points: [131, 134, 4, -1, 1, 3]',
                              f'auf_region_points: {new_region_points}')
    cb_p_ = cb_p_text.replace('auf_region_points: [131, 134, 4, -1, 1, 4]',
                              f'auf_region_points: {new_region_points}')
    ca_p_ = ca_p_.replace(r'cat_folder_path: gaia_folder_{}', 'cat_folder_path: a_cat')
    cb_p_ = cb_p_.replace(r'cat_folder_path: wise_folder_{}', 'cat_folder_path: b_cat')

    os.makedirs('new_test_path_9', exist_ok=True)
    cm = CrossMatch(mock_filename(cm_p_.encode("utf-8")), mock_filename(ca_p_.encode("utf-8")),
                    mock_filename(cb_p_.encode("utf-8")))
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
