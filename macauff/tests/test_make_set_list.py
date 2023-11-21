# Licensed under a 3-clause BSD style license - see LICENSE
'''
Tests for the "make_set_list" module.
'''

import os
import pytest
import numpy as np

from ..make_set_list import _initial_group_numbering, set_list


def test_initial_group_numbering():
    a_num = np.array([1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 0, 0, 0, 1, 1])
    b_num = np.array([2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 0, 0, 1, 1])
    a_overlaps = -1*np.ones((2, 20), int)
    for _i, _inds in enumerate([[0], [1], [2], [3, 18], [4], [5], [6], [7], [8], [9], [10],
                               [11], [12, 17], [13], [14], [], [], [], [0], [12]]):
        # Unlike in test_group_sources and TestOverlap, these should corretctly
        # be zero-indexed since we convert back from one-indexing during the
        # larger assigning of overlap indices from get_overlap_indices.
        a_overlaps[:len(_inds), _i] = np.array(_inds)
    b_overlaps = -1*np.ones((2, 19), int)
    for _i, _inds in enumerate([[0, 18], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10],
                               [11], [12, 19], [13], [14], [], [], [12], [3]]):
        b_overlaps[:len(_inds), _i] = np.array(_inds)
    os.makedirs('./group', exist_ok=True)
    agroup, bgroup = _initial_group_numbering(a_overlaps, b_overlaps, a_num, b_num, '.')

    assert np.all(agroup == np.array(
        [18, 1, 2, 19, 3, 4, 5, 6, 7, 8, 9, 10, 20, 11, 12, 13, 14, 15, 18, 20]))
    assert np.all(bgroup == np.array(
        [18, 1, 2, 19, 3, 4, 5, 6, 7, 8, 9, 10, 20, 11, 12, 16, 17, 20, 19]))


def test_set_list_maximum_exceeded():
    os.makedirs('./group', exist_ok=True)
    for i, (N_a, N_b) in enumerate(zip([21, 10, 7], [5, 10, 6])):
        a_overlaps = np.empty((N_b, N_a+2), int)
        a_overlaps[:, :-2] = np.arange(N_b).reshape(-1, 1)
        a_overlaps[:, -2:] = -1
        a_overlaps[0, -2] = N_b
        b_overlaps = np.empty((N_a, N_b+2), int)
        b_overlaps[:, :-1] = np.arange(N_a).reshape(-1, 1)
        b_overlaps[:, -2:] = -1
        b_overlaps[0, -2] = N_a

        a_num = np.append(np.array([N_b]*N_a), [1, 0])
        b_num = np.append(np.array([N_a]*N_b), [1, 0])

        if i != 2:
            with pytest.warns(UserWarning, match='1 island, containing {}/{} catalogue a and '
                              '{}/{} catalogue b stars'.format(N_a, N_a+2, N_b, N_b+2)):
                alist, blist, agrplen, bgrplen, areject, breject = set_list(
                    a_overlaps, b_overlaps, a_num, b_num, '.', 2)
        else:
            with pytest.warns(None) as record:
                alist, blist, agrplen, bgrplen = set_list(
                    a_overlaps, b_overlaps, a_num, b_num, '.', 2)
            # Should be empty if no warnings were raised.
            assert not record
        if i != 2:
            assert np.all(agrplen == np.array([1, 1, 0]))
            assert np.all(bgrplen == np.array([1, 0, 1]))
            assert np.all(alist == np.array([[N_a, N_a+1, -1]]))
            assert np.all(blist == np.array([[N_b, -1, N_b+1]]))
        else:
            assert np.all(agrplen == np.array([1, 1, 0, 7]))
            assert np.all(bgrplen == np.array([1, 0, 1, 6]))
            # Here we can assume a hard-coded N_a=7, N_b=6.
            assert np.all(alist == np.array(
                [[N_a, N_a+1, -1, 0], [-1, -1, -1, 1], [-1, -1, -1, 2], [-1, -1, -1, 3],
                 [-1, -1, -1, 4], [-1, -1, -1, 5], [-1, -1, -1, 6]]))
            assert np.all(blist == np.array(
                [[N_b, -1, N_b+1, 0], [-1, -1, -1, 1], [-1, -1, -1, 2], [-1, -1, -1, 3],
                 [-1, -1, -1, 4], [-1, -1, -1, 5]]))
