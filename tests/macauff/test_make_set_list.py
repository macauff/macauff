# Licensed under a 3-clause BSD style license - see LICENSE
'''
Tests for the "make_set_list" module.
'''

import os
import warnings

import numpy as np
import pytest

from macauff.make_set_list import _initial_group_numbering, set_list


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
    agroup, bgroup = _initial_group_numbering(a_overlaps, b_overlaps, a_num, b_num)

    assert np.all(agroup == np.array(
        [18, 1, 2, 19, 3, 4, 5, 6, 7, 8, 9, 10, 20, 11, 12, 13, 14, 15, 18, 20]))
    assert np.all(bgroup == np.array(
        [18, 1, 2, 19, 3, 4, 5, 6, 7, 8, 9, 10, 20, 11, 12, 16, 17, 20, 19]))


def test_initial_group_numbering_catch_recursion():
    a_num = np.array([2] * 600)
    b_num = np.array([2] * 600)
    a_overlaps = np.empty((2, 600), int)
    a_overlaps[0] = np.arange(600, dtype=int)
    a_overlaps[1] = np.arange(600, dtype=int)
    b_overlaps = np.empty((2, 600), int)
    b_overlaps[0] = np.arange(1, 601, dtype=int)
    b_overlaps[1] = np.arange(1, 601, dtype=int)
    b_overlaps = np.arange(1, 601, dtype=int).reshape(1, -1)
    b_overlaps[:, -1] = -1

    agroup, bgroup, nfl = _initial_group_numbering(a_overlaps, b_overlaps, a_num, b_num)

    fake_agroup = np.ones(600, int)
    fake_agroup[:116] = 0
    fake_bgroup = np.ones(600, int)
    fake_bgroup[:116] = 0
    assert np.all(agroup == fake_agroup)
    assert np.all(bgroup == fake_bgroup)
    assert np.all(nfl[0] == np.arange(0, 116))
    assert np.all(nfl[1] == np.arange(0, 116))


def test_set_list_catch_recursion():
    a_num = np.array([2] * 600)
    b_num = np.array([2] * 600)
    a_overlaps = np.empty((2, 600), int)
    a_overlaps[0] = np.arange(600, dtype=int)
    a_overlaps[1] = np.arange(600, dtype=int)
    b_overlaps = np.empty((2, 600), int)
    b_overlaps[0] = np.arange(1, 601, dtype=int)
    b_overlaps[1] = np.arange(1, 601, dtype=int)
    b_overlaps[:, -1] = -1

    alist, blist, _, bgrplen, arej, brej = set_list(a_overlaps, b_overlaps, a_num, b_num, 2)
    assert len(alist) == 0
    assert len(bgrplen) == 0
    assert np.all(arej == np.concatenate((np.arange(117, 600), np.arange(117))))
    assert np.all(brej == np.concatenate((np.arange(117, 600), np.arange(117))))

    # Move one pointing so we get a single valid island at the end.
    x = np.copy(a_overlaps[:, 114])
    a_overlaps[:, 114:-1] = a_overlaps[:, 115:]
    a_overlaps[:, -1] = x
    b_overlaps[:, 113] = 599
    b_overlaps[:, 114:-1] = b_overlaps[:, 114:-1] - 1
    b_overlaps[:, -1] = 598

    alist, blist, _, bgrplen, arej, brej = set_list(a_overlaps, b_overlaps, a_num, b_num, 2)
    # Changing the order of sequences above fails all but the final six
    # objects in each catalogue now. So we have one valid island and
    # some rejections.
    assert np.all(alist == np.array([597, 598]).reshape(-1, 1))
    assert np.all(blist == np.array([597, 598, 599]).reshape(-1, 1))
    assert np.all(arej == np.append(np.arange(597), 599))
    assert np.all(brej == np.arange(597))


def test_set_list_maximum_exceeded():
    os.makedirs('./group', exist_ok=True)
    for i, (n_a, n_b) in enumerate(zip([21, 10, 7], [5, 10, 6])):
        a_overlaps = np.empty((n_b, n_a+2), int)
        a_overlaps[:, :-2] = np.arange(n_b).reshape(-1, 1)
        a_overlaps[:, -2:] = -1
        a_overlaps[0, -2] = n_b
        b_overlaps = np.empty((n_a, n_b+2), int)
        b_overlaps[:, :-1] = np.arange(n_a).reshape(-1, 1)
        b_overlaps[:, -2:] = -1
        b_overlaps[0, -2] = n_a

        a_num = np.append(np.array([n_b]*n_a), [1, 0])
        b_num = np.append(np.array([n_a]*n_b), [1, 0])

        if i != 2:
            with pytest.warns(UserWarning, match=f'1 island, containing {n_a}/{n_a+2} catalogue a and '
                              f'{n_b}/{n_b+2} catalogue b stars'):
                alist, blist, agrplen, bgrplen, _, _ = set_list(a_overlaps, b_overlaps, a_num, b_num, 2)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                # pylint: disable-next=unbalanced-tuple-unpacking
                alist, blist, agrplen, bgrplen = set_list(a_overlaps, b_overlaps, a_num, b_num, 2)
        if i != 2:
            assert np.all(agrplen == np.array([1, 1, 0]))
            assert np.all(bgrplen == np.array([1, 0, 1]))
            assert np.all(alist == np.array([[n_a, n_a+1, -1]]))
            assert np.all(blist == np.array([[n_b, -1, n_b+1]]))
        else:
            assert np.all(agrplen == np.array([1, 1, 0, 7]))
            assert np.all(bgrplen == np.array([1, 0, 1, 6]))
            # Here we can assume a hard-coded N_a=7, N_b=6.
            assert np.all(alist == np.array(
                [[n_a, n_a+1, -1, 0], [-1, -1, -1, 1], [-1, -1, -1, 2], [-1, -1, -1, 3],
                 [-1, -1, -1, 4], [-1, -1, -1, 5], [-1, -1, -1, 6]]))
            assert np.all(blist == np.array(
                [[n_b, -1, n_b+1, 0], [-1, -1, -1, 1], [-1, -1, -1, 2], [-1, -1, -1, 3],
                 [-1, -1, -1, 4], [-1, -1, -1, 5]]))
