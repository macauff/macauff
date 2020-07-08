# Licensed under a 3-clause BSD style license - see LICENSE
'''
Tests for the "matching" module.
'''

import pytest
import os
from numpy.testing import assert_almost_equal
import numpy as np

from ..matching import CrossMatch


def test_crossmatch_run_input():
    with pytest.raises(FileNotFoundError):
        cm = CrossMatch('./file.txt')

    cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data/metadata.txt'))
    assert cm.run_auf is False
    assert cm.run_group is False
    assert cm.run_cf is True
    assert cm.run_star is True

    # List of simple one line config file replacements for error message checking
    f = open(os.path.join(os.path.dirname(__file__), 'data/metadata.txt')).readlines()
    for old_line, new_line, match_text in zip(['run_cf = yes', 'run_auf = no', 'run_auf = no'],
                                              ['', 'run_auf = aye\n', 'run_auf = yes\n'],
                                              ['Missing key', 'Boolean flag key not set',
                                               'Inconsistency between run/no run']):
        idx = np.where([old_line in line for line in f])[0][0]
        CrossMatch._replace_line(cm, os.path.join(os.path.dirname(__file__), 'data/metadata.txt'),
                                 idx, new_line, out_file=os.path.join(os.path.dirname(__file__),
                                                                      'data/metadata_.txt'))

        with pytest.raises(ValueError, match=match_text):
            cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data/metadata_.txt'))


def test_crossmatch_auf_input():
    cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data/metadata.txt'))
    assert cm.auf_region_frame == 'equatorial'
    assert_almost_equal(cm.auf_region_points,
                        np.array([[131, -1], [132, -1], [133, -1], [134, -1],
                                  [131, 0], [132, 0], [133, 0], [134, 0],
                                  [131, 1], [132, 1], [133, 1], [134, 1]]))

    # List of simple one line config file replacements for error message checking
    f = open(os.path.join(os.path.dirname(__file__), 'data/metadata.txt')).readlines()
    for old_line, new_line, match_text in zip(
        ['auf_region_type = rectangle', 'auf_region_type = rectangle',
         'auf_region_points = 131 134 4 -1 1 3', 'auf_region_points = 131 134 4 -1 1 3',
         'auf_region_frame = equatorial'],
        ['', 'auf_region_type = triangle\n', 'auf_region_points = 131 134 4 -1 1 a\n',
         'auf_region_points = 131 134 4 -1 1\n', 'auf_region_frame = ecliptic\n'],
        ['Missing key', "should either be 'rectangle' or", 'should be 6 numbers',
         'should be 6 numbers', "should either be 'equatorial' or"]):
        idx = np.where([old_line in line for line in f])[0][0]
        CrossMatch._replace_line(cm, os.path.join(os.path.dirname(__file__), 'data/metadata.txt'),
                                 idx, new_line, out_file=os.path.join(os.path.dirname(__file__),
                                                                      'data/metadata_.txt'))

        with pytest.raises(ValueError, match=match_text):
            cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data/metadata_.txt'))

    # Check correct and incorrect auf_region_points when auf_region_type is 'points'
    idx = np.where(['auf_region_type = rectangle' in line for line in f])[0][0]
    CrossMatch._replace_line(cm, os.path.join(os.path.dirname(__file__), 'data/metadata.txt'),
                             idx, 'auf_region_type = points\n',
                             out_file=os.path.join(os.path.dirname(__file__),
                                                   'data/metadata_.txt'))

    idx = np.where(['auf_region_points = 131 134 4 -1 1 3' in line for line in f])[0][0]
    CrossMatch._replace_line(cm, os.path.join(os.path.dirname(__file__), 'data/metadata_.txt'),
                             idx, 'auf_region_points = (131, 0), (133, 0), (132, -1)\n',
                             out_file=os.path.join(os.path.dirname(__file__),
                                                   'data/metadata_2.txt'))

    cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data/metadata_2.txt'))
    assert_almost_equal(cm.auf_region_points, np.array([[131, 0], [133, 0], [132, -1]]))

    old_line = 'auf_region_points = 131 134 4 -1 1 3'
    for new_line in ['auf_region_points = (131, 0), (131, )\n',
                     'auf_region_points = (131, 0), (131, 1, 2)\n',
                     'auf_region_points = (131, 0), (131, a)\n']:
        idx = np.where([old_line in line for line in f])[0][0]
        CrossMatch._replace_line(cm, os.path.join(os.path.dirname(__file__), 'data/metadata_.txt'),
                                 idx, new_line, out_file=os.path.join(os.path.dirname(__file__),
                                                                      'data/metadata_2.txt'))

        with pytest.raises(ValueError):
            cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data/metadata_2.txt'))

    # Check galactic run is also fine
    idx = np.where(['auf_region_frame = equatorial' in line for line in f])[0][0]
    CrossMatch._replace_line(cm, os.path.join(os.path.dirname(__file__), 'data/metadata.txt'),
                             idx, 'auf_region_frame = galactic\n',
                             out_file=os.path.join(os.path.dirname(__file__),
                                                   'data/metadata_.txt'))

    cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data/metadata_.txt'))

    assert cm.auf_region_frame == 'galactic'
    assert_almost_equal(cm.auf_region_points,
                        np.array([[131, -1], [132, -1], [133, -1], [134, -1],
                                  [131, 0], [132, 0], [133, 0], [134, 0],
                                  [131, 1], [132, 1], [133, 1], [134, 1]]))

    # Check single-length point grids are fine
    idx = np.where(['auf_region_points = 131 134 4 -1 1 3' in line for line in f])[0][0]
    CrossMatch._replace_line(cm, os.path.join(os.path.dirname(__file__), 'data/metadata.txt'),
                             idx, 'auf_region_points = 131 131 1 0 0 1\n',
                             out_file=os.path.join(os.path.dirname(__file__),
                                                   'data/metadata_.txt'))

    cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data/metadata_.txt'))

    assert_almost_equal(cm.auf_region_points, np.array([[131, 0]]))

    idx = np.where(['auf_region_type = rectangle' in line for line in f])[0][0]
    CrossMatch._replace_line(cm, os.path.join(os.path.dirname(__file__), 'data/metadata.txt'),
                             idx, 'auf_region_type = points\n',
                             out_file=os.path.join(os.path.dirname(__file__),
                                                   'data/metadata_.txt'))

    idx = np.where(['auf_region_points = 131 134 4 -1 1 3' in line for line in f])[0][0]
    CrossMatch._replace_line(cm, os.path.join(os.path.dirname(__file__), 'data/metadata_.txt'),
                             idx, 'auf_region_points = (131, 0)\n',
                             out_file=os.path.join(os.path.dirname(__file__),
                                                   'data/metadata_2.txt'))

    cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data/metadata_2.txt'))
    assert_almost_equal(cm.auf_region_points, np.array([[131, 0]]))
