# Licensed under a 3-clause BSD style license - see LICENSE
'''
Tests for the "matching" module.
'''

import pytest
import os
import numpy as np

from ..matching import CrossMatch


def test_crossmatch_input():
    with pytest.raises(FileNotFoundError):
        cm = CrossMatch('./file.txt')

    cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data/metadata.txt'))
    assert cm.run_auf is False
    assert cm.run_group is False
    assert cm.run_cf is True
    assert cm.run_star is True

    f = open(os.path.join(os.path.dirname(__file__), 'data/metadata.txt')).readlines()
    for old_line, new_line in zip(['run_cf = yes', 'run_auf = no', 'run_auf = no'],
                                  ['', 'run_auf = aye\n', 'run_auf = yes\n']):
        miss_key_idx = np.where([old_line in line for line in f])[0][0]
        CrossMatch._replace_line(cm, os.path.join(os.path.dirname(__file__), 'data/metadata.txt'),
                                 miss_key_idx, new_line,
                                 out_file=os.path.join(os.path.dirname(__file__),
                                                       'data/metadata_.txt'))

        with pytest.raises(ValueError):
            cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data/metadata_.txt'))
