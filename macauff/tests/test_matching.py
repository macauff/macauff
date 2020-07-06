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
    miss_key_idx = np.where(['run_cf = yes' in line for line in f])[0][0]
    CrossMatch._replace_line(cm, os.path.join(os.path.dirname(__file__), 'data/metadata.txt'),
                             miss_key_idx, '',
                             out_file=os.path.join(os.path.dirname(__file__),
                                                   'data/metadata_missing_key.txt'))

    with pytest.raises(ValueError):
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data/metadata_missing_key.txt'))

    bad_type_idx = np.where(['run_auf = no' in line for line in f])[0][0]
    CrossMatch._replace_line(cm, os.path.join(os.path.dirname(__file__), 'data/metadata.txt'),
                             bad_type_idx, 'run_auf = aye\n',
                             out_file=os.path.join(os.path.dirname(__file__),
                                                   'data/metadata_bad_type.txt'))

    with pytest.raises(ValueError):
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data/metadata_bad_type.txt'))

    bad_order_idx = np.where(['run_auf = no' in line for line in f])[0][0]
    CrossMatch._replace_line(cm, os.path.join(os.path.dirname(__file__), 'data/metadata.txt'),
                             bad_order_idx, 'run_auf = yes\n',
                             out_file=os.path.join(os.path.dirname(__file__),
                                                   'data/metadata_bad_order.txt'))

    with pytest.raises(ValueError):
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data/metadata_bad_order.txt'))
