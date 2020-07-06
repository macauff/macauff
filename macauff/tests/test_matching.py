# Licensed under a 3-clause BSD style license - see LICENSE
'''
Tests for the "matching" module.
'''

import pytest
import os

from ..matching import CrossMatch


def test_crossmatch_input():
    with pytest.raises(FileNotFoundError):
        cm = CrossMatch('./file.txt')

    cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data/metadata.txt'))
    assert cm.run_auf is False
    assert cm.run_group is False
    assert cm.run_cf is True
    assert cm.run_star is True

    with pytest.raises(ValueError):
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data/metadata_missing_key.txt'))

    with pytest.raises(ValueError):
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data/metadata_bad_type.txt'))

    with pytest.raises(ValueError):
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data/metadata_bad_order.txt'))
