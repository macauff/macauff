# Licensed under a 3-clause BSD style license - see LICENSE
'''
Tests for the "matching" module.
'''

import pytest

from ..matching import CrossMatch


def test_crossmatch_input():
    with pytest.raises(FileNotFoundError):
        cm = CrossMatch('./file.txt')
