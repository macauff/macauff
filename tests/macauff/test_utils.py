# Licensed under a 3-clause BSD style license - see LICENSE
'''
Utility functions for test modules.
'''

import os


def mock_filename(content: bytes) -> int:
    r, w = os.pipe()
    with open(w, "wb") as wf:
        wf.write(content)
    return r
