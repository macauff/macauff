# Licensed under a 3-clause BSD style license - see LICENSE
'''
Tests for the "perturbation_auf" module.
'''

import pytest
import os
from numpy.testing import assert_almost_equal
import numpy as np

from ..matching import CrossMatch