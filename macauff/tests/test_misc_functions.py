# Licensed under a 3-clause BSD style license - see LICENSE
'''
Tests for the "misc_functions" module.
'''

import numpy as np
from numpy.testing import assert_allclose
import scipy.special

from ..misc_functions_fortran import misc_functions_fortran as mff


def test_calc_j0():
    r = np.linspace(0, 5, 5000)
    rho = np.linspace(0, 100, 5000)
    j0 = mff.calc_j0(r, rho)
    assert_allclose(j0, scipy.special.j0(2 * np.pi * r.reshape(-1, 1) * rho.reshape(1, -1)),
                    rtol=1e-5)
