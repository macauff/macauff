# Licensed under a 3-clause BSD style license - see LICENSE
'''
Tests for the "derive_psf_auf_params" module.
'''

import os

import numpy as np
from numpy.testing import assert_allclose

from macauff.derive_psf_auf_params import FitPSFPerturbations


def test_derive_psf_auf_params():
    data_save_folder = 'auf_data'
    a = FitPSFPerturbations(psf_fwhm=6.1, d_di=0.1, d_Li=0.1, n_pool=1,
                            data_save_folder=data_save_folder, plot_save_folder='auf_plot')
    a(run_initial_Ld=True, run_skew_fit=True, run_polynomial_fit=True, make_fit_plots=True,
      draw_sim_num=100)

    assert os.path.isfile('auf_plot/dd_params_visualisation.pdf')
    psf_sig = 6.1 / (2 * np.sqrt(2 * np.log(2)))
    dd = np.load('{}/dd_Ld.npy'.format(data_save_folder))
    x_div_sig = dd[0, :, 1] / psf_sig
    y_div_sig = dd[0, :, 0] / psf_sig
    # At small enough relative perturber fluxes, dx ~ f x exp(-0.25 d^2/sig^2)
    assert_allclose(y_div_sig, 0.15 * x_div_sig * np.exp(-0.25 * x_div_sig**2),
                    rtol=0.11, atol=6e-4)

    Ns = np.load('{}/Ns.npy'.format(data_save_folder))
    dd_ic = np.load('{}/dd_ic.npy'.format(data_save_folder))
    l_cut = np.load('{}/l_cut.npy'.format(data_save_folder))
    dd_params_full = np.load('{}/dd_params_full.npy'.format(data_save_folder))
    N_ind = np.unravel_index(np.argmin(dd_ic[:, 0] + 2*Ns), dd_ic[:, 1].shape)[0]
    ddparams = a.return_ddparams(0.15, l_cut, dd_params_full, Ns[N_ind], N_ind)
    new_ydivsig = a.dd_combined_fit(ddparams, x_div_sig, 0.15, l_cut[2])
    assert_allclose(y_div_sig, new_ydivsig, rtol=0.11, atol=6e-4)
