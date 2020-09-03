# Licensed under a 3-clause BSD style license - see LICENSE
'''
This module provides the high-level framework for performing catalogue-catalogue cross-matches.
'''

import numpy as np

__all__ = ['create_perturb_auf']


def create_perturb_auf(auf_folder, filters, auf_points, psf_fwhms, tri_download_flag, ax_lims,
                       r, dr, rho, drho, which_cat):
    """
    Function to perform the creation of the blended object perturbation component
    of the AUF.

    auf_folder : string
        The overall folder into which to create filter-pointing folders and save
        individual simulation files.
    filters : list of strings or numpy.ndarray of strings
        An array containing the list of filters in this catalogue to create
        simulated AUF components for.
    auf_points : numpy.ndarray
        Two-dimensional array containing pairs of coordinates at which to evaluate
        the perturbation AUF components.
    psf_fwhms : numpy.ndarray
        Array of full width at half-maximums for each filter in ``filters``.
    tri_download_flag : boolean
        A ``True``/``False`` flag, whether to re-download TRILEGAL simulated star
        counts or not if a simulation already exists in a given folder.
    ax_lims : numpy.ndarray
        Array containing the four sky coordinate limits of the cross-match region.
    r : numpy.ndarray
        The real-space coordinates for the Hankel transformations used in AUF-AUF
        convolution.
    dr : numpy.ndarray
        The spacings between ``r`` elements.
    rho : numpy.ndarray
        The fourier-space coordinates for Hankel transformations.
    drho : numpy.ndarray
        The spacings between ``rho`` elements.
    """
