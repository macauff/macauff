# Licensed under a 3-clause BSD style license - see LICENSE
'''
This module provides the framework to handle the creation of the perturbation
component of the astrometric uncertainty function.
'''

import os
import sys
import numpy as np

from .misc_functions_fortran import misc_functions_fortran as mff

__all__ = ['make_perturb_aufs']


def make_perturb_aufs(auf_folder, cat_folder, filters, auf_points, ax_lims, r, dr, rho,
                      drho, which_cat, include_perturb_auf, n_sources, mem_chunk_num,
                      psf_fwhms=None, tri_download_flag=False):
    """
    Function to perform the creation of the blended object perturbation component
    of the AUF.

    Parameters
    ----------
    auf_folder : string
        The overall folder into which to create filter-pointing folders and save
        individual simulation files.
    cat_folder : string
        The folder that the photometric catalogue being simulated for perturbation
        AUF component is stored in.
    filters : list of strings or numpy.ndarray of strings
        An array containing the list of filters in this catalogue to create
        simulated AUF components for.
    auf_points : numpy.ndarray
        Two-dimensional array containing pairs of coordinates at which to evaluate
        the perturbation AUF components.
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
    which_cat : string
        Indicator as to whether these perturbation AUFs are for catalogue "a"
        or catalogue "b" within the cross-match process.
    include_perturb_auf : boolean
        ``True`` or ``False`` flag indicating whether perturbation component of the
        AUF should be used or not within the cross-match process.
    n_sources : int
        Number of sources in the main catalogue to be simulated.
    mem_chunk_num : int
        Number of individual sub-sections to break catalogue into for memory
        saving purposes.
    psf_fwhms : numpy.ndarray, optional
        Array of full width at half-maximums for each filter in ``filters``. Only
        required if ``include_perturb_auf`` is True; defaults to ``None``.
    tri_download_flag : boolean, optional
        A ``True``/``False`` flag, whether to re-download TRILEGAL simulated star
        counts or not if a simulation already exists in a given folder. Only
        needed if ``include_perturb_auf`` is True.
    """
    print('Creating empirical crowding AUFs for catalogue "{}"...'.format(which_cat))
    sys.stdout.flush()

    # TODO: assign these variables as input args.
    # The number of simulated PSFs to create for statistical purposes
    N_trials = 1000000
    # Magnitude offsets corresponding to relative fluxes of perturbing sources; here
    # dm of 2.5 is 10% relative flux and dm = 5 corresponds to 1% relative flux. Used
    # to inform the fraction of simulations with a contaminant above these relative
    # fluxes.
    delta_mag_cuts = np.array([2.5, 5])

    # Store the length of the density-magnitude combinations in each sky/filter
    # combination for future loading purposes.
    arraylengths = np.lib.format.open_memmap('{}/arraylengths.npy'.format(auf_folder), mode='w+',
                                             dtype=int, shape=(len(filters), len(auf_points)),
                                             fortran_order=True)

    for i in range(len(auf_points)):
        ax1, ax2 = auf_points[i]
        ax_folder = '{}/{}/{}'.format(auf_folder, ax1, ax2)
        if not os.path.exists(ax_folder):
            os.makedirs(ax_folder, exist_ok=True)
        if include_perturb_auf:
            # TODO: download TRILEGAL simulation here
            raise NotImplementedError("Perturbation AUF components are not currently "
                                      "included in the cross-match process.")

        for j in range(len(filters)):
            filt = filters[j]

            filt_folder = '{}/{}'.format(ax_folder, filt)
            if not os.path.exists(filt_folder):
                os.makedirs(filt_folder, exist_ok=True)

            if include_perturb_auf:
                # TODO: extend with create_single_auf
                raise NotImplementedError("Perturbation AUF components are not currently "
                                          "included in the cross-match process.")
            else:
                # Without the simulations to force local normalising density N or
                # individual source brightness magnitudes, we can simply combine
                # all data into a single "bin".
                num_N_mag = 1
                # In cases where we do not want to use the perturbation AUF component,
                # we currently don't have separate functions, but instead set up dummy
                # functions and variables to pass what mathematically amounts to
                # "nothing" through the cross-match. Here we would use fortran
                # subroutines to create the perturbation simulations, so we make
                # f-ordered dummy parameters.
                Frac = np.zeros((len(delta_mag_cuts), num_N_mag), float, order='F')
                np.save('{}/frac.npy'.format(filt_folder), Frac)
                Flux = np.zeros(num_N_mag, float, order='F')
                np.save('{}/flux.npy'.format(filt_folder), Flux)
                # Remember that r is bins, so the evaluations at bin middle are one
                # shorter in length.
                offset = np.zeros((len(r)-1, num_N_mag), float, order='F')
                # Fix offsets such that the probability density function looks like
                # a delta function, such that a two-dimensional circular coordinate
                # integral would evaluate to one at every point, cf. ``cumulative``.
                offset[0, :] = 1 / (2 * np.pi * (r[0] + dr[0]/2) * dr[0])
                np.save('{}/offset.npy'.format(filt_folder), offset)
                # The cumulative integral of a delta function is always unity.
                cumulative = np.ones((len(r)-1, num_N_mag), float, order='F')
                np.save('{}/cumulative.npy'.format(filt_folder), cumulative)
                # The Hankel transform of a delta function is a flat line; this
                # then preserves the convolution being multiplication in fourier
                # space, as F(x) x 1 = F(x), similar to how f(x) * d(0) = f(x).
                fourieroffset = np.ones((len(rho)-1, num_N_mag), float, order='F')
                np.save('{}/fourier.npy'.format(filt_folder), fourieroffset)
                # Both normalising density and magnitude arrays can be proxied
                # with a dummy parameter, as any minimisation of N-m distance
                # must pick the single value anyway.
                Narray = np.array([[1]], float)
                np.save('{}/N.npy'.format(filt_folder), Narray)
                magarray = np.array([[1]], float)
                np.save('{}/mag.npy'.format(filt_folder), magarray)
            arraylengths[j, i] = len(Narray)

    # Once the individual AUF simulations are saved, we also need to calculate
    # the indices each source references when slicing into the 4-D cubes
    # created by [1-D array] x N-m combination x filter x sky position iteration.

    print('Creating crowding AUFs indices for catalogue "{}"...'.format(which_cat))
    sys.stdout.flush()

    modelrefinds = np.lib.format.open_memmap('{}/modelrefinds.npy'.format(auf_folder),
                                             mode='w+', dtype=int, shape=(3, n_sources),
                                             fortran_order=True)

    for cnum in range(0, mem_chunk_num):
        lowind = np.floor(n_sources*cnum/mem_chunk_num).astype(int)
        highind = np.floor(n_sources*(cnum+1)/mem_chunk_num).astype(int)
        a = np.load('{}/con_cat_astro.npy'.format(cat_folder), mmap_mode='r')[lowind:highind]
        magref = np.load('{}/magref.npy'.format(cat_folder), mmap_mode='r')[lowind:highind]
        # As we chunk in even steps through the files this is simple for now,
        # but could be replaced with a more complex mapping in the future.
        indexmap = np.arange(lowind, highind, 1)

        if include_perturb_auf:
            # TODO: load 3-D cube of N-m combinations for unique sky/filter pairs.
            raise NotImplementedError("Perturbation AUF components are not currently "
                                      "included in the cross-match process.")
        else:
            # For the case that we do not use the perturbation AUF component,
            # our dummy N-m files are all one-length arrays, so we can
            # trivially index them, regardless of specifics.
            modelrefinds[0, indexmap] = 0

        # The mapping of which filter to use is straightforward: simply pick
        # the filter index of the "best" filter for each source, from magref.
        modelrefinds[1, indexmap] = magref

        # Which sky position to use is more complex; this involves determining
        # the smallest great-circle distance to each auf_point AUF mapping for
        # each source.
        modelrefinds[2, indexmap] = mff.find_nearest_point(a[:, 0], a[:, 1],
                                                           auf_points[:, 0], auf_points[:, 1])
