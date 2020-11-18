# Licensed under a 3-clause BSD style license - see LICENSE
'''
This module provides miscellaneous scripts, called in other parts of the cross-match
framework.
'''

import numpy as np


def create_fourier_offsets_grid(auf_folder_path, auf_pointings, filt_names, rho):
    '''
    Minor function to offload the creation of a 4-D array from a series of 2-D
    arrays.

    Parameters
    ----------
    auf_folder_path : string
        Location of the top-level folder in which all fourier grids are saved.
    auf_pointings : numpy.ndarray
        Two-dimensional array with the sky coordinates of each pointing used
        in the perturbation AUF component creation.
    filt_names : list or numpy.ndarray
        List of ordered filters for the given catalogue.
    rho : numpy.ndarray
        Array of values used to create the fourier-space description of the
        perturbation AUFs in question that are to be loaded.
    '''
    arraylengths = np.load('{}/arraylengths.npy'.format(auf_folder_path))
    longestNm = np.amax(arraylengths)
    fouriergrid = np.lib.format.open_memmap('{}/fourier_grid.npy'.format(
        auf_folder_path), mode='w+', dtype=float, shape=(len(rho)-1, longestNm, len(filt_names),
                                                         len(auf_pointings)), fortran_order=True)
    fouriergrid[:, :, :, :] = -1
    for j in range(0, len(auf_pointings)):
        ax1, ax2 = auf_pointings[j]
        for i in range(0, len(filt_names)):
            filt = filt_names[i]
            fourier = np.load('{}/{}/{}/{}/fourier.npy'.format(auf_folder_path,
                              ax1, ax2, filt))
            fouriergrid[:, :arraylengths[i, j], i, j] = fourier
    del arraylengths, longestNm, fouriergrid


def load_small_ref_ind_fourier_grid(modrefind, auf_folder_path):
    '''
    Function to create reference index arrays out of a larger array, based on
    the mappings from the original reference index array into a larger grid,
    such that the corresponding cutout reference index now maps onto the smaller
    cutout 4-D array.

    Parameters
    ----------
    modrefind : numpy.ndarray
        The reference index array that maps into saved array ``fourier_grid``
        for each source in the given catalogue.
    auf_folder_path : string
        Location of the folder in which ``fourier_grid`` is stored.

    Returns
    -------
    fouriergrid : numpy.ndarray
        The small cutout of ``fourier_grid``, containing only the appropriate
        indices for AUF pointing, filter, etc.
    modrefindsmall : numpy.ndarray
        The corresponding mappings for each source onto ``fouriergrid``, such
        that each source still points to the correct entry that it did in
        ``fourier_grid``.
    '''
    nmuniqueind, nmnewind = np.unique(modrefind[0, :], return_inverse=True)
    filtuniqueind, filtnewind = np.unique(modrefind[1, :], return_inverse=True)
    axuniqueind, axnewind = np.unique(modrefind[2, :], return_inverse=True)
    fouriergrid = np.asfortranarray(np.load('{}/fourier_grid.npy'.format(
                                    auf_folder_path), mmap_mode='r')[:, :, :,
                                    axuniqueind][:, :, filtuniqueind, :][:,
                                    nmuniqueind, :, :])
    modrefindsmall = np.empty((3, modrefind.shape[1]), int, order='F')
    del modrefind
    modrefindsmall[0, :] = nmnewind
    modrefindsmall[1, :] = filtnewind
    modrefindsmall[2, :] = axnewind

    return fouriergrid, modrefindsmall


def hav_dist_constant_lat(x_lon, x_lat, lon):
    '''
    Computes the Haversine formula in the limit that sky separation is only
    determined by longitudinal separation (i.e., delta-lat is zero).

    Parameters
    ----------
    x_lon : float
        Sky coordinate of the source in question, in degrees.
    x_lat : float
        Orthogonal sky coordinate of the source, in degrees.
    lon : float
        Longitudinal sky coordinate to calculate the "horizontal" sky separation
        of the source to.

    Returns
    -------
    dist : float
        Horizontal sky separation between source and given ``lon``, in degrees.
    '''

    dist = np.degrees(2 * np.arcsin(np.abs(np.cos(np.radians(x_lat)) *
                                           np.sin(np.radians((x_lon - lon)/2)))))

    return dist
