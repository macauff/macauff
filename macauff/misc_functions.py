# Licensed under a 3-clause BSD style license - see LICENSE
'''
This module provides miscellaneous scripts, called in other parts of the cross-match
framework.
'''

import numpy as np


def create_auf_params_grid(auf_folder_path, auf_pointings, filt_names, array_name,
                           len_first_axis=None):
    '''
    Minor function to offload the creation of a 3-D or 4-D array from a series
    of 2-D arrays.

    Parameters
    ----------
    auf_folder_path : string
        Location of the top-level folder in which all fourier grids are saved.
    auf_pointings : numpy.ndarray
        Two-dimensional array with the sky coordinates of each pointing used
        in the perturbation AUF component creation.
    filt_names : list or numpy.ndarray
        List of ordered filters for the given catalogue.
    array_name : string
        The name of the individually-saved arrays, one per sub-folder, to turn
        into a 3-D or 4-D array.
    len_first_axis : integer, optional
        Length of the initial axis of the 4-D array. If not provided or is
        ``None``, final array is assumed to be 3-D instead.
    '''
    arraylengths = np.load('{}/arraylengths.npy'.format(auf_folder_path))
    longestNm = np.amax(arraylengths)
    if len_first_axis is None:
        grid = np.lib.format.open_memmap('{}/{}_grid.npy'.format(
            auf_folder_path, array_name), mode='w+', dtype=float, shape=(
            longestNm, len(filt_names), len(auf_pointings)), fortran_order=True)
        grid[:, :, :] = -1
    else:
        grid = np.lib.format.open_memmap('{}/{}_grid.npy'.format(
            auf_folder_path, array_name), mode='w+', dtype=float, shape=(
            len_first_axis, longestNm, len(filt_names), len(auf_pointings)), fortran_order=True)
        grid[:, :, :, :] = -1
    for j in range(0, len(auf_pointings)):
        ax1, ax2 = auf_pointings[j]
        for i in range(0, len(filt_names)):
            filt = filt_names[i]
            single_array = np.load('{}/{}/{}/{}/{}.npy'.format(auf_folder_path,
                                   ax1, ax2, filt, array_name))
            if len_first_axis is None:
                grid[:arraylengths[i, j], i, j] = single_array
            else:
                grid[:, :arraylengths[i, j], i, j] = single_array
    del arraylengths, longestNm, grid


def load_small_ref_auf_grid(modrefind, auf_folder_path, file_name_prefixes):
    '''
    Function to create reference index arrays out of larger arrays, based on
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
    file_name_prefixes : list
        Prefixes of the files stored in ``auf_folder_path`` -- the parts before
        "_grid" -- to be loaded as sub-arrays and returned.

    Returns
    -------
    small_grids : list of numpy.ndarray
        Small cutouts of ``*_grid`` files defined by ``file_name_prefixes``,
        containing only the appropriate indices for AUF pointing, filter, etc.
    modrefindsmall : numpy.ndarray
        The corresponding mappings for each source onto ``fouriergrid``, such
        that each source still points to the correct entry that it did in
        ``fourier_grid``.
    '''
    nmuniqueind, nmnewind = np.unique(modrefind[0, :], return_inverse=True)
    filtuniqueind, filtnewind = np.unique(modrefind[1, :], return_inverse=True)
    axuniqueind, axnewind = np.unique(modrefind[2, :], return_inverse=True)
    small_grids = []
    for name in file_name_prefixes:
        if len(np.load('{}/{}_grid.npy'.format(auf_folder_path, name), mmap_mode='r').shape) == 4:
            small_grids.append(np.asfortranarray(np.load('{}/{}_grid.npy'.format(
                auf_folder_path, name), mmap_mode='r')[:, :, :, axuniqueind][
                :, :, filtuniqueind, :][:, nmuniqueind, :, :]))
        else:
            small_grids.append(np.asfortranarray(np.load('{}/{}_grid.npy'.format(
                auf_folder_path, name), mmap_mode='r')[:, :, axuniqueind][
                :, filtuniqueind, :][nmuniqueind, :, :]))
    modrefindsmall = np.empty((3, modrefind.shape[1]), int, order='F')
    del modrefind
    modrefindsmall[0, :] = nmnewind
    modrefindsmall[1, :] = filtnewind
    modrefindsmall[2, :] = axnewind

    return small_grids, modrefindsmall


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
