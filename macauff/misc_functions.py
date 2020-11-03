# Licensed under a 3-clause BSD style license - see LICENSE
'''
This module provides miscellaneous scripts, called in other parts of the cross-match
framework.
'''

import numpy as np


def create_cumulative_offsets_grid(auf_folder_path, auf_pointings, filt_names, r):
    arraylengths = np.load('{}/arraylengths.npy'.format(auf_folder_path))
    longestNm = np.amax(arraylengths)
    cumulatoffgrids = np.lib.format.open_memmap('{}/cumulative_grid.npy'.format(
        auf_folder_path), mode='w+', dtype=float, shape=(len(r)-1, longestNm, len(filt_names),
                                                         len(auf_pointings)), fortran_order=True)
    cumulatoffgrids[:, :, :, :] = -1
    for j in range(0, len(auf_pointings)):
        ax1, ax2 = auf_pointings[j]
        for i in range(0, len(filt_names)):
            filt = filt_names[i]
            cumulatoffgrid = np.load('{}/{}/{}/{}/cumulative.npy'.format(auf_folder_path,
                                     ax1, ax2, filt))
            cumulatoffgrids[:, :arraylengths[i, j], i, j] = cumulatoffgrid
    del arraylengths, longestNm, cumulatoffgrids


def load_small_ref_ind_cumulative_grid(modrefind, auf_folder_path):
    nmuniqueind, nmnewind = np.unique(modrefind[0, :], return_inverse=True)
    filtuniqueind, filtnewind = np.unique(modrefind[1, :], return_inverse=True)
    axuniqueind, axnewind = np.unique(modrefind[2, :], return_inverse=True)
    cumulatoffgrid = np.asfortranarray(np.load('{}/cumulative_grid.npy'.format(
                                       auf_folder_path), mmap_mode='r')[:, :, :,
                                       axuniqueind][:, :, filtuniqueind, :][:,
                                       nmuniqueind, :, :])
    modrefindsmall = np.empty((3, modrefind.shape[1]), int, order='F')
    del modrefind
    modrefindsmall[0, :] = nmnewind
    modrefindsmall[1, :] = filtnewind
    modrefindsmall[2, :] = axnewind

    return cumulatoffgrid, modrefindsmall


def hav_dist_constant_lat(x_lon, x_lat, lon):
    '''
    Computes the Haversine formula in the limit that sky separation is only
    determined by longitudinal separation (i.e., delta-lat is zero).
    '''

    dist = np.degrees(2 * np.arcsin(np.abs(np.cos(np.radians(x_lat)) *
                                           np.sin(np.radians((x_lon - lon)/2)))))

    return dist
