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
