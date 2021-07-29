# Licensed under a 3-clause BSD style license - see LICENSE
'''
This module provides functionality to convert input photometric catalogues
to the required numpy binary files used within the cross-match process.
'''

import numpy as np
from numpy.lib.format import open_memmap
import pandas as pd


def csv_to_npy(input_folder, input_filename, output_folder, astro_cols, photo_cols, bestindex_col,
               header=False):
    '''
    Convert a .csv file representation of a photometric catalogue into the
    appropriate .npy binary files used in the cross-matching process.

    Parameters
    ----------
    input_folder : string
        Folder on disk where the catalogue .csv file is stored.
    input_filename : string
        Name of the .csv file, without the extension, to convert to binary files.
    output_folder : string
        Folder on disk of where to save the .npy versions of the catalogue.
    astro_cols : list or numpy.array of integers
        List of zero-indexed columns in the input catalogue representing the
        three required astrometric parameters, two orthogonal sky axis
        coordinates and a single, circular astrometric precision.
    photo_cols : list or numpy.array of integers
        List of zero-indexed columns in the input catalogue representing the
        magnitudes of each photometric source to be used in the cross-matching.
    bestindex_col : integer
        Zero-indexed column of the flag indicating which of the available
        photometric brightnesses (represented by ``photo_cols``) is the
        preferred choice -- usually the most precise and highest quality
        detection.
    header : boolean, optional
        Flag indicating whether the .csv file has a first line with the names
        of the columns in it, or whether the first line of the file is the first
        line of the dataset.
    '''
    astro_cols, photo_cols = np.array(astro_cols), np.array(photo_cols)
    with open('{}/{}.csv'.format(input_folder, input_filename)) as fp:
        n_rows = 0 if not header else -1
        for _ in fp:
            n_rows += 1

    astro = open_memmap('{}/con_cat_astro.npy'.format(output_folder), mode='w+', dtype=float,
                        shape=(n_rows, 3))
    photo = open_memmap('{}/con_cat_photo.npy'.format(output_folder), mode='w+', dtype=float,
                        shape=(n_rows, len(photo_cols)))
    best_index = open_memmap('{}/magref.npy'.format(output_folder), mode='w+', dtype=int,
                             shape=(n_rows,))

    used_cols = np.concatenate((astro_cols, photo_cols, [bestindex_col]))
    new_astro_cols = np.array([np.where(used_cols == a)[0][0] for a in astro_cols])
    new_photo_cols = np.array([np.where(used_cols == a)[0][0] for a in photo_cols])
    new_bestindex_col = np.where(used_cols == bestindex_col)[0][0]
    n = 0
    for chunk in pd.read_csv('{}/{}.csv'.format(input_folder, input_filename), chunksize=100000,
                             usecols=used_cols, header=None if not header else 0):
        astro[n:n+chunk.shape[0]] = chunk.values[:, new_astro_cols]
        photo[n:n+chunk.shape[0]] = chunk.values[:, new_photo_cols]
        best_index[n:n+chunk.shape[0]] = chunk.values[:, new_bestindex_col]
        n += chunk.shape[0]

    return
