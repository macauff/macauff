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


def npy_to_csv(input_csv_folders, input_match_folder, output_folder, csv_filenames,
               output_filenames, column_name_lists, csv_col_name_or_num_lists,
               extra_col_cat_names, mem_chunk_num, headers=[False, False]):
    '''
    Function to convert output .npy files, as created during the cross-match
    process, and create a .csv file of matches and non-matches, combining columns
    from the original .csv catalogues.

    Parameters
    ----------
    input_csv_folders : list of strings
        List of the folders in which the two respective .csv file catalogues
        are stored.
    input_match_folder : string
        Folder in which all intermediate cross-match files have been stored.
    output_folder : string
        Folder into which to save the resulting .csv output files.
    csv_filenames : list of strings
        List of names, minus extensions, of the two catalogues that were matched.
    output_filenames : list of strings
        List of the names, minus extensions, out to which to save merged
        datasets.
    column_name_lists : list of list or array of strings
        List containing two lists of strings, one per catalogue. Each inner list
        should contain the names of the columns in each respective catalogue to
        be included in the merged dataset -- its ID or designation, two
        orthogonal sky positions, and N magnitudes, as were originally used
        in the matching process, and likely used in ``csv_to_npy``.
    csv_col_name_or_num_lists : list of list or array of integers
        List containing two lists or arrays of integers, one per catalogue,
        with the zero-index column integers corresponding to those columns listed
        in ``column_name_lists``.
    extra_col_cat_names : list of strings
        List of two strings, one per catalogue, indicating the names of the two
        catalogues, to append to the front of the derived contamination-based
        values included in the output datasets.
    mem_chunk_num : integer
        Indicator of how many subsets of the catalogue to load each catalogue in,
        for memory related issues.
    headers : list of booleans, optional
        List of two boolmean flags, one per catalogue, indicating whether the
        original input .csv file for this catalogue had a header which provides
        names for each column on its first line, or whether its first line is the
        first line of the data.
    '''
    # Need IDs/coordinates x2, mags (xN), then our columns: match probability, average
    # contaminant flux, eta/xi, and then M contaminant fractions for M relative fluxes.
    # TODO: un-hardcode number of relative contaminant fractions
    # TODO: remove photometric likelihood when not used.
    cols = np.append(np.append(column_name_lists[0], column_name_lists[1]),
                     ['MATCH_P', 'ETA', 'XI', '{}_AVG_CONT'.format(extra_col_cat_names[0]),
                      '{}_AVG_CONT'.format(extra_col_cat_names[1]),
                      '{}_CONT_F1'.format(extra_col_cat_names[0]),
                      '{}_CONT_F10'.format(extra_col_cat_names[0]),
                      '{}_CONT_F1'.format(extra_col_cat_names[1]),
                      '{}_CONT_F10'.format(extra_col_cat_names[1])])
    ac = np.load('{}/pairing/ac.npy'.format(input_match_folder), mmap_mode='r')
    bc = np.load('{}/pairing/bc.npy'.format(input_match_folder), mmap_mode='r')
    p = np.load('{}/pairing/pc.npy'.format(input_match_folder), mmap_mode='r')
    eta = np.load('{}/pairing/eta.npy'.format(input_match_folder), mmap_mode='r')
    xi = np.load('{}/pairing/xi.npy'.format(input_match_folder), mmap_mode='r')
    a_avg_cont = np.load('{}/pairing/acontamflux.npy'.format(input_match_folder), mmap_mode='r')
    b_avg_cont = np.load('{}/pairing/bcontamflux.npy'.format(input_match_folder), mmap_mode='r')
    acontprob = np.load('{}/pairing/pacontam.npy'.format(input_match_folder), mmap_mode='r')
    bcontprob = np.load('{}/pairing/pbcontam.npy'.format(input_match_folder), mmap_mode='r')
    # TODO: generalise so that other columns than designation+position+magnitudes
    # can be kept.
    n_amags, n_bmags = len(column_name_lists[0]) - 3, len(column_name_lists[1]) - 3
    cat_a = pd.read_csv('{}/{}.csv'.format(input_csv_folders[0], csv_filenames[0]),
                        memory_map=True, header=None if not headers[0] else 0,
                        usecols=csv_col_name_or_num_lists[0], names=column_name_lists[0])
    cat_b = pd.read_csv('{}/{}.csv'.format(input_csv_folders[1], csv_filenames[1]),
                        memory_map=True, header=None if not headers[1] else 0,
                        usecols=csv_col_name_or_num_lists[1], names=column_name_lists[1])
    n_matches = len(ac)
    match_df = pd.DataFrame(columns=cols, index=np.arange(0, n_matches))

    for cnum in range(0, mem_chunk_num):
        lowind = np.floor(n_matches*cnum/mem_chunk_num).astype(int)
        highind = np.floor(n_matches*(cnum+1)/mem_chunk_num).astype(int)
        for i in column_name_lists[0]:
            match_df[i].iloc[lowind:highind] = cat_a[i].iloc[ac[lowind:highind]].values
        for i in column_name_lists[1]:
            match_df[i].iloc[lowind:highind] = cat_b[i].iloc[bc[lowind:highind]].values
        match_df.iloc[lowind:highind, 6+n_amags+n_bmags] = p[lowind:highind]
        match_df.iloc[lowind:highind, 6+n_amags+n_bmags+1] = eta[lowind:highind]
        match_df.iloc[lowind:highind, 6+n_amags+n_bmags+2] = xi[lowind:highind]
        match_df.iloc[lowind:highind, 6+n_amags+n_bmags+3] = a_avg_cont[lowind:highind]
        match_df.iloc[lowind:highind, 6+n_amags+n_bmags+4] = b_avg_cont[lowind:highind]
        for i in range(acontprob.shape[1]):
            match_df.iloc[lowind:highind, 6+n_amags+n_bmags+5+i] = acontprob[lowind:highind, i]
        for i in range(bcontprob.shape[1]):
            match_df.iloc[lowind:highind, 6+n_amags+n_bmags+5+acontprob.shape[1]+i] = bcontprob[
                lowind:highind, i]

    match_df.to_csv('{}/{}.csv'.format(output_folder, output_filenames[0]), encoding='utf-8',
                    index=False, header=False)

    # For non-match, ID/coordinates/mags, then island probability.
    # TODO: add average contaminant flux recording to non-match outputs.
    af = np.load('{}/pairing/af.npy'.format(input_match_folder), mmap_mode='r')
    p = np.load('{}/pairing/pfa.npy'.format(input_match_folder), mmap_mode='r')
    cols = np.append(column_name_lists[0], ['MATCH_P'])
    n_anonmatches = len(af)
    a_nonmatch_df = pd.DataFrame(columns=cols, index=np.arange(0, n_anonmatches))
    for cnum in range(0, mem_chunk_num):
        lowind = np.floor(n_anonmatches*cnum/mem_chunk_num).astype(int)
        highind = np.floor(n_anonmatches*(cnum+1)/mem_chunk_num).astype(int)
        for i in column_name_lists[0]:
            a_nonmatch_df[i].iloc[lowind:highind] = cat_a[i].iloc[af[lowind:highind]].values
        a_nonmatch_df.iloc[lowind:highind, 3+n_amags] = p[lowind:highind]

    a_nonmatch_df.to_csv('{}/{}.csv'.format(output_folder, output_filenames[1]), encoding='utf-8',
                         index=False, header=False)

    bf = np.load('{}/pairing/bf.npy'.format(input_match_folder), mmap_mode='r')
    p = np.load('{}/pairing/pfb.npy'.format(input_match_folder), mmap_mode='r')
    cols = np.append(column_name_lists[1], ['MATCH_P'])
    n_bnonmatches = len(bf)
    b_nonmatch_df = pd.DataFrame(columns=cols, index=np.arange(0, n_bnonmatches))
    for cnum in range(0, mem_chunk_num):
        lowind = np.floor(n_bnonmatches*cnum/mem_chunk_num).astype(int)
        highind = np.floor(n_bnonmatches*(cnum+1)/mem_chunk_num).astype(int)
        for i in column_name_lists[1]:
            b_nonmatch_df[i].iloc[lowind:highind] = cat_b[i].iloc[bf[lowind:highind]].values
        b_nonmatch_df.iloc[lowind:highind, 3+n_bmags] = p[lowind:highind]

    b_nonmatch_df.to_csv('{}/{}.csv'.format(output_folder, output_filenames[2]), encoding='utf-8',
                         index=False, header=False)

    return
