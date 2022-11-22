# Licensed under a 3-clause BSD style license - see LICENSE
'''
This module provides functionality to convert input photometric catalogues
to the required numpy binary files used within the cross-match process.
'''

import os
import numpy as np
from numpy.lib.format import open_memmap
import pandas as pd

from .misc_functions import _load_rectangular_slice, _create_rectangular_slice_arrays
from .misc_functions_fortran import misc_functions_fortran as mff

__all__ = ['csv_to_npy', 'rect_slice_npy', 'npy_to_csv', 'rect_slice_csv']


def csv_to_npy(input_folder, input_filename, output_folder, astro_cols, photo_cols, bestindex_col,
               chunk_overlap_col, header=False, process_uncerts=False, astro_sig_fits_filepath=None,
               cat_in_radec=None):
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
        coordinates and a single, circular astrometric precision. The precision
        must be the last column, with the first two columns of ``astro_cols``
        matching in order to the first two columns of the output astrometry
        binary file, if ``process_uncerts`` is ``True``.
    photo_cols : list or numpy.array of integers
        List of zero-indexed columns in the input catalogue representing the
        magnitudes of each photometric source to be used in the cross-matching.
    bestindex_col : integer
        Zero-indexed column of the flag indicating which of the available
        photometric brightnesses (represented by ``photo_cols``) is the
        preferred choice -- usually the most precise and highest quality
        detection.
    chunk_overlap_col : integer
        Zero-indexed column in the .csv file containing an integer representation
        of the boolean representation of whether sources are in the "halo"
        (``1`` in the .csv) or "core" (``0``) of the region. If ``None`` then all
        objects are assumed to be in the core.
    header : boolean, optional
        Flag indicating whether the .csv file has a first line with the names
        of the columns in it, or whether the first line of the file is the first
        line of the dataset.
    process_uncerts : boolean, optional
        Determines whether uncertainties are re-processed in light of astrometric
        fitting on large scales.
    astro_sig_fits_filepath : string, optional
        Location on disk of the two saved files that contain the parameters
        that describe best-fit astrometric precision as a function of quoted
        astrometric precision. Must be provided if ``process_uncerts`` is ``True``.
    cat_in_radec : boolean, optional
        If ``process_uncerts`` is ``True``, must be provided, and either be
        ``True`` or ``False``, indicating whether the catalogue being processed
        is in RA-Dec coordinates or not. If ``True``, coordinates of mid-points
        for derivations of ``m`` and ``n`` for quoted-fit uncertainty relations
        will be converted from Right Ascension and Declination to Galactic
        Longitude and Latitude for the purposes of nearest-neighbour use.
    '''
    if not (process_uncerts is True or process_uncerts is False):
        raise ValueError("process_uncerts must either be True or False.")
    if process_uncerts and astro_sig_fits_filepath is None:
        raise ValueError("astro_sig_fits_filepath must given if process_uncerts is True.")
    if process_uncerts and cat_in_radec is None:
        raise ValueError("cat_in_radec must given if process_uncerts is True.")
    if process_uncerts and not (cat_in_radec is True or cat_in_radec is False):
        raise ValueError("If process_uncerts is True, cat_in_radec must either be True or False.")
    if process_uncerts and not os.path.exists(astro_sig_fits_filepath):
        raise ValueError("process_uncerts is True but astro_sig_fits_filepath does not exist. "
                         "Please ensure file path is correct.")
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
    chunk_overlap = open_memmap('{}/in_chunk_overlap.npy'.format(output_folder), mode='w+',
                                dtype=bool, shape=(n_rows,))

    if process_uncerts:
        m_sigs = np.load('{}/m_sigs_array.npy'.format(astro_sig_fits_filepath))
        n_sigs = np.load('{}/n_sigs_array.npy'.format(astro_sig_fits_filepath))
        mn_coords = np.empty((len(m_sigs), 2), float)
        mn_coords[:, 0] = np.load('{}/lmids.npy'.format(astro_sig_fits_filepath))
        mn_coords[:, 1] = np.load('{}/bmids.npy'.format(astro_sig_fits_filepath))
        if cat_in_radec:
            # Convert mn_coords to RA/Dec if catalogue is in Equatorial coords.
            from astropy.coordinates import SkyCoord
            a = SkyCoord(l=mn_coords[:, 0], b=mn_coords[:, 1], unit='deg', frame='galactic')
            mn_coords[:, 0] = a.icrs.ra.degree
            mn_coords[:, 1] = a.icrs.dec.degree

    used_cols = np.concatenate((astro_cols, photo_cols, [bestindex_col]))
    if chunk_overlap_col is not None:
        used_cols = np.concatenate((used_cols, [chunk_overlap_col]))
    new_astro_cols = np.array([np.where(used_cols == a)[0][0] for a in astro_cols])
    new_photo_cols = np.array([np.where(used_cols == a)[0][0] for a in photo_cols])
    new_bestindex_col = np.where(used_cols == bestindex_col)[0][0]
    if chunk_overlap_col is not None:
        new_chunk_overlap_col = np.where(used_cols == chunk_overlap_col)[0][0]
    n = 0
    for chunk in pd.read_csv('{}/{}.csv'.format(input_folder, input_filename), chunksize=100000,
                             usecols=used_cols, header=None if not header else 0):
        if not process_uncerts:
            astro[n:n+chunk.shape[0]] = chunk.values[:, new_astro_cols]
        else:
            astro[n:n+chunk.shape[0], 0] = chunk.values[:, new_astro_cols[0]]
            astro[n:n+chunk.shape[0], 1] = chunk.values[:, new_astro_cols[1]]
            old_sigs = chunk.values[:, new_astro_cols[2]]
            sig_mn_inds = mff.find_nearest_point(chunk.values[:, new_astro_cols[0]],
                                                 chunk.values[:, new_astro_cols[1]],
                                                 mn_coords[:, 0], mn_coords[:, 1])
            new_sigs = np.sqrt((m_sigs[sig_mn_inds]*old_sigs)**2 + n_sigs[sig_mn_inds]**2)
            astro[n:n+chunk.shape[0], 2] = new_sigs
        photo[n:n+chunk.shape[0]] = chunk.values[:, new_photo_cols]
        best_index[n:n+chunk.shape[0]] = chunk.values[:, new_bestindex_col]
        if chunk_overlap_col is not None:
            chunk_overlap[n:n+chunk.shape[0]] = chunk.values[:, new_chunk_overlap_col].astype(bool)
        else:
            chunk_overlap[n:n+chunk.shape[0]] = False

        n += chunk.shape[0]

    return


def npy_to_csv(input_csv_folders, input_match_folder, output_folder, csv_filenames,
               output_filenames, column_name_lists, column_num_lists, extra_col_cat_names,
               mem_chunk_num, input_npy_folders, headers=[False, False], extra_col_name_lists=None,
               extra_col_num_lists=None):
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
    column_num_lists : list of list or array of integers
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
    input_npy_folders : list of strings
        List of folders in which the respective catalogues' .npy files were
        saved to, as part of ``csv_to_npy``, in the same order as
        ``input_csv_folders``. If a catalogue did not have its uncertainties
        processed, its entry should be None, so a case where both catalogues in
        a match had their uncertainties treated as given would be
        ``[None, None]``.
    headers : list of booleans, optional
        List of two boolmean flags, one per catalogue, indicating whether the
        original input .csv file for this catalogue had a header which provides
        names for each column on its first line, or whether its first line is the
        first line of the data.
    extra_col_name_lists : list of list or array of strings, optional
        If not ``None``, should be a list of two lists of strings, one per
        catalogue. As with ``column_name_lists``, these should be names of
        columns from their respective catalogue in ``csv_filenames``, to be
        included in the output merged datasets.
    extra_col_num_lists : list of list or array of integer, optional
        If not ``None``, should be a list of two lists of strings, analagous
        to ``column_num_lists``, providing the column indices for additional
        catalogue columns in the original .csv files to be included in the
        output datafiles.
    '''
    # Need IDs/coordinates x2, mags (xN), then our columns: match probability, average
    # contaminant flux, eta/xi, and then M contaminant fractions for M relative fluxes.
    # TODO: un-hardcode number of relative contaminant fractions
    # TODO: remove photometric likelihood when not used.
    our_columns = ['MATCH_P', 'SEPARATION', 'ETA', 'XI',
                   '{}_AVG_CONT'.format(extra_col_cat_names[0]),
                   '{}_AVG_CONT'.format(extra_col_cat_names[1]),
                   '{}_CONT_F1'.format(extra_col_cat_names[0]),
                   '{}_CONT_F10'.format(extra_col_cat_names[0]),
                   '{}_CONT_F1'.format(extra_col_cat_names[1]),
                   '{}_CONT_F10'.format(extra_col_cat_names[1])]
    cols = np.append(np.append(column_name_lists[0], column_name_lists[1]), our_columns)
    if ((extra_col_name_lists is None and extra_col_num_lists is not None) or
            (extra_col_name_lists is not None and extra_col_num_lists is None)):
        raise UserWarning("extra_col_name_lists and extra_col_num_lists either both "
                          "need to be None, or both need to not be None.")
    if extra_col_num_lists is not None:
        cols = np.append(np.append(cols, extra_col_name_lists[0]), extra_col_name_lists[1])
    ac = np.load('{}/pairing/ac.npy'.format(input_match_folder), mmap_mode='r')
    bc = np.load('{}/pairing/bc.npy'.format(input_match_folder), mmap_mode='r')
    p = np.load('{}/pairing/pc.npy'.format(input_match_folder), mmap_mode='r')
    eta = np.load('{}/pairing/eta.npy'.format(input_match_folder), mmap_mode='r')
    xi = np.load('{}/pairing/xi.npy'.format(input_match_folder), mmap_mode='r')
    a_avg_cont = np.load('{}/pairing/acontamflux.npy'.format(input_match_folder), mmap_mode='r')
    b_avg_cont = np.load('{}/pairing/bcontamflux.npy'.format(input_match_folder), mmap_mode='r')
    acontprob = np.load('{}/pairing/pacontam.npy'.format(input_match_folder), mmap_mode='r')
    bcontprob = np.load('{}/pairing/pbcontam.npy'.format(input_match_folder), mmap_mode='r')
    seps = np.load('{}/pairing/crptseps.npy'.format(input_match_folder), mmap_mode='r')

    if input_npy_folders[0] is not None:
        cols = np.append(cols, ['{}_FIT_SIG'.format(extra_col_cat_names[0])])
        a_concatastro = np.load('{}/con_cat_astro.npy'.format(input_npy_folders[0]), mmap_mode='r')
    if input_npy_folders[1] is not None:
        cols = np.append(cols, ['{}_FIT_SIG'.format(extra_col_cat_names[1])])
        b_concatastro = np.load('{}/con_cat_astro.npy'.format(input_npy_folders[1]), mmap_mode='r')

    n_amags, n_bmags = len(column_name_lists[0]) - 3, len(column_name_lists[1]) - 3
    if extra_col_num_lists is None:
        a_cols = column_num_lists[0]
        b_cols = column_num_lists[1]
        a_names = column_name_lists[0]
        b_names = column_name_lists[1]
    else:
        a_cols = np.append(column_num_lists[0], extra_col_num_lists[0]).astype(int)
        b_cols = np.append(column_num_lists[1], extra_col_num_lists[1]).astype(int)
        a_names = np.append(column_name_lists[0], extra_col_name_lists[0])
        b_names = np.append(column_name_lists[1], extra_col_name_lists[1])
    a_names, b_names = np.array(a_names)[np.argsort(a_cols)], np.array(b_names)[np.argsort(b_cols)]

    cat_a = pd.read_csv('{}/{}.csv'.format(input_csv_folders[0], csv_filenames[0]),
                        memory_map=True, header=None if not headers[0] else 0,
                        usecols=a_cols, names=a_names)
    cat_b = pd.read_csv('{}/{}.csv'.format(input_csv_folders[1], csv_filenames[1]),
                        memory_map=True, header=None if not headers[1] else 0,
                        usecols=b_cols, names=b_names)
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
        match_df.iloc[lowind:highind, 6+n_amags+n_bmags+1] = seps[lowind:highind]
        match_df.iloc[lowind:highind, 6+n_amags+n_bmags+2] = eta[lowind:highind]
        match_df.iloc[lowind:highind, 6+n_amags+n_bmags+3] = xi[lowind:highind]
        match_df.iloc[lowind:highind, 6+n_amags+n_bmags+4] = a_avg_cont[lowind:highind]
        match_df.iloc[lowind:highind, 6+n_amags+n_bmags+5] = b_avg_cont[lowind:highind]
        for i in range(acontprob.shape[1]):
            match_df.iloc[lowind:highind, 6+n_amags+n_bmags+6+i] = acontprob[lowind:highind, i]
        for i in range(bcontprob.shape[1]):
            match_df.iloc[lowind:highind, 6+n_amags+n_bmags+6+acontprob.shape[1]+i] = bcontprob[
                lowind:highind, i]
        if extra_col_name_lists is not None:
            for i in extra_col_name_lists[0]:
                match_df[i].iloc[lowind:highind] = cat_a[i].iloc[ac[lowind:highind]].values
            for i in extra_col_name_lists[1]:
                match_df[i].iloc[lowind:highind] = cat_b[i].iloc[bc[lowind:highind]].values

        # FIT_SIG are the last 0-2 columns, after [ID+coords(x2)+mag(xN)]x2 +
        # Q match-made columns, plus len(extra_col_name_lists)x2.
        if input_npy_folders[0] is not None:
            ind = (len(column_name_lists[0]) + len(column_name_lists[1]) + len(our_columns) +
                   (len(extra_col_name_lists[0]) if extra_col_name_lists is not None else 0) +
                   (len(extra_col_name_lists[1]) if extra_col_name_lists is not None else 0))
            match_df.iloc[lowind:highind, ind] = a_concatastro[ac[lowind:highind], 2]
        if input_npy_folders[1] is not None:
            # Here we also need to check if catalogue "a" has processed uncertainties too.
            _dx = 1 if input_npy_folders[0] is not None else 0
            ind = (len(column_name_lists[0]) + len(column_name_lists[1]) + len(our_columns) +
                   (len(extra_col_name_lists[0]) if extra_col_name_lists is not None else 0) +
                   (len(extra_col_name_lists[1]) if extra_col_name_lists is not None else 0)) + _dx
            match_df.iloc[lowind:highind, ind] = b_concatastro[bc[lowind:highind], 2]

    match_df.to_csv('{}/{}.csv'.format(output_folder, output_filenames[0]), encoding='utf-8',
                    index=False, header=False)

    # For non-match, ID/coordinates/mags, then island probability + average
    # contamination.
    af = np.load('{}/pairing/af.npy'.format(input_match_folder), mmap_mode='r')
    a_avg_cont = np.load('{}/pairing/afieldflux.npy'.format(input_match_folder), mmap_mode='r')
    p = np.load('{}/pairing/pfa.npy'.format(input_match_folder), mmap_mode='r')
    seps = np.load('{}/pairing/afieldseps.npy'.format(input_match_folder), mmap_mode='r')
    afeta = np.load('{}/pairing/afieldeta.npy'.format(input_match_folder), mmap_mode='r')
    afxi = np.load('{}/pairing/afieldxi.npy'.format(input_match_folder), mmap_mode='r')
    our_columns = ['MATCH_P', 'NNM_SEPARATION', 'NNM_ETA', 'NNM_XI',
                   '{}_AVG_CONT'.format(extra_col_cat_names[0])]
    cols = np.append(column_name_lists[0], our_columns)
    if extra_col_num_lists is not None:
        cols = np.append(cols, extra_col_name_lists[0])
    if input_npy_folders[0] is not None:
        cols = np.append(cols, ['{}_FIT_SIG'.format(extra_col_cat_names[0])])
        a_concatastro = np.load('{}/con_cat_astro.npy'.format(input_npy_folders[0]), mmap_mode='r')
    n_anonmatches = len(af)
    a_nonmatch_df = pd.DataFrame(columns=cols, index=np.arange(0, n_anonmatches))
    for cnum in range(0, mem_chunk_num):
        lowind = np.floor(n_anonmatches*cnum/mem_chunk_num).astype(int)
        highind = np.floor(n_anonmatches*(cnum+1)/mem_chunk_num).astype(int)
        for i in column_name_lists[0]:
            a_nonmatch_df[i].iloc[lowind:highind] = cat_a[i].iloc[af[lowind:highind]].values
        a_nonmatch_df.iloc[lowind:highind, 3+n_amags] = p[lowind:highind]
        a_nonmatch_df.iloc[lowind:highind, 3+n_amags+1] = seps[lowind:highind]
        a_nonmatch_df.iloc[lowind:highind, 3+n_amags+2] = afeta[lowind:highind]
        a_nonmatch_df.iloc[lowind:highind, 3+n_amags+3] = afxi[lowind:highind]
        a_nonmatch_df.iloc[lowind:highind, 3+n_amags+4] = a_avg_cont[lowind:highind]
        if extra_col_name_lists is not None:
            for i in extra_col_name_lists[0]:
                a_nonmatch_df[i].iloc[lowind:highind] = cat_a[i].iloc[af[lowind:highind]].values

        if input_npy_folders[0] is not None:
            ind = (len(column_name_lists[0]) + len(our_columns) +
                   (len(extra_col_name_lists[0]) if extra_col_name_lists is not None else 0))
            a_nonmatch_df.iloc[lowind:highind, ind] = a_concatastro[af[lowind:highind], 2]

    a_nonmatch_df.to_csv('{}/{}.csv'.format(output_folder, output_filenames[1]), encoding='utf-8',
                         index=False, header=False)

    bf = np.load('{}/pairing/bf.npy'.format(input_match_folder), mmap_mode='r')
    b_avg_cont = np.load('{}/pairing/bfieldflux.npy'.format(input_match_folder), mmap_mode='r')
    p = np.load('{}/pairing/pfb.npy'.format(input_match_folder), mmap_mode='r')
    seps = np.load('{}/pairing/bfieldseps.npy'.format(input_match_folder), mmap_mode='r')
    bfeta = np.load('{}/pairing/bfieldeta.npy'.format(input_match_folder), mmap_mode='r')
    bfxi = np.load('{}/pairing/bfieldxi.npy'.format(input_match_folder), mmap_mode='r')
    our_columns = ['MATCH_P', 'NNM_SEPARATION', 'NNM_ETA', 'NNM_XI',
                   '{}_AVG_CONT'.format(extra_col_cat_names[1])]
    cols = np.append(column_name_lists[1], our_columns)
    if extra_col_num_lists is not None:
        cols = np.append(cols, extra_col_name_lists[1])
    if input_npy_folders[1] is not None:
        cols = np.append(cols, ['{}_FIT_SIG'.format(extra_col_cat_names[1])])
        b_concatastro = np.load('{}/con_cat_astro.npy'.format(input_npy_folders[1]), mmap_mode='r')
    n_bnonmatches = len(bf)
    b_nonmatch_df = pd.DataFrame(columns=cols, index=np.arange(0, n_bnonmatches))
    for cnum in range(0, mem_chunk_num):
        lowind = np.floor(n_bnonmatches*cnum/mem_chunk_num).astype(int)
        highind = np.floor(n_bnonmatches*(cnum+1)/mem_chunk_num).astype(int)
        for i in column_name_lists[1]:
            b_nonmatch_df[i].iloc[lowind:highind] = cat_b[i].iloc[bf[lowind:highind]].values
        b_nonmatch_df.iloc[lowind:highind, 3+n_bmags] = p[lowind:highind]
        b_nonmatch_df.iloc[lowind:highind, 3+n_bmags+1] = seps[lowind:highind]
        b_nonmatch_df.iloc[lowind:highind, 3+n_bmags+2] = bfeta[lowind:highind]
        b_nonmatch_df.iloc[lowind:highind, 3+n_bmags+3] = bfxi[lowind:highind]
        b_nonmatch_df.iloc[lowind:highind, 3+n_bmags+4] = b_avg_cont[lowind:highind]
        if extra_col_name_lists is not None:
            for i in extra_col_name_lists[1]:
                b_nonmatch_df[i].iloc[lowind:highind] = cat_b[i].iloc[bf[lowind:highind]].values

        if input_npy_folders[1] is not None:
            ind = (len(column_name_lists[1]) + len(our_columns) +
                   (len(extra_col_name_lists[1]) if extra_col_name_lists is not None else 0))
            b_nonmatch_df.iloc[lowind:highind, ind] = b_concatastro[bf[lowind:highind], 2]

    b_nonmatch_df.to_csv('{}/{}.csv'.format(output_folder, output_filenames[2]), encoding='utf-8',
                         index=False, header=False)

    return


def rect_slice_csv(input_folder, output_folder, input_filename, output_filename, rect_coords,
                   padding, astro_cols, mem_chunk_num, header=False):
    '''
    Convenience function to take a small rectangular slice of a larger .csv catalogue,
    based on its given orthogonal sky coordinates in the large catalogue.

    Parameters
    ----------
    input_folder : string
        Folder in which the larger .csv catalogue is stored.
    output_folder : string
        Folder into which to save the cutout catalogue.
    input_filename : string
        Name, minus .csv extension, of the larger catalogue file.
    output_filename : string
        Name, minus .csv extension, of the cutout catalogue.
    rect_coords : list or array of floats
        List of coordinates inside which to take the subset catalogue. Should
        be of the kind [lower_ax1, upper_ax1, lower_ax2, upper_ax2], where ax1
        is e.g. Right Ascension and ax2 is e.g. Declination.
    padding : float
        Amount of additional on-sky area to permit outside of ``rect_coords``.
        In these cases sources must be within ``padding`` distance of the
        rectangle defined by ``rect_coords`` by the Haversine formula.
    astro_cols : list or array of integers
        List of zero-index columns representing the orthogonal sky axes, in the
        sense of [ax1, ax2], with ax1 being e.g. Galactic Longitude and ax2 being
        e.g. Galactic Latitude, as with ``rect_coords``.
    mem_chunk_num : integer
        Integer representing the number of sub-slices of the catalogue to load,
        in cases where the larger file is larger than available memory.
    header : boolean, optional
        Flag indicating whether .csv file has a header line, giving names of
        each column, or if the first line of the file is the first line of the
        dataset.
    '''
    with open('{}/{}.csv'.format(input_folder, input_filename)) as fp:
        n_rows = 0 if not header else -1
        for _ in fp:
            n_rows += 1
    small_astro = open_memmap('{}/temp_astro.npy'.format(input_folder), mode='w+', dtype=float,
                              shape=(n_rows, 2))

    n = 0
    for chunk in pd.read_csv('{}/{}.csv'.format(input_folder, input_filename), chunksize=100000,
                             usecols=astro_cols, header=None if not header else 0):
        small_astro[n:n+chunk.shape[0]] = chunk.values
        n += chunk.shape[0]

    _create_rectangular_slice_arrays(input_folder, '', n_rows)
    memmap_arrays = []
    for n in ['1', '2', '3', '4', 'combined']:
        memmap_arrays.append(np.lib.format.open_memmap('{}/{}_temporary_sky_slice_{}.npy'.format(
                             input_folder, '', n), mode='r+', dtype=bool, shape=(n_rows,)))
    _load_rectangular_slice(input_folder, '', small_astro, rect_coords[0], rect_coords[1],
                            rect_coords[2], rect_coords[3], padding, memmap_arrays)

    n_inside_rows = 0
    combined_memmap = memmap_arrays[4]
    for cnum in range(0, mem_chunk_num):
        lowind = np.floor(n_rows*cnum/mem_chunk_num).astype(int)
        highind = np.floor(n_rows*(cnum+1)/mem_chunk_num).astype(int)
        n_inside_rows += np.sum(combined_memmap[lowind:highind])
    df_orig = pd.read_csv('{}/{}.csv'.format(input_folder, input_filename), nrows=1,
                          header=None if not header else 0)
    df = pd.DataFrame(columns=df_orig.columns, index=np.arange(0, n_inside_rows))

    counter = 0
    outer_counter = 0
    chunksize = 100000
    for chunk in pd.read_csv('{}/{}.csv'.format(input_folder, input_filename), chunksize=chunksize,
                             header=None if not header else 0):
        inside_n = np.sum(combined_memmap[outer_counter:outer_counter+chunksize])
        df.iloc[counter:counter+inside_n] = chunk.values[
            combined_memmap[outer_counter:outer_counter+chunksize]]
        counter += inside_n
        outer_counter += chunksize

    df.to_csv('{}/{}.csv'.format(output_folder, output_filename), encoding='utf-8', index=False,
              header=False)

    for n in ['1', '2', '3', '4', 'combined']:
        os.remove('{}/{}_temporary_sky_slice_{}.npy'.format(input_folder, '', n))
    os.remove('{}/temp_astro.npy'.format(input_folder))

    return


def rect_slice_npy(input_folder, output_folder, rect_coords, padding, mem_chunk_num):
    '''
    Convenience function to take a small rectangular slice of a larger catalogue,
    represented by three or four binary .npy files, based on its given orthogonal
    sky coordinates in the large catalogue.

    Parameters
    ----------
    input_folder : string
        Folder in which the larger .npy files representing the catalogue
        are stored.
    output_folder : string
        Folder into which to save the cutout catalogue .npy files.
    rect_coords : list or array of floats
        List of coordinates inside which to take the subset catalogue. Should
        be of the kind [lower_ax1, upper_ax1, lower_ax2, upper_ax2], where ax1
        is e.g. Right Ascension and ax2 is e.g. Declination.
    padding : float
        Amount of additional on-sky area to permit outside of ``rect_coords``.
        In these cases sources must be within ``padding`` distance of the
        rectangle defined by ``rect_coords`` by the Haversine formula.
    astro_cols : list or array of integers
        List of zero-index columns representing the orthogonal sky axes, in the
        sense of [ax1, ax2], with ax1 being e.g. Galactic Longitude and ax2 being
        e.g. Galactic Latitude, as with ``rect_coords``.
    mem_chunk_num : integer
        Integer representing the number of sub-slices of the catalogue to load,
        in cases where the larger file is larger than available memory.
    '''
    astro = np.load('{}/con_cat_astro.npy'.format(input_folder), mmap_mode='r')
    photo = np.load('{}/con_cat_photo.npy'.format(input_folder), mmap_mode='r')
    best_index = np.load('{}/magref.npy'.format(input_folder), mmap_mode='r')
    n_rows = len(astro)
    _create_rectangular_slice_arrays(input_folder, '', n_rows)
    memmap_arrays = []
    for n in ['1', '2', '3', '4', 'combined']:
        memmap_arrays.append(np.lib.format.open_memmap('{}/{}_temporary_sky_slice_{}.npy'.format(
                             input_folder, '', n), mode='r+', dtype=bool, shape=(n_rows,)))
    _load_rectangular_slice(input_folder, '', astro, rect_coords[0], rect_coords[1],
                            rect_coords[2], rect_coords[3], padding, memmap_arrays)

    n_inside_rows = 0
    combined_memmap = memmap_arrays[4]
    for cnum in range(0, mem_chunk_num):
        lowind = np.floor(n_rows*cnum/mem_chunk_num).astype(int)
        highind = np.floor(n_rows*(cnum+1)/mem_chunk_num).astype(int)
        n_inside_rows += np.sum(combined_memmap[lowind:highind])

    small_astro = open_memmap('{}/con_cat_astro.npy'.format(output_folder), mode='w+', dtype=float,
                              shape=(n_inside_rows, 3))
    small_photo = open_memmap('{}/con_cat_photo.npy'.format(output_folder), mode='w+', dtype=float,
                              shape=(n_inside_rows, photo.shape[1]))
    small_best_index = open_memmap('{}/magref.npy'.format(output_folder), mode='w+', dtype=int,
                                   shape=(n_inside_rows,))
    small_chunk_overlap = open_memmap('{}/in_chunk_overlap.npy'.format(output_folder), mode='w+',
                                      dtype=bool, shape=(n_inside_rows,))

    counter = 0
    for cnum in range(0, mem_chunk_num):
        lowind = np.floor(n_rows*cnum/mem_chunk_num).astype(int)
        highind = np.floor(n_rows*(cnum+1)/mem_chunk_num).astype(int)
        inside_n = np.sum(combined_memmap[lowind:highind])
        small_astro[counter:counter+inside_n] = astro[lowind:highind][
            combined_memmap[lowind:highind]]
        small_photo[counter:counter+inside_n] = photo[lowind:highind][
            combined_memmap[lowind:highind]]
        small_best_index[counter:counter+inside_n] = best_index[lowind:highind][
            combined_memmap[lowind:highind]]
        # Always assume that a cutout is a single "visit" with no chunk "halo".
        small_chunk_overlap[counter:counter:inside_n] = False
        counter += inside_n

    for n in ['1', '2', '3', '4', 'combined']:
        os.remove('{}/{}_temporary_sky_slice_{}.npy'.format(input_folder, '', n))

    return
