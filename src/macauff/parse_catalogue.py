# Licensed under a 3-clause BSD style license - see LICENSE
'''
This module provides functionality to convert input photometric catalogues
to the required numpy binary files used within the cross-match process.
'''

import os

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from numpy.lib.format import open_memmap

# pylint: disable=import-error,no-name-in-module
from macauff.misc_functions import _load_rectangular_slice
from macauff.misc_functions_fortran import misc_functions_fortran as mff

# pylint: enable=import-error,no-name-in-module

__all__ = ['csv_to_npy', 'rect_slice_npy', 'npy_to_csv', 'rect_slice_csv']


def csv_to_npy(input_folder, input_filename, output_folder, astro_cols, photo_cols, bestindex_col,
               chunk_overlap_col, header=False, process_uncerts=False, astro_sig_fits_filepath=None,
               cat_in_radec=None, mn_in_radec=None):
    '''
    Convert a .csv file representation of a photometric catalogue into the
    appropriate .npy binary files used in the cross-matching process.

    Parameters
    ----------
    input_folder : string
        Folder on disk where the catalogue .csv file is stored.
    input_filename : string
        Name of the CSV file, including extension, to convert to binary files.
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
        will be converted from Galactic Longitude and Latitude to Right Ascension
        and Declination for the purposes of nearest-neighbour use, if
        ``mn_in_radec`` is ``False`` (and m-n coordinates are in l/b).
    mn_in_radec : boolean, optional
        If ``process_uncerts`` is ``True``, must be provided, and similar to
        ``cat_in_radec`` is a flag indcating whether the coordinates used to
        compute m-n scaling relations are in RA/Dec or not. If ``mn_in_radec``
        disagrees with ``cat_in_radec`` then m-n coordinates will be converted
        to the coordinate system of the catalogue.
    '''
    if not (process_uncerts is True or process_uncerts is False):
        raise ValueError("process_uncerts must either be True or False.")
    if process_uncerts and astro_sig_fits_filepath is None:
        raise ValueError("astro_sig_fits_filepath must given if process_uncerts is True.")
    if process_uncerts and cat_in_radec is None:
        raise ValueError("cat_in_radec must given if process_uncerts is True.")
    if process_uncerts and mn_in_radec is None:
        raise ValueError("mn_in_radec must given if process_uncerts is True.")
    if process_uncerts and not (cat_in_radec is True or cat_in_radec is False):
        raise ValueError("If process_uncerts is True, cat_in_radec must either be True or False.")
    if process_uncerts and not (mn_in_radec is True or mn_in_radec is False):
        raise ValueError("If process_uncerts is True, mn_in_radec must either be True or False.")
    if process_uncerts and not os.path.exists(astro_sig_fits_filepath):
        raise ValueError("process_uncerts is True but astro_sig_fits_filepath does not exist. "
                         "Please ensure file path is correct.")
    astro_cols, photo_cols = np.array(astro_cols), np.array(photo_cols)
    with open(f'{input_folder}/{input_filename}', encoding='utf-8') as fp:
        n_rows = 0 if not header else -1
        for _ in fp:
            n_rows += 1

    astro = open_memmap(f'{output_folder}/con_cat_astro.npy', mode='w+', dtype=float,
                        shape=(n_rows, 3))
    photo = open_memmap(f'{output_folder}/con_cat_photo.npy', mode='w+', dtype=float,
                        shape=(n_rows, len(photo_cols)))
    best_index = open_memmap(f'{output_folder}/magref.npy', mode='w+', dtype=int,
                             shape=(n_rows,))
    chunk_overlap = open_memmap(f'{output_folder}/in_chunk_overlap.npy', mode='w+',
                                dtype=bool, shape=(n_rows,))

    if process_uncerts:
        m_sigs = np.load(f'{astro_sig_fits_filepath}/m_sigs_array.npy')
        n_sigs = np.load(f'{astro_sig_fits_filepath}/n_sigs_array.npy')
        mn_coords = np.empty((len(m_sigs), 2), float)
        mn_coords[:, 0] = np.load(f'{astro_sig_fits_filepath}/snr_mag_params.npy')[0, :, 3]
        mn_coords[:, 1] = np.load(f'{astro_sig_fits_filepath}/snr_mag_params.npy')[0, :, 4]
        if cat_in_radec and not mn_in_radec:
            # Convert mn_coords to RA/Dec if catalogue is in Equatorial coords.
            a = SkyCoord(l=mn_coords[:, 0], b=mn_coords[:, 1], unit='deg', frame='galactic')
            mn_coords[:, 0] = a.icrs.ra.degree
            mn_coords[:, 1] = a.icrs.dec.degree
        if not cat_in_radec and mn_in_radec:
            # Convert mn_coords to l/b if catalogue is in Galactic coords.
            a = SkyCoord(ra=mn_coords[:, 0], dec=mn_coords[:, 1], unit='deg', frame='icrs')
            mn_coords[:, 0] = a.galactic.l.degree
            mn_coords[:, 1] = a.galactic.b.degree

    used_cols = np.concatenate((astro_cols, photo_cols, [bestindex_col]))
    if chunk_overlap_col is not None:
        used_cols = np.concatenate((used_cols, [chunk_overlap_col]))
    new_astro_cols = np.array([np.where(used_cols == a)[0][0] for a in astro_cols])
    new_photo_cols = np.array([np.where(used_cols == a)[0][0] for a in photo_cols])
    new_bestindex_col = np.where(used_cols == bestindex_col)[0][0]
    if chunk_overlap_col is not None:
        new_chunk_overlap_col = np.where(used_cols == chunk_overlap_col)[0][0]
    n = 0
    for chunk in pd.read_csv(f'{input_folder}/{input_filename}', chunksize=100000,
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


# pylint: disable-next=dangerous-default-value,too-many-locals,too-many-statements
def npy_to_csv(input_csv_folders, input_match_folder, output_folder, csv_filenames,
               output_filenames, column_name_lists, column_num_lists, extra_col_cat_names,
               input_npy_folders, headers=[False, False], extra_col_name_lists=[None, None],
               extra_col_num_lists=[None, None]):
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
        List of names, including extensions, of the two catalogues that were matched.
    output_filenames : list of strings
        List of the names, including extensions, out to which to save merged
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
    extra_col_name_lists : list of list or array of strings, or None, optional
        Should be a list of two lists of strings, one per catalogue. As with
        ``column_name_lists``, these should be names of columns from their
        respective catalogue in ``csv_filenames``, to be included in the output
        merged datasets. For a particular catalogue, if no extra columns should
        be included, put ``None`` in that entry. For example, to only include
        an extra single column ``Q`` for the second catalogue,
        ``extra_col_name_lists=[None, ['Q']]``.
    extra_col_num_lists : list of list or array of integer, or None, optional
        Should be a list of two lists of strings, analagous to
        ``column_num_lists``, providing the column indices for additional
        catalogue columns in the original .csv files to be included in the
        output datafiles. Like ``extra_col_name_lists``, for either catalogue
        ``None`` can be entered for no additional columns; for the above example
        we would use ``extra_col_num_lists=[None, [7]]``.
    '''
    # Need IDs/coordinates x2, mags (xN), then our columns: match probability, average
    # contaminant flux, eta/xi, and then M contaminant fractions for M relative fluxes.
    # TODO: un-hardcode number of relative contaminant fractions.  pylint: disable=fixme
    # TODO: remove photometric likelihood when not used.  pylint: disable=fixme
    our_columns = ['MATCH_P', 'SEPARATION', 'ETA', 'XI',
                   f'{extra_col_cat_names[0]}_AVG_CONT', f'{extra_col_cat_names[1]}_AVG_CONT',
                   f'{extra_col_cat_names[0]}_CONT_F1', f'{extra_col_cat_names[0]}_CONT_F10',
                   f'{extra_col_cat_names[1]}_CONT_F1', f'{extra_col_cat_names[1]}_CONT_F10']
    cols = np.append(np.append(column_name_lists[0], column_name_lists[1]), our_columns)
    for i, entry in zip([0, 1], ['1st', '2nd']):
        if ((extra_col_name_lists[i] is None and extra_col_num_lists[i] is not None) or
                (extra_col_name_lists[i] is not None and extra_col_num_lists[i] is None)):
            raise UserWarning("extra_col_name_lists and extra_col_num_lists either both "
                              f"need to be None, or both need to not be None, for the {entry} "
                              "catalogue.")
        if extra_col_num_lists[i] is not None:
            cols = np.append(cols, extra_col_name_lists[i])
    ac = np.load(f'{input_match_folder}/ac.npy')
    bc = np.load(f'{input_match_folder}/bc.npy')
    p = np.load(f'{input_match_folder}/pc.npy')
    eta = np.load(f'{input_match_folder}/eta.npy')
    xi = np.load(f'{input_match_folder}/xi.npy')
    a_avg_cont = np.load(f'{input_match_folder}/acontamflux.npy')
    b_avg_cont = np.load(f'{input_match_folder}/bcontamflux.npy')
    acontprob = np.load(f'{input_match_folder}/pacontam.npy')
    bcontprob = np.load(f'{input_match_folder}/pbcontam.npy')
    seps = np.load(f'{input_match_folder}/crptseps.npy')

    if input_npy_folders[0] is not None:
        cols = np.append(cols, [f'{extra_col_cat_names[0]}_FIT_SIG'])
        a_concatastro = np.load(f'{input_npy_folders[0]}/con_cat_astro.npy')
    if input_npy_folders[1] is not None:
        cols = np.append(cols, [f'{extra_col_cat_names[1]}_FIT_SIG'])
        b_concatastro = np.load(f'{input_npy_folders[1]}/con_cat_astro.npy')

    n_amags, n_bmags = len(column_name_lists[0]) - 3, len(column_name_lists[1]) - 3
    if extra_col_num_lists[0] is None:
        a_cols = column_num_lists[0]
        a_names = column_name_lists[0]
    else:
        a_cols = np.append(column_num_lists[0], extra_col_num_lists[0]).astype(int)
        a_names = np.append(column_name_lists[0], extra_col_name_lists[0])
    if extra_col_num_lists[1] is None:
        b_cols = column_num_lists[1]
        b_names = column_name_lists[1]
    else:
        b_cols = np.append(column_num_lists[1], extra_col_num_lists[1]).astype(int)
        b_names = np.append(column_name_lists[1], extra_col_name_lists[1])
    a_names, b_names = np.array(a_names)[np.argsort(a_cols)], np.array(b_names)[np.argsort(b_cols)]

    cat_a = pd.read_csv(f'{input_csv_folders[0]}/{csv_filenames[0]}', memory_map=True,
                        header=None if not headers[0] else 0, usecols=a_cols, names=a_names)
    cat_b = pd.read_csv(f'{input_csv_folders[1]}/{csv_filenames[1]}', memory_map=True,
                        header=None if not headers[1] else 0, usecols=b_cols, names=b_names)
    n_matches = len(ac)
    match_df = pd.DataFrame(columns=cols, index=np.arange(0, n_matches))

    for i in column_name_lists[0]:
        match_df.loc[:, i] = cat_a.loc[ac, i].values
    for i in column_name_lists[1]:
        match_df.loc[:, i] = cat_b.loc[bc, i].values
    match_df.iloc[:, 6+n_amags+n_bmags] = p
    match_df.iloc[:, 6+n_amags+n_bmags+1] = seps
    match_df.iloc[:, 6+n_amags+n_bmags+2] = eta
    match_df.iloc[:, 6+n_amags+n_bmags+3] = xi
    match_df.iloc[:, 6+n_amags+n_bmags+4] = a_avg_cont
    match_df.iloc[:, 6+n_amags+n_bmags+5] = b_avg_cont
    for i in range(acontprob.shape[0]):
        match_df.iloc[:, 6+n_amags+n_bmags+6+i] = acontprob[i, :]
    for i in range(bcontprob.shape[0]):
        match_df.iloc[:, 6+n_amags+n_bmags+6+acontprob.shape[0]+i] = bcontprob[i, :]
    if extra_col_name_lists[0] is not None:
        for i in extra_col_name_lists[0]:
            match_df.loc[:, i] = cat_a.loc[ac, i].values
    if extra_col_name_lists[1] is not None:
        for i in extra_col_name_lists[1]:
            match_df.loc[:, i] = cat_b.loc[bc, i].values

    # FIT_SIG are the last 0-2 columns, after [ID+coords(x2)+mag(xN)]x2 +
    # Q match-made columns, plus len(extra_col_name_lists)x2.
    if input_npy_folders[0] is not None:
        ind = (len(column_name_lists[0]) + len(column_name_lists[1]) + len(our_columns) +
               (len(extra_col_name_lists[0]) if extra_col_name_lists[0] is not None else 0) +
               (len(extra_col_name_lists[1]) if extra_col_name_lists[1] is not None else 0))
        match_df.iloc[:, ind] = a_concatastro[ac, 2]
    if input_npy_folders[1] is not None:
        # Here we also need to check if catalogue "a" has processed uncertainties too.
        _dx = 1 if input_npy_folders[0] is not None else 0
        ind = (len(column_name_lists[0]) + len(column_name_lists[1]) + len(our_columns) +
               (len(extra_col_name_lists[0]) if extra_col_name_lists[0] is not None else 0) +
               (len(extra_col_name_lists[1]) if extra_col_name_lists[1] is not None else 0)) + _dx
        match_df.iloc[:, ind] = b_concatastro[bc, 2]

    match_df.to_csv(f'{output_folder}/{output_filenames[0]}', encoding='utf-8', index=False, header=False)

    # For non-match, ID/coordinates/mags, then island probability + average
    # contamination.
    af = np.load(f'{input_match_folder}/af.npy')
    a_avg_cont = np.load(f'{input_match_folder}/afieldflux.npy')
    p = np.load(f'{input_match_folder}/pfa.npy')
    seps = np.load(f'{input_match_folder}/afieldseps.npy')
    afeta = np.load(f'{input_match_folder}/afieldeta.npy')
    afxi = np.load(f'{input_match_folder}/afieldxi.npy')
    our_columns = ['MATCH_P', 'NNM_SEPARATION', 'NNM_ETA', 'NNM_XI', f'{extra_col_cat_names[0]}_AVG_CONT']
    cols = np.append(column_name_lists[0], our_columns)
    if extra_col_num_lists[0] is not None:
        cols = np.append(cols, extra_col_name_lists[0])
    if input_npy_folders[0] is not None:
        cols = np.append(cols, [f'{extra_col_cat_names[0]}_FIT_SIG'])
        a_concatastro = np.load(f'{input_npy_folders[0]}/con_cat_astro.npy')
    n_anonmatches = len(af)
    a_nonmatch_df = pd.DataFrame(columns=cols, index=np.arange(0, n_anonmatches))
    for i in column_name_lists[0]:
        a_nonmatch_df.loc[:, i] = cat_a.loc[af, i].values
    a_nonmatch_df.iloc[:, 3+n_amags] = p
    a_nonmatch_df.iloc[:, 3+n_amags+1] = seps
    a_nonmatch_df.iloc[:, 3+n_amags+2] = afeta
    a_nonmatch_df.iloc[:, 3+n_amags+3] = afxi
    a_nonmatch_df.iloc[:, 3+n_amags+4] = a_avg_cont
    if extra_col_name_lists[0] is not None:
        for i in extra_col_name_lists[0]:
            a_nonmatch_df.loc[:, i] = cat_a.loc[af, i].values

    if input_npy_folders[0] is not None:
        ind = (len(column_name_lists[0]) + len(our_columns) +
               (len(extra_col_name_lists[0]) if extra_col_name_lists[0] is not None else 0))
        a_nonmatch_df.iloc[:, ind] = a_concatastro[af, 2]

    a_nonmatch_df.to_csv(f'{output_folder}/{output_filenames[1]}', encoding='utf-8',
                         index=False, header=False)

    bf = np.load(f'{input_match_folder}/bf.npy')
    b_avg_cont = np.load(f'{input_match_folder}/bfieldflux.npy')
    p = np.load(f'{input_match_folder}/pfb.npy')
    seps = np.load(f'{input_match_folder}/bfieldseps.npy')
    bfeta = np.load(f'{input_match_folder}/bfieldeta.npy')
    bfxi = np.load(f'{input_match_folder}/bfieldxi.npy')
    our_columns = ['MATCH_P', 'NNM_SEPARATION', 'NNM_ETA', 'NNM_XI', f'{extra_col_cat_names[1]}_AVG_CONT']
    cols = np.append(column_name_lists[1], our_columns)
    if extra_col_num_lists[1] is not None:
        cols = np.append(cols, extra_col_name_lists[1])
    if input_npy_folders[1] is not None:
        cols = np.append(cols, [f'{extra_col_cat_names[1]}_FIT_SIG'])
        b_concatastro = np.load(f'{input_npy_folders[1]}/con_cat_astro.npy')
    n_bnonmatches = len(bf)
    b_nonmatch_df = pd.DataFrame(columns=cols, index=np.arange(0, n_bnonmatches))
    for i in column_name_lists[1]:
        b_nonmatch_df.loc[:, i] = cat_b.loc[bf, i].values
    b_nonmatch_df.iloc[:, 3+n_bmags] = p
    b_nonmatch_df.iloc[:, 3+n_bmags+1] = seps
    b_nonmatch_df.iloc[:, 3+n_bmags+2] = bfeta
    b_nonmatch_df.iloc[:, 3+n_bmags+3] = bfxi
    b_nonmatch_df.iloc[:, 3+n_bmags+4] = b_avg_cont
    if extra_col_name_lists[1] is not None:
        for i in extra_col_name_lists[1]:
            b_nonmatch_df.loc[:, i] = cat_b.loc[bf, i].values

    if input_npy_folders[1] is not None:
        ind = (len(column_name_lists[1]) + len(our_columns) +
               (len(extra_col_name_lists[1]) if extra_col_name_lists[1] is not None else 0))
        b_nonmatch_df.iloc[:, ind] = b_concatastro[bf, 2]

    b_nonmatch_df.to_csv(f'{output_folder}/{output_filenames[2]}', encoding='utf-8',
                         index=False, header=False)


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
        Name, including extension, of the larger catalogue file.
    output_filename : string
        Name, including extension, of the cutout catalogue.
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
    with open(f'{input_folder}/{input_filename}', encoding='utf-8') as fp:
        n_rows = 0 if not header else -1
        for _ in fp:
            n_rows += 1
    small_astro = open_memmap(f'{input_folder}/temp_astro.npy', mode='w+', dtype=float, shape=(n_rows, 2))

    n = 0
    for chunk in pd.read_csv(f'{input_folder}/{input_filename}', chunksize=100000, usecols=astro_cols,
                             header=None if not header else 0):
        small_astro[n:n+chunk.shape[0]] = chunk.values
        n += chunk.shape[0]

    sky_cut = _load_rectangular_slice(small_astro, rect_coords[0], rect_coords[1], rect_coords[2],
                                      rect_coords[3], padding)

    n_inside_rows = 0
    for cnum in range(0, mem_chunk_num):
        lowind = np.floor(n_rows*cnum/mem_chunk_num).astype(int)
        highind = np.floor(n_rows*(cnum+1)/mem_chunk_num).astype(int)
        n_inside_rows += np.sum(sky_cut[lowind:highind])
    df_orig = pd.read_csv(f'{input_folder}/{input_filename}', nrows=1, header=None if not header else 0)
    df = pd.DataFrame(columns=df_orig.columns, index=np.arange(0, n_inside_rows))

    counter = 0
    outer_counter = 0
    chunksize = 100000
    for chunk in pd.read_csv(f'{input_folder}/{input_filename}', chunksize=chunksize,
                             header=None if not header else 0):
        inside_n = np.sum(sky_cut[outer_counter:outer_counter+chunksize])
        df.iloc[counter:counter+inside_n] = chunk.values[
            sky_cut[outer_counter:outer_counter+chunksize]]
        counter += inside_n
        outer_counter += chunksize

    df.to_csv(f'{output_folder}/{output_filename}', encoding='utf-8', index=False,
              header=False)

    os.remove(f'{input_folder}/temp_astro.npy')


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
    astro = np.load(f'{input_folder}/con_cat_astro.npy', mmap_mode='r')
    photo = np.load(f'{input_folder}/con_cat_photo.npy', mmap_mode='r')
    best_index = np.load(f'{input_folder}/magref.npy', mmap_mode='r')
    n_rows = len(astro)
    sky_cut = _load_rectangular_slice(astro, rect_coords[0], rect_coords[1], rect_coords[2], rect_coords[3],
                                      padding)

    n_inside_rows = 0
    for cnum in range(0, mem_chunk_num):
        lowind = np.floor(n_rows*cnum/mem_chunk_num).astype(int)
        highind = np.floor(n_rows*(cnum+1)/mem_chunk_num).astype(int)
        n_inside_rows += int(np.sum(sky_cut[lowind:highind]))

    small_astro = open_memmap(f'{output_folder}/con_cat_astro.npy', mode='w+', dtype=float,
                              shape=(n_inside_rows, 3))
    small_photo = open_memmap(f'{output_folder}/con_cat_photo.npy', mode='w+', dtype=float,
                              shape=(n_inside_rows, photo.shape[1]))
    small_best_index = open_memmap(f'{output_folder}/magref.npy', mode='w+', dtype=int,
                                   shape=(n_inside_rows,))
    small_chunk_overlap = open_memmap(f'{output_folder}/in_chunk_overlap.npy', mode='w+',
                                      dtype=bool, shape=(n_inside_rows,))

    counter = 0
    for cnum in range(0, mem_chunk_num):
        lowind = np.floor(n_rows*cnum/mem_chunk_num).astype(int)
        highind = np.floor(n_rows*(cnum+1)/mem_chunk_num).astype(int)
        inside_n = np.sum(sky_cut[lowind:highind])
        small_astro[counter:counter+inside_n] = astro[lowind:highind][
            sky_cut[lowind:highind]]
        small_photo[counter:counter+inside_n] = photo[lowind:highind][
            sky_cut[lowind:highind]]
        small_best_index[counter:counter+inside_n] = best_index[lowind:highind][
            sky_cut[lowind:highind]]
        # Always assume that a cutout is a single "visit" with no chunk "halo".
        small_chunk_overlap[counter:counter:inside_n] = False
        counter += inside_n
