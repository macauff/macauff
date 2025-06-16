# Licensed under a 3-clause BSD style license - see LICENSE
'''
This module provides functionality to convert input photometric catalogues
to the required numpy binary files used within the cross-match process.
'''

import os

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy import units as u

from numpy.lib.format import open_memmap

# pylint: disable=import-error,no-name-in-module
from macauff.misc_functions import _load_rectangular_slice
from macauff.misc_functions_fortran import misc_functions_fortran as mff

# pylint: enable=import-error,no-name-in-module

__all__ = ['csv_to_npy', 'rect_slice_npy', 'npy_to_csv', 'rect_slice_csv']


def csv_to_npy(input_filename, astro_cols, photo_cols, bestindex_col,
               chunk_overlap_col, snr_cols=None, header=False, process_uncerts=False,
               astro_sig_fits_filepath=None, cat_in_radec=None, mn_in_radec=None):
    '''
    Convert a .csv file representation of a photometric catalogue into the
    appropriate .npy binary files used in the cross-matching process.

    Parameters
    ----------
    input_filename : string
        Location on disk, including extension, where the catalogue .csv file is
        stored that is to be converted into numpy arrays.
    astro_cols : list or numpy.array of integers
        List of zero-indexed columns in the input catalogue representing the
        three required astrometric parameters, two orthogonal sky axis
        coordinates and a single, circular astrometric precision. The first two
        columns of ``astro_cols`` must match in order the first two columns of
        the output astrometry binary file. For cases where ``process_uncerts``
        is ``True``, the last ``N`` columns must either be a single astrometric
        uncertainty, for when catalogue-given astrometric uncertainties are
        available and were used to parameterise the position-residuals in the
        catalogue, or a list of ``N`` photometric uncertainty bands, in which
        case there will be ``N`` band-based parameterisations of the astrometry
        and the ``bestindex_col`` flags will be used to determine which
        parameterisation to use for each source individually.
    photo_cols : list or numpy.ndarray of integers
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
    snr_cols : list or numpy.ndarray of integers
        List of zero-indexed columns in the input catalogue representing the
        signal-to-noise ratios of each detection corresponding to those same
        magnitudes in ``photo_cols``.
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

    Returns
    -------
    astro : numpy.ndarray
        Three-elements per source, shape ``(N, 3)``, longitude, latitude, and
        (circular) astrometric uncertainty for every object in the catalogue.
    photo : numpy.ndarray
        Photometry of all objects in the catalogue, also length ``N`` in its
        first axis and then ``M`` photometric bands per object.
    best_index : numpy.ndarray
        Indices, ``0``-``M-1``, indicating which of the ``M`` detections is the
        preferred band for every object.
    chunk_overlap : numpy.ndarray
        Boolean flag, indicating whether an object is in the "chunk" or whether
        it has been included in a halo around the primary chunk objects for
        match purposes, but is a primary detection in a different chunk of this
        catalogue.
    snrs : numpy.ndarray, optional
        If ``snr_cols`` are provided, also returns the signal-to-noise ratios for
        each ``photo`` detection for each source.
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
    with open(input_filename, encoding='utf-8') as fp:
        n_rows = 0 if not header else -1
        for _ in fp:
            n_rows += 1

    astro = np.empty((n_rows, 3), float)
    photo = np.empty((n_rows, len(photo_cols)), float)
    best_index = np.empty(n_rows, int)
    chunk_overlap = np.empty(n_rows, bool)
    if snr_cols is not None:
        snrs = np.empty((n_rows, len(photo_cols)), float)

    if process_uncerts:
        mn_sigs = np.load(f'{astro_sig_fits_filepath}/mn_sigs_array.npy')
        mn_coords = np.empty((len(mn_sigs), 2), float)
        # Check the shape of mn_sigs for a third dimension which signals that
        # we need to perform per-band parameterisation.
        if len(mn_sigs.shape) == 3:
            per_band_param = True
            mn_coords[:, 0] = np.copy(mn_sigs[:, 0, 2])
            mn_coords[:, 1] = np.copy(mn_sigs[:, 0, 3])
        else:
            per_band_param = False
            mn_coords[:, 0] = np.copy(mn_sigs[:, 2])
            mn_coords[:, 1] = np.copy(mn_sigs[:, 3])
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
    if snr_cols is not None:
        used_cols = np.concatenate((used_cols, snr_cols))
    used_cols = np.sort(used_cols)
    new_astro_cols = np.array([np.where(used_cols == a)[0][0] for a in astro_cols])
    new_photo_cols = np.array([np.where(used_cols == a)[0][0] for a in photo_cols])
    new_bestindex_col = np.where(used_cols == bestindex_col)[0][0]
    if chunk_overlap_col is not None:
        new_chunk_overlap_col = np.where(used_cols == chunk_overlap_col)[0][0]
    if snr_cols is not None:
        new_snr_cols = np.array([np.where(used_cols == a)[0][0] for a in snr_cols])
    n = 0
    for chunk in pd.read_csv(input_filename, chunksize=100000,
                             usecols=used_cols, header=None if not header else 0):
        best_index[n:n+chunk.shape[0]] = chunk.values[:, new_bestindex_col]
        if not process_uncerts:
            astro[n:n+chunk.shape[0]] = chunk.values[:, new_astro_cols]
        else:
            astro[n:n+chunk.shape[0], 0] = chunk.values[:, new_astro_cols[0]]
            astro[n:n+chunk.shape[0], 1] = chunk.values[:, new_astro_cols[1]]
            sig_mn_inds = mff.find_nearest_point(chunk.values[:, new_astro_cols[0]],
                                                 chunk.values[:, new_astro_cols[1]],
                                                 mn_coords[:, 0], mn_coords[:, 1])
            if not per_band_param:  # pylint: disable=possibly-used-before-assignment
                old_sigs = chunk.values[:, new_astro_cols[2]]
                # pylint: disable-next=possibly-used-before-assignment
                new_sigs = np.sqrt((mn_sigs[sig_mn_inds, 0]*old_sigs)**2 + mn_sigs[sig_mn_inds, 1]**2)
            else:
                new_sigs = np.empty(chunk.shape[0], float)
                for i in range(chunk.shape[0]):
                    old_sig = chunk.values[i, new_astro_cols[2+best_index[i]]]
                    new_sig = np.sqrt((mn_sigs[sig_mn_inds[i], best_index[i], 0]*old_sig)**2 +
                                      mn_sigs[sig_mn_inds[i], best_index[i], 1]**2)
                    new_sigs[i] = new_sig
            astro[n:n+chunk.shape[0], 2] = new_sigs
        photo[n:n+chunk.shape[0]] = chunk.values[:, new_photo_cols]
        if chunk_overlap_col is not None:
            chunk_overlap[n:n+chunk.shape[0]] = chunk.values[:, new_chunk_overlap_col].astype(bool)
        else:
            chunk_overlap[n:n+chunk.shape[0]] = False
        if snr_cols is not None:
            snrs[n:n+chunk.shape[0]] = chunk.values[:, new_snr_cols]

        n += chunk.shape[0]

    if snr_cols is not None:
        return astro, photo, best_index, chunk_overlap, snrs
    return astro, photo, best_index, chunk_overlap


# pylint: disable-next=dangerous-default-value,too-many-locals,too-many-statements
def npy_to_csv(input_csv_file_paths, cm, output_folder, output_filenames, column_name_lists,
               column_num_lists, extra_col_cat_names, correct_astro_flags, headers=[False, False],
               extra_col_name_lists=[None, None], extra_col_num_lists=[None, None], file_extension=''):
    '''
    Function to convert output .npy files, as created during the cross-match
    process, and create a .csv file of matches and non-matches, combining columns
    from the original .csv catalogues.

    Parameters
    ----------
    input_csv_file_paths : list of strings
        List of the locations in which the two respective .csv file catalogues
        are stored, including filename and extension.
    cm : Class
        ``CrossMatch`` class, containing all of the necessary arrays of derived
        information from a cross-match run to save out to files.
    output_folder : string
        Folder into which to save the resulting .csv output files.
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
    correct_astro_flags : list of booleans
        Flags, in the same order as ``input_csv_file_paths``, for whether to
        save uncorrected and corrected astrometric uncertainties for each
        catalogue respectively. If a catalogue did not have its uncertainties
        processed, its entry should be False, so a case where both catalogues in
        a match had their uncertainties treated as given would be
        ``[False, False]``.
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
    file_extension : string, optional
        Additional string to insert into loaded cross-match-specific files (such
        as ``ac.npy``) and into saved files. Defaults to empty string, but should
        be given for cases of "with and without photometry" match runs, where
        a single cross-match run is used to create two separate match tables, and
        hence two separate output sets of .csv files.
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

    ac = getattr(cm, f'ac{file_extension}')
    bc = getattr(cm, f'bc{file_extension}')
    p = getattr(cm, f'pc{file_extension}')
    eta = getattr(cm, f'eta{file_extension}')
    xi = getattr(cm, f'xi{file_extension}')
    a_avg_cont = getattr(cm, f'acontamflux{file_extension}')
    b_avg_cont = getattr(cm, f'bcontamflux{file_extension}')
    acontprob = getattr(cm, f'pacontam{file_extension}')
    bcontprob = getattr(cm, f'pbcontam{file_extension}')
    seps = getattr(cm, f'crptseps{file_extension}')

    if correct_astro_flags[0]:
        cols = np.append(cols, [f'{extra_col_cat_names[0]}_FIT_SIG'])
        a_concatastro = cm.a_astro
    if correct_astro_flags[1]:
        cols = np.append(cols, [f'{extra_col_cat_names[1]}_FIT_SIG'])
        b_concatastro = cm.b_astro

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

    cat_a = pd.read_csv(input_csv_file_paths[0], memory_map=True,
                        header=None if not headers[0] else 0, usecols=a_cols, names=a_names)
    cat_b = pd.read_csv(input_csv_file_paths[1], memory_map=True,
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
    if correct_astro_flags[0]:
        ind = (len(column_name_lists[0]) + len(column_name_lists[1]) + len(our_columns) +
               (len(extra_col_name_lists[0]) if extra_col_name_lists[0] is not None else 0) +
               (len(extra_col_name_lists[1]) if extra_col_name_lists[1] is not None else 0))
        match_df.iloc[:, ind] = a_concatastro[ac, 2]
    if correct_astro_flags[1]:
        # Here we also need to check if catalogue "a" has processed uncertainties too.
        _dx = 1 if correct_astro_flags[0] else 0
        ind = (len(column_name_lists[0]) + len(column_name_lists[1]) + len(our_columns) +
               (len(extra_col_name_lists[0]) if extra_col_name_lists[0] is not None else 0) +
               (len(extra_col_name_lists[1]) if extra_col_name_lists[1] is not None else 0)) + _dx
        match_df.iloc[:, ind] = b_concatastro[bc, 2]

    if file_extension == '':
        _output_filename = output_filenames[0]
    else:
        # Insert the file_extension keyword into the middle of
        # /path/to/file/foo.bar.
        f, f_ext = os.path.splitext(output_filenames[0])
        _output_filename = f + file_extension + f_ext
    match_df.to_csv(f'{output_folder}/{_output_filename}', encoding='utf-8', index=False, header=False)

    # For non-match, ID/coordinates/mags, then island probability + average
    # contamination.
    af = getattr(cm, f'af{file_extension}')
    a_avg_cont = getattr(cm, f'afieldflux{file_extension}')
    p = getattr(cm, f'pfa{file_extension}')
    seps = getattr(cm, f'afieldseps{file_extension}')
    afeta = getattr(cm, f'afieldeta{file_extension}')
    afxi = getattr(cm, f'afieldxi{file_extension}')
    our_columns = ['MATCH_P', 'NNM_SEPARATION', 'NNM_ETA', 'NNM_XI', f'{extra_col_cat_names[0]}_AVG_CONT']
    cols = np.append(column_name_lists[0], our_columns)
    if extra_col_num_lists[0] is not None:
        cols = np.append(cols, extra_col_name_lists[0])
    if correct_astro_flags[0]:
        cols = np.append(cols, [f'{extra_col_cat_names[0]}_FIT_SIG'])
        a_concatastro = cm.a_astro
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

    if correct_astro_flags[0]:
        ind = (len(column_name_lists[0]) + len(our_columns) +
               (len(extra_col_name_lists[0]) if extra_col_name_lists[0] is not None else 0))
        a_nonmatch_df.iloc[:, ind] = a_concatastro[af, 2]

    if file_extension == '':
        _output_filename = output_filenames[1]
    else:
        # Insert the file_extension keyword into the middle of
        # /path/to/file/foo.bar.
        f, f_ext = os.path.splitext(output_filenames[1])
        _output_filename = f + file_extension + f_ext
    a_nonmatch_df.to_csv(f'{output_folder}/{_output_filename}', encoding='utf-8',
                         index=False, header=False)

    bf = getattr(cm, f'bf{file_extension}')
    b_avg_cont = getattr(cm, f'bfieldflux{file_extension}')
    p = getattr(cm, f'pfb{file_extension}')
    seps = getattr(cm, f'bfieldseps{file_extension}')
    bfeta = getattr(cm, f'bfieldeta{file_extension}')
    bfxi = getattr(cm, f'bfieldxi{file_extension}')
    our_columns = ['MATCH_P', 'NNM_SEPARATION', 'NNM_ETA', 'NNM_XI', f'{extra_col_cat_names[1]}_AVG_CONT']
    cols = np.append(column_name_lists[1], our_columns)
    if extra_col_num_lists[1] is not None:
        cols = np.append(cols, extra_col_name_lists[1])
    if correct_astro_flags[1]:
        cols = np.append(cols, [f'{extra_col_cat_names[1]}_FIT_SIG'])
        b_concatastro = cm.b_astro
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

    if correct_astro_flags[1]:
        ind = (len(column_name_lists[1]) + len(our_columns) +
               (len(extra_col_name_lists[1]) if extra_col_name_lists[1] is not None else 0))
        b_nonmatch_df.iloc[:, ind] = b_concatastro[bf, 2]

    if file_extension == '':
        _output_filename = output_filenames[2]
    else:
        # Insert the file_extension keyword into the middle of
        # /path/to/file/foo.bar.
        f, f_ext = os.path.splitext(output_filenames[2])
        _output_filename = f + file_extension + f_ext
    b_nonmatch_df.to_csv(f'{output_folder}/{_output_filename}', encoding='utf-8',
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
    small_astro = np.empty((n_rows, 2), float)

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


def apply_proper_motion(lon, lat, pm_lon, pm_lat, ref_epoch, move_to_epoch, coord_system):
    '''
    Functionality to apply proper motions to a dataset with measured on-sky
    stellar drift.

    Parameters
    ----------
    lon : numpy.ndarray of floats
        Longitude coordinate of each object. Should either be Right Ascension
        or galactic longitude, depending on ``coord_system``.
    lat: numpy.ndarray of floats
        Declination or galactic latitude (depending on ``coord_system``) of
        each object to be moved to a new epoch.
    pm_lon : numpy.ndarray of floats
        Longitudinal drift of each object, corresponding element by element
        to ``lon`` and ``lat``. Must be in units of arcseconds per year, and
        account for the latitudinal projection effect (e.g., the "cos dec"
        effect) already.
    pm_lat : numpy.ndarray of floats
        The latitudinal drift of the objects, in arcseconds per year.
    ref_epoch : numpy.ndarray of strings or string
        The date, or dates, of all observations. Can either be a single value
        or an array/list of epochs, element-wise with ``lon`` et al. Must be
        accepted by ``SkyCoord`` as a valid format, such as ``JXXXX.YYY` or
        ``YYYY-MM-DD``.
    move_to_epoch : string
        The new date to which all sources should have their positions moved to.
        Must be a valid astropy ``Time`` format, expecting either ``JXXXX.YYY``
        or ``YYYY-MM-DD`.
    coord_system : string
        String to determine which coordinate system the data are in, either
        ``equatorial``, in which case the ICRS frame is used internally, or
        ``galactic``, where the galactic coordinate system will be applied.

    Returns
    -------
    new_lon : numpy.ndarray of floats
        The new longitudinal coordinates of each object in ``move_to_epoch``
        dates.
    new_lat : numpy.ndarray of floats
        Latitude (Dec or b) of objects at the new epoch.
    '''
    if coord_system == 'galactic':
        c = SkyCoord(l=lon * u.degree, b=lat * u.degree, frame='galactic', distance=1e9*u.kpc,
                     pm_l_cosb=pm_lon * u.arcsecond / u.year, pm_b=pm_lat * u.arcsecond / u.year,
                     obstime=ref_epoch)
    else:
        c = SkyCoord(ra=lon * u.degree, dec=lat * u.degree, frame='icrs', distance=1e9*u.kpc,
                     pm_ra_cosdec=pm_lon * u.arcsecond / u.year, pm_dec=pm_lat * u.arcsecond / u.year,
                     obstime=ref_epoch)

    d = c.apply_space_motion(new_obstime=Time(move_to_epoch))

    if coord_system == 'galactic':
        new_lon, new_lat = d.l.degree, d.b.degree
    else:
        new_lon, new_lat = d.ra.degree, d.dec.degree

    return new_lon, new_lat
