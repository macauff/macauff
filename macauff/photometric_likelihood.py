# Licensed under a 3-clause BSD style license - see LICENSE
'''
This module provides the framework for the creation of the photometric likelihoods
used in the cross-matching of the two catalogues.
'''

import os
import sys
import numpy as np

from .misc_functions import map_large_index_to_small_index, _load_single_sky_slice
from .misc_functions_fortran import misc_functions_fortran as mff
from .photometric_likelihood_fortran import photometric_likelihood_fortran as plf

__all__ = ['compute_photometric_likelihoods']


def compute_photometric_likelihoods(joint_folder_path, a_cat_folder_path, b_cat_folder_path,
                                    afilts, bfilts, mem_chunk_num, cf_points, cf_areas,
                                    include_phot_like, use_phot_priors, bright_frac=None,
                                    field_frac=None):
    '''
    Derives the photometric likelihoods and priors for use in the catalogue
    cross-match process.

    Parameters
    ----------
    joint_folder_path : string
        The folder where all folders and files created during the cross-match
        process are stored.
    a_cat_folder_path : string
        The folder where the input data for catalogue "a" are located.
    b_cat_folder_path : string
        The location of catalogue "b"'s input data.
    afilts : list of string
        A list of the filters in catalogue "a"'s photometric data file.
    bfilts : list of string
        List of catalogue "b"'s filters.
    mem_chunk_num : integer
        Fraction of input datasets to load at once, in the case of data larger
        than the memory of the system.
    cf_points : numpy.ndarray
        The on-sky coordinates that define the locations of each small
        set of sources to be used to derive the relative match and non-match
        photometric likelihoods.
    cf_areas : numpy.ndarray
        The areas of closest on-sky separation surrounding each point in
        ``cf_points``, used to normalise numbers of sources to sky densities.
    include_phot_like : boolean
        Flag to indicate whether to derive astrophysical likelihoods ``c`` and
        ``f``, based on the common coevality of sources of given magnitudes.
    use_phot_priors : boolean
        Indicator as to whether to use astrophysical priors, based on the common
        number of likely matches and non-matches in each ``cf_points`` area, or
        use naive, asymmetric priors solely based on number density of sources.
    bright_frac : float, optional
        Expected fraction of sources inside the "bright" error circles used to
        construct the counterpart distribution, to correct for missing numbers.
        If ``include_phot_like`` or ``use_phot_prior`` is True then this must
        be supplied, otherwise it can be omitted.
    field_frac : float, optional
        Expected fraction of sources inside the "field" error circles used to
        construct the counterpart distribution, to correct for missing numbers.
        If ``include_phot_like`` or ``use_phot_prior`` is True then this must
        be supplied, otherwise it can be omitted.
    '''

    if bright_frac is None and (include_phot_like or use_phot_priors):
        raise ValueError("bright_frac must be supplied if include_phot_like or use_phot_priors "
                         "is set to True. Please supply an appropriate fraction.")
    if field_frac is None and (include_phot_like or use_phot_priors):
        raise ValueError("field_frac must be supplied if include_phot_like or use_phot_priors "
                         "is set to True. Please supply an appropriate fraction.")

    print("Creating c(m, m) and f(m)...")

    len_a = len(np.load('{}/con_cat_astro.npy'.format(a_cat_folder_path), mmap_mode='r'))
    len_b = len(np.load('{}/con_cat_astro.npy'.format(b_cat_folder_path), mmap_mode='r'))

    print("Distributing sources into sky slices...")
    sys.stdout.flush()

    a_sky_inds = distribute_sky_indices(joint_folder_path, a_cat_folder_path, 'a', mem_chunk_num,
                                        cf_points)
    b_sky_inds = distribute_sky_indices(joint_folder_path, b_cat_folder_path, 'b', mem_chunk_num,
                                        cf_points)

    print("Making bins...")
    sys.stdout.flush()

    abinlengths, abinsarray, longabinlen = create_magnitude_bins(
        cf_points, afilts, mem_chunk_num, joint_folder_path, a_cat_folder_path, 'a', a_sky_inds,
        include_phot_like or use_phot_priors)
    bbinlengths, bbinsarray, longbbinlen = create_magnitude_bins(
        cf_points, bfilts, mem_chunk_num, joint_folder_path, b_cat_folder_path, 'b', b_sky_inds,
        include_phot_like or use_phot_priors)

    print("Calculating PDFs...")
    sys.stdout.flush()

    c_priors = np.lib.format.open_memmap(
        '{}/phot_like/c_priors.npy'.format(joint_folder_path), mode='w+', dtype=float,
        shape=(len(bfilts), len(afilts), len(cf_points)), fortran_order=True)
    fa_priors = np.lib.format.open_memmap(
        '{}/phot_like/fa_priors.npy'.format(joint_folder_path), mode='w+', dtype=float,
        shape=(len(bfilts), len(afilts), len(cf_points)), fortran_order=True)
    fb_priors = np.lib.format.open_memmap(
        '{}/phot_like/fb_priors.npy'.format(joint_folder_path), mode='w+', dtype=float,
        shape=(len(bfilts), len(afilts), len(cf_points)), fortran_order=True)
    c_array = np.lib.format.open_memmap(
        '{}/phot_like/c_array.npy'.format(joint_folder_path), mode='w+', dtype=float,
        shape=(longbbinlen-1, longabinlen-1, len(bfilts), len(afilts),
               len(cf_points)), fortran_order=True)
    fa_array = np.lib.format.open_memmap(
        '{}/phot_like/fa_array.npy'.format(joint_folder_path), mode='w+', dtype=float,
        shape=(longabinlen-1, len(bfilts), len(afilts), len(cf_points)), fortran_order=True)
    fb_array = np.lib.format.open_memmap(
        '{}/phot_like/fb_array.npy'.format(joint_folder_path), mode='w+', dtype=float,
        shape=(longbbinlen-1, len(bfilts), len(afilts), len(cf_points)), fortran_order=True)

    # Within each loop, since we've already assigned all sources to have a closest
    # c/f sky position pointing to be assigned to, we simply slice within the _load
    # functions on ID, and for the initial mem_chunk_num stepping load in the range
    # [m_, m_+mem_chunk_num).
    for m_ in range(0, len(cf_points), mem_chunk_num):
        a_multi_return = _load_multiple_sky_slice(joint_folder_path, 'a', m_, m_+mem_chunk_num,
                                                  a_cat_folder_path, a_sky_inds,
                                                  include_phot_like or use_phot_priors)
        if include_phot_like or use_phot_priors:
            (a_small_photo, a_sky_ind_small, a_small_astro, a_blen_small, a_inds_small,
             a_size_small) = a_multi_return
        else:
            a_small_photo, a_sky_ind_small = a_multi_return
        del a_multi_return

        b_multi_return = _load_multiple_sky_slice(joint_folder_path, 'b', m_, m_+mem_chunk_num,
                                                  b_cat_folder_path, b_sky_inds,
                                                  include_phot_like or use_phot_priors)
        if include_phot_like or use_phot_priors:
            (b_small_photo, b_sky_ind_small, b_small_astro, b_blen_small, b_inds_small,
             b_size_small) = b_multi_return
            del b_blen_small
        else:
            b_small_photo, b_sky_ind_small = b_multi_return
        del b_multi_return

        for m in range(m_, min(len(cf_points), m_+mem_chunk_num)):
            area = cf_areas[m]
            a_sky_cut = _load_single_sky_slice(
                joint_folder_path, 'a', m, a_sky_ind_small)
            a_photo_cut = a_small_photo[a_sky_cut]
            if include_phot_like or use_phot_priors:
                a_astro_cut, a_blen_cut, a_inds_cut, a_size_cut = (
                    a_small_astro[a_sky_cut], a_blen_small[a_sky_cut], a_inds_small[:, a_sky_cut],
                    a_size_small[a_sky_cut])

            b_sky_cut = _load_single_sky_slice(
                joint_folder_path, 'b', m, b_sky_ind_small)
            b_photo_cut = b_small_photo[b_sky_cut]
            if include_phot_like or use_phot_priors:
                b_astro_cut, b_inds_cut, b_size_cut = (
                    b_small_astro[b_sky_cut], b_inds_small[:, b_sky_cut], b_size_small[b_sky_cut])

                # Return the overall to subarray mapping of *_inds_cut, as well
                # as the unique values of *_inds_cut, for use in creating the
                # opposing view slices into the other catalogue, for each
                # catalogue. Note that the lengths are swapped in the two calls,
                # as the a_inds map into length of catalogue "b" and vice versa.
                a_inds_map, a_inds_cut_unique = map_large_index_to_small_index(
                    a_inds_cut, len_b, '{}/phot_like'.format(joint_folder_path))
                b_inds_map, b_inds_cut_unique = map_large_index_to_small_index(
                    b_inds_cut, len_a, '{}/phot_like'.format(joint_folder_path))

                # Combined with b_inds_map, this subarray of the catalogue gives
                # an array that has every "a" source that overlaps the "b"
                # subarray objects. Note we use b_inds_cut_unique for catalogue
                # "a" sources.
                a_astro_ind = np.load('{}/con_cat_astro.npy'.format(a_cat_folder_path),
                                      mmap_mode='r')[b_inds_cut_unique]
                b_astro_ind = np.load('{}/con_cat_astro.npy'.format(b_cat_folder_path),
                                      mmap_mode='r')[a_inds_cut_unique]
                a_photo_ind = np.load('{}/con_cat_photo.npy'.format(a_cat_folder_path),
                                      mmap_mode='r')[b_inds_cut_unique]
                b_photo_ind = np.load('{}/con_cat_photo.npy'.format(b_cat_folder_path),
                                      mmap_mode='r')[a_inds_cut_unique]
                a_flen_ind = np.load('{}/group/aflen.npy'.format(joint_folder_path),
                                     mmap_mode='r')[b_inds_cut_unique]
                b_flen_ind = np.load('{}/group/bflen.npy'.format(joint_folder_path),
                                     mmap_mode='r')[a_inds_cut_unique]

            for i in range(0, len(afilts)):
                if not include_phot_like and not use_phot_priors:
                    a_num_photo_cut = np.sum(~np.isnan(a_photo_cut[:, i]))
                    Na = a_num_photo_cut / area
                else:
                    a_bins = abinsarray[:abinlengths[i, m], i, m]
                    a_mag = a_photo_cut[:, i]
                    a_flags = ~np.isnan(a_mag)
                    a_flags_ind = ~np.isnan(a_photo_ind[:, i])
                for j in range(0, len(bfilts)):
                    if not include_phot_like and not use_phot_priors:
                        b_num_photo_cut = np.sum(~np.isnan(b_photo_cut[:, j]))
                        Nb = b_num_photo_cut / area
                        # Without using photometric-based priors, all we can
                        # do is set the prior on one catalogue to 0.5 -- that
                        # is, equal chance of match or non-match; for this we
                        # use the less dense of the two catalogues as our
                        # "one-sided" match. Then, accordingly, we update the
                        # "field" source density of the more dense catalogue
                        # with its corresponding density, based on the input
                        # density and the counterpart density calculated.
                        c_prior = min(Na, Nb) / 2
                        fa_prior = Na - c_prior
                        fb_prior = Nb - c_prior
                        # To fake no photometric likelihoods, simply set all
                        # values to one, to cancel in the ratio later.
                        c_like, fa_like, fb_like = 1, 1, 1
                    else:
                        b_bins = bbinsarray[:bbinlengths[j, m], j, m]
                        b_mag = b_photo_cut[:, j]
                        b_flags = ~np.isnan(b_mag)
                        b_mag_ind = b_photo_ind[:, j]
                        b_flags_ind = ~np.isnan(b_mag_ind)

                        c_prior, c_like, fa_prior, fa_like, fb_prior, fb_like = create_c_and_f(
                            a_astro_cut, b_astro_cut, a_mag, b_mag, a_inds_map, a_size_cut,
                            b_inds_map, b_size_cut, a_blen_cut, a_bins, b_bins, bright_frac,
                            field_frac, a_flags, b_flags, a_astro_ind, b_astro_ind, a_flen_ind,
                            b_flen_ind, a_flags_ind, b_flags_ind, b_mag_ind, area)
                    if use_phot_priors and not include_phot_like:
                        # If we only used the create_c_and_f routine to derive
                        # priors, then quickly update likelihoods here.
                        c_like, fa_like, fb_like = 1, 1, 1

                    c_priors[j, i, m] = c_prior
                    fa_priors[j, i, m] = fa_prior
                    fb_priors[j, i, m] = fb_prior
                    c_array[:bbinlengths[j, m]-1,
                            :abinlengths[i, m]-1, j, i, m] = c_like
                    fa_array[:abinlengths[i, m]-1, j, i, m] = fa_like
                    fb_array[:bbinlengths[j, m]-1, j, i, m] = fb_like

    os.system('rm {}/a_small_sky_slice.npy'.format(joint_folder_path))
    os.system('rm {}/b_small_sky_slice.npy'.format(joint_folder_path))

    # *binsarray is passed back from create_magnitude_bins as a memmapped array,
    # but *binlengths is just a numpy array, so quickly save these before returning.
    np.save('{}/phot_like/abinlengths.npy'.format(joint_folder_path), abinlengths)
    np.save('{}/phot_like/bbinlengths.npy'.format(joint_folder_path), bbinlengths)

    return


def distribute_sky_indices(joint_folder_path, cat_folder, name, mem_chunk_num, cf_points):
    '''
    Function to calculate the nearest on-sky photometric likelihood point for
    each catalogue source.

    Parameters
    ----------
    joint_folder_path : string
        Top-level folder path for the common files created during the cross-match
        process.
    cat_folder : string
        The location of a given catalogue's input data files.
    name : string
        Representation of whether we are calculating sky indices for catalogue
        "a", or catalogue "b".
    mem_chunk_num : integer
        Number of sub-sets to break larger data files down to, to preserve memory.
    cf_points : numpy.ndarray
        The two-point sky coordinates for each point to be used as a central
        point of a small sky area, for calculating "counterpart" and "field"
        photometric likelihoods.

    Returns
    -------
    sky_inds : numpy.ndarray
        The indices, matching ``cf_points``, of the closest sky position for each
        source in this catalogue.
    '''
    n_sources = len(np.load('{}/con_cat_astro.npy'.format(cat_folder), mmap_mode='r'))
    sky_inds = np.lib.format.open_memmap('{}/phot_like/{}_sky_inds.npy'.format(
                                         joint_folder_path, name), mode='w+', dtype=int,
                                         shape=(n_sources,), fortran_order=True)
    for cnum in range(0, mem_chunk_num):
        lowind = np.floor(n_sources*cnum/mem_chunk_num).astype(int)
        highind = np.floor(n_sources*(cnum+1)/mem_chunk_num).astype(int)
        a = np.load('{}/con_cat_astro.npy'.format(cat_folder), mmap_mode='r')[lowind:highind]
        sky_inds[lowind:highind] = mff.find_nearest_point(a[:, 0], a[:, 1],
                                                          cf_points[:, 0], cf_points[:, 1])

    return sky_inds


def create_magnitude_bins(cf_points, filts, mem_chunk_num, joint_folder_path,
                          cat_folder_path, cat_type, sky_inds, load_extra_arrays):
    '''
    Creates the N-dimensional arrays of single-band photometric bins, and
    corresponding array lengths.

    Parameters
    ----------
    cf_points : numpy.ndarray
        List of the two-dimensional on-sky coordinates defining the centers
        of each cutout for which the photometric likelihoods should be
        calculated.
    filts : list of strings
        List of the filters to create magnitude bins for in this catalogue.
    mem_chunk_num : integer
        Number of sub-sets to break larger catalogues down into, for memory
        saving purposes.
    joint_folder_path : string
        Location of top-level folder into which all intermediate files are
        saved for the cross-match process.
    cat_folder_path : string
        Location of the input data for this catalogue.
    cat_type : string
        String to indicate which catalogue we are creating bins for, either
        "a", or "b".
    sky_inds : numpy.ndarray
        Array of indices, showing which on-sky photometric point, from
        ``cf_points``, each source in the catalogue is closest to.
    load_extra_arrays : boolean
        Flag to indicate whether the photometric information is being used in
        the cross-match process, and whether to load additional arrays accordingly.

    Returns
    -------
    binlengths : numpy.ndarray
        Two-dimensional array, indicating the length of the magnitude bins for
        each filter-sky coordinate combination.
    binsarray : numpy.ndarray
        Three-dimensional array, containing the values of the magnitude bins for
        the filter-sky combinations.
    longbinlen : integer
        Value of the largest of all filter-sky combinations of ``binlengths``.
    '''
    binlengths = np.empty((len(filts), len(cf_points)), int)

    for m_ in range(0, len(cf_points), mem_chunk_num):
        a_multi_return = _load_multiple_sky_slice(
            joint_folder_path, cat_type, m_, m_+mem_chunk_num, cat_folder_path, sky_inds,
            load_extra_arrays)
        if load_extra_arrays:
            (a_phot_, sky_inds_, _, _, _, _) = a_multi_return
        else:
            a_phot_, sky_inds_ = a_multi_return
        del a_multi_return
        for m in range(m_, min(len(cf_points), m_+mem_chunk_num)):
            sky_cut = _load_single_sky_slice(
                joint_folder_path, cat_type, m, sky_inds_)
            for i in range(0, len(filts)):
                a = a_phot_[sky_cut, i]
                if np.sum(~np.isnan(a)) > 0:
                    f = make_bins(a[~np.isnan(a)])
                else:
                    f = np.array([0])
                del a
                binlengths[i, m] = len(f)

    longbinlen = np.amax(binlengths)
    binsarray = np.lib.format.open_memmap(
        '{}/phot_like/{}binsarray.npy'.format(joint_folder_path, cat_type), mode='w+', dtype=float,
        shape=(longbinlen, len(filts), len(cf_points)), fortran_order=True)
    binsarray[:, :, :] = -1
    for m_ in range(0, len(cf_points), mem_chunk_num):
        a_multi_return = _load_multiple_sky_slice(
            joint_folder_path, cat_type, m_, m_+mem_chunk_num, cat_folder_path, sky_inds,
            load_extra_arrays)
        if load_extra_arrays:
            (a_phot_, sky_inds_, _, _, _, _) = a_multi_return
        else:
            a_phot_, sky_inds_ = a_multi_return
        for m in range(m_, min(len(cf_points), m_+mem_chunk_num)):
            sky_cut = _load_single_sky_slice(
                joint_folder_path, cat_type, m, sky_inds_)
            for i in range(0, len(filts)):
                a = a_phot_[sky_cut, i]
                if np.sum(~np.isnan(a)) > 0:
                    f = make_bins(a[~np.isnan(a)])
                else:
                    f = np.array([0])
                del a
                binsarray[:binlengths[i, m], i, m] = f

    return binlengths, binsarray, longbinlen


def make_bins(input_mags):
    '''
    Calculate bins for a catalogue's magnitude distribution, ensuring all stars
    are in histogram bins of sufficient number statistics.

    Parameters
    ----------
    input_mags : numpy.ndarray
        Array of magnitudes of given filter from the specific catalogue, to be
        placed in a histogram.

    Returns
    -------
    output_bins : numpy.ndarray
        Bins for the given catalogue-filter combination that produce robust
        numbers of sources within each magnitude interval.
    '''
    minamag = np.amin(input_mags)
    maxamag = np.amax(input_mags)
    da = 0.1
    maxa = da*np.ceil(maxamag/da)
    mina = da*np.floor(minamag/da)
    # If min/max magnitudes that define magnitude bins happen to lie exactly
    # on a bin edge (i.e., maxamag % da == 0), then just pad bin edge slightly.
    if np.abs(mina - minamag) < 1e-5:
        mina -= 1e-4
    if np.abs(maxa - maxamag) < 1e-5:
        maxa += 1e-4
    na = int(np.ceil((maxa - mina)/da) + 1)
    output_bins = np.linspace(mina, maxa, na)

    hist, output_bins = np.histogram(input_mags, bins=output_bins)
    smalllist = []
    # Minimum number statistics in each 1-D bin.
    minnum = 250

    for i in range(0, len(output_bins)-1):
        if hist[i] < minnum:
            smalllist.extend([i])
    smalllist = np.array(smalllist)
    dellist = []
    if len(smalllist) > 0:
        for i in smalllist:
            if i not in dellist:
                flag = 0
                for j in range(i+1, len(output_bins)-1):
                    if np.sum(hist[i:j+1]) > minnum:
                        dellist.extend([k for k in range(i+1, j+1)])
                        flag = 1
                        break
                if flag == 0:
                    dellist.extend([k for k in range(i+1, len(output_bins)-1)])
    output_bins = np.delete(output_bins, dellist)

    return output_bins


def _load_multiple_sky_slice(joint_folder_path, cat_name, ind1, ind2, cat_folder_path, sky_inds,
                             load_extra_arrays):
    '''
    Function to, in a memmap-friendly way, return a sub-set of the photometry
    of a given catalogue.

    Parameters
    ----------
    joint_folder_path : string
        Folder in which common cross-match intermediate data files are stored.
    cat_name : string
        String defining whether this function was called on catalogue "a" or "b".
    ind1 : float
        The lower of the two sky indices, as defined in ``distribute_sky_indices``,
        to return a sub-set of the larger catalogue between. This value represents
        the index of a given on-sky position, used to construct the "counterpart"
        and "field" likelihoods.
    ind2 : float
        The upper of the sky indices, defining the sub-set of the photometric
        array to return.
    cat_folder_path : string
        The folder defining where this particular catalogue is stored.
    sky_inds : numpy.ndarray
        The given catalogue's ``distribute_sky_indices`` values, to compare
        with ``ind1`` and ``ind2``.
    load_extra_arrays : boolean
        Flag to indicate whether the photometric information is being used in
        the cross-match process, and whether to load additional arrays accordingly.

    Returns
    -------
    photo_cutout : numpy.ndarray
        A sub-set of the photometry of the given catalogue, those points which are
        astrometrically closest to the sky indices between ``ind1`` and ``ind2``.
    sky_ind_cutout : numpy.ndarray
        The reduced ``sky_inds`` array, containing only those between ``ind1`` and
        ``ind2``.
    list_of_arrays : list of numpy.ndarrays
        Depending on whether ``load_extra_arrays`` is ``True`` or not, this list
        contains either just a cutout of the photometric data for this catalogue
        and the corresponding subset of the sky index array, or it also contains
        subsets of the astrometry array, and "bright" source error circle length,
        and overlap index and size arrays.
    '''
    sky_cut = np.lib.format.open_memmap('{}/{}_temporary_sky_slice_combined.npy'.format(
        joint_folder_path, cat_name), mode='w+', dtype=bool, shape=(len(sky_inds),))

    di = max(1, len(sky_inds) // 20)

    for i in range(0, len(sky_inds), di):
        sky_cut[i:i+di] = (sky_inds[i:i+di] >= ind1) & (sky_inds[i:i+di] < ind2)

    if load_extra_arrays:
        astro_cutout = np.load('{}/con_cat_astro.npy'.format(cat_folder_path),
                               mmap_mode='r')[sky_cut]
        a_blen_cutout = np.load('{}/group/{}blen.npy'.format(joint_folder_path, cat_name),
                                mmap_mode='r')[sky_cut]
        a_inds_cutout = np.load('{}/group/{}inds.npy'.format(joint_folder_path, cat_name),
                                mmap_mode='r')[:, sky_cut]
        a_size_cutout = np.load('{}/group/{}size.npy'.format(joint_folder_path, cat_name),
                                mmap_mode='r')[sky_cut]

    photo_cutout = np.load('{}/con_cat_photo.npy'.format(cat_folder_path), mmap_mode='r')[sky_cut]
    sky_ind_cutout = np.load('{}/phot_like/{}_sky_inds.npy'.format(joint_folder_path, cat_name),
                             mmap_mode='r')[sky_cut]

    if load_extra_arrays:
        list_of_arrays = (photo_cutout, sky_ind_cutout, astro_cutout, a_blen_cutout, a_inds_cutout,
                          a_size_cutout)
    else:
        list_of_arrays = (photo_cutout, sky_ind_cutout)

    os.system('rm {}/{}_temporary_sky_slice_combined.npy'.format(joint_folder_path, cat_name))

    return list_of_arrays


def create_c_and_f(a_astro, b_astro, a_mag, b_mag, a_inds, a_size, b_inds, b_size, a_blen,
                   a_bins, b_bins, bright_frac, field_frac, a_flags, b_flags, a_astro_ind,
                   b_astro_ind, a_flen_ind, b_flen_ind, a_flags_ind, b_flags_ind, b_mag_ind, area):
    '''
    Functionality to create the photometric likelihood and priors from a set
    of photometric data in a given pair of filters.

    Parameters
    ----------
    a_astro : numpy.ndarray
        Array of astrometric parameters for all catalogue "a" sources in this
        given sky slice.
    b_astro : numpy.ndarray
        Astrometric parameters for small sky region catalogue "b" objects.
    a_mag : numpy.ndarray
        Catalogue "a" magnitudes for sky area.
    b_mag : numpy.ndarray
        Catalogue "b" magnitudes for sources in sky region.
    a_inds : numpy.ndarray
        Indices into catalogue "b" for each "a" object, indicating potential
        overlaps in counterparts.
    a_size : numpy.ndarray
        The number of potential overlapping sources in catalogue "b", for each
        catalogue "a" object.
    b_inds : numpy.ndarray
        Overlap indices into catalogue "a" for each catalogue "b" object.
    b_size : numpy.ndarray
        Number of overlaps from catalogue "b" into catalogue "a".
    a_blen : numpy.ndarray
        The "bright" error circle radius, integrating the joint AUF convolution
        out to ``expected_frac`` for largest "a"-"b" potential pairing, for each
        catalogue "a" object.
    a_bins : numpy.ndarray
        Array containing the magnitude bins into which to place catalogue "a"
        sources.
    b_bins : numpy.ndarray
        Array containing the magnitude bins into which to place catalogue "b"
        sources.
    bright_frac : float
        Fraction of total probability integral to consider potential counterparts
        out to, when considering potential overlaps between catalogue-catalogue
        pairings.
    field_frac : float
        Fraction of total probability integral out to which sources are removed
        from consideration as "field" sources (i.e., they are not assumed to be
        ruled out as potential counterparts), when considering potential overlaps
        in pairs of objects between the two catalogues.
    a_flags : numpy.ndarray
        Boolean flags for whether a source in catalogue "a" has a detected
        magnitude in ``a_mag``.
    b_flags : numpy.ndarray
        Detection flags for catalogue "b" sources in ``b_mag``.
    a_astro_ind : numpy.ndarray
        Astrometric information for all catalogue "a" sources with at least one
        overlap with catalogue "b" sources from ``b_astro``.
    b_astro_ind : numpy.ndarray
        Astrometric information for all catalogue "b" sources with an overlap
        with at least one catalogue "a" source in ``a_astro``.
    a_flen_ind : numpy.ndarray
        Largest joint AUF integral error circle radius for the "field" source
        integral fraction, for each catalogue "a" object in ``a_astro_ind``.
    b_flen_ind : numpy.ndarray
        Maximum AUF integral distance for each source in ``b_astro_ind``.
    a_flags_ind : numpy.ndarray
        Boolean flags for catalogue "a" sources which have an overlap with any
        catalogue "b" objects that are in this sky region.
    b_flags_ind : numpy.ndarray
        Boolean detection flags for all catalogue "b" sources for which any
        sources in catalogue "a" have a potential overlap in this sky region.
    b_mag_ind : numpy.ndarray
        All source magnitudes for which the catalogue "a" subset of sources
        have an overlap in catalogue "b".
    area : float
        Area of sky region for which photometric likelihood and prior are
        being calculated, in square degrees.

    Returns
    -------
    Nc : float
        The prior density of counterpart sources between the catalogues.
    cdmdm : numpy.ndarray
        Two-dimensional array of the photometric likelihood of counterpart between
        the two catalogues.
    Nfa : float
        So-called "field" source density in catalogue "a".
    fa : numpy.ndarray
        Probability density array of field sources for catalogue "a".
    Nfb : float
        Field source density prior for catalogue "b".
    fb : numpy.ndarray
        Field source PDF for catalogue "b".
    '''
    a_hist, a_bins = np.histogram(a_mag[a_flags], bins=a_bins)
    pa = a_hist/(np.sum(a_hist)*np.diff(a_bins))

    a_cuts = np.zeros((len(a_bins)-1, len(a_astro)), bool)
    for i in range(0, len(a_astro)):
        if a_flags[i]:
            q = np.where(a_mag[i] >= a_bins[:-1])[0][-1]
            a_cuts[q, i] = 1

    b_cuts = np.zeros((len(b_bins)-1, len(b_astro)), bool)
    for i in range(0, len(b_astro)):
        if b_flags[i]:
            q = np.where(b_mag[i] >= b_bins[:-1])[0][-1]
            b_cuts[q, i] = 1

    # get_field_dists allows for magnitude slicing, to get f(m | m) instead of f(m),
    # but when we do want f(m) we just pass two impossible magnitudes as the limits.
    a_mask, a_area = plf.get_field_dists(a_astro[:, 0], a_astro[:, 1], b_astro_ind[:, 0],
                                         b_astro_ind[:, 1], a_inds, a_size, b_flen_ind, a_flags,
                                         b_flags_ind, b_mag, -999, 999)
    b_mask, b_area = plf.get_field_dists(b_astro[:, 0], b_astro[:, 1], a_astro_ind[:, 0],
                                         a_astro_ind[:, 1], b_inds, b_size, a_flen_ind, b_flags,
                                         a_flags_ind, a_mag, -999, 999)
    a_mask = a_mask.astype(bool)
    b_mask = b_mask.astype(bool)
    a_left = a_mag[a_mask]
    b_left = b_mag[b_mask]
    hist, a_bins = np.histogram(a_left, bins=a_bins)
    Num_fa = np.sum(a_mask)

    fa = hist / (np.sum(hist)*np.diff(a_bins))

    hist, b_bins = np.histogram(b_left, bins=b_bins)
    Num_fb = np.sum(b_mask)

    fb = hist / (np.sum(hist)*np.diff(b_bins))

    Nfa = Num_fa/(area - a_area)
    Nfb = Num_fb/(area - b_area)

    bm = np.empty((len(b_bins)-1, len(a_bins)-1), float, order='F')
    z = np.empty(len(a_bins)-1, float)

    mag_mask, aa = plf.brightest_mag(a_astro[:, 0], a_astro[:, 1], b_astro_ind[:, 0],
                                     b_astro_ind[:, 1], a_mag, b_mag_ind, a_inds, a_size, a_blen,
                                     a_flags, b_flags_ind, a_bins)
    mag_mask = mag_mask.astype(bool)
    for i in range(0, len(a_bins)-1):
        hist, b_bins = np.histogram(b_mag_ind[mag_mask[:, i]], bins=b_bins)
        q = np.sum(mag_mask[:, i])
        if q > 0:
            bm[:, i] = hist/(np.diff(b_bins)*q)
        else:
            bm[:, i] = 0
        z[i] = np.sum(hist)/np.sum(a_cuts[i])
    cdmdm = np.empty((len(b_bins)-1, len(a_bins)-1), float, order='F')
    for i in range(0, len(a_bins)-1):
        bmask, barea = plf.get_field_dists(
            b_astro[:, 0], b_astro[:, 1], a_astro[:, 0], a_astro[:, 1], b_inds, b_size, a_flen_ind,
            b_flags, a_flags_ind, a_mag, a_bins[i], a_bins[i+1])
        bmask = bmask.astype(bool)
        b_left = b_mag[bmask]
        hist, b_bins = np.histogram(b_left, bins=b_bins)
        _Num_fb = np.sum(b_mask)

        _fb = hist / (np.sum(hist)*np.diff(b_bins))
        _Nfb = _Num_fb/(area - barea)
        Fm = np.append(0, np.cumsum(_fb[:-1] * np.diff(b_bins[:-1])))
        for j in range(0, len(b_bins)-1):
            Cm = np.sum(cdmdm[:j, i]*np.diff(b_bins[:j+1]))
            cdmdm[j, i] = max(0, z[i]*bm[j, i]*np.exp(aa[i]*_Nfb*Fm[j]) - (1-Cm)*aa[i]*_Nfb*_fb[j])

    zc = np.sum(cdmdm*np.diff(b_bins).reshape(-1, 1), axis=0)
    frac = zc/bright_frac
    density_of_inputs = np.sum(a_cuts, axis=1)/area
    Nc = np.sum(frac*density_of_inputs)

    integral = 0
    for i in range(0, len(a_bins)-1):
        cdmdm[:, i] *= pa[i] / (a_bins[i+1] - a_bins[i])
        integral = integral + np.sum(cdmdm[:, i]*np.diff(b_bins))*(a_bins[i+1] - a_bins[i])
    if integral > 0:
        cdmdm /= integral

    # Correct the field priors for the fraction of counterparts that get left
    # in their "cutout" circle, by the fact that we don't use the entire integral:
    Nfa = Nfa - (1 - field_frac)*Nc
    Nfb = Nfb - (1 - field_frac)*Nc

    return Nc, cdmdm, Nfa, fa, Nfb, fb
