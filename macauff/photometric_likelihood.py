# Licensed under a 3-clause BSD style license - see LICENSE
'''
This module provides the framework for the creation of the photometric likelihoods
used in the cross-matching of the two catalogues.
'''

import sys
import numpy as np

from .misc_functions import map_large_index_to_small_index, StageData
from .misc_functions_fortran import misc_functions_fortran as mff
from .photometric_likelihood_fortran import photometric_likelihood_fortran as plf

__all__ = ['compute_photometric_likelihoods']


def compute_photometric_likelihoods(joint_folder_path, a_cat_folder_path, b_cat_folder_path,
                                    afilts, bfilts, cf_points, cf_areas, include_phot_like,
                                    use_phot_priors, group_sources_data, bright_frac=None,
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
    group_sources_data : class.StageData
        Object containing all outputs from ``make_island_groupings``
        TODO Improve description
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

    len_a = len(np.load('{}/con_cat_astro.npy'.format(a_cat_folder_path)))
    len_b = len(np.load('{}/con_cat_astro.npy'.format(b_cat_folder_path)))

    print("Distributing sources into sky slices...")
    sys.stdout.flush()

    a_astro = np.load('{}/con_cat_astro.npy'.format(a_cat_folder_path))
    b_astro = np.load('{}/con_cat_astro.npy'.format(b_cat_folder_path))
    a_photo = np.load('{}/con_cat_photo.npy'.format(a_cat_folder_path))
    b_photo = np.load('{}/con_cat_photo.npy'.format(b_cat_folder_path))

    a_sky_inds = mff.find_nearest_point(a_astro[:, 0], a_astro[:, 1], cf_points[:, 0],
                                        cf_points[:, 1])
    b_sky_inds = mff.find_nearest_point(b_astro[:, 0], b_astro[:, 1], cf_points[:, 0],
                                        cf_points[:, 1])

    print("Making bins...")
    sys.stdout.flush()

    abinlengths, abinsarray, longabinlen = create_magnitude_bins(
        cf_points, afilts, a_photo, joint_folder_path, a_cat_folder_path, 'a', a_sky_inds,
        include_phot_like or use_phot_priors, group_sources_data.ablen,
        group_sources_data.ainds, group_sources_data.asize)
    bbinlengths, bbinsarray, longbbinlen = create_magnitude_bins(
        cf_points, bfilts, b_photo, joint_folder_path, b_cat_folder_path, 'b', b_sky_inds,
        include_phot_like or use_phot_priors, group_sources_data.bblen,
        group_sources_data.binds, group_sources_data.bsize)

    print("Calculating PDFs...")
    sys.stdout.flush()

    c_priors = np.zeros(dtype=float, shape=(len(bfilts), len(afilts), len(cf_points)), order='F')
    fa_priors = np.zeros(dtype=float, shape=(len(bfilts), len(afilts), len(cf_points)), order='F')
    fb_priors = np.zeros(dtype=float, shape=(len(bfilts), len(afilts), len(cf_points)), order='F')
    c_array = np.zeros(dtype=float, shape=(longbbinlen-1, longabinlen-1, len(bfilts), len(afilts),
                       len(cf_points)), order='F')
    fa_array = np.zeros(dtype=float, shape=(longabinlen-1, len(bfilts), len(afilts),
                                            len(cf_points)), order='F')
    fb_array = np.zeros(dtype=float, shape=(longbbinlen-1, len(bfilts), len(afilts),
                                            len(cf_points)), order='F')

    ablen = group_sources_data.ablen
    ainds = group_sources_data.ainds
    binds = group_sources_data.binds
    asize = group_sources_data.asize
    bsize = group_sources_data.bsize

    for m in range(0, len(cf_points)):
        area = cf_areas[m]
        a_sky_cut = a_sky_inds == m
        a_photo_cut = a_photo[a_sky_cut]
        if include_phot_like or use_phot_priors:
            a_astro_cut, a_blen_cut, a_inds_cut, a_size_cut = (
                a_astro[a_sky_cut], ablen[a_sky_cut], ainds[:, a_sky_cut], asize[a_sky_cut])

        b_sky_cut = b_sky_inds == m
        b_photo_cut = b_photo[b_sky_cut]
        if include_phot_like or use_phot_priors:
            b_astro_cut, b_inds_cut, b_size_cut = (
                b_astro[b_sky_cut], binds[:, b_sky_cut], bsize[b_sky_cut])

            # Return the overall to subarray mapping of *_inds_cut, as well
            # as the unique values of *_inds_cut, for use in creating the
            # opposing view slices into the other catalogue, for each
            # catalogue. Note that the lengths are swapped in the two calls,
            # as the a_inds map into length of catalogue "b" and vice versa.
            a_inds_map, a_inds_cut_unique = map_large_index_to_small_index(a_inds_cut, len_b)
            b_inds_map, b_inds_cut_unique = map_large_index_to_small_index(b_inds_cut, len_a)

            # Combined with b_inds_map, this subarray of the catalogue gives
            # an array that has every "a" source that overlaps the "b"
            # subarray objects. Note we use b_inds_cut_unique for catalogue
            # "a" sources.
            a_astro_ind = a_astro[b_inds_cut_unique]
            b_astro_ind = b_astro[a_inds_cut_unique]
            a_photo_ind = a_photo[b_inds_cut_unique]
            b_photo_ind = b_photo[a_inds_cut_unique]

            a_flen_ind = group_sources_data.aflen[b_inds_cut_unique]
            b_flen_ind = group_sources_data.bflen[a_inds_cut_unique]

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
                    c_like, fa_like, fb_like = (1-1e-10)**2, 1-1e-10, 1-1e-10
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
                    c_like, fa_like, fb_like = (1-1e-10)**2, 1-1e-10, 1-1e-10

                # Have to add a very small "fire extinguisher" value to all
                # likelihoods and priors, to avoid ever having exactly zero
                # value in either, which would mean all island permutations
                # were rejected.
                c_priors[j, i, m] = c_prior + 1e-100
                fa_priors[j, i, m] = fa_prior + 1e-10
                fb_priors[j, i, m] = fb_prior + 1e-10
                # c should be equal to f^2 in the limit of indifference, so
                # add 1e-10 and (1e-10)^2 to f and c respectively.
                c_array[:bbinlengths[j, m]-1,
                        :abinlengths[i, m]-1, j, i, m] = c_like + 1e-100
                fa_array[:abinlengths[i, m]-1, j, i, m] = fa_like + 1e-10
                fb_array[:bbinlengths[j, m]-1, j, i, m] = fb_like + 1e-10

    phot_like_data = StageData(abinsarray=abinsarray, abinlengths=abinlengths,
                               bbinsarray=bbinsarray, bbinlengths=bbinlengths,
                               a_sky_inds=a_sky_inds, b_sky_inds=b_sky_inds,
                               c_priors=c_priors, c_array=c_array,
                               fa_priors=fa_priors, fa_array=fa_array,
                               fb_priors=fb_priors, fb_array=fb_array)
    return phot_like_data


def create_magnitude_bins(cf_points, filts, a_photo, joint_folder_path,
                          cat_folder_path, cat_type, sky_inds, load_extra_arrays,
                          blen_cutout, inds_cutout, size_cutout):
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
    a_photo : numpy.ndarray
        Photometric detections from which to create magnitude bins.
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
    blen_cutout : numpy.ndarray
        Output from ``group_sources``. Used only when ``load_extra_arrays`` is True.
    inds_cutout : numpy.ndarray
        Output from ``group_sources``. Used only when ``load_extra_arrays`` is True.
    size_cutout : numpy.ndarray
        Output from ``group_sources``. Used only when ``load_extra_arrays`` is True.

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

    for m in range(0, len(cf_points)):
        sky_cut = sky_inds == m
        for i in range(0, len(filts)):
            a = a_photo[sky_cut, i]
            if np.sum(~np.isnan(a)) > 0:
                f = make_bins(a[~np.isnan(a)])
            else:
                f = np.array([0])
            del a
            binlengths[i, m] = len(f)

    longbinlen = np.amax(binlengths)

    binsarray = np.full(dtype=float, shape=(longbinlen, len(filts), len(cf_points)), fill_value=-1, order='F')
    for m in range(0, len(cf_points)):
        sky_cut = sky_inds == m
        for i in range(0, len(filts)):
            a = a_photo[sky_cut, i]
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
    na = int(np.ceil((maxa - mina)/da) + 1)
    output_bins = np.linspace(mina, maxa, na)
    # If min/max magnitudes that define magnitude bins happen to lie exactly
    # on a bin edge (i.e., maxamag % da == 0), then just pad bin edge slightly.
    if np.abs(mina - minamag) < 1e-5:
        output_bins[0] -= 1e-4
    if np.abs(maxa - maxamag) < 1e-5:
        output_bins[-1] += 1e-4

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

    a_cuts = plf.find_mag_bin_inds(a_mag, a_flags, a_bins)

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
