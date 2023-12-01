# Licensed under a 3-clause BSD style license - see LICENSE
'''
This module provides the framework for grouping sources from two photometric catalogues into
distinct "islands" of sources, along with calculating whether they are within overlap for
various photometric integral purposes.
'''

import sys
import multiprocessing
import itertools
import numpy as np

from macauff.misc_functions import (load_small_ref_auf_grid, hav_dist_constant_lat,
                             _load_rectangular_slice, StageData)
from macauff.group_sources_fortran import group_sources_fortran as gsf
from macauff.make_set_list import set_list

__all__ = ['make_island_groupings']


def make_island_groupings(joint_folder_path, a_cat_folder_path, b_cat_folder_path,
                          a_auf_pointings, b_auf_pointings, a_filt_names, b_filt_names, a_title,
                          b_title, a_modelrefinds, b_modelrefinds, r, dr, rho, drho, j1s, max_sep,
                          ax_lims, int_fracs, include_phot_like, use_phot_priors,
                          n_pool, a_perturb_auf_outputs, b_perturb_auf_outputs):
    '''
    Function to handle the creation of "islands" of astrometrically coeval
    sources, and identify which overlap to some probability based on their
    combined AUFs.

    Parameters
    ----------
    joint_folder_path : string
        Folder on disk containing the files related to the cross-match between
        the two catalogues.
    a_cat_folder_path : string
        Folder on disk where catalogue "a" files have been stored.
    b_cat_folder_path : string
        Folder on disk where catalogue "b" files are saved.
    a_auf_pointings : 2-D numpy.ndarray
        Array containing the listings of longitude, latitude pointings at which
        the perturbation AUF components were computed for catalogue "a".
    b_auf_pointings : 2-D numpy.ndarray
        Array containing the listings of longitude, latitude pointings at which
        the perturbation AUF components were computed for catalogue "b".
    a_filt_names : list of string
        List of ordered names for filters used in catalogue "a" cross-match.
    b_filt_names : list of string
        List of filters in catalogue "b" matching.
    a_title : string
        Name used to describe catalogue "a" in the cross-match.
    b_title : string
        Catalogue "b" description, for identifying its given folder.
    a_modelrefinds : numpy.ndarray
        Catalogue "a" modelrefinds array output from ``create_perturb_auf``.
        TODO Improve description
    b_modelrefinds : numpy.ndarray
        Catalogue "b" modelrefinds array output from ``create_perturb_auf``.
        TODO Improve description
    r : numpy.ndarray
        Array of real-space distances, in arcseconds, used in the evaluation of
        convolved AUF integrals; represent bin edges.
    dr : numpy.ndarray
        Widths of real-space bins in ``r``. Will have shape one shorter than ``r``,
        due to ``r`` requiring an additional right-hand bin edge.
    rho : numpy.ndarray
        Fourier-space array, used in handling the Hankel transformation for
        convolution of AUFs. As with ``r``, represents bin edges.
    drho : numpy.ndarray
        Array representing the bin widths of ``rho``. As with ``dr``, is one
        shorter than ``rho`` due to its additional bin edge.
    j1s : 2-D numpy.ndarray
        Array holding the evaluations of the Bessel Function of First kind of
        First Order, evaluated at all ``r`` and ``rho`` bin-middle combination.
    max_sep : float
        The maximum allowed sky separation between two sources in opposing
        catalogues for consideration as potential counterparts.
    ax_lims : list of floats, or numpy.ndarray
        The four limits of the cross-match between catalogues "a" and "b",
        as lower and upper longitudinal coordinate, lower and upper latitudinal
        coordinates respectively.
    int_fracs : list of floats, or numpy.ndarray
        List of integral limits used in evaluating probability of match based on
        separation distance.
    include_phot_like : boolean
        Flag indicating whether to perform additional computations required for
        the future calculation of photometric likelihoods.
    use_phot_priors : boolean
        Flag indicating whether to calcualte additional parameters needed to
        calculate photometric-information dependent priors for cross-matching.
    n_pool : integer
        Number of multiprocessing pools to use when parallelising.
    a_perturb_auf_outputs : dictionary
        Dict containing the results from the previous step of the cross-match,
        the simulations of the perturbation component of catalogue a's AUF.
    b_perturb_auf_outputs : dictionary
        Dict containing the results from the previous step of the cross-match,
        the simulations of the perturbation component of catalogue b's AUF.
    '''

    # Convert from arcseconds to degrees internally.
    max_sep = np.copy(max_sep) / 3600
    print("Creating catalogue islands and overlaps...")
    sys.stdout.flush()

    print("Calculating maximum overlap...")
    sys.stdout.flush()

    # The initial step to create island groupings is to find the largest number
    # of overlaps for a single source, to minimise the size of the array of
    # overlap indices. To do so, we load small-ish chunks of the sky, with
    # padding in one catalogue to ensure all pairings can be found, and total
    # the number of overlaps for each object across all sky slices.

    ax1_skip, ax2_skip = 8, 8
    ax1_loops = np.linspace(ax_lims[0], ax_lims[1], 41)
    # Force the sub-division of the sky area in question to be 1600 chunks, or
    # roughly quarter square degree chunks, whichever is larger in area.
    if ax1_loops[1] - ax1_loops[0] < 0.25:
        ax1_loops = np.linspace(ax_lims[0], ax_lims[1],
                                int(np.ceil((ax_lims[1] - ax_lims[0])/0.25) + 1))
    ax2_loops = np.linspace(ax_lims[2], ax_lims[3], 41)
    if ax2_loops[1] - ax2_loops[0] < 0.25:
        ax2_loops = np.linspace(ax_lims[2], ax_lims[3],
                                int(np.ceil((ax_lims[3] - ax_lims[2])/0.25) + 1))
    ax1_sparse_loops = ax1_loops[::ax1_skip]
    if ax1_sparse_loops[-1] != ax1_loops[-1]:
        ax1_sparse_loops = np.append(ax1_sparse_loops, ax1_loops[-1])
    ax2_sparse_loops = ax2_loops[::ax2_skip]
    if ax2_sparse_loops[-1] != ax2_loops[-1]:
        ax2_sparse_loops = np.append(ax2_sparse_loops, ax2_loops[-1])

    # Load the astrometry of each catalogue for slicing.
    a_full = np.load('{}/con_cat_astro.npy'.format(a_cat_folder_path))
    b_full = np.load('{}/con_cat_astro.npy'.format(b_cat_folder_path))

    asize = np.zeros(dtype=int, shape=(len(a_full),))
    bsize = np.zeros(dtype=int, shape=(len(b_full),))

    for i, (ax1_sparse_start, ax1_sparse_end) in enumerate(zip(ax1_sparse_loops[:-1],
                                                               ax1_sparse_loops[1:])):
        for j, (ax2_sparse_start, ax2_sparse_end) in enumerate(zip(ax2_sparse_loops[:-1],
                                                                   ax2_sparse_loops[1:])):
            a_big_sky_cut = _load_rectangular_slice('', a_full, ax1_sparse_start, ax1_sparse_end,
                                                    ax2_sparse_start, ax2_sparse_end, 0)
            b_big_sky_cut = _load_rectangular_slice('', b_full, ax1_sparse_start, ax1_sparse_end,
                                                    ax2_sparse_start, ax2_sparse_end, max_sep)
            a_cutout = a_full[a_big_sky_cut]
            b_cutout = b_full[b_big_sky_cut]

            a_sky_inds = np.arange(0, len(a_full))[a_big_sky_cut]
            b_sky_inds = np.arange(0, len(b_full))[b_big_sky_cut]
            for ax1_start, ax1_end in zip(ax1_loops[i*ax1_skip:(i+1)*ax1_skip],
                                          ax1_loops[i*ax1_skip+1:(i+1)*ax1_skip+1]):
                for ax2_start, ax2_end in zip(ax2_loops[j*ax2_skip:(j+1)*ax2_skip],
                                              ax2_loops[j*ax2_skip+1:(j+1)*ax2_skip+1]):
                    ax_cutout = [ax1_start, ax1_end, ax2_start, ax2_end]
                    a, afouriergrid, amodrefindsmall, a_cut = _load_fourier_grid_cutouts(
                        a_cutout, ax_cutout, joint_folder_path, a_cat_folder_path,
                        a_perturb_auf_outputs, 0, 'a', a_big_sky_cut, a_modelrefinds)
                    b, bfouriergrid, bmodrefindsmall, b_cut = _load_fourier_grid_cutouts(
                        b_cutout, ax_cutout, joint_folder_path, b_cat_folder_path,
                        b_perturb_auf_outputs, max_sep, 'b', b_big_sky_cut, b_modelrefinds)

                    if len(a) > 0 and len(b) > 0:
                        overlapa, overlapb = gsf.get_max_overlap(
                            a[:, 0], a[:, 1], b[:, 0], b[:, 1], max_sep, a[:, 2], b[:, 2],
                            r[:-1]+dr/2, rho[:-1], drho, j1s, afouriergrid, bfouriergrid,
                            amodrefindsmall, bmodrefindsmall, int_fracs[2])
                        a_cut2 = a_sky_inds[a_cut]
                        b_cut2 = b_sky_inds[b_cut]

                        asize[a_cut2] = asize[a_cut2] + overlapa
                        bsize[b_cut2] = bsize[b_cut2] + overlapb

    amaxsize = int(np.amax(asize))
    bmaxsize = int(np.amax(bsize))
    del (overlapa, overlapb, a, b, a_cut, b_cut, amodrefindsmall, bmodrefindsmall,
         afouriergrid, bfouriergrid)

    print("Truncating star overlaps by AUF integral...")
    sys.stdout.flush()

    ainds = np.zeros(dtype=int, shape=(amaxsize, len(a_full)), order='F')
    binds = np.zeros(dtype=int, shape=(bmaxsize, len(b_full)), order='F')

    ainds[:, :] = -1
    binds[:, :] = -1
    asize[:] = 0
    bsize[:] = 0

    for i, (ax1_sparse_start, ax1_sparse_end) in enumerate(zip(ax1_sparse_loops[:-1],
                                                               ax1_sparse_loops[1:])):
        for j, (ax2_sparse_start, ax2_sparse_end) in enumerate(zip(ax2_sparse_loops[:-1],
                                                                   ax2_sparse_loops[1:])):
            a_big_sky_cut = _load_rectangular_slice('', a_full, ax1_sparse_start, ax1_sparse_end,
                                                    ax2_sparse_start, ax2_sparse_end, 0)
            b_big_sky_cut = _load_rectangular_slice('', b_full, ax1_sparse_start, ax1_sparse_end,
                                                    ax2_sparse_start, ax2_sparse_end, max_sep)
            a_cutout = a_full[a_big_sky_cut]
            b_cutout = b_full[b_big_sky_cut]

            a_sky_inds = np.arange(0, len(a_full))[a_big_sky_cut]
            b_sky_inds = np.arange(0, len(b_full))[b_big_sky_cut]
            for ax1_start, ax1_end in zip(ax1_loops[i*ax1_skip:(i+1)*ax1_skip],
                                          ax1_loops[i*ax1_skip+1:(i+1)*ax1_skip+1]):
                for ax2_start, ax2_end in zip(ax2_loops[j*ax2_skip:(j+1)*ax2_skip],
                                              ax2_loops[j*ax2_skip+1:(j+1)*ax2_skip+1]):
                    ax_cutout = [ax1_start, ax1_end, ax2_start, ax2_end]
                    a, afouriergrid, amodrefindsmall, a_cut = _load_fourier_grid_cutouts(
                        a_cutout, ax_cutout, joint_folder_path, a_cat_folder_path,
                        a_perturb_auf_outputs, 0, 'a', a_big_sky_cut, a_modelrefinds)
                    b, bfouriergrid, bmodrefindsmall, b_cut = _load_fourier_grid_cutouts(
                        b_cutout, ax_cutout, joint_folder_path, b_cat_folder_path,
                        b_perturb_auf_outputs, max_sep, 'b', b_big_sky_cut, b_modelrefinds)

                    if len(a) > 0 and len(b) > 0:
                        indicesa, indicesb, overlapa, overlapb = gsf.get_overlap_indices(
                            a[:, 0], a[:, 1], b[:, 0], b[:, 1], max_sep, amaxsize, bmaxsize,
                            a[:, 2], b[:, 2], r[:-1]+dr/2, rho[:-1], drho, j1s, afouriergrid,
                            bfouriergrid, amodrefindsmall, bmodrefindsmall, int_fracs[2])

                        a_cut2 = a_sky_inds[a_cut]
                        b_cut2 = b_sky_inds[b_cut]

                        for k in range(0, len(a_cut2)):
                            ainds[asize[a_cut2[k]]:asize[a_cut2[k]]+overlapa[k], a_cut2[k]] = \
                                b_cut2[indicesa[:overlapa[k], k] - 1]
                        for k in range(0, len(b_cut2)):
                            binds[bsize[b_cut2[k]]:bsize[b_cut2[k]]+overlapb[k], b_cut2[k]] = \
                                a_cut2[indicesb[:overlapb[k], k] - 1]

                        asize[a_cut2] = asize[a_cut2] + overlapa
                        bsize[b_cut2] = bsize[b_cut2] + overlapb

    del (a_cut, a_cut2, b_cut, b_cut2, indicesa, indicesb, overlapa, overlapb, a, b,
         amodrefindsmall, bmodrefindsmall, afouriergrid, bfouriergrid)

    print("Cleaning overlaps...")
    sys.stdout.flush()

    ainds, asize = _clean_overlaps(ainds, asize, n_pool)
    binds, bsize = _clean_overlaps(binds, bsize, n_pool)

    print("Calculating integral lengths...")
    sys.stdout.flush()

    if include_phot_like or use_phot_priors:
        a_err = a_full[:, 2]

        b_err = b_full[:, 2]

        a_fouriergrid = a_perturb_auf_outputs['fourier_grid']
        b_fouriergrid = b_perturb_auf_outputs['fourier_grid']

        a_int_lens = gsf.get_integral_length(
            a_err, b_err, r[:-1]+dr/2, rho[:-1], drho, j1s, a_fouriergrid, b_fouriergrid,
            a_modelrefinds, b_modelrefinds, ainds, asize, int_fracs[0:2])
        ablen = a_int_lens[:, 0]
        aflen = a_int_lens[:, 1]

        b_int_lens = gsf.get_integral_length(
            b_err, a_err, r[:-1]+dr/2, rho[:-1], drho, j1s, b_fouriergrid, a_fouriergrid,
            b_modelrefinds, a_modelrefinds, binds, bsize, int_fracs[0:2])
        bblen = b_int_lens[:, 0]
        bflen = b_int_lens[:, 1]

    print("Maximum overlaps are:", amaxsize, bmaxsize)
    sys.stdout.flush()

    print("Finding unique sets...")
    sys.stdout.flush()

    set_list_items = set_list(ainds, binds, asize, bsize, joint_folder_path, n_pool)
    if len(set_list_items) == 6:
        alist, blist, agrplen, bgrplen, areject, breject = set_list_items
        reject_flag = True
    else:
        alist, blist, agrplen, bgrplen = set_list_items
        reject_flag = False

    # The final act of creating island groups is to clear out any sources too
    # close to the edge of the catalogue -- defined by its rectangular extend.
    # TODO: add flag for allowing the keeping of potentially incomplete islands
    # in the main catalogue; here we default to, and only allow, their removal.

    passed_check = np.zeros(dtype=bool, shape=(alist.shape[1],))
    failed_check = np.ones(dtype=bool, shape=(alist.shape[1],))

    num_a_failed_checks = 0
    num_b_failed_checks = 0

    # Here, since we know no source can be outside of extent, we can simply
    # look at whether any source has a sky separation of less than max_sep
    # from any of the four lines defining extent in orthogonal sky axes.
    counter = np.arange(0, alist.shape[1])
    expand_constants = [itertools.repeat(item) for item in [
        a_full, b_full, alist, blist, agrplen, bgrplen, ax_lims, max_sep]]
    iter_group = zip(counter, *expand_constants)
    # Initialise the multiprocessing loop setup:
    pool = multiprocessing.Pool(n_pool)
    for return_items in pool.imap_unordered(_distance_check, iter_group,
                                            chunksize=max(1, len(counter) // n_pool)):
        i, dist_check, a, b = return_items
        if dist_check:
            passed_check[i] = 1
            failed_check[i] = 0
        else:
            # While "good" islands just need their total number incrementing
            # for the group, "failed" islands we need to track the number of
            # sources in each catalogue for.
            num_a_failed_checks += len(a)
            num_b_failed_checks += len(b)

    pool.close()
    pool.join()

    # If set_list returned any rejected sources, then add any sources too close
    # to match extent to those now. Ensure that we only reject the unique source IDs
    # across each island group, ignoring the "default" -1 index.
    if reject_flag:
        a_first_rejected_len = len(areject)
    else:
        a_first_rejected_len = 0
    if num_a_failed_checks + a_first_rejected_len > 0:
        reject_a = np.zeros(dtype=int, shape=(num_a_failed_checks+a_first_rejected_len,))
    if reject_flag:
        reject_a[num_a_failed_checks:] = areject
    if reject_flag:
        b_first_rejected_len = len(breject)
    else:
        b_first_rejected_len = 0
    if num_b_failed_checks + b_first_rejected_len > 0:
        reject_b = np.zeros(dtype=int, shape=(num_b_failed_checks+b_first_rejected_len,))
    if reject_flag:
        reject_b[num_b_failed_checks:] = breject

    if reject_flag:
        alist_reject = alist[:, failed_check]
        reject_a[:num_a_failed_checks] = alist_reject[alist_reject > -1]
        blist_reject = blist[:, failed_check]
        reject_b[:num_b_failed_checks] = blist_reject[blist_reject > -1]
    else:
        reject_a = alist[:, failed_check]
        reject_a = reject_a[reject_a > -1]
        reject_b = blist[:, failed_check]
        reject_b = reject_b[reject_b > -1]

    # This should basically be alist = alist[:, passed_check] and
    # agrplen = agrplen[passed_check], simply removing those above failed
    # islands from the list, analagous to the same functionality in set_list.
    alist = alist[:, passed_check]
    agrplen = agrplen[passed_check]
    blist = blist[:, passed_check]
    bgrplen = bgrplen[passed_check]
    # Only return aflen and bflen if they were created
    if not (include_phot_like or use_phot_priors):
        ablen = bblen = None
        aflen = bflen = None
    # Only return reject counts if they were created
    if num_a_failed_checks + a_first_rejected_len > 0:
        lenrejecta = len(reject_a)
        # Save rejects output files.
        np.save('{}/reject/reject_a.npy'.format(joint_folder_path), reject_a)
    else:
        lenrejecta = 0
    if num_b_failed_checks + b_first_rejected_len > 0:
        lenrejectb = len(reject_b)
        np.save('{}/reject/reject_b.npy'.format(joint_folder_path), reject_b)
    else:
        lenrejectb = 0

    group_sources_data = StageData(ablen=ablen, bblen=bblen, ainds=ainds, binds=binds,
                                   asize=asize, bsize=bsize, aflen=aflen, bflen=bflen,
                                   alist=alist, blist=blist, agrplen=agrplen, bgrplen=bgrplen,
                                   lenrejecta=lenrejecta, lenrejectb=lenrejectb)
    return group_sources_data


def _load_fourier_grid_cutouts(a, sky_rect_coords, joint_folder_path, cat_folder_path,
                               perturb_auf_outputs, padding, cat_name, large_sky_slice,
                               modelrefinds):
    '''
    Function to load a sub-set of a given catalogue's astrometry, slicing it
    in a given sky coordinate rectangle, and load the appropriate sub-array
    of the perturbation AUF's fourier-space PDF.

    Parameters
    ----------
    a : numpy.ndarray
        Array containing the full entries for a given catalogue.
    sky_rect_coords : numpy.ndarray or list
        Array with the rectangular extents of the cutout to be performed, in the
        order lower longitudinal coordinate, upper longitudinal coordinate,
        lower latitudinal coordinate, and upper latitudinal coordinate.
    joint_folder_path : string
        Folder on disk indicating where to store files related to the joint
        cross-match being performed.
    cat_folder_path : string
        Location on disk where catalogues for the same dataset given in ``a``
        are stored.
    perturb_auf_outputs : dictionary
        Results from the simulations of catalogue ``a``'s AUF extensions.
    padding : float
        Maximum allowed sky separation the "wrong" side of ``sky_rect_coords``,
        allowing for an increase in sky box size which ensures that all overlaps
        get caught in ``get_max_overlap`` and ``get_max_indices``.
    cat_name : string
        String indicating whether we are loading cutouts from catalogue "a" or
        "b".
    large_sky_slice : boolean
        Slice array containing the ``True`` and ``False`` elements of which
        elements of the full catalogue, in ``con_cat_astro.npy``, are in ``a``.
    modelrefinds : numpy.ndarray
        The modelrefinds array output from ``create_perturb_auf``.
        TODO Improve description
    '''

    lon1, lon2, lat1, lat2 = sky_rect_coords

    sky_cut = _load_rectangular_slice(cat_name, a, lon1, lon2, lat1, lat2, padding)

    a_cutout = np.load('{}/con_cat_astro.npy'.format(cat_folder_path))[large_sky_slice][sky_cut]

    modrefind = modelrefinds[:, large_sky_slice][:, sky_cut]

    [fouriergrid], modrefindsmall = load_small_ref_auf_grid(modrefind, perturb_auf_outputs,
                                                            ['fourier'])

    return a_cutout, fouriergrid, modrefindsmall, sky_cut


def _clean_overlaps(inds, size, n_pool):
    '''
    Convenience function to parse either catalogue's indices array for
    duplicate references to the opposing array on a per-source basis,
    and filter duplications.

    Parameters
    ----------
    inds : numpy.ndarray
        Array containing the indices of overlap between this catalogue, for each
        source, and the opposing catalogue, including potential duplication.
    size : numpy.ndarray
        Array containing the number of overlaps between this catalogue and the
        opposing catalogue prior to duplication removal.
    n_pool : integer
        Number of multiprocessing threads to use.

    Returns
    -------
    inds : numpy.ndarray
        The unique indices of overlap into the opposing catalogue for each
        source in a given catalogue, stripped of potential duplicates.
    size : numpy.ndarray
        Newly updated ``size`` array, containing the lengths of the unique
        indices of overlap into the opposing catalogue for each source.
    '''
    maxsize = 0
    size[:] = 0
    pool = multiprocessing.Pool(n_pool)
    counter = np.arange(0, inds.shape[1])
    iter_group = zip(counter, itertools.repeat(inds))
    for return_items in pool.imap_unordered(_calc_unique_inds, iter_group,
                                            chunksize=max(1, len(counter) // n_pool)):
        i, unique_inds = return_items
        y = len(unique_inds)
        inds[:y, i] = unique_inds
        inds[y:, i] = -1
        if y > maxsize:
            maxsize = y
        size[i] = y

    pool.close()
    pool.join()

    inds = np.asfortranarray(inds[:maxsize, :])

    return inds, size


def _calc_unique_inds(iterable):
    i, inds = iterable
    return i, np.unique(inds[inds[:, i] > -1, i])


def _distance_check(iterable):
    i, a_, b_, alist_1, blist_1, agrplen_small, bgrplen_small, ax_lims, max_sep = iterable
    subset = alist_1[:agrplen_small[i], i]
    a = a_[subset]
    subset = blist_1[:bgrplen_small[i], i]
    b = b_[subset]
    meets_min_distance = np.zeros(len(a)+len(b), bool)
    # Do not check for longitudinal "extent" small separations for cases
    # where all 0-360 degrees are included, as this will result in no loss
    # of sources from consideration, with the 0->360 wraparound of
    # coordinates. In either case if there is a small slice of sky not
    # considered, however, we must remove sources near the "empty" strip.
    # Likely redundant, explicitly check for either limits being inside of
    # 0-360 exactly by the top edge being <360 but also the bottom edge
    # being positive or negative -- i.e., not exactly zero.
    if ax_lims[0] > 0 or ax_lims[0] < 0 or ax_lims[1] < 360:
        for lon in ax_lims[:2]:
            # The Haversine formula doesn't care if lon < 0 or if lon ~ 360,
            # so no need to consider ax_lims that straddle 0 longitude here.
            is_within_dist_of_lon = (
                hav_dist_constant_lat(a[:, 0], a[:, 1], lon) <= max_sep)
            # Progressively update the boolean for each source in the group
            # for each distance check for the four extents.
            meets_min_distance[:len(a)] = (meets_min_distance[:len(a)] |
                                           is_within_dist_of_lon)
    # Similarly, if either "latitude" is set to 90 degrees, we cannot have
    # lack of up-down missing sources, so we must check (individually this
    # time) for whether we should skip this check.
    for lat in ax_lims[2:]:
        if np.abs(lat) < 90:
            is_within_dist_of_lat = (np.abs(a[:, 1] - lat) <= max_sep)
            meets_min_distance[:len(a)] = (meets_min_distance[:len(a)] |
                                           is_within_dist_of_lat)

    # Because all sources in BOTH catalogues must pass, we continue
    # to update meets_min_distance for catalogue "b" as well.
    if ax_lims[0] > 0 or ax_lims[0] < 0 or ax_lims[1] < 360:
        for lon in ax_lims[:2]:
            is_within_dist_of_lon = (
                hav_dist_constant_lat(b[:, 0], b[:, 1], lon) <= max_sep)
            meets_min_distance[len(a):] = (meets_min_distance[len(a):] |
                                           is_within_dist_of_lon)
    for lat in ax_lims[2:]:
        if np.abs(lat) < 90:
            is_within_dist_of_lat = (np.abs(b[:, 1] - lat) <= max_sep)
            meets_min_distance[len(a):] = (meets_min_distance[len(a):] |
                                           is_within_dist_of_lat)

    return [i, np.all(meets_min_distance == 0), a, b]
