# Licensed under a 3-clause BSD style license - see LICENSE
'''
This module provides the framework for grouping sources from two photometric catalogues into
distinct "islands" of sources, along with calculating whether they are within overlap for
various photometric integral purposes.
'''

import datetime
import itertools
import multiprocessing
import sys

import numpy as np

# pylint: disable=import-error,no-name-in-module
from macauff.group_sources_fortran import group_sources_fortran as gsf
from macauff.make_set_list import set_list
from macauff.misc_functions import (
    SharedNumpyArray,
    _load_rectangular_slice,
    convex_hull_area,
    load_small_ref_auf_grid,
)
from macauff.misc_functions_fortran import misc_functions_fortran as mff

# pylint: enable=import-error,no-name-in-module

__all__ = ['make_island_groupings']


# pylint: disable-next=too-many-arguments,too-many-locals,too-many-statements
def make_island_groupings(cm):
    '''
    Function to handle the creation of "islands" of astrometrically coeval
    sources, and identify which overlap to some probability based on their
    combined AUFs.

    Parameters
    ----------
    cm : Class
        The cross-match wrapper, containing all of the necessary metadata to
        perform the cross-match and determine match islands.
    '''

    # Convert from arcseconds to degrees internally.
    max_sep = np.copy(cm.pos_corr_dist) / 3600
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{t} Rank {cm.rank}, chunk {cm.chunk_id}: Creating catalogue islands and overlaps...")
    sys.stdout.flush()

    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{t} Rank {cm.rank}, chunk {cm.chunk_id}: Calculating maximum overlap...")
    sys.stdout.flush()

    # The initial step to create island groupings is to find the largest number
    # of overlaps for a single source, to minimise the size of the array of
    # overlap indices. To do so, we load small-ish chunks of the sky, with
    # padding in one catalogue to ensure all pairings can be found, and total
    # the number of overlaps for each object across all sky slices.

    ax1_skip, ax2_skip = 8, 8
    ax1_loops = np.linspace(cm.cross_match_extent[0], cm.cross_match_extent[1], 41)
    # Force the sub-division of the sky area in question to be 1600 chunks, or
    # roughly quarter square degree chunks, whichever is larger in area.
    if ax1_loops[1] - ax1_loops[0] < 0.25:
        ax1_loops = np.linspace(cm.cross_match_extent[0], cm.cross_match_extent[1],
                                int(np.ceil((cm.cross_match_extent[1] - cm.cross_match_extent[0])/0.25) + 1))
    ax2_loops = np.linspace(cm.cross_match_extent[2], cm.cross_match_extent[3], 41)
    if ax2_loops[1] - ax2_loops[0] < 0.25:
        ax2_loops = np.linspace(cm.cross_match_extent[2], cm.cross_match_extent[3],
                                int(np.ceil((cm.cross_match_extent[3] - cm.cross_match_extent[2])/0.25) + 1))
    ax1_sparse_loops = ax1_loops[::ax1_skip]
    if ax1_sparse_loops[-1] != ax1_loops[-1]:
        ax1_sparse_loops = np.append(ax1_sparse_loops, ax1_loops[-1])
    ax2_sparse_loops = ax2_loops[::ax2_skip]
    if ax2_sparse_loops[-1] != ax2_loops[-1]:
        ax2_sparse_loops = np.append(ax2_sparse_loops, ax2_loops[-1])

    # Load the astrometry of each catalogue for slicing.
    a_full = cm.a_astro
    b_full = cm.b_astro

    asize = np.zeros(dtype=int, shape=(len(a_full),))
    bsize = np.zeros(dtype=int, shape=(len(b_full),))

    for i, (ax1_sparse_start, ax1_sparse_end) in enumerate(zip(ax1_sparse_loops[:-1],
                                                               ax1_sparse_loops[1:])):
        for j, (ax2_sparse_start, ax2_sparse_end) in enumerate(zip(ax2_sparse_loops[:-1],
                                                                   ax2_sparse_loops[1:])):
            a_big_sky_cut = _load_rectangular_slice(a_full, ax1_sparse_start, ax1_sparse_end,
                                                    ax2_sparse_start, ax2_sparse_end, 0)
            b_big_sky_cut = _load_rectangular_slice(b_full, ax1_sparse_start, ax1_sparse_end,
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
                        a_cutout, ax_cutout, cm.a_perturb_auf_outputs, 0, a_big_sky_cut, cm.a_modelrefinds)
                    b, bfouriergrid, bmodrefindsmall, b_cut = _load_fourier_grid_cutouts(
                        b_cutout, ax_cutout, cm.b_perturb_auf_outputs, max_sep,
                        b_big_sky_cut, cm.b_modelrefinds)
                    if len(a) > 0 and len(b) > 0:
                        overlapa, overlapb = gsf.get_max_overlap(
                            a[:, 0], a[:, 1], b[:, 0], b[:, 1], max_sep, a[:, 2], b[:, 2],
                            cm.r[:-1]+cm.dr/2, cm.rho[:-1], cm.drho, cm.j1s, afouriergrid, bfouriergrid,
                            amodrefindsmall, bmodrefindsmall, cm.int_fracs[2])
                        a_cut2 = a_sky_inds[a_cut]
                        b_cut2 = b_sky_inds[b_cut]

                        asize[a_cut2] = asize[a_cut2] + overlapa
                        bsize[b_cut2] = bsize[b_cut2] + overlapb

    amaxsize = int(np.amax(asize))
    bmaxsize = int(np.amax(bsize))
    del (overlapa, overlapb, a, b, a_cut, b_cut, amodrefindsmall, bmodrefindsmall,
         afouriergrid, bfouriergrid)

    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{t} Rank {cm.rank}, chunk {cm.chunk_id}: Truncating star overlaps by AUF integral...")
    sys.stdout.flush()

    ainds = np.zeros(dtype=int, shape=(amaxsize, len(a_full)), order='F')
    binds = np.zeros(dtype=int, shape=(bmaxsize, len(b_full)), order='F')

    auf_cdf_a = np.zeros(dtype=float, shape=(amaxsize, len(a_full)), order='F')
    auf_cdf_b = np.zeros(dtype=float, shape=(bmaxsize, len(b_full)), order='F')

    ainds[:, :] = -1
    binds[:, :] = -1
    auf_cdf_a[:, :] = -1
    auf_cdf_b[:, :] = -1
    asize[:] = 0
    bsize[:] = 0

    # pylint: disable-next=too-many-nested-blocks
    for i, (ax1_sparse_start, ax1_sparse_end) in enumerate(zip(ax1_sparse_loops[:-1],
                                                               ax1_sparse_loops[1:])):
        for j, (ax2_sparse_start, ax2_sparse_end) in enumerate(zip(ax2_sparse_loops[:-1],
                                                                   ax2_sparse_loops[1:])):
            a_big_sky_cut = _load_rectangular_slice(a_full, ax1_sparse_start, ax1_sparse_end,
                                                    ax2_sparse_start, ax2_sparse_end, 0)
            b_big_sky_cut = _load_rectangular_slice(b_full, ax1_sparse_start, ax1_sparse_end,
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
                        a_cutout, ax_cutout, cm.a_perturb_auf_outputs, 0, a_big_sky_cut, cm.a_modelrefinds)
                    b, bfouriergrid, bmodrefindsmall, b_cut = _load_fourier_grid_cutouts(
                        b_cutout, ax_cutout, cm.b_perturb_auf_outputs, max_sep,
                        b_big_sky_cut, cm.b_modelrefinds)

                    if len(a) > 0 and len(b) > 0:
                        (indicesa, indicesb, overlapa, overlapb,
                         aufcdfa, aufcdfb) = gsf.get_overlap_indices(
                            a[:, 0], a[:, 1], b[:, 0], b[:, 1], max_sep, amaxsize, bmaxsize,
                            a[:, 2], b[:, 2], cm.r[:-1]+cm.dr/2, cm.rho[:-1], cm.drho, cm.j1s, afouriergrid,
                            bfouriergrid, amodrefindsmall, bmodrefindsmall, cm.int_fracs[2])

                        a_cut2 = a_sky_inds[a_cut]
                        b_cut2 = b_sky_inds[b_cut]

                        for k, _acut2 in enumerate(a_cut2):
                            ainds[asize[_acut2]:asize[_acut2]+overlapa[k], _acut2] = \
                                b_cut2[indicesa[:overlapa[k], k] - 1]
                            auf_cdf_a[asize[_acut2]:asize[_acut2]+overlapa[k], _acut2] = \
                                aufcdfa[:overlapa[k], k]
                        for k, _bcut2 in enumerate(b_cut2):
                            binds[bsize[_bcut2]:bsize[_bcut2]+overlapb[k], _bcut2] = \
                                a_cut2[indicesb[:overlapb[k], k] - 1]
                            auf_cdf_b[bsize[_bcut2]:bsize[_bcut2]+overlapb[k], _bcut2] = \
                                aufcdfb[:overlapb[k], k]

                        asize[a_cut2] = asize[a_cut2] + overlapa
                        bsize[b_cut2] = bsize[b_cut2] + overlapb

    del (a_cut, a_cut2, b_cut, b_cut2, indicesa, indicesb, overlapa, overlapb, a, b,
         amodrefindsmall, bmodrefindsmall, afouriergrid, bfouriergrid)

    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{t} Rank {cm.rank}, chunk {cm.chunk_id}: Cleaning overlaps...")
    sys.stdout.flush()

    ainds, asize, auf_cdf_a = _clean_overlaps(ainds, asize, auf_cdf_a, cm.n_pool, cm.chunk_id)
    binds, bsize, auf_cdf_b = _clean_overlaps(binds, bsize, auf_cdf_b, cm.n_pool, cm.chunk_id)

    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{t} Rank {cm.rank}, chunk {cm.chunk_id}: Calculating integral lengths...")
    sys.stdout.flush()

    if cm.include_phot_like or cm.use_phot_priors:
        a_err = a_full[:, 2]

        b_err = b_full[:, 2]

        a_fouriergrid = cm.a_perturb_auf_outputs['fourier_grid']
        b_fouriergrid = cm.b_perturb_auf_outputs['fourier_grid']

        a_int_areas = gsf.get_integral_length(
            a_err, b_err, cm.r[:-1]+cm.dr/2, cm.rho[:-1], cm.drho, cm.j1s, a_fouriergrid, b_fouriergrid,
            cm.a_modelrefinds, cm.b_modelrefinds, ainds, asize, cm.int_fracs[0:2])
        ab_area = a_int_areas[:, 0]
        af_area = a_int_areas[:, 1]

        b_int_areas = gsf.get_integral_length(
            b_err, a_err, cm.r[:-1]+cm.dr/2, cm.rho[:-1], cm.drho, cm.j1s, b_fouriergrid, a_fouriergrid,
            cm.b_modelrefinds, cm.a_modelrefinds, binds, bsize, cm.int_fracs[0:2])
        bb_area = b_int_areas[:, 0]
        bf_area = b_int_areas[:, 1]

    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{t} Rank {cm.rank}, chunk {cm.chunk_id}: Maximum overlaps are:", amaxsize, bmaxsize)
    sys.stdout.flush()

    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{t} Rank {cm.rank}, chunk {cm.chunk_id}: Finding unique sets...")
    sys.stdout.flush()

    set_list_items = set_list(ainds, binds, asize, bsize, cm.n_pool)
    if len(set_list_items) == 6:
        alist, blist, agrplen, bgrplen, areject, breject = set_list_items
        reject_flag = True
    else:
        alist, blist, agrplen, bgrplen = set_list_items  # pylint: disable=unbalanced-tuple-unpacking
        reject_flag = False

    # The final act of creating island groups is to clear out any sources too
    # close to the edge of the catalogue -- defined by its rectangular extend.
    # pylint: disable-next=fixme
    # TODO: add flag for allowing the keeping of potentially incomplete islands
    # in the main catalogue; here we default to, and only allow, their removal.

    passed_check = np.zeros(dtype=bool, shape=(alist.shape[1],))
    failed_check = np.ones(dtype=bool, shape=(alist.shape[1],))

    num_a_failed_checks = 0
    num_b_failed_checks = 0

    _, a_hull_points, a_hull_x_shift = convex_hull_area(
        a_full[:, 0], a_full[:, 1], return_hull=True)
    _, b_hull_points, b_hull_x_shift = convex_hull_area(
        b_full[:, 0], b_full[:, 1], return_hull=True)

    # "Pad factor" to allow for a percentage of missing search circle in cases
    # where e.g. we are 0-360 wrapped and have a "missing" meridian slice.
    # acceptable_lost_circle_frac here is equal to A_segment / A_circle, giving
    # an inflation factor for our search radius x \equiv d/r.
    acceptable_lost_circle_frac = 0.1
    x = np.linspace(0, 1, 1001)
    f = np.arccos(x) - x * np.sqrt(1 - x**2)
    ind = np.argmin(np.abs(f - np.pi * acceptable_lost_circle_frac))
    new_max_sep = max_sep / x[ind]

    seed = np.random.default_rng().choice(100000, size=(mff.get_random_seed_size(), len(a_full)))
    a_hull_radius_overlap = mff.get_circle_area_overlap(
        a_full[:, 0] + a_hull_x_shift, a_full[:, 1], new_max_sep,
        np.append(a_hull_points[:, 0], a_hull_points[0, 0]),
        np.append(a_hull_points[:, 1], a_hull_points[0, 1]), seed)
    a_hull_overlap_frac = a_hull_radius_overlap / (np.pi * new_max_sep**2)
    seed = np.random.default_rng().choice(100000, size=(mff.get_random_seed_size(), len(b_full)))
    b_hull_radius_overlap = mff.get_circle_area_overlap(
        b_full[:, 0] + b_hull_x_shift, b_full[:, 1], new_max_sep,
        np.append(b_hull_points[:, 0], b_hull_points[0, 0]),
        np.append(b_hull_points[:, 1], b_hull_points[0, 1]), seed)
    b_hull_overlap_frac = b_hull_radius_overlap / (np.pi * new_max_sep**2)

    for i in range(alist.shape[1]):
        ahof = a_hull_overlap_frac[alist[:agrplen[i], i]]
        bhof = b_hull_overlap_frac[blist[:bgrplen[i], i]]

        # A failed check would be any where *hof is too small, and hence we
        # lost some of the circle to being outside of the hull. If no objects
        # have fractions that are below some critical value then we can keep
        # the island.
        dist_check = np.all(ahof > (1 - acceptable_lost_circle_frac)) & np.all(
            bhof > (1 - acceptable_lost_circle_frac))

        if dist_check:
            passed_check[i] = 1
            failed_check[i] = 0
        else:
            # While "good" islands just need their total number incrementing
            # for the group, "failed" islands we need to track the number of
            # sources in each catalogue for.
            num_a_failed_checks += agrplen[i]
            num_b_failed_checks += bgrplen[i]

    # If set_list returned any rejected sources, then add any sources too close
    # to match extent to those now. Ensure that we only reject the unique source IDs
    # across each island group, ignoring the "default" -1 index.
    if reject_flag:
        a_first_rejected_len = len(areject)  # pylint: disable=possibly-used-before-assignment
    else:
        a_first_rejected_len = 0
    if num_a_failed_checks + a_first_rejected_len > 0:
        reject_a = np.zeros(dtype=int, shape=(num_a_failed_checks+a_first_rejected_len,))
    if reject_flag:
        reject_a[num_a_failed_checks:] = areject  # pylint: disable=used-before-assignment
    if reject_flag:
        b_first_rejected_len = len(breject)  # pylint: disable=possibly-used-before-assignment
    else:
        b_first_rejected_len = 0
    if num_b_failed_checks + b_first_rejected_len > 0:
        reject_b = np.zeros(dtype=int, shape=(num_b_failed_checks+b_first_rejected_len,))
    if reject_flag:
        reject_b[num_b_failed_checks:] = breject  # pylint: disable=used-before-assignment

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
    # Only return a[bf]_area and b[bf]_area if they were created
    if not (cm.include_phot_like or cm.use_phot_priors):
        ab_area = bb_area = None
        af_area = bf_area = None
        auf_cdf_a = auf_cdf_b = None
    # Only return reject counts if they were created
    if num_a_failed_checks + a_first_rejected_len > 0:
        lenrejecta = len(reject_a)
        # Save rejects output files.
        cm.reject_a = reject_a
    else:
        lenrejecta = 0
        cm.reject_a = None
    if num_b_failed_checks + b_first_rejected_len > 0:
        lenrejectb = len(reject_b)
        cm.reject_b = reject_b
    else:
        lenrejectb = 0
        cm.reject_b = None

    cm.ab_area = ab_area  # pylint: disable=possibly-used-before-assignment
    cm.bb_area = bb_area  # pylint: disable=possibly-used-before-assignment
    cm.ainds = ainds
    cm.binds = binds
    cm.asize = asize
    cm.bsize = bsize
    cm.af_area = af_area  # pylint: disable=possibly-used-before-assignment
    cm.bf_area = bf_area  # pylint: disable=possibly-used-before-assignment
    cm.alist = alist
    cm.blist = blist
    cm.agrplen = agrplen
    cm.bgrplen = bgrplen
    cm.lenrejecta = lenrejecta
    cm.lenrejectb = lenrejectb
    cm.auf_cdf_a = auf_cdf_a
    cm.auf_cdf_b = auf_cdf_b


def _load_fourier_grid_cutouts(a, sky_rect_coords, perturb_auf_outputs, padding,
                               large_sky_slice, modelrefinds):
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
    perturb_auf_outputs : dictionary
        Results from the simulations of catalogue ``a``'s AUF extensions.
    padding : float
        Maximum allowed sky separation the "wrong" side of ``sky_rect_coords``,
        allowing for an increase in sky box size which ensures that all overlaps
        get caught in ``get_max_overlap`` and ``get_max_indices``.
    large_sky_slice : boolean
        Slice array containing the ``True`` and ``False`` elements of which
        elements of the full catalogue, in ``con_cat_astro.npy``, are in ``a``.
    modelrefinds : numpy.ndarray
        The modelrefinds array output from ``create_perturb_auf``.
        TODO Improve description
    '''

    lon1, lon2, lat1, lat2 = sky_rect_coords

    sky_cut = _load_rectangular_slice(a, lon1, lon2, lat1, lat2, padding)

    a_cutout = a[sky_cut]

    modrefind = modelrefinds[:, large_sky_slice][:, sky_cut]

    [fouriergrid], modrefindsmall = load_small_ref_auf_grid(modrefind, perturb_auf_outputs,
                                                            ['fourier'])

    return a_cutout, fouriergrid, modrefindsmall, sky_cut


def _clean_overlaps(inds, size, cdf, n_pool, n_mem):
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
    cdf : numpy.ndarray
        Array of the cumulative distribution functions of each ``inds`` pair
        overlap between the two catalogues, evaluating the chance of counterpart
        between objects based on sky position, position precision, etc.
    n_pool : integer
        Number of multiprocessing threads to use.
    n_mem : integer
        Unique value, used to ensure shared-memory operations do not clash.

    Returns
    -------
    inds : numpy.ndarray
        The unique indices of overlap into the opposing catalogue for each
        source in a given catalogue, stripped of potential duplicates.
    cdf: numpy.ndarray
        ``cdf`` filtered by the unique indices in ``inds``.
    size : numpy.ndarray
        Newly updated ``size`` array, containing the lengths of the unique
        indices of overlap into the opposing catalogue for each source.
    '''
    maxsize = 0
    size[:] = 0
    counter = np.arange(0, inds.shape[1])

    shared_inds = SharedNumpyArray(inds, f'inds_{n_mem}')
    shared_cdf = SharedNumpyArray(cdf, f'cdf_{n_mem}')

    iter_group = zip(counter, itertools.repeat(shared_inds), itertools.repeat(shared_cdf))
    with multiprocessing.Pool(n_pool) as pool:
        for return_items in pool.imap_unordered(_calc_unique_inds, iter_group,
                                                chunksize=max(1, len(counter) // n_pool)):
            i, unique_inds, unique_cdf = return_items
            y = len(unique_inds)
            inds[:y, i] = unique_inds
            inds[y:, i] = -1
            cdf[:y, i] = unique_cdf
            cdf[y:, i] = -1
            maxsize = max(maxsize, y)
            size[i] = y

    pool.join()

    for _shared in [shared_inds, shared_cdf]:
        _shared.unlink()

    inds = np.asfortranarray(inds[:maxsize, :])
    cdf = np.asfortranarray(cdf[:maxsize, :])

    return inds, size, cdf


def _calc_unique_inds(iterable):
    i, inds, cdf = iterable
    x, inds_for_x = np.unique(inds.read()[inds.read()[:, i] > -1, i], return_index=True)
    return i, x, cdf.read()[inds.read()[:, i] > -1, i][inds_for_x]
