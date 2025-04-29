# Licensed under a 3-clause BSD style license - see LICENSE
'''
This module provides the framework for grouping sources from two photometric catalogues into
distinct "islands" of sources, along with calculating whether they are within overlap for
various photometric integral purposes.
'''

import datetime
import sys

import numpy as np

# pylint: disable=import-error,no-name-in-module
from macauff.group_sources_fortran import group_sources_fortran as gsf
from macauff.make_set_list import set_list
from macauff.misc_functions import calculate_overlap_counts, convex_hull_area
from macauff.misc_functions_fortran import misc_functions_fortran as mff

# pylint: enable=import-error,no-name-in-module

__all__ = ['make_island_groupings']


# pylint: disable-next=too-many-locals,too-many-statements
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

    # Load the astrometry of each catalogue.
    a_full = cm.a_astro
    b_full = cm.b_astro

    # The initial step to create island groupings is to find the largest number
    # of overlaps for a single source, to minimise the size of the array of
    # overlap indices.

    _ainds = calculate_overlap_counts(a_full, b_full, -999, 999, max_sep, cm.n_pool, np.nan, 0, 1,
                                      cm.cf_region_frame, 'array', cm.chunk_id)
    _binds = calculate_overlap_counts(b_full, a_full, -999, 999, max_sep, cm.n_pool, np.nan, 0, 1,
                                      cm.cf_region_frame, 'array', cm.chunk_id)

    amaxsize = int(np.amax([len(x) for x in _ainds]))
    bmaxsize = int(np.amax([len(x) for x in _binds]))

    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{t} Rank {cm.rank}, chunk {cm.chunk_id}: Truncating star overlaps by AUF integral...")
    sys.stdout.flush()

    ainds = np.ones(dtype=int, shape=(amaxsize, len(a_full)), order='F') * -1
    binds = np.ones(dtype=int, shape=(bmaxsize, len(b_full)), order='F') * -1
    # Populate the indices into a fortran-acceptable grid, rather than nested list.
    for i, x in enumerate(_ainds):
        ainds[:len(x), i] = x
    for i, x in enumerate(_binds):
        binds[:len(x), i] = x

    asize = np.array([np.sum(ainds[:, i] >= 0) for i in range(ainds.shape[1])])
    bsize = np.array([np.sum(binds[:, i] >= 0) for i in range(binds.shape[1])])

    ainds, binds, asize, bsize, auf_cdf_a, auf_cdf_b = gsf.get_overlap_indices(
        a_full[:, 0], a_full[:, 1], b_full[:, 0], b_full[:, 1], ainds, asize, binds, bsize, amaxsize,
        bmaxsize, a_full[:, 2], b_full[:, 2], cm.r[:-1]+cm.dr/2, cm.rho[:-1], cm.drho, cm.j1s,
        cm.a_perturb_auf_outputs['fourier_grid'], cm.b_perturb_auf_outputs['fourier_grid'], cm.a_modelrefinds,
        cm.b_modelrefinds, cm.int_fracs[2])

    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{t} Rank {cm.rank}, chunk {cm.chunk_id}: Cleaning overlaps...")
    sys.stdout.flush()

    # Clean arrays for any potential unnecessary extra rows caused by the largest
    # overlap number being reduced due to the additional criteria for match in
    # get_overlap_indices vs a naive radius-based search. In some cases we will
    # then have e.g. np.all(ainds[-1, :] == -1) evaluating to True, and might
    # as well get rid of the extraneous column.
    ainds = np.asfortranarray(ainds[:np.amax(asize), :])
    binds = np.asfortranarray(binds[:np.amax(bsize), :])

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
    cm.a_hull_points = a_hull_points
    cm.a_hull_x_shift = a_hull_x_shift
    cm.b_hull_points = b_hull_points
    cm.b_hull_x_shift = b_hull_x_shift

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
