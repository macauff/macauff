# Licensed under a 3-clause BSD style license - see LICENSE
'''
This module provides the framework for the creation of the photometric likelihoods
used in the cross-matching of the two catalogues.
'''

import datetime
import sys

import numpy as np
from scipy.optimize import minimize

# pylint: disable=import-error,no-name-in-module
from macauff.misc_functions import coord_inside_convex_hull
from macauff.misc_functions_fortran import misc_functions_fortran as mff
from macauff.photometric_likelihood_fortran import photometric_likelihood_fortran as plf

# pylint: enable=import-error,no-name-in-module

__all__ = ['compute_photometric_likelihoods']


# pylint: disable-next=too-many-locals
def compute_photometric_likelihoods(cm):
    '''
    Derives the photometric likelihoods and priors for use in the catalogue
    cross-match process.

    Parameters
    ----------
    cm : Class
        The cross-match wrapper, containing all of the necessary metadata to
        perform the cross-match and determine photometric likelihoods.
    '''

    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{t} Rank {cm.rank}, chunk {cm.chunk_id}: Creating c(m, m) and f(m)...")

    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{t} Rank {cm.rank}, chunk {cm.chunk_id}: Distributing sources into sky slices...")
    sys.stdout.flush()

    a_astro = cm.a_astro
    b_astro = cm.b_astro
    a_photo = cm.a_photo
    b_photo = cm.b_photo
    auf_cdf_a = cm.auf_cdf_a
    auf_cdf_b = cm.auf_cdf_b

    a_sky_inds = mff.find_nearest_point(a_astro[:, 0], a_astro[:, 1], cm.cf_region_points[:, 0],
                                        cm.cf_region_points[:, 1])
    b_sky_inds = mff.find_nearest_point(b_astro[:, 0], b_astro[:, 1], cm.cf_region_points[:, 0],
                                        cm.cf_region_points[:, 1])

    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{t} Rank {cm.rank}, chunk {cm.chunk_id}: Making bins...")
    sys.stdout.flush()

    abinlengths, abinsarray, longabinlen = create_magnitude_bins(cm.cf_region_points, cm.a_filt_names,
                                                                 a_photo, a_sky_inds)
    bbinlengths, bbinsarray, longbbinlen = create_magnitude_bins(cm.cf_region_points, cm.b_filt_names,
                                                                 b_photo, b_sky_inds)

    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{t} Rank {cm.rank}, chunk {cm.chunk_id}: Calculating PDFs...")
    sys.stdout.flush()

    c_priors = np.zeros(dtype=float, shape=(len(cm.b_filt_names), len(cm.a_filt_names),
                                            len(cm.cf_region_points)), order='F')
    fa_priors = np.zeros(dtype=float, shape=(len(cm.b_filt_names), len(cm.a_filt_names),
                                             len(cm.cf_region_points)), order='F')
    fb_priors = np.zeros(dtype=float, shape=(len(cm.b_filt_names), len(cm.a_filt_names),
                                             len(cm.cf_region_points)), order='F')
    c_array = np.zeros(dtype=float, shape=(longbbinlen-1, longabinlen-1, len(cm.b_filt_names),
                                           len(cm.a_filt_names), len(cm.cf_region_points)), order='F')
    fa_array = np.zeros(dtype=float, shape=(longabinlen-1, len(cm.b_filt_names), len(cm.a_filt_names),
                                            len(cm.cf_region_points)), order='F')
    fb_array = np.zeros(dtype=float, shape=(longbbinlen-1, len(cm.b_filt_names), len(cm.a_filt_names),
                                            len(cm.cf_region_points)), order='F')

    a_in_joint_area = np.array([
        coord_inside_convex_hull([a_astro[p, 0] + cm.a_hull_x_shift, a_astro[p, 1]], cm.a_hull_points) &
        coord_inside_convex_hull([a_astro[p, 0] + cm.b_hull_x_shift, a_astro[p, 1]], cm.b_hull_points)
        for p in range(len(a_astro))])
    b_in_joint_area = np.array([
        coord_inside_convex_hull([b_astro[p, 0] + cm.a_hull_x_shift, b_astro[p, 1]], cm.a_hull_points) &
        coord_inside_convex_hull([b_astro[p, 0] + cm.b_hull_x_shift, b_astro[p, 1]], cm.b_hull_points)
        for p in range(len(b_astro))])
    for m in range(0, len(cm.cf_region_points)):
        area = cm.cf_areas[m]
        a_sky_cut = (a_sky_inds == m) & a_in_joint_area
        a_photo_cut = a_photo[a_sky_cut]
        if cm.include_phot_like or cm.use_phot_priors:
            a_auf_cdf = auf_cdf_a[:, a_sky_cut]
            a_b_area_cut, a_f_area_cut, a_inds_cut, a_size_cut = (
                cm.ab_area[a_sky_cut], cm.af_area[a_sky_cut], cm.ainds[:, a_sky_cut], cm.asize[a_sky_cut])

        b_sky_cut = (b_sky_inds == m) & b_in_joint_area
        b_photo_cut = b_photo[b_sky_cut]
        if cm.include_phot_like or cm.use_phot_priors:
            b_auf_cdf = auf_cdf_b[:, b_sky_cut]
            b_b_area_cut, b_f_area_cut, b_inds_cut, b_size_cut = (
                cm.bb_area[b_sky_cut], cm.bf_area[b_sky_cut], cm.binds[:, b_sky_cut], cm.bsize[b_sky_cut])

        for i in range(0, len(cm.a_filt_names)):
            if not cm.include_phot_like and not cm.use_phot_priors:
                a_num_photo_cut = np.sum(~np.isnan(a_photo_cut[:, i]))
                na = a_num_photo_cut / area
            else:
                a_bins = abinsarray[:abinlengths[i, m], i, m]
                a_mag = a_photo[:, i]
                a_mag_cut = a_photo_cut[:, i]
                a_flags = ~np.isnan(a_mag)
                a_flags_cut = a_flags[a_sky_cut]
            for j in range(0, len(cm.b_filt_names)):
                if not cm.include_phot_like and not cm.use_phot_priors:
                    b_num_photo_cut = np.sum(~np.isnan(b_photo_cut[:, j]))
                    nb = b_num_photo_cut / area
                    # Without using photometric-based priors, all we can
                    # do is set the prior on one catalogue to 0.5 -- that
                    # is, equal chance of match or non-match; for this we
                    # use the less dense of the two catalogues as our
                    # "one-sided" match. Then, accordingly, we update the
                    # "field" source density of the more dense catalogue
                    # with its corresponding density, based on the input
                    # density and the counterpart density calculated.
                    c_prior = min(na, nb) / 2
                    fa_prior = na - c_prior
                    fb_prior = nb - c_prior
                    # To fake no photometric likelihoods, simply set all
                    # values to one, to cancel in the ratio later.
                    c_like, fa_like, fb_like = (1-1e-10)**2, 1-1e-10, 1-1e-10
                else:
                    b_bins = bbinsarray[:bbinlengths[j, m], j, m]
                    b_mag = b_photo[:, j]
                    b_mag_cut = b_photo_cut[:, j]
                    b_flags = ~np.isnan(b_mag)
                    b_flags_cut = b_flags[b_sky_cut]

                    bright_frac, field_frac = cm.int_fracs[[0, 1]]
                    c_prior, c_like, fa_prior, fa_like, fb_prior, fb_like = create_c_and_f(
                        # pylint: disable-next=possibly-used-before-assignment
                        a_mag, b_mag, a_mag_cut, b_mag_cut, a_inds_cut, a_size_cut, b_inds_cut, b_size_cut,
                        # pylint: disable-next=possibly-used-before-assignment
                        a_b_area_cut, b_b_area_cut, a_f_area_cut, b_f_area_cut, a_auf_cdf, b_auf_cdf, a_bins,
                        # pylint: disable-next=possibly-used-before-assignment
                        b_bins, bright_frac, field_frac, a_flags, b_flags, a_flags_cut, b_flags_cut, area)
                if cm.use_phot_priors and not cm.include_phot_like:
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
    cm.abinsarray = abinsarray
    cm.abinlengths = abinlengths
    cm.bbinsarray = bbinsarray
    cm.bbinlengths = bbinlengths
    cm.a_sky_inds = a_sky_inds
    cm.b_sky_inds = b_sky_inds
    cm.c_priors = c_priors
    cm.c_array = c_array
    cm.fa_priors = fa_priors
    cm.fa_array = fa_array
    cm.fb_priors = fb_priors
    cm.fb_array = fb_array


def create_magnitude_bins(cf_points, filts, a_photo, sky_inds):
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
    sky_inds : numpy.ndarray
         Array of indices, showing which on-sky photometric point, from
        ``cf_points``, each source in the catalogue is closest to.

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
                        dellist.extend(list(range(i+1, j+1)))
                        flag = 1
                        break
                if flag == 0:
                    dellist.extend(list(range(i+1, len(output_bins)-1)))
    output_bins = np.delete(output_bins, dellist)

    return output_bins


# pylint: disable-next=too-many-locals,too-many-arguments,too-many-statements
def create_c_and_f(a_mag, b_mag, a_mag_cut, b_mag_cut, a_inds, a_size, b_inds, b_size, a_b_area, b_b_area,
                   a_f_area, b_f_area, auf_cdf_a, auf_cdf_b, a_bins, b_bins, bright_frac, field_frac, a_flags,
                   b_flags, a_flags_cut, b_flags_cut, area):
    '''
    Functionality to create the photometric likelihood and priors from a set
    of photometric data in a given pair of filters.

    Parameters
    ----------
    a_mag : numpy.ndarray
        Catalogue "a" magnitudes for the full catalogue, such that indexing from
        e.g. ``b_inds`` is correct.
    b_mag : numpy.ndarray
        Catalogue "b" magnitudes for indexing purposes.
    a_mag_cut : numpy.ndarray
        Catalogue "a" magnitudes for sky area in which to determine photometric
        priors and likelihoods.
    b_mag_cut : numpy.ndarray
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
    a_b_area : numpy.ndarray
        The "bright" error circle area, integrating the joint AUF convolution
        out to ``expected_frac`` for largest "a"-"b" potential pairing, for each
        catalogue "a" object.
    b_b_area : numpy.narray
        Catalogue b's "bright" error circle area.
    a_f_area : numpy.ndarray
        The "field" error circle area, integrating the joint AUF convolution
        out to ``expected_frac`` for largest "a"-"b" potential pairing, for each
        catalogue "a" object.
    b_f_area : numpy.ndarray
        Catalogue b's "field" error circle area.
    auf_cdf_a : numpy.ndarray
        Evaluations of the astrometric uncertainty function, integrated out to
        the separation of the potential overlaps to all potential counterparts
        to all catalogue "a" objects.
    auf_cdf_b : numpy.ndarray
        Evaluations of the CDF of the AUF, for all potential counterparts in
        catalogue "a", for each catalogue "b" source.
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
    a_flags_cut : numpy.ndarray
        Boolean flags for whether a source in catalogue "a" has a detected
        magnitude in ``a_mag_cut``.
    b_flags_cut : numpy.ndarray
        Detection flags for catalogue "b" sources in ``b_mag_cut``.
    area : float
        Area of sky region for which photometric likelihood and prior are
        being calculated, in square degrees.

    Returns
    -------
    nc : float
        The prior density of counterpart sources between the catalogues.
    cdmdm : numpy.ndarray
        Two-dimensional array of the photometric likelihood of counterpart between
        the two catalogues.
    nfa : float
        So-called "field" source density in catalogue "a".
    fa : numpy.ndarray
        Probability density array of field sources for catalogue "a".
    nfb : float
        Field source density prior for catalogue "b".
    fb : numpy.ndarray
        Field source PDF for catalogue "b".
    '''
    a_hist, a_bins = np.histogram(a_mag_cut[a_flags_cut], bins=a_bins)
    pa = a_hist/(np.sum(a_hist)*np.diff(a_bins))

    a_cuts = plf.find_mag_bin_inds(a_mag_cut, a_flags_cut, a_bins)

    # get_field_dists allows for magnitude slicing, to get f(m | m) instead of f(m),
    # but when we do want f(m) we just pass two impossible magnitudes as the limits.
    # Note that the "primary" catalogue passed is the *_mag_cut array, matching
    # *_inds in length, but the second catalogue has to be the non-cut array such that
    # counterpart indexes are valid.
    a_masks = plf.get_field_dists(auf_cdf_a, a_inds, a_size, np.array([bright_frac, field_frac]), a_flags_cut,
                                  a_mag_cut, b_flags, b_mag, -999, 999, -999, 999)
    b_masks = plf.get_field_dists(auf_cdf_b, b_inds, b_size, np.array([bright_frac, field_frac]), b_flags_cut,
                                  b_mag_cut, a_flags, a_mag, -999, 999, -999, 999)

    # Generate our chosen field-frac's field source histograms.
    a_mask = a_masks[:, 1].astype(bool)
    b_mask = b_masks[:, 1].astype(bool)
    a_left = a_mag_cut[a_mask]
    b_left = b_mag_cut[b_mask]

    hist, a_bins = np.histogram(a_left, bins=a_bins)
    fa = hist / (np.sum(hist)*np.diff(a_bins))

    hist, b_bins = np.histogram(b_left, bins=b_bins)
    fb = hist / (np.sum(hist)*np.diff(b_bins))

    # Calculate the Nc and two Nf values from our biased densities.
    # These are generated from the measured density being counterparts (X)
    # and field density (Y), mitigated by deliberated removed objects inside
    # our chosen counterpart CDF fraction F and unintentional removal due to
    # random-chance alignment.
    measured_density_a = np.sum(a_masks, axis=0) / area
    measured_density_b = np.sum(b_masks, axis=0) / area
    # rho = x / a, sig_x = sqrt(x), sig_rho = sig_x * drho/dx = sig_x / a
    # sig_rho = sqrt(x) / a = sqrt(x / a) / sqrt(a) = sqrt(rho) / sqrt(a)
    # = sqrt(rho / a).
    meas_dens_a_uncert = np.sqrt(measured_density_a / area)
    meas_dens_b_uncert = np.sqrt(measured_density_b / area)

    # Values for the case where we don't remove anything, and F = 0. This
    # is used to get a handle on the total combined value of our two contributing
    # densities, X and Y.
    tot_density_a = np.sum(a_flags_cut) / area
    tot_dens_a_uncert = np.sqrt(tot_density_a / area)
    tot_density_b = np.sum(b_flags_cut) / area
    tot_dens_b_uncert = np.sqrt(tot_density_b / area)

    measured_density_a = np.array([tot_density_a, *measured_density_a])
    measured_density_b = np.array([tot_density_b, *measured_density_b])
    # Add a floor of 1/sq deg in quadrature to avoid empty measurements
    # causing NaNs.
    meas_dens_a_uncert = np.sqrt(np.array([tot_dens_a_uncert, *meas_dens_a_uncert])**2 + 1**2)
    meas_dens_b_uncert = np.sqrt(np.array([tot_dens_b_uncert, *meas_dens_b_uncert])**2 + 1**2)

    # Filter for completely isolated sources, which will have zero area,
    # to take a meaningful sample.
    pc = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
    if np.sum(a_b_area > 0) > 0:
        aba_perc = np.percentile(a_b_area[a_b_area > 0], pc)
    else:
        aba_perc = np.zeros(len(pc), float)
    if np.sum(a_f_area > 0) > 0:
        afa_perc = np.percentile(a_f_area[a_f_area > 0], pc)
    else:
        afa_perc = np.zeros(len(pc), float)
    if np.sum(b_b_area > 0) > 0:
        bba_perc = np.percentile(b_b_area[b_b_area > 0], pc)
    else:
        bba_perc = np.zeros(len(pc), float)
    if np.sum(b_f_area > 0) > 0:
        bfa_perc = np.percentile(b_f_area[b_f_area > 0], pc)
    else:
        bfa_perc = np.zeros(len(pc), float)
    avg_a_areas = np.array([aba_perc, afa_perc]).T
    avg_b_areas = np.array([bba_perc, bfa_perc]).T

    # With a percentile in steps of 10 percentage points, each area will be
    # equally weighted with 1/10th of the objects.
    wht_a_areas = 0.1 * np.ones_like(avg_a_areas)
    wht_b_areas = 0.1 * np.ones_like(avg_b_areas)

    res = minimize(calculate_prior_densities, args=(measured_density_a, measured_density_b,
                   meas_dens_a_uncert, meas_dens_b_uncert, np.array([bright_frac, field_frac]), avg_a_areas,
                   avg_b_areas, wht_a_areas, wht_b_areas), x0=[1, tot_density_a, tot_density_b], jac=True,
                   method='L-BFGS-B', bounds=[(0, None), (0, None), (0, None)])

    nc, nfa, nfb = res.x

    bm = np.empty((len(b_bins)-1, len(a_bins)-1), float, order='F')
    z = np.empty(len(a_bins)-1, float)

    # Average area here might be wrong too, but since we deal in small magnitude
    # slices the approximations should hold. Again, both a-catalogue magnitude
    # arrays need to be the "cut" versions to match a_size's shape, but catalogue
    # b versions need to be full to ensure cross-catalogue indexing works.
    mag_mask, aa = plf.brightest_mag(auf_cdf_a, a_mag_cut, b_mag, a_inds, a_size, bright_frac, a_b_area,
                                     a_flags_cut, b_flags, a_bins)
    mag_mask = mag_mask.astype(bool)
    for i in range(0, len(a_bins)-1):
        hist, b_bins = np.histogram(b_mag[mag_mask[:, i]], bins=b_bins)
        q = np.sum(mag_mask[:, i])
        if q > 0:
            bm[:, i] = hist/(np.diff(b_bins)*q)
        else:
            bm[:, i] = 0
        z[i] = np.sum(hist)/np.sum(a_cuts[i])
    cdmdm = np.empty((len(b_bins)-1, len(a_bins)-1), float, order='F')
    for i in range(0, len(a_bins)-1):
        # Within the loop, we only want to consider a catalogue, effectively, of
        # "a" sources with magitude [a_bins[i], a_bins[i+1]]. We therefore
        # get_field_dists with a slice in catalogue a, but no slice in b.
        b_masks = plf.get_field_dists(
            auf_cdf_b, b_inds, b_size, np.array([bright_frac, field_frac]), b_flags_cut, b_mag_cut, a_flags,
            a_mag, -999, 999, a_bins[i], a_bins[i+1])
        # However, to calculate the opposite masking -- and get biased-density
        # measurements of the "field" density of surviving objects in catalogue
        # "a" with narrow-range brightnesses -- we slice in a still, and not in b,
        # remembering that in the function call a and b will be swapped, with "a"
        # now the first-passed catalogue (either a or b).
        a_masks = plf.get_field_dists(
            auf_cdf_a, a_inds, a_size, np.array([bright_frac, field_frac]), a_flags_cut, a_mag_cut, b_flags,
            b_mag, a_bins[i], a_bins[i+1], -999, 999)

        a_mag_filter = (a_mag_cut >= a_bins[i]) & (a_mag_cut <= a_bins[i+1])

        measured_density_a = np.sum(a_masks, axis=0) / area
        measured_density_b = np.sum(b_masks, axis=0) / area
        meas_dens_a_uncert = np.sqrt(measured_density_a / area)
        meas_dens_b_uncert = np.sqrt(measured_density_b / area)
        tot_density_a = np.sum(a_flags_cut & a_mag_filter) / area
        tot_dens_a_uncert = np.sqrt(tot_density_a / area)
        tot_density_b = np.sum(b_flags_cut) / area
        tot_dens_b_uncert = np.sqrt(tot_density_b / area)

        measured_density_a = np.array([tot_density_a, *measured_density_a])
        measured_density_b = np.array([tot_density_b, *measured_density_b])
        meas_dens_a_uncert = np.sqrt(np.array([tot_dens_a_uncert, *meas_dens_a_uncert])**2 + 1**2)
        meas_dens_b_uncert = np.sqrt(np.array([tot_dens_b_uncert, *meas_dens_b_uncert])**2 + 1**2)

        # Also filter the a-catalogue objects for being within our magnitude
        # range, but don't do anything to b, so re-use the previous one.
        if np.sum((a_b_area > 0) & a_mag_filter) > 0:
            aba_perc = np.percentile(a_b_area[(a_b_area > 0) & a_mag_filter], pc)
        else:
            aba_perc = np.zeros(len(pc), float)
        if np.sum((a_f_area > 0) & a_mag_filter) > 0:
            afa_perc = np.percentile(a_f_area[(a_f_area > 0) & a_mag_filter], pc)
        else:
            afa_perc = np.zeros(len(pc), float)
        avg_a_areas = np.array([aba_perc, afa_perc]).T
        wht_a_areas = 0.1 * np.ones_like(avg_a_areas)

        res = minimize(calculate_prior_densities, args=(measured_density_a, measured_density_b,
                       meas_dens_a_uncert, meas_dens_b_uncert, np.array([bright_frac, field_frac]),
                       avg_a_areas, avg_b_areas, wht_a_areas, wht_b_areas),
                       x0=[1, tot_density_a, tot_density_b], jac=True, method='L-BFGS-B',
                       bounds=[(0, None), (0, None), (0, None)])

        _, _, _nfb = res.x

        bmask = b_masks[:, 1].astype(bool)
        b_left = b_mag_cut[bmask]
        hist, b_bins = np.histogram(b_left, bins=b_bins)
        _fb = hist / (np.sum(hist)*np.diff(b_bins))

        fm = np.append(0, np.cumsum(_fb[:-1] * np.diff(b_bins[:-1])))
        for j in range(0, len(b_bins)-1):
            cm = np.sum(cdmdm[:j, i]*np.diff(b_bins[:j+1]))
            cdmdm[j, i] = max(0, z[i]*bm[j, i]*np.exp(aa[i]*_nfb*fm[j]) - (1-cm)*aa[i]*_nfb*_fb[j])
        # What we get at the end is, as per Naylor et al. (2013), Zc c. They
        # convert directly to X c, but we have split that out above into Nc
        # already. Here we therefore don't care about the constant in front
        # of c(m | m), and simply normalise.
        if np.sum(cdmdm[:, i] * np.diff(b_bins)) > 0:
            cdmdm[:, i] /= np.sum(cdmdm[:, i] * np.diff(b_bins))

    integral = 0
    for i in range(0, len(a_bins)-1):
        cdmdm[:, i] *= pa[i]
        integral = integral + np.sum(cdmdm[:, i]*np.diff(b_bins))*(a_bins[i+1] - a_bins[i])
    if integral > 0:
        cdmdm /= integral

    # If any density is NaN, or Nc and at least one of Nfa or Nfb are
    # non-positive, then we have to warn and quit, since we won't be able to
    # recover that below. Similarly, we can't allow for a dependency-based
    # density condition where Nc is set based on one Nf and then the other Nf
    # is set based on Nc, since we don't know which counterpart density to use.
    # pylint: disable-next=too-many-boolean-expressions
    if (np.any(np.isnan([nc, nfa, nfb])) or (nc <= 0 and (nfa <= 0 or nfb <= 0)) or
            (nfa <= 0.01 * nc and nc <= 0.01 * nfb) or (nfb <= 0.01 * nc and nc <= 0.01 * nfa)):
        raise ValueError("Incorrect prior densities, unable to process chunk.")
    # Otherwise, if simply set to 1% the lowest of the (valid) field densities.
    if nfa > 0 and nfb > 0 and nc <= 0.01 * min(nfa, nfb):
        nc = 0.01 * min(nfa, nfb)
    # We would have conditions something like nfb <= 0 and nc <= 0.01 * nfa
    # and nfa > 0, to test the asymmetric cases, implicitly requiring nc > 0 to
    # avoid the raise above, but these will always trigger one of the latter
    # two error-criteria above.

    # If either field density is too small, set to 1% the counterpart density,
    # similarly. nfa <= 0.01 * nc implies nc > 0.01 * nfb or we'd've triggered
    # the latter two error-criteria again, though, so nc > 0 requires nfb > 0.
    if nfa <= 0.01 * nc and nc > 0:
        nfa = 0.01 * nc
    if nfb <= 0.01 * nc and nc > 0:
        nfb = 0.01 * nc

    return nc, cdmdm, nfa, fa, nfb, fb


def calculate_prior_densities(model_densities, rho, phi, o_rho, o_phi, fs, as_rho, as_phi, ws_rho, ws_phi):
    '''
    Calculate the joint-catalogue counterpart density and separate catalogue
    non-matched ("field") densities of two catalogues based on a series of
    measured densities. In each case, objects within a particular integrated
    fraction of object pairs being counterparts given their separation and
    corresponding positional uncertainties are removed from consideration,
    and "surviving" object densities are computed, to break the degeneracy
    between counterparts and non-counterparts.

    Parameters
    ----------
    model_densities : list or numpy.ndarray
        Three-element array with counterpart, catalogue a's field density,
        and catalogue b's field density in, evaluated by the minimisation.
    rho : list or numpy.ndarray
        Measured densities for catalogue a, at each fraction in ``fs``.
    phi : list or numpy.ndarray
        Catalogue b density measurements, corresponding to ``fs``.
    o_rho : list or numpy.ndarray
        Uncertainty in measured ``rho`` densities.
    o_phi : list or numpy.ndarray
        ``phi`` uncertainties.
    fs : list or numpy.ndarray
        Each CDF fraction used to remove potential counterparts (true or
        otherwise) in measuring ``rho`` and ``phi``.
    as_rho : list of lists or numpy.ndarray
        Average "error circle" area, the average area for a radius out to which
        all objects in catalogue a integrated to get to each ``fs`` CDF, for
        each CDF fraction. A weighted distribution is given for each ``fs``,
        corresponding to ``ws_rho``.
    as_phi : list of lists or numpy.ndarray
        Average catalogue b error circle area, corresponding to each
        ``fs`` CDF with an associated weight in ``ws_phi``.
    ws_rho : list of lists or numpy.ndarray
        Array of shape ``(X, Y)``, with fractions ``fs`` of length ``Y``. Each
        fraction has a set of error circle areas with corresponding weights,
        such that a sum along ``ws_rho[:, i]`` is unity, weighting the ``i``th
        fraction's error circles.
    ws_phi : list of lists or numpy.ndarray
        The joint error circle-fraction weights for catalogue ``phi``.

    Returns
    -------
    lst_sq : float
        The chi-squared fit between the model measured-densities, evaluated at
        each ``fs``, and ``rho`` and ``phi``.
    lst_sq_grad : numpy.ndarray
        The gradient of ``lst_sq`` with respect to each element of
        ``model_densities``.
    '''
    def calculate_dens_and_grad(t, u, v, f_s, a_s, w_s):
        '''
        Calculate the model for measured number counts based on
        joint-counterpart and randomly aligned source densities of two
        catalogues.

        Parameters
        ----------
        t : float
            Density, sources per square degree, of counterparts
        u : float
            Catalogue a's field density.
        v : float
            Catalogue b's field density.
        f_s : list or numpy.ndarray
            Set of CDF fractions at which catalogue densities have been sampled,
            removing some percentage of counterparts from the measured densities.
        a_s : list of lists or numpy.ndarray
            Set of average "error circle" areas, shape ``(X, Y)``, with ``X`` the
            number of different error circle areas to weight by and ``Y``
            corresponding to each ``f_s`` integral, which accounts for some
            percentage of removed chance-alignment sources from the measured
            density.
        w_s : list of lists or numpy.ndarray
            Two-dimensional array, shape ``(X, Y)``, the weights for each area
            for each set of fractions ``f_s``.

        Returns
        -------
        weighted_model : float
            The expected calculated overlap density between the two catalogues
            based on counterpart and non-matching densities and percentile
            integration for removal from measurements.
        weighted_model_grad : numpy.ndarray
            The gradient of the model with respect to ``t``, ``u``, and ``v``.
        '''
        weighted_model = np.zeros((len(f_s)), float)
        weighted_model_grad = np.zeros((3, len(f_s)), float)
        for a, w in zip(a_s, w_s):
            # d/d[tv] o_m_e = -pi r^2 o_m_e = -a o_m_e
            o_m_e = 1 - e(t, v, a)
            model = w * (((1 - f_s) * t + u) * o_m_e)

            dmodel_dt = w * ((1 - f_s) * o_m_e - a * model)
            dmodel_du = w * o_m_e
            dmodel_dv = w * -a * model

            weighted_model += model
            weighted_model_grad += np.array([dmodel_dt, dmodel_du, dmodel_dv])

        return weighted_model, weighted_model_grad

    def e(g, h, a):
        '''
        Calculate CDF of randomly aligned nearest-neighbour distribution.

        Parameters
        ----------
        g : float
            Density of one dataset.
        h : float
            Density of second dataset.
        a : float
            Area enclosed by radius out to which to consider potential
            neighbours.

        Returns
        -------
        float
            CDF, 1 - exp(-pi r^2 (g + h)).
        '''
        return 1 - np.exp(-a * (g + h))

    x, y, z = model_densities
    # The first measurement we make is for F=0, not passed through the fs array,
    # but IS passed through rho and phi as their first elements. This is a
    # simple function, and gives P = X + Y, Q = X + Z.
    p_0 = x + y
    q_0 = x + z
    p_0_grad = np.array([1, 1, 0])
    q_0_grad = np.array([1, 0, 1])
    # Flip y and z for the two calls!
    p, p_grad = calculate_dens_and_grad(x, y, z, fs, as_rho, ws_rho)
    q, q_grad = calculate_dens_and_grad(x, z, y, fs, as_phi, ws_phi)
    # Reverse the q_grad order, so that what is actually y is the 2nd element.
    q_grad = np.array([q_grad[0], q_grad[2], q_grad[1]])

    p = np.array([p_0, *p])
    q = np.array([q_0, *q])
    p_grad = np.array([[p_0_grad[0], *p_grad[0]], [p_0_grad[1], *p_grad[1]], [p_0_grad[2], *p_grad[2]]])
    q_grad = np.array([[q_0_grad[0], *q_grad[0]], [q_0_grad[1], *q_grad[1]], [q_0_grad[2], *q_grad[2]]])

    lst_sq = np.sum((p - rho)**2 / o_rho**2 + (q - phi)**2 / o_phi**2)
    lst_sq_grad = np.empty(3, float)
    lst_sq_grad[0] = np.sum(2 * (p - rho) / o_rho**2 * p_grad[0] + 2 * (q - phi) / o_phi**2 * q_grad[0])
    lst_sq_grad[1] = np.sum(2 * (p - rho) / o_rho**2 * p_grad[1] + 2 * (q - phi) / o_phi**2 * q_grad[1])
    lst_sq_grad[2] = np.sum(2 * (p - rho) / o_rho**2 * p_grad[2] + 2 * (q - phi) / o_phi**2 * q_grad[2])

    return lst_sq, lst_sq_grad
