# Licensed under a 3-clause BSD style license - see LICENSE
'''
This module provides the framework to handle the creation of the perturbation
component of the astrometric uncertainty function.
'''

# pylint: disable=too-many-lines
# pylint: disable=duplicate-code

import datetime
import os
import signal
import sys
import timeit

import numpy as np
import requests
from scipy.stats import binned_statistic

# pylint: disable=import-error,no-name-in-module
from macauff.galaxy_counts import create_galaxy_counts
from macauff.get_trilegal_wrapper import get_trilegal
from macauff.misc_functions import (
    convex_hull_area,
    create_auf_params_grid,
    create_densities,
    find_model_counts_corrections,
    generate_avs_inside_hull,
)
from macauff.misc_functions_fortran import misc_functions_fortran as mff
from macauff.perturbation_auf_fortran import perturbation_auf_fortran as paf

# pylint: enable=import-error,no-name-in-module

__all__ = ['make_perturb_aufs', 'create_single_perturb_auf']


# pylint: disable-next=too-many-locals,too-many-statements
def make_perturb_aufs(cm, which_cat):
    r"""
    cm : Class
        The cross-match wrapper, containing all of the necessary metadata to
        perform the cross-match and determine photometric likelihoods.
    which_cat : string
        Indicator as to whether these perturbation AUFs are for catalogue "a"
        or catalogue "b" within the cross-match process, to ensure the correct
        attributes are accessed.
    """
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f'{t} Rank {cm.rank}, chunk {cm.chunk_id}: Creating perturbation AUFs sky indices for '
          f'catalogue "{which_cat}"...')
    sys.stdout.flush()

    # Extract the single-catalogue values for determining the perturbation
    # component of the AUF.
    auf_file_path = getattr(cm, f'{which_cat}_auf_file_path')
    auf_points = getattr(cm, f'{which_cat}_auf_region_points')
    filters = getattr(cm, f'{which_cat}_filt_names')
    if cm.include_perturb_auf:
        auf_region_frame = getattr(cm, f'{which_cat}_auf_region_frame')
        density_radius = getattr(cm, f'{which_cat}_dens_dist')
        tri_download_flag = getattr(cm, f'{which_cat}_download_tri')
        tri_set_name = getattr(cm, f'{which_cat}_tri_set_name')
        tri_filt_num = getattr(cm, f'{which_cat}_tri_filt_num')
        tri_maglim_faint = getattr(cm, f'{which_cat}_tri_maglim_faint')
        tri_num_faint = getattr(cm, f'{which_cat}_tri_num_faint')
        fit_gal_flag = getattr(cm, f'{which_cat}_fit_gal_flag')
        num_trials = cm.num_trials
        psf_fwhms = getattr(cm, f'{which_cat}_psf_fwhms')
        tri_filt_names = getattr(cm, f'{which_cat}_tri_filt_names')
        d_mag = cm.d_mag
        delta_mag_cuts = cm.delta_mag_cuts
        run_fw = getattr(cm, f'{which_cat}_run_fw_auf')
        run_psf = getattr(cm, f'{which_cat}_run_psf_auf')
        al_avs = getattr(cm, f'{which_cat}_gal_al_avs')
        if run_psf:
            dd_params = getattr(cm, 'dd_params')
            l_cut = getattr(cm, 'l_cut')
        else:
            # Fake arrays to pass only to run_fw that fortran will accept:
            dd_params = np.zeros((1, 1), float)
            l_cut = np.zeros((1), float)
        if fit_gal_flag:
            cmau_array = cm.gal_cmau_array
            wavs = getattr(cm, f'{which_cat}_gal_wavs')
            z_maxs = getattr(cm, f'{which_cat}_gal_zmax')
            nzs = getattr(cm, f'{which_cat}_gal_nzs')
            alpha0 = cm.gal_alpha0
            alpha1 = cm.gal_alpha1
            alpha_weight = cm.gal_alphaweight
            ab_offsets = getattr(cm, f'{which_cat}_gal_aboffsets')
            filter_names = getattr(cm, f'{which_cat}_gal_filternames')
            saturation_magnitudes = getattr(cm, f'{which_cat}_saturation_magnitudes')

        # Extract either dummy or real TRILEGAL histogram lists.
        dens_hist_tri = getattr(cm, f'{which_cat}_dens_hist_tri_list')
        tri_model_mags = getattr(cm, f'{which_cat}_tri_model_mags_list')
        tri_model_mag_mids = getattr(cm, f'{which_cat}_tri_model_mag_mids_list')
        tri_model_mags_interval = getattr(cm, f'{which_cat}_tri_model_mags_interval_list')
        tri_n_bright_sources_star = getattr(cm, f'{which_cat}_tri_n_bright_sources_star_list')

        n_pool = cm.n_pool

    a_tot_astro = getattr(cm, f'{which_cat}_astro')
    if cm.include_perturb_auf:
        a_tot_photo = getattr(cm, f'{which_cat}_photo')
        a_tot_snr = getattr(cm, f'{which_cat}_snr')

    n_sources = len(a_tot_astro)

    modelrefinds = np.zeros(dtype=int, shape=(3, n_sources), order='f')

    # Which sky position to use is more complex; this involves determining
    # the smallest great-circle distance to each auf_point AUF mapping for
    # each source.
    modelrefinds[2, :] = mff.find_nearest_point(a_tot_astro[:, 0], a_tot_astro[:, 1],
                                                auf_points[:, 0], auf_points[:, 1])

    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f'{t} Rank {cm.rank}, chunk {cm.chunk_id}: Creating empirical perturbation AUF PDFs for '
          f'catalogue "{which_cat}"...')
    sys.stdout.flush()

    # Store the length of the density-magnitude combinations in each sky/filter
    # combination for future loading purposes.
    arraylengths = np.zeros(dtype=int, shape=(len(filters), len(auf_points)), order='f')

    if cm.include_perturb_auf:
        local_n = np.zeros(dtype=float, shape=(len(a_tot_astro), len(filters)))

    perturb_auf_outputs = {}

    for i, auf_point in enumerate(auf_points):
        ax1, ax2 = auf_point
        if auf_file_path is not None:
            new_auf_file_path = auf_file_path.format(ax1, ax2)
            if not os.path.exists(os.path.dirname(new_auf_file_path)):
                os.makedirs(os.path.dirname(new_auf_file_path), exist_ok=True)
        else:
            new_auf_file_path = None

        if cm.include_perturb_auf:
            sky_cut = modelrefinds[2, :] == i
            med_index_slice = np.arange(0, len(local_n))[sky_cut]
            a_photo_cut = a_tot_photo[sky_cut]
            a_astro_cut = a_tot_astro[sky_cut]
            a_snr_cut = a_tot_snr[sky_cut]

            if len(a_astro_cut) > 0:
                dens_mags = np.empty(len(filters), float)
                for j in range(len(dens_mags)):  # pylint: disable=consider-using-enumerate
                    # Take the "density" magnitude (i.e., the faint limit down to
                    # which to integrate counts per square degree per magnitude) from
                    # the data, with a small allowance for completeness limit turnover.
                    hist, bins = np.histogram(a_photo_cut[~np.isnan(a_photo_cut[:, j]) &
                                                          ~np.isnan(a_snr_cut[:, j]), j], bins='auto')
                    # TODO: relax half-mag cut, make input parameter  pylint: disable=fixme
                    dens_mags[j] = (bins[:-1]+np.diff(bins)/2)[np.argmax(hist)] - 0.5

                # Calculate the area of the current patch, assuming it is
                # sufficiently convex to be defineable by its convex hull.
                sky_area, hull_points, hull_x_shift = convex_hull_area(
                    a_astro_cut[:, 0], a_astro_cut[:, 1], return_hull=True)

                # Also, before beginning the loop of filters, sample the V-band
                # extinction across the region.
                avs = generate_avs_inside_hull(hull_points, hull_x_shift, auf_region_frame)

        # If there are no sources in this entire section of sky, we don't need
        # to bother downloading any TRILEGAL simulations since we'll auto-fill
        # dummy data (and never use it) in the filter loop.
        if auf_file_path is not None:
            x, y = os.path.splitext(new_auf_file_path)
            full_file = x + '_faint' + y
        if auf_file_path is not None and cm.include_perturb_auf and len(a_astro_cut) > 0 and (
                # pylint: disable-next=possibly-used-before-assignment
                tri_download_flag or not os.path.isfile(full_file)):

            data_bright_dens = np.array([np.sum(~np.isnan(a_photo_cut[:, q]) & ~np.isnan(a_snr_cut[:, q]) &
                                         (a_photo_cut[:, q] <= dens_mags[q])) / sky_area
                                        for q in range(len(dens_mags))])
            # TODO: un-hardcode min_bright_tri_number  pylint: disable=fixme
            min_bright_tri_number = 1000
            min_area = max(min_bright_tri_number / data_bright_dens)
            # Hard-coding the AV=1 trick to allow for using av_grid later.
            # pylint: disable-next=possibly-used-before-assignment
            download_trilegal_simulation(full_file, tri_set_name, ax1, ax2, tri_filt_num,
                                         # pylint: disable-next=possibly-used-before-assignment
                                         auf_region_frame, tri_maglim_faint, min_area,
                                         # pylint: disable-next=possibly-used-before-assignment
                                         av=1, sigma_av=0, total_objs=tri_num_faint,
                                         rank=cm.rank, chunk_id=cm.chunk_id)
        for j, filt in enumerate(filters):
            perturb_auf_combo = f'{ax1}-{ax2}-{filt}'

            if cm.include_perturb_auf:
                good_mag_snr_slice = ~np.isnan(a_photo_cut[:, j]) & ~np.isnan(a_snr_cut[:, j])
                a_astro = a_astro_cut[good_mag_snr_slice]
                a_photo = a_photo_cut[good_mag_snr_slice, j]
                a_snr = a_snr_cut[good_mag_snr_slice, j]
                if len(a_photo) == 0:
                    arraylengths[j, i] = 0
                    # If no sources in this AUF-filter combination, we need to
                    # fake some dummy variables for use in the 3/4-D grids below.
                    # See below, in include_perturb_auf is False, for meanings.
                    num_n_mag = 1
                    frac = np.zeros((1, num_n_mag), float, order='F')
                    flux = np.zeros(num_n_mag, float, order='F')
                    offset = np.zeros((len(cm.r)-1, num_n_mag), float, order='F')
                    offset[0, :] = 1 / (2 * np.pi * (cm.r[0] + cm.dr[0]/2) * cm.dr[0])
                    cumulative = np.ones((len(cm.r)-1, num_n_mag), float, order='F')
                    fourieroffset = np.ones((len(cm.rho)-1, num_n_mag), float, order='F')
                    narray = np.array([[1]], float)
                    magarray = np.array([[1]], float)
                    # Make a second dictionary with the single pointing-filter
                    # combination in it.
                    single_perturb_auf_output = {}
                    for name, entry in zip(
                            ['frac', 'flux', 'offset', 'cumulative', 'fourier', 'Narray',
                             'magarray'], [frac, flux, offset, cumulative, fourieroffset, narray,
                                           magarray]):
                        single_perturb_auf_output[name] = entry
                    perturb_auf_outputs[perturb_auf_combo] = single_perturb_auf_output
                    continue
                # Should be x[:, 0] = ax1, x[:, 1] = ax2, x[:, 2] = mag, for
                # create_densities' API.
                x = np.vstack((a_astro[:, 0], a_astro[:, 1], a_photo)).T
                localn = create_densities(x, -999, dens_mags[j], hull_points, hull_x_shift, density_radius,
                                          n_pool, 2, 0, 1, auf_region_frame, cm.chunk_id)
                # Because we always calculate the density from the full
                # catalogue, using just the astrometry, we should be able
                # to just over-write this N times if there happen to be N
                # good detections of a source.
                local_n[med_index_slice[good_mag_snr_slice], j] = localn
                if fit_gal_flag:
                    single_perturb_auf_output = create_single_perturb_auf(
                        cm.r, cm.dr, cm.j0s, num_trials, psf_fwhms[j], dens_mags[j], a_photo, a_snr, localn,
                        d_mag, delta_mag_cuts, dd_params, l_cut, run_fw, run_psf, al_avs[j], avs,
                        fit_gal_flag, sky_area, saturation_magnitudes[j], cmau_array, wavs[j], z_maxs[j],
                        nzs[j], alpha0, alpha1, alpha_weight, ab_offsets[j], filter_names[j],
                        tri_file_path=new_auf_file_path, filt_header=tri_filt_names[j],
                        dens_hist_tri=dens_hist_tri[j], model_mags=tri_model_mags[j],
                        model_mag_mids=tri_model_mag_mids[j], model_mags_interval=tri_model_mags_interval[j],
                        n_bright_sources_star=tri_n_bright_sources_star[j])
                else:
                    single_perturb_auf_output = create_single_perturb_auf(
                        cm.r, cm.dr, cm.j0s, num_trials, psf_fwhms[j], dens_mags[j], a_photo, a_snr, localn,
                        d_mag, delta_mag_cuts, dd_params, l_cut, run_fw, run_psf, al_avs[j], avs,
                        fit_gal_flag, tri_file_path=new_auf_file_path, filt_header=tri_filt_names[j],
                        dens_hist_tri=dens_hist_tri[j], model_mags=tri_model_mags[j],
                        model_mag_mids=tri_model_mag_mids[j], model_mags_interval=tri_model_mags_interval[j],
                        n_bright_sources_star=tri_n_bright_sources_star[j])
                perturb_auf_outputs[perturb_auf_combo] = single_perturb_auf_output
            else:
                # Without the simulations to force local normalising density N or
                # individual source brightness magnitudes, we can simply combine
                # all data into a single "bin".
                num_n_mag = 1
                # In cases where we do not want to use the perturbation AUF component,
                # we currently don't have separate functions, but instead set up dummy
                # functions and variables to pass what mathematically amounts to
                # "nothing" through the cross-match. Here we would use fortran
                # subroutines to create the perturbation simulations, so we make
                # f-ordered dummy parameters.
                frac = np.zeros((1, num_n_mag), float, order='F')
                flux = np.zeros(num_n_mag, float, order='F')
                # Remember that r is bins, so the evaluations at bin middle are one
                # shorter in length.
                offset = np.zeros((len(cm.r)-1, num_n_mag), float, order='F')
                # Fix offsets such that the probability density function looks like
                # a delta function, such that a two-dimensional circular coordinate
                # integral would evaluate to one at every point, cf. ``cumulative``.
                offset[0, :] = 1 / (2 * np.pi * (cm.r[0] + cm.dr[0]/2) * cm.dr[0])
                # The cumulative integral of a delta function is always unity.
                cumulative = np.ones((len(cm.r)-1, num_n_mag), float, order='F')
                # The Hankel transform of a delta function is a flat line; this
                # then preserves the convolution being multiplication in fourier
                # space, as F(x) x 1 = F(x), similar to how f(x) * d(0) = f(x).
                fourieroffset = np.ones((len(cm.rho)-1, num_n_mag), float, order='F')
                # Both normalising density and magnitude arrays can be proxied
                # with a dummy parameter, as any minimisation of N-m distance
                # must pick the single value anyway.
                narray = np.array([[1]], float)
                magarray = np.array([[1]], float)
                single_perturb_auf_output = {}
                for name, entry in zip(
                        ['frac', 'flux', 'offset', 'cumulative', 'fourier', 'Narray', 'magarray'],
                        [frac, flux, offset, cumulative, fourieroffset, narray, magarray]):
                    single_perturb_auf_output[name] = entry
                perturb_auf_outputs[perturb_auf_combo] = single_perturb_auf_output
            arraylengths[j, i] = len(perturb_auf_outputs[perturb_auf_combo]['Narray'])

    if cm.include_perturb_auf:
        longestnm = np.amax(arraylengths)

        narrays = np.full(dtype=float, shape=(longestnm, len(filters), len(auf_points)),
                          order='F', fill_value=-1)

        magarrays = np.full(dtype=float, shape=(longestnm, len(filters), len(auf_points)),
                            order='F', fill_value=-1)

        for i, auf_point in enumerate(auf_points):
            ax1, ax2 = auf_point
            for j, filt in enumerate(filters):
                if arraylengths[j, i] == 0:
                    continue
                perturb_auf_combo = f'{ax1}-{ax2}-{filt}'
                narray = perturb_auf_outputs[perturb_auf_combo]['Narray']
                magarray = perturb_auf_outputs[perturb_auf_combo]['magarray']
                narrays[:arraylengths[j, i], j, i] = narray
                magarrays[:arraylengths[j, i], j, i] = magarray

    # Once the individual AUF simulations are saved, we also need to calculate
    # the indices each source references when slicing into the 4-D cubes
    # created by [1-D array] x N-m combination x filter x sky position iteration.

    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f'{t} Rank {cm.rank}, chunk {cm.chunk_id}: Creating perturbation AUFs filter indices for '
          f'catalogue "{which_cat}"...')
    sys.stdout.flush()

    if cm.include_perturb_auf:
        a = getattr(cm, f'{which_cat}_photo')
        localn = local_n
    magref = getattr(cm, f'{which_cat}_magref')

    if cm.include_perturb_auf:
        for i in range(0, len(a)):
            axind = modelrefinds[2, i]
            filterind = magref[i]
            nmind = np.argmin((localn[i, filterind] - narrays[:arraylengths[filterind, axind],
                                                              filterind, axind])**2 +
                              (a[i, filterind] - magarrays[:arraylengths[filterind, axind],
                                                           filterind, axind])**2)
            modelrefinds[0, i] = nmind
    else:
        # For the case that we do not use the perturbation AUF component,
        # our dummy N-m files are all one-length arrays, so we can
        # trivially index them, regardless of specifics.
        modelrefinds[0, :] = 0

    # The mapping of which filter to use is straightforward: simply pick
    # the filter index of the "best" filter for each source, from magref.
    modelrefinds[1, :] = magref

    if not cm.include_perturb_auf:
        n_fracs = 2  # TODO: generalise once delta_mag_cuts is user-inputtable.  pylint: disable=fixme
    else:
        n_fracs = len(delta_mag_cuts)
    # Create the 4-D grids that house the perturbation AUF fourier-space
    # representation.
    perturb_auf_outputs['fourier_grid'] = create_auf_params_grid(
        perturb_auf_outputs, auf_points, filters, 'fourier', arraylengths, len(cm.rho)-1)
    # Create the estimated levels of flux contamination and fraction of
    # contaminated source grids.
    perturb_auf_outputs['frac_grid'] = create_auf_params_grid(
        perturb_auf_outputs, auf_points, filters, 'frac', arraylengths, n_fracs)
    perturb_auf_outputs['flux_grid'] = create_auf_params_grid(
        perturb_auf_outputs, auf_points, filters, 'flux', arraylengths)

    if cm.include_perturb_auf:
        del narrays, magarrays

    return modelrefinds, perturb_auf_outputs


def download_trilegal_simulation(tri_file_path, tri_filter_set, ax1, ax2, mag_num, region_frame,
                                 mag_lim, min_area, total_objs=1.5e6, av=None, sigma_av=0.1,
                                 rank=None, chunk_id=None):
    '''
    Get a single Galactic sightline TRILEGAL simulation of an appropriate sky
    size, and save it in a folder for use in the perturbation AUF simulations.

    Parameters
    ----------
    tri_file_path : string
        The location on disk into which to save the TRILEGAL file.
    tri_filter_set : string
        The name of the filterset, as given by the TRILEGAL input form.
    ax1 : float
        The first axis position of the sightline to be simulated, in the frame
        determined by ``region_frame``.
    ax2 : float
        The second axis position of the TRILEGAL sightline.
    mag_num : integer
        The zero-indexed filter number in the ``tri_filter_set`` list of filters
        which decides the limiting magnitude down to which tosimulate the
        Galactic sources.
    region_frame : string
        Frame, either equatorial or galactic, of the cross-match being performed,
        indicating whether ``ax1`` and ``ax2`` are in Right Ascension and
        Declination or Galactic Longitude and Latitude.
    mag_lim : float
        Magnitude down to which to generate sources for the simulation.
    min_area : float
        Smallest requested area, based on the density of catalogue objects
        per unit area above specified brightness limits and a minimum
        acceptable number of simulated objects above those same limits.
    total_objs : integer, optional
        The approximate number of objects to simulate in a TRILEGAL sightline,
        affecting how large an area to request a simulated Galactic region of.
    av : float, optional
        If specified, pass a pre-determined value of infinite-Av to the simulation
        API; otherwise pass its own "default" value and request it derive one
        internally.
    sigma_av : float, optional
        If given, bypasses the default value specified in ~`macauff.get_trilegal`,
        setting the fractional scaling around `av` in which to randomise
        extinction values.
    rank: string, optional
        If running a parallelised cross-match, pass through the appropriate MPI
        worker rank for print statements. Otherwise defaults to ``None``.
    chunk_id: string, optional
        If running a parallelised cross-match, pass through the appropriate
        "chunk" ID for print statements. Otherwise defaults to ``None``.
    '''
    class TimeoutException(Exception):  # pylint: disable=missing-class-docstring
        pass

    def timeout_handler(signum, frame):
        raise TimeoutException

    areaflag = 0
    triarea = min(10, min_area)
    galactic_flag = region_frame == 'galactic'
    # To avoid a loop where we start at some area, halve repeatedly until
    # the API call limit is satisfied, but then get nobjs < total_objs and
    # try to scale back up again only to time out, only allow that to happen
    # if we haven't halved our area within the loop at all.
    area_halved = False
    while areaflag == 0:
        start = timeit.default_timer()
        result = "timeout"
        try:
            nocomm_count = 0
            while result in ("timeout", "nocomm"):
                signal.signal(signal.SIGALRM, timeout_handler)
                # Set a 11 minute "timer" to raise an error if get_trilegal takes
                # longer than, as this indicates the API call has run out of CPU
                # time on the other end. As get_trilegal has an internal "busy"
                # tone, we need to reset this alarm for each call, if we don't
                # get a "good" result from the function call.
                signal.alarm(11*60)
                av_inf, result = get_trilegal(
                    tri_file_path, ax1, ax2, galactic=galactic_flag, filterset=tri_filter_set, area=triarea,
                    maglim=mag_lim, magnum=mag_num, av=av, sigma_av=sigma_av)
                if result == "nocomm":
                    nocomm_count += 1
                # 11 minute timer allows for 5 loops of two-minute waits for
                # a response from the server.
                if nocomm_count >= 5:
                    raise requests.exceptions.HTTPError("TRILEGAL server has not communicated "
                                                        f"in {nocomm_count} attempts.")
        except TimeoutException:
            triarea /= 2
            area_halved = True
            end = timeit.default_timer()
            t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f'{t} Rank {rank}, chunk {chunk_id}: TRILEGAL call time: {end-start:.2f}')
            t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{t} Rank {rank}, chunk {chunk_id}: Timed out, halving area")
            continue
        else:
            end = timeit.default_timer()
            t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f'{t} Rank {rank}, chunk {chunk_id}: TRILEGAL call time: {end-start:.2f}')
            signal.alarm(0)
        with open(tri_file_path, "r", encoding='utf-8') as f:
            contents = f.readlines()
        # Two comment lines; one at the top and one at the bottom - we add a
        # third in a moment, however
        nobjs = len(contents) - 2
        # If too few stars then increase by factor 10 and loop, or scale to give
        # about total_objs stars and come out of area increase loop --
        # simulations can't be more than 10 sq deg, so accept if that's as large
        # as we can go.
        if nobjs < 10000 and not area_halved:
            t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{t} Rank {rank}, chunk {chunk_id}: Too few numbers, increasing area...")
            triarea = min(10, triarea*10)
            accept_results = False
            # If we can't multiple by 10 since we get to 10 sq deg area, then
            # we can just quit immediately since we can't do any better.
            if triarea == 10:
                areaflag = 1
        # If number counts are too low for either nobjs < X comparison but
        # the area had to be reduced by 50% previously, just accept the area
        # we got, since it's basically the best the TRILEGAL API will provide.
        elif nobjs < total_objs and not area_halved:
            t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{t} Rank {rank}, chunk {chunk_id}: Scaling area to total_objs counts...")
            triarea = min(10, triarea / nobjs * total_objs)
            areaflag = 1
            accept_results = False
        else:
            t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{t} Rank {rank}, chunk {chunk_id}: Sufficient counts or area halved, "
                  "accepting run...")
            areaflag = 1
            accept_results = True
        if not accept_results:
            os.system(f'rm {tri_file_path}')
    if not accept_results:
        result = "timeout"
        while result == "timeout":
            av_inf, result = get_trilegal(
                tri_file_path, ax1, ax2, galactic=galactic_flag, filterset=tri_filter_set, area=triarea,
                maglim=mag_lim, magnum=mag_num, av=av, sigma_av=sigma_av)
    with open(tri_file_path, "r", encoding='utf-8') as f:
        contents = f.readlines()
    contents.insert(0, f'#area = {triarea} sq deg\n#Av at infinity = {av_inf}\n')
    with open(tri_file_path, "w", encoding='utf-8') as f:
        contents = "".join(contents)
        f.write(contents)


# pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
def create_single_perturb_auf(r, dr, j0s, num_trials, psf_fwhm, density_mag, a_photo, a_snr,
                              localn, d_mag, mag_cut, dd_params, l_cut, run_fw, run_psf, al_av,
                              avs, fit_gal_flag, sky_area=None, saturation_magnitude=None, cmau_array=None,
                              wav=None, z_max=None, nz=None, alpha0=None, alpha1=None, alpha_weight=None,
                              ab_offset=None, filter_name=None, tri_file_path=None, filt_header=None,
                              dens_hist_tri=None, model_mags=None, model_mag_mids=None,
                              model_mags_interval=None, n_bright_sources_star=None):
    r'''
    Creates the associated parameters for describing a single perturbation AUF
    component, for a single sky position.

    Parameters
    ----------
    r : numpy.ndarray
        Array of real-space positions.
    dr : numpy.ndarray
        Array of the bin sizes of each ``r`` position.
    j0s : numpy.ndarray
        The Bessel Function of First Kind of Zeroth Order, evaluated at all
        ``r``-``rho`` combinations.
    num_trials : integer
        The number of realisations of blended contaminant sources to draw
        when simulating perturbations of source positions.
    psf_fwhm : float
        The full-width at half maxima of the ``filt`` filter.
    density_mag : float
        The limiting magnitude above which to consider local normalising densities,
        corresponding to the ``filt`` bandpass.
    a_photo : numpy.ndarray
        The photometry of each source for which simulated perturbations should be
        made.
    a_snr : numpy.ndarray
        The signal-to-noise ratios of each source in ``a_photo``.
    localn : numpy.ndarray
        The local normalising densities for each source.
    d_mag : float
        The interval at which to bin the magnitudes of a given set of objects,
        for the creation of the appropriate brightness/density combinations to
        simulate.
    mag_cut : numpy.ndarray or list of floats
        The magnitude offsets -- or relative fluxes -- above which to keep track of
        the fraction of objects suffering from a contaminating source.
    dd_params : numpy.ndarray
        Polynomial fits for the various parameters controlling the background
        limited PSF-fit algorithm for centroid perturbations.
    l_cut : numpy.ndarray or list
        Relative flux cutoffs for which algorithm to use in the background
        limited PSF-fit algorithm case.
    run_fw : bool
        Flag indicating whether to run the "flux-weighted" version of the
        perturbation algorithm.
    run_psf : bool
        Flag indicating whether to run the "background-dominated PSF" version
        of the perturbation algorithm.
    al_av : float
        Reddening vector for the filter, :math:`\frac{A_\lambda}{A_V}`.
    avs : list or numpy.ndarray of floats
        Sampling of V-band extinctions from within the region for which we wish
        to define the AUF perturbation components.
    fit_gal_flag : bool
        Flag to indicate whether to simulate galaxy counts for the purposes of
        simulating the perturbation component of the AUF.
    sky_area : float, optional
        Area of the region in question, in square degrees. Required if
        ``fit_gal_flag`` is ``True``.
    saturation_magnitude : float, optional
        Magnitude at which the given filter experiences source dropout due to
        saturation effects. Required if ``fit_gal_flag`` is ``True``.
    cmau_array : numpy.ndarray, optional
        Array holding the c/m/a/u values that describe the parameterisation
        of the Schechter functions with wavelength, following Wilson (2022, RNAAS,
        6, 60) [1]_. Shape should be `(5, 2, 4)`, with 5 parameters for both blue
        and red galaxies.
    wav : float, optional
        Wavelength, in microns, of the filter of the current observations.
    z_max : float, optional
        Maximum redshift to simulate differential galaxy counts out to.
    nz : int, optional
        Number of redshifts to simulate, to dictate resolution of Schechter
        functions used to generate differential galaxy counts.
    alpha0 : list of numpy.ndarray or numpy.ndarray, optional
        Zero-redshift indices used to calculate Dirichlet SED coefficients,
        used within the differential galaxy count simulations. Should either be
        a two-element list or shape ``(2, 5)`` array. See [2]_ and [3]_ for
        more details.
    alpha1 : list of numpy.ndarray or numpy.ndarray, optional
        Dirichlet SED coefficients at z=1.
    alpha_weight : list of numpy.ndarray or numpy.ndarray, optional
        Weights used to derive the ``kcorrect`` coefficients within the
        galaxy count framework.
    ab_offset : float, optional
        The zero point difference between the chosen filter and the AB system,
        for conversion of simulated galaxy counts from AB magnitudes. Should
        be of the convention m = m_AB - ab_offset
    filter_name : string, optional
        The ``speclite`` style ``group_name-band_name`` name for the filter,
        for use in the creation of simulated galaxy counts.
    tri_file_path : string or None, optional
        Location on disk where the TRILEGAL datafile is stored, and where the
        individual filter-specific perturbation AUF simulations should be saved.
        Must be provided if not providing pre-computed TRILEGAL magnitude counts.
    filt_header : float or None, optional
        The filter name, as given by the TRILEGAL datafile, for this simulation.
        Must be provided along with ``tri_folder``, or must be ``None`` if
        ``tri_folder`` is ``None``.
    dens_hist_tri : list or numpy.ndarray or None, optional
        A pre-computed list of densities (per square degree per magnitude) of
        simulated star counts, such as from TRILEGAL, and the output from
        ``make_tri_counts``. Must be provided if ``tri_folder`` is ``None`` and
        be ``None`` is ``tri_folder`` is specified.
    model_mags : list or numpy.ndarray or None, optional
        Left-hand bin edges of magnitudes of star differential source counts.
        Must be given if ``dens_hist_tri`` is given, otherwise be ``None``.
    model_mag_mids : list or numpy.ndarray or None, optional
        Magnitude bin centres of star differential source counts. Must be given
        if ``dens_hist_tri`` is given, otherwise be ``None``.
    model_mags_interval : list or numpy.ndarray or None, optional
        Widths of magnitude bins of star differential source counts. Must be
        given if ``dens_hist_tri`` is given, otherwise be ``None``.
    n_bright_sources_star : integer or None, optional
        Number of sources above a specified magnitude limit ``density_mag``
        for resolution purposes, as given by ``make_tri_counts``. Must be given
        if ``dens_hist_tri`` is given, otherwise be ``None``.

    Returns
    -------
    count_array : numpy.ndarray
        The simulated local normalising densities that were used to simulate
        potential perturbation distributions.

    References
    ----------
    .. [1] Wilson T. J. (2022), RNAAS, 6, 60
    .. [2] Herbel J., Kacprzak T., Amara A., et al. (2017), JCAP, 8, 35
    .. [3] Blanton M. R., Roweis S. (2007), AJ, 133, 734

    '''
    # TODO: extend to allow a Galactic source model that doesn't depend on TRILEGAL  pylint: disable=fixme
    if tri_file_path is not None:
        (dens_hist_tri, model_mags, model_mag_mids, model_mags_interval, _,
         n_bright_sources_star) = make_tri_counts(
            tri_file_path, filt_header, d_mag, np.amin(a_photo), density_mag, al_av=al_av, av_grid=avs)

    log10y_tri = -np.inf * np.ones_like(dens_hist_tri)
    log10y_tri[dens_hist_tri > 0] = np.log10(dens_hist_tri[dens_hist_tri > 0])

    mag_slice = model_mags+model_mags_interval <= density_mag
    tri_count = np.sum(10**log10y_tri[mag_slice] * model_mags_interval[mag_slice])

    if fit_gal_flag:
        al_grid = al_av * avs
        z_array = np.linspace(0, z_max, nz)
        gal_dens = create_galaxy_counts(cmau_array, model_mag_mids, z_array, wav, alpha0, alpha1,
                                        alpha_weight, ab_offset, filter_name, al_grid)
        gal_count = np.sum(gal_dens[mag_slice] * model_mags_interval[mag_slice])
        log10y_gal = -np.inf * np.ones_like(log10y_tri)
        log10y_gal[gal_dens > 0] = np.log10(gal_dens[gal_dens > 0])
    else:
        gal_count = 0
        log10y_gal = -np.inf * np.ones_like(log10y_tri)

        # If we're not generating galaxy counts, we have to solely rely on
        # TRILEGAL counting statistics, so we only want to keep populated bins.
        hc = np.where(dens_hist_tri > 0)[0]
        model_mag_mids = model_mag_mids[hc]
        model_mags_interval = model_mags_interval[hc]
        log10y_tri = log10y_tri[hc]

    # If we have the two-component model for source counts, then we have to
    # allow for their relative normalisation factors to not be right, and hence
    # perform a quick least-squares fit to the ensemble counts now to re-weight
    # the two components. If just one is used, then with the re-normalisation
    # within paf.perturb_aufs this doesn't matter, so we set corrections to 1/0
    # respectively.
    if fit_gal_flag:
        _data_hist, _data_bins = np.histogram(a_photo, bins='auto')
        d_hc = np.where(_data_hist > 3)[0]
        data_hist = _data_hist[d_hc]
        data_dbins = np.diff(_data_bins)[d_hc]
        data_bins = _data_bins[d_hc]

        data_uncert = np.sqrt(data_hist) / data_dbins / sky_area
        data_hist = data_hist / data_dbins / sky_area
        data_loghist = np.log10(data_hist)
        data_dloghist = 1/np.log(10) * data_uncert / data_hist

        # pylint: disable-next=fixme
        # TODO: make the half-mag offset flexible, passing from CrossMatch and/or
        # directly into AstrometricCorrections.
        maxmag = _data_bins[:-1][np.argmax(_data_hist)] - 0.5

        q = (data_bins <= maxmag) & (data_bins >= saturation_magnitude)
        tri_corr, gal_corr = find_model_counts_corrections(data_loghist[q], data_dloghist[q],
                                                           data_bins[q]+data_dbins[q]/2, 10**log10y_tri,
                                                           10**log10y_gal, model_mags_interval)
    else:
        tri_corr, gal_corr = 1, 0
    model_count = tri_count * tri_corr + gal_count * gal_corr

    if fit_gal_flag:
        # If we have both galaxies and stars to consider, both can be sufficiently
        # high number counts to make a valid model density. Simply scale the
        # number of simulated Galactic sources by the ratio of bright-magnitude
        # densities to get an effective "number" of galaxies.
        n_bright_sources_gal = int(gal_count / tri_count * n_bright_sources_star)
        tot_n_bright_sources = n_bright_sources_star + n_bright_sources_gal
    else:
        # More straightforward, without any galaxy counts we simply check for
        # if we returned a good number of Galactic sources in our simulation.
        tot_n_bright_sources = n_bright_sources_star

    if tot_n_bright_sources < 100:
        raise ValueError("The number of simulated objects in this sky patch is too low to "
                         "reliably derive a model source density. Please include "
                         "more simulated objects.")

    log10y = np.log10(10**log10y_tri * tri_corr + 10**log10y_gal * gal_corr)

    # Set a magnitude bin width of 0.25 mags, to avoid oversampling.
    dmag = 0.25
    mag_min = dmag * np.floor(np.amin(a_photo)/dmag)
    mag_max = dmag * np.ceil(np.amax(a_photo)/dmag)
    magbins = np.arange(mag_min, mag_max+1e-10, dmag)
    # For local densities, we want a percentage offset, given that we're in
    # logarithmic bins, accepting a log-difference maximum. This is slightly
    # lop-sided, but for 20% results in +18%/-22% limits, which is fine.
    dlogn = 0.2
    lognvals = np.log(localn)
    logn_min = dlogn * np.floor(np.amin(lognvals)/dlogn)
    logn_max = dlogn * np.ceil(np.amax(lognvals)/dlogn)
    lognbins = np.arange(logn_min, logn_max+1e-10, dlogn)

    counts, lognbins, magbins = np.histogram2d(lognvals, a_photo, bins=[lognbins, magbins])
    ni, magi = np.where(counts > 0)
    mag_array = 0.5*(magbins[1:]+magbins[:-1])[magi]
    count_array = np.exp(0.5*(lognbins[1:]+lognbins[:-1])[ni])

    psf_r = 1.185 * psf_fwhm

    n_sources_inv_max_snr = 1000
    inv_max_snr = 1 / np.percentile(a_snr[np.argsort(a_snr)][:n_sources_inv_max_snr], 50)
    snr, _, _ = binned_statistic(a_photo, a_snr, statistic='median', bins=magbins)
    snr = snr[magi]

    b = 0.05
    dm_max = _calculate_magnitude_offsets(count_array, mag_array, b, snr, model_mag_mids, log10y,
                                          model_mags_interval, psf_r, model_count)

    seed = np.random.default_rng().choice(100000, size=(mff.get_random_seed_size(),
                                                        len(count_array)))

    psf_sig = psf_fwhm / (2 * np.sqrt(2 * np.log(2)))

    if run_fw:
        frac_fw, flux_fw, fourieroffset_fw, offset_fw, cumulative_fw = paf.perturb_aufs(
            count_array, mag_array, r[:-1]+dr/2, dr, r, j0s.T,
            model_mag_mids, model_mags_interval, log10y, model_count,
            (dm_max/d_mag).astype(int), mag_cut, psf_r, psf_sig, num_trials, seed, dd_params,
            l_cut, 'fw')
    if run_psf:
        frac_psf, flux_psf, fourieroffset_psf, offset_psf, cumulative_psf = paf.perturb_aufs(
            count_array, mag_array, r[:-1]+dr/2, dr, r, j0s.T,
            model_mag_mids, model_mags_interval, log10y, model_count,
            (dm_max/d_mag).astype(int), mag_cut, psf_r, psf_sig, num_trials, seed, dd_params,
            l_cut, 'psf')

    if run_fw and run_psf:
        h = 1 - np.sqrt(1 - np.minimum(np.ones_like(snr), inv_max_snr**2 * snr**2))
        flux = h * flux_fw + (1 - h) * flux_psf  # pylint: disable=possibly-used-before-assignment
        h = h.reshape(1, -1)
        frac = h * frac_fw + (1 - h) * frac_psf  # pylint: disable=possibly-used-before-assignment
        offset = h * offset_fw + (1 - h) * offset_psf  # pylint: disable=possibly-used-before-assignment
        # pylint: disable-next=possibly-used-before-assignment
        cumulative = h * cumulative_fw + (1 - h) * cumulative_psf
        # pylint: disable-next=possibly-used-before-assignment
        fourieroffset = h * fourieroffset_fw + (1 - h) * fourieroffset_psf
    elif run_fw:
        flux = flux_fw
        frac = frac_fw
        offset = offset_fw
        cumulative = cumulative_fw
        fourieroffset = fourieroffset_fw
    else:
        flux = flux_psf
        frac = frac_psf
        offset = offset_psf
        cumulative = cumulative_psf
        fourieroffset = fourieroffset_psf

    single_perturb_auf_output = {}
    for name, entry in zip(
            ['frac', 'flux', 'offset', 'cumulative', 'fourier', 'Narray', 'magarray'],
            [frac, flux, offset, cumulative, fourieroffset, count_array, mag_array]):
        single_perturb_auf_output[name] = entry

    return single_perturb_auf_output


# pylint: disable=too-many-locals,too-many-statements
def make_tri_counts(trifilepath, trifiltname, dm, brightest_source_mag,
                    density_mag, use_bright=False, use_faint=True, al_av=None, av_grid=None):
    """
    Combine TRILEGAL simulations for a given line of sight in the Galaxy, using
    both a "bright" simulation, with a brighter magnitude limit that allows for
    more detail in the lower-number statistic bins, and a "faint" or full
    simulation down to the faint limit to capture the full source count
    distribution for the filter.

    Parameters
    ----------
    trifilepath : string
        Location on disk into which the TRILEGAL simulations are saved, to
        which "_bright" and "_faint" will be added for the two runs
        respectively, as needed.
    trifiltname : string
        The individual filter within ``trifilterset`` being used for generating
        differential source counts.
    dm : float
        Width of the bins into which to place simulated magnitudes.
    brightest_source_mag : float
        Magnitude in the appropriate ``trifiltname`` bandpass of the brightest
        source that these simulations are relevant for.
    density_mag : float
        The magnitude at which the counts of the corresponding dataset this
        TRILEGAL simulation is for turns over, suffering completeness limit
        effects.
    use_bright : boolean, optional
        Controls whether we load a "bright" set of TRILEGAL sources or not.
    use_faint : boolean, optional
        Determines whether we use a larger dynamic range, fainter TRILEGAL
        simulation to create a histogram of source counts.
    al_av : float, optional
        Differential extinction vector relative to the V-band. If given,
        ``av_grid`` must also be provided; together these will be used
        to manually extinct the TRILEGAL counts (assumed to be subject to zero
        reddening) to simulate differential extinction within the region.
    av_grid : numpy.ndarray, optional
        Grid of extinctions across the region TRILEGAL simulations are valid
        for. Must be provided if ``al_av`` is given.

    Returns
    -------
    dens : numpy.ndarray
        The probability density function of the resulting merged differential
        source counts from the two TRILEGAL simulations, weighted by their
        counting-statistic bin uncertainties.
    tri_mags : numpy.ndarray
        The left-hand bin edges of all bins used to generate ``dens``.
    tri_mags_mids : numpy.ndarray
        Middle of each bin generating ``dens``.
    dtri_mags : numpy.ndarray
        Bin widths of all bins corresponding to each element of ``dens``.
    uncert : numpy.ndarray
        Propagated Poissonian uncertainties of the PDF of ``dens``, using the
        weighted average of the individual uncertainties of each run for every
        bin in ``dens``.
    num_bright_obj : integer
        Number of simulated objects above the given ``density_mag`` brightness
        limit.
    """
    if not use_bright and not use_faint:
        raise ValueError("use_bright and use_faint cannot both be 'False'.")
    if (al_av is None and av_grid is not None) or (al_av is not None and av_grid is None):
        raise ValueError("If one of al_av or av_grid is provided the other must be given as well.")
    if use_faint:
        x, y = os.path.splitext(trifilepath)
        full_file = x + "_faint" + y
        with open(full_file, "r", encoding='utf-8') as f:
            area_line = f.readline()
            av_line = f.readline()
        # #area = {} sq deg, #Av at infinity = {} should be the first two lines, so
        # just split that by whitespace
        bits = area_line.split(' ')
        tri_area_faint = float(bits[2])
        bits = av_line.split(' ')
        tri_av_inf_faint = float(bits[4])
        if tri_av_inf_faint < 0.1 and av_grid is not None:
            raise ValueError("tri_av_inf_faint cannot be smaller than 0.1 while using av_grid.")
        tri_faint = np.genfromtxt(full_file, delimiter=None, names=True, comments='#', skip_header=2,
                                  usecols=[trifiltname, 'Av'])

    if use_bright:
        x, y = os.path.splitext(trifilepath)
        full_file = x + "_bright" + y
        with open(full_file, "r", encoding='utf-8') as f:
            area_line = f.readline()
            av_line = f.readline()
        bits = area_line.split(' ')
        tri_area_bright = float(bits[2])
        bits = av_line.split(' ')
        tri_av_inf_bright = float(bits[4])
        if tri_av_inf_bright < 0.1 and av_grid is not None:
            raise ValueError("tri_av_inf_bright cannot be smaller than 0.1 while using av_grid.")
        tri_bright = np.genfromtxt(full_file, delimiter=None, names=True, comments='#', skip_header=2,
                                   usecols=[trifiltname, 'Av'])

    if use_faint:
        tridata_faint = tri_faint[:][trifiltname]
        tri_av_faint = np.amax(tri_faint[:]['Av'])
        if al_av is not None:
            avs_faint = tri_faint[:]['Av']
        del tri_faint
    if use_bright:
        tridata_bright = tri_bright[:][trifiltname]
        tri_av_bright = np.amax(tri_bright[:]['Av'])
        if al_av is not None:
            avs_bright = tri_bright[:]['Av']
        del tri_bright

    minmag = dm * np.floor(brightest_source_mag/dm)
    if use_bright and use_faint:
        # pylint: disable-next=possibly-used-before-assignment
        maxmag = dm * np.ceil(max(np.amax(tridata_faint), np.amax(tridata_bright))/dm)
    elif use_bright:
        maxmag = dm * np.ceil(np.amax(tridata_bright)/dm)
    elif use_faint:
        maxmag = dm * np.ceil(np.amax(tridata_faint)/dm)
    if al_av is None:
        tri_mags = np.arange(minmag, maxmag+1e-10, dm)  # pylint: disable=possibly-used-before-assignment
    else:
        # Pad the brightest magnitude (minmag) by the possibility of AV=0,
        # scaled to current reddening vector.
        if use_bright and use_faint:
            # pylint: disable-next=possibly-used-before-assignment
            tri_mags = np.arange(minmag-al_av*max(tri_av_faint, tri_av_bright), maxmag+1e-10, dm)
        elif use_bright:
            tri_mags = np.arange(minmag-al_av*tri_av_bright, maxmag+1e-10, dm)
        elif use_faint:
            tri_mags = np.arange(minmag-al_av*tri_av_faint, maxmag+1e-10, dm)
    tri_mags_mids = tri_mags[:-1]+np.diff(tri_mags)/2  # pylint: disable=used-before-assignment
    if use_faint:
        if al_av is None:
            hist, tri_mags = np.histogram(tridata_faint, bins=tri_mags)
        else:
            hist = np.zeros((len(tri_mags) - 1), int)
            for av in av_grid:
                # Take the ratio of AVs for scaling (i.e., if we'd run TRILEGAL
                # with AV=1 but av_grid[0] = 2, we get 2x the extinction at each
                # distance we'd otherwise have found. Or, if AV=1,
                # av_grid[1]=0.25, then we have a quarter the infinite-distance
                # extinction and hence 25% the extinction applied to the source.
                av_ratio = av / tri_av_inf_faint
                # Apply the correction. Here if av_grid[i] = AV then we do
                # nothing; otherwise av_ratio = 2 gives an extra 100% AV,
                # and e.g. av_ratio = 0.25 subtracts three-quarters of the
                # applied AV value. These are scaled to the correct extinction
                # vector subsequently.
                m = tridata_faint + (av_ratio - 1) * avs_faint * al_av
                y, _ = np.histogram(m, bins=tri_mags)
                hist += y
        hc_faint = hist > 3
        dens_faint = hist / np.diff(tri_mags) / tri_area_faint
        dens_uncert_faint = np.sqrt(hist) / np.diff(tri_mags) / tri_area_faint
        # Account for summing NxM Avs here by dividing out len(av_grid).
        if av_grid is not None:
            dens_faint = dens_faint / len(av_grid)
            dens_uncert_faint = dens_uncert_faint / np.sqrt(len(av_grid))
        dens_uncert_faint[dens_uncert_faint == 0] = 1e10
        # Now check whether there are sufficient sources at the bright end of
        # the simulation, counting the sources brighter than density_mag.
        num_bright_obj_faint = np.sum(hist[tri_mags[:-1] < density_mag])
        if av_grid is not None:
            num_bright_obj_faint /= len(av_grid)
    if use_bright:
        if al_av is None:
            hist, tri_mags = np.histogram(tridata_bright, bins=tri_mags)
        else:
            hist = np.zeros((len(tri_mags) - 1), int)
            for av in av_grid:
                av_ratio = av / tri_av_inf_bright
                m = tridata_bright + (av_ratio - 1) * avs_bright * al_av
                y, _ = np.histogram(m, bins=tri_mags)
                hist += y
        hc_bright = hist > 3
        dens_bright = hist / np.diff(tri_mags) / tri_area_bright
        dens_uncert_bright = np.sqrt(hist) / np.diff(tri_mags) / tri_area_bright
        if av_grid is not None:
            dens_bright = dens_bright / len(av_grid)
            dens_uncert_bright = dens_uncert_bright / np.sqrt(len(av_grid))
        dens_uncert_bright[dens_uncert_bright == 0] = 1e10
        num_bright_obj_bright = np.sum(hist[tri_mags[:-1] < density_mag])
        if av_grid is not None:
            num_bright_obj_bright /= len(av_grid)
    if use_bright and use_faint:
        # Assume that the number of objects in the bright dataset is truncated such
        # that it should be most dense at its faintest magnitude, and ignore cases
        # where objects may have "scattered" outside of that limit. These are most
        # likely to be objects in magnitudes that don't define the TRILEGAL cutoff,
        # where differential reddening can make a few of them slightly fainter than
        # average.
        bright_cutoff_mag = tri_mags[1:][np.argmax(hist)]
        dens_uncert_bright[tri_mags[1:] > bright_cutoff_mag] = 1e10
        w_f, w_b = 1 / dens_uncert_faint**2, 1 / dens_uncert_bright**2
        dens = (dens_bright * w_b + dens_faint * w_f) / (w_b + w_f)
        dens_uncert = (dens_uncert_bright * w_b + dens_uncert_faint * w_f) / (w_b + w_f)
        hc = hc_bright | hc_faint  # pylint: disable=possibly-used-before-assignment

        num_bright_obj = max(num_bright_obj_faint, num_bright_obj_bright)
    elif use_bright:
        dens = dens_bright
        dens_uncert = dens_uncert_bright
        hc = hc_bright

        num_bright_obj = num_bright_obj_bright
    elif use_faint:
        dens = dens_faint
        dens_uncert = dens_uncert_faint
        hc = hc_faint

        num_bright_obj = num_bright_obj_faint

    dens = dens[hc]  # pylint: disable=used-before-assignment,possibly-used-before-assignment
    dtri_mags = np.diff(tri_mags)[hc]
    tri_mags_mids = tri_mags_mids[hc]
    tri_mags = tri_mags[:-1][hc]
    uncert = dens_uncert[hc]  # pylint: disable=possibly-used-before-assignment

    # pylint: disable-next=possibly-used-before-assignment
    return dens, tri_mags, tri_mags_mids, dtri_mags, uncert, num_bright_obj


def _calculate_magnitude_offsets(count_array, mag_array, b, snr, model_mag_mids, log10y,
                                 model_mags_interval, r, n_norm):
    '''
    Derive minimum relative fluxes, or largest magnitude offsets, down to which
    simulated perturbers need to be simulated, based on both considerations of
    their flux relative to the noise of the primary object and the fraction of
    simulations in which there is no simulated perturbation.

    Parameters
    ----------
    count_array : numpy.ndarray
        Local normalising densities of simulations.
    mag_array : numpy.ndarray
        Magnitudes of central objects to have perturbations simulated for.
    b : float
        Fraction of ``snr`` the flux of the perturber should be; e.g. for
        1/20th ``B`` should be 0.05.
    snr : numpy.ndarray
        Theoretical signal-to-noise ratios of each object in ``mag_array``.
    model_mag_mids : numpy.ndarray
        Model magnitudes for simulated densities of background objects.
    log10y : numpy.ndarray
        log-10 source densities of simulated objects in the given line of sight.
    model_mags_interval : numpy.ndarray
        Widths of the bins for each ``log10y``.
    r : float
        Radius of the PSF of the given simulation, in arcseconds.
    n_norm : float
        Normalising local density of simulations, to scale to each
        ``count_array``.

    Returns
    -------
    dm : numpy.ndarray
        Maximum magnitude offset required for simulations, based on SNR and
        empty simulation fraction.
    '''
    flim = b / snr
    dm_max_snr = -2.5 * np.log10(flim)

    dm_max_no_perturb = np.empty_like(mag_array)
    for i, mag in enumerate(mag_array):
        q = model_mag_mids >= mag
        if np.sum(q) == 0:
            dm_max_no_perturb[i] = 0
            continue
        _x = model_mag_mids[q]
        _y = 10**log10y[q] * model_mags_interval[q] * np.pi * (r/3600)**2 * count_array[i] / n_norm

        # Convolution of Poissonian distributions each with l_i is a Poissonian
        # with mean of sum_i l_i.
        lamb = np.cumsum(_y)
        # CDF of Poissonian is regularised gamma Q(floor(k + 1), lambda), and we
        # want k = 0; we wish to find the dm that gives sufficiently large lambda
        # that k = 0 only occurs <= x% of the time. If lambda is too small then
        # k = 0 is too likely. P(X <= 0; lambda) = exp(-lambda).
        # For 1% chance of no perturber we want 0.01 = exp(-lambda); rearranging
        # lambda = -ln(0.01).
        q = np.where(lamb >= -np.log(0.01))[0]
        if len(q) > 0:
            dm_max_no_perturb[i] = _x[q[0]] - mag
        else:
            # In the case that we can't go deep enough in our simulated counts to
            # get <1% chance of no perturber, just do the best we can.
            dm_max_no_perturb[i] = _x[-1] - mag

    dm = np.maximum(dm_max_snr, dm_max_no_perturb)

    return dm
