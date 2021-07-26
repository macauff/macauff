# Licensed under a 3-clause BSD style license - see LICENSE
'''
This module provides the framework to handle the creation of the perturbation
component of the astrometric uncertainty function.
'''

import os
import sys
import numpy as np

from .misc_functions import (_load_single_sky_slice, _load_rectangular_slice,
                             _create_rectangular_slice_arrays)
from .misc_functions_fortran import misc_functions_fortran as mff
from .get_trilegal_wrapper import get_trilegal
from .perturbation_auf_fortran import perturbation_auf_fortran as paf

__all__ = ['make_perturb_aufs', 'create_single_perturb_auf']


def make_perturb_aufs(auf_folder, cat_folder, filters, auf_points, r, dr, rho,
                      drho, which_cat, include_perturb_auf, mem_chunk_num,
                      tri_download_flag=False, delta_mag_cuts=None, psf_fwhms=None,
                      tri_set_name=None, tri_filt_num=None, tri_filt_names=None,
                      auf_region_frame=None, num_trials=None, j0s=None, density_mags=None,
                      dm_max=None, d_mag=None, compute_local_density=None, density_radius=None):
    """
    Function to perform the creation of the blended object perturbation component
    of the AUF.

    Parameters
    ----------
    auf_folder : string
        The overall folder into which to create filter-pointing folders and save
        individual simulation files.
    cat_folder : string
        The folder that the photometric catalogue being simulated for perturbation
        AUF component is stored in.
    filters : list of strings or numpy.ndarray of strings
        An array containing the list of filters in this catalogue to create
        simulated AUF components for.
    auf_points : numpy.ndarray
        Two-dimensional array containing pairs of coordinates at which to evaluate
        the perturbation AUF components.
    r : numpy.ndarray
        The real-space coordinates for the Hankel transformations used in AUF-AUF
        convolution.
    dr : numpy.ndarray
        The spacings between ``r`` elements.
    rho : numpy.ndarray
        The fourier-space coordinates for Hankel transformations.
    drho : numpy.ndarray
        The spacings between ``rho`` elements.
    which_cat : string
        Indicator as to whether these perturbation AUFs are for catalogue "a"
        or catalogue "b" within the cross-match process.
    include_perturb_auf : boolean
        ``True`` or ``False`` flag indicating whether perturbation component of the
        AUF should be used or not within the cross-match process.
    mem_chunk_num : int
        Number of individual sub-sections to break catalogue into for memory
        saving purposes.
    tri_download_flag : boolean, optional
        A ``True``/``False`` flag, whether to re-download TRILEGAL simulated star
        counts or not if a simulation already exists in a given folder. Only
        needed if ``include_perturb_auf`` is True.
    delta_mag_cuts : numpy.ndarray, optional
        Array of magnitude offsets corresponding to relative fluxes of perturbing
        sources, for consideration of relative contamination chances. Must be given
        if ``include_perturb_auf`` is ``True``.
    psf_fwhms : numpy.ndarray, optional
        Array of full width at half-maximums for each filter in ``filters``. Only
        required if ``include_perturb_auf`` is True; defaults to ``None``.
    tri_set_name : string, optional
        Name of the filter set to generate simulated TRILEGAL Galactic source
        counts from. If ``include_perturb_auf`` and ``tri_download_flag`` are
        ``True``, this must be set.
    tri_filt_num : string, optional
        Column number of the filter defining the magnitude limit of simulated
        TRILEGAL Galactic sources. If ``include_perturb_auf`` and
        ``tri_download_flag`` are ``True``, this must be set.
    tri_filt_names : list or array of strings, optional
        List of filter names in the TRILEGAL filterset defined in ``tri_set_name``,
        in the same order as provided in ``psf_fwhms``. If ``include_perturb_auf``
        and ``tri_download_flag`` are ``True``, this must be set.
    auf_region_frame : string, optional
        Coordinate reference frame in which sky coordinates are defined, either
        ``equatorial`` or ``galactic``, used to define the coordinates TRILEGAL
        simulations are generated in. If ``include_perturb_auf`` and
        ``tri_download_flag`` are ``True``, this must be set.
    num_trials : integer, optional
        The number of simulated PSFs to compute in the derivation of the
        perturbation component of the AUF. Must be given if ``include_perturb_auf``
        is ``True``.
    j0s : numpy.ndarray, optional
        The Bessel Function of First Kind of Zeroth Order evaluated at each
        ``r``-``rho`` combination. Must be given if ``include_perturb_auf``
        is ``True``.
    density_mags : numpy.ndarray, optional
        The faintest magnitude down to which to consider number densities of
        objects for normalisation purposes, in each filter, in the same order
        as ``psf_fwhms``. Must be given if ``include_perturb_auf`` is ``True``.
        Must be the same as were used to calculate ``count_density`` in
        ``calculate_local_density``, if ``compute_local_density`` is ``False``
        and pre-computed densities are being used.
    dm_max : float, optional
        The maximum magnitude difference, or relative flux, down to which to
        consider simulated blended sources. Must be given if
        ``include_perturb_auf`` is ``True``.
    d_mag : float, optional
        The resolution at which to create the TRILEGAL source density distribution.
        Must be provided if ``include_perturb_auf`` is ``True``.
    compute_local_density : boolean, optional
        Flag to indicate whether to calculate local source densities during
        the cross-match process, or whether to use pre-calculated values. Must
        be provided if ``include_perturb_auf`` is ``True``.
    density_radius : float, optional
        The astrometric distance, in degrees, within which to consider numbers
        of internal catalogue sources, from which to calculate local density.
        Must be given if both ``include_perturb_auf`` and
        ``compute_local_density`` are both ``True``.
    """
    if include_perturb_auf and tri_download_flag and tri_set_name is None:
        raise ValueError("tri_set_name must be given if include_perturb_auf and tri_download_flag "
                         "are both True.")
    if include_perturb_auf and tri_download_flag and tri_filt_num is None:
        raise ValueError("tri_filt_num must be given if include_perturb_auf and tri_download_flag "
                         "are both True.")
    if include_perturb_auf and tri_download_flag and auf_region_frame is None:
        raise ValueError("auf_region_frame must be given if include_perturb_auf and "
                         "tri_download_flag are both True.")

    if include_perturb_auf and tri_filt_names is None:
        raise ValueError("tri_filt_names must be given if include_perturb_auf is True.")
    if include_perturb_auf and delta_mag_cuts is None:
        raise ValueError("delta_mag_cuts must be given if include_perturb_auf is True.")
    if include_perturb_auf and psf_fwhms is None:
        raise ValueError("psf_fwhms must be given if include_perturb_auf is True.")
    if include_perturb_auf and num_trials is None:
        raise ValueError("num_trials must be given if include_perturb_auf is True.")
    if include_perturb_auf and j0s is None:
        raise ValueError("j0s must be given if include_perturb_auf is True.")
    if include_perturb_auf and density_mags is None:
        raise ValueError("density_mags must be given if include_perturb_auf is True.")
    if include_perturb_auf and dm_max is None:
        raise ValueError("dm_max must be given if include_perturb_auf is True.")
    if include_perturb_auf and d_mag is None:
        raise ValueError("d_mag must be given if include_perturb_auf is True.")
    if include_perturb_auf and compute_local_density is None:
        raise ValueError("compute_local_density must be given if include_perturb_auf is True.")
    if include_perturb_auf and compute_local_density and density_radius is None:
        raise ValueError("density_radius must be given if include_perturb_auf and "
                         "compute_local_density are both True.")

    print('Creating perturbation AUFs sky indices for catalogue "{}"...'.format(which_cat))
    sys.stdout.flush()

    n_sources = len(np.load('{}/con_cat_astro.npy'.format(cat_folder), mmap_mode='r'))

    modelrefinds = np.lib.format.open_memmap('{}/modelrefinds.npy'.format(auf_folder),
                                             mode='w+', dtype=int, shape=(3, n_sources),
                                             fortran_order=True)

    for cnum in range(0, mem_chunk_num):
        lowind = np.floor(n_sources*cnum/mem_chunk_num).astype(int)
        highind = np.floor(n_sources*(cnum+1)/mem_chunk_num).astype(int)
        a = np.load('{}/con_cat_astro.npy'.format(cat_folder), mmap_mode='r')[lowind:highind]
        # As we chunk in even steps through the files this is simple for now,
        # but could be replaced with a more complex mapping in the future.
        indexmap = np.arange(lowind, highind, 1)

        # Which sky position to use is more complex; this involves determining
        # the smallest great-circle distance to each auf_point AUF mapping for
        # each source.
        modelrefinds[2, indexmap] = mff.find_nearest_point(a[:, 0], a[:, 1],
                                                           auf_points[:, 0], auf_points[:, 1])

    print('Creating empirical perturbation AUFs for catalogue "{}"...'.format(which_cat))
    sys.stdout.flush()

    # Store the length of the density-magnitude combinations in each sky/filter
    # combination for future loading purposes.
    arraylengths = np.lib.format.open_memmap('{}/arraylengths.npy'.format(auf_folder), mode='w+',
                                             dtype=int, shape=(len(filters), len(auf_points)),
                                             fortran_order=True)

    # Overload compute_local_density if it is False but local_N does not exist.
    if not compute_local_density and not os.path.isfile('{}/local_N.npy'.format(auf_folder)):
        compute_local_density = True

    if include_perturb_auf:
        a_tot_photo = np.load('{}/con_cat_photo.npy'.format(cat_folder), mmap_mode='r')
        if compute_local_density:
            a_tot_astro = np.load('{}/con_cat_astro.npy'.format(cat_folder), mmap_mode='r')
            # Set up the temporary sky slice memmap arrays quickly, as they will
            # be needed in calculate_local_density later.
            _create_rectangular_slice_arrays(auf_folder, '', len(a_tot_astro))
            memmap_slice_arrays = []
            for n in ['1', '2', '3', '4', 'combined']:
                memmap_slice_arrays.append(np.lib.format.open_memmap(
                    '{}/{}_temporary_sky_slice_{}.npy'.format(auf_folder, '', n), mode='r+',
                    dtype=bool, shape=(len(a_tot_astro),)))

    if compute_local_density and include_perturb_auf:
        local_N = np.lib.format.open_memmap('{}/local_N.npy'.format(auf_folder), mode='w+',
                                            dtype=float, shape=(len(a_tot_astro),))

    for i in range(len(auf_points)):
        ax1, ax2 = auf_points[i]
        ax_folder = '{}/{}/{}'.format(auf_folder, ax1, ax2)
        if not os.path.exists(ax_folder):
            os.makedirs(ax_folder, exist_ok=True)

        if include_perturb_auf and (tri_download_flag or not
                                    os.path.isfile('{}/trilegal_auf_simulation.dat'
                                                   .format(ax_folder))):
            download_trilegal_simulation(ax_folder, tri_set_name, ax1, ax2, tri_filt_num,
                                         auf_region_frame)

        if include_perturb_auf:
            sky_cut = _load_single_sky_slice(auf_folder, '', i, modelrefinds[2, :])
            a_photo_cut = a_tot_photo[sky_cut]
            if compute_local_density:
                a_astro_cut = a_tot_astro[sky_cut]

        for j in range(len(filters)):
            filt = filters[j]

            filt_folder = '{}/{}'.format(ax_folder, filt)
            if not os.path.exists(filt_folder):
                os.makedirs(filt_folder, exist_ok=True)

            if include_perturb_auf:
                good_mag_slice = ~np.isnan(a_photo_cut[:, j])
                a_photo = a_photo_cut[good_mag_slice, j]
                if len(a_photo) == 0:
                    arraylengths[j, i] = 0
                    continue
                if compute_local_density:
                    localN = calculate_local_density(
                        a_astro_cut[good_mag_slice], a_tot_astro, a_tot_photo[:, j],
                        auf_folder, cat_folder, density_radius, density_mags[j],
                        memmap_slice_arrays)
                    # Because we always calculate the density from the full
                    # catalogue, using just the astrometry, we should be able
                    # to just over-write this N times if there happen to be N
                    # good detections of a source.
                    local_N[sky_cut][good_mag_slice] = localN
                else:
                    localN = np.load('{}/local_N.npy'.format(auf_folder),
                                     mmap_mode='r')[sky_cut][good_mag_slice, j]
                Narray = create_single_perturb_auf(
                    ax_folder, filters[j], r, dr, rho, drho, j0s, num_trials, psf_fwhms[j],
                    tri_filt_names[j], density_mags[j], a_photo, localN, dm_max, d_mag,
                    delta_mag_cuts)
            else:
                # Without the simulations to force local normalising density N or
                # individual source brightness magnitudes, we can simply combine
                # all data into a single "bin".
                num_N_mag = 1
                # In cases where we do not want to use the perturbation AUF component,
                # we currently don't have separate functions, but instead set up dummy
                # functions and variables to pass what mathematically amounts to
                # "nothing" through the cross-match. Here we would use fortran
                # subroutines to create the perturbation simulations, so we make
                # f-ordered dummy parameters.
                Frac = np.zeros((1, num_N_mag), float, order='F')
                np.save('{}/frac.npy'.format(filt_folder), Frac)
                Flux = np.zeros(num_N_mag, float, order='F')
                np.save('{}/flux.npy'.format(filt_folder), Flux)
                # Remember that r is bins, so the evaluations at bin middle are one
                # shorter in length.
                offset = np.zeros((len(r)-1, num_N_mag), float, order='F')
                # Fix offsets such that the probability density function looks like
                # a delta function, such that a two-dimensional circular coordinate
                # integral would evaluate to one at every point, cf. ``cumulative``.
                offset[0, :] = 1 / (2 * np.pi * (r[0] + dr[0]/2) * dr[0])
                np.save('{}/offset.npy'.format(filt_folder), offset)
                # The cumulative integral of a delta function is always unity.
                cumulative = np.ones((len(r)-1, num_N_mag), float, order='F')
                np.save('{}/cumulative.npy'.format(filt_folder), cumulative)
                # The Hankel transform of a delta function is a flat line; this
                # then preserves the convolution being multiplication in fourier
                # space, as F(x) x 1 = F(x), similar to how f(x) * d(0) = f(x).
                fourieroffset = np.ones((len(rho)-1, num_N_mag), float, order='F')
                np.save('{}/fourier.npy'.format(filt_folder), fourieroffset)
                # Both normalising density and magnitude arrays can be proxied
                # with a dummy parameter, as any minimisation of N-m distance
                # must pick the single value anyway.
                Narray = np.array([[1]], float)
                np.save('{}/N.npy'.format(filt_folder), Narray)
                magarray = np.array([[1]], float)
                np.save('{}/mag.npy'.format(filt_folder), magarray)
            arraylengths[j, i] = len(Narray)

    if include_perturb_auf:
        longestNm = np.amax(arraylengths)

        Narrays = np.lib.format.open_memmap('{}/narrays.npy'.format(auf_folder), mode='w+',
                                            dtype=float, shape=(longestNm, len(filters),
                                            len(auf_points)), fortran_order=True)
        Narrays[:, :, :] = -1
        magarrays = np.lib.format.open_memmap('{}/magarrays.npy'.format(auf_folder), mode='w+',
                                              dtype=float, shape=(longestNm, len(filters),
                                              len(auf_points)), fortran_order=True)
        magarrays[:, :, :] = -1
        for i in range(len(auf_points)):
            ax1, ax2 = auf_points[i]
            ax_folder = '{}/{}/{}'.format(auf_folder, ax1, ax2)
            for j in range(len(filters)):
                if arraylengths[j, i] == 0:
                    continue
                filt = filters[j]
                filt_folder = '{}/{}'.format(ax_folder, filt)
                Narray = np.load('{}/N.npy'.format(filt_folder))
                magarray = np.load('{}/mag.npy'.format(filt_folder))
                Narrays[:arraylengths[j, i], j, i] = Narray
                magarrays[:arraylengths[j, i], j, i] = magarray

    # Once the individual AUF simulations are saved, we also need to calculate
    # the indices each source references when slicing into the 4-D cubes
    # created by [1-D array] x N-m combination x filter x sky position iteration.

    print('Creating perturbation AUFs filter indices for catalogue "{}"...'.format(which_cat))
    sys.stdout.flush()

    for cnum in range(0, mem_chunk_num):
        lowind = np.floor(n_sources*cnum/mem_chunk_num).astype(int)
        highind = np.floor(n_sources*(cnum+1)/mem_chunk_num).astype(int)
        if include_perturb_auf:
            a = np.load('{}/con_cat_photo.npy'.format(cat_folder), mmap_mode='r')[lowind:highind]
            local_N = np.load('{}/local_N.npy'.format(auf_folder), mmap_mode='r')[lowind:highind]
        magref = np.load('{}/magref.npy'.format(cat_folder), mmap_mode='r')[lowind:highind]
        # As we chunk in even steps through the files this is simple for now,
        # but could be replaced with a more complex mapping in the future.
        indexmap = np.arange(lowind, highind, 1)

        if include_perturb_auf:
            for i in range(0, len(a)):
                axind = modelrefinds[2, indexmap[i]]
                filterind = magref[i]
                Nmind = np.argmin((local_N[i] - Narrays[:arraylengths[filterind, axind],
                                                        filterind, axind])**2 +
                                  (a[i, filterind] - magarrays[:arraylengths[filterind, axind],
                                                               filterind, axind])**2)
                modelrefinds[0, indexmap[i]] = Nmind
        else:
            # For the case that we do not use the perturbation AUF component,
            # our dummy N-m files are all one-length arrays, so we can
            # trivially index them, regardless of specifics.
            modelrefinds[0, indexmap] = 0

        # The mapping of which filter to use is straightforward: simply pick
        # the filter index of the "best" filter for each source, from magref.
        modelrefinds[1, indexmap] = magref

    if include_perturb_auf:
        del Narrays, magarrays
        os.remove('{}/narrays.npy'.format(auf_folder))
        os.remove('{}/magarrays.npy'.format(auf_folder))

        # Delete sky slices used to make fourier cutouts.
        os.system('rm {}/*temporary_sky_slice*.npy'.format(auf_folder))
        os.system('rm {}/_small_sky_slice.npy'.format(auf_folder))


def download_trilegal_simulation(tri_folder, tri_filter_set, ax1, ax2, mag_num, region_frame,
                                 total_objs=1.5e6):
    '''
    Get a single Galactic sightline TRILEGAL simulation of an appropriate sky
    size, and save it in a folder for use in the perturbation AUF simulations.

    Parameters
    ----------
    tri_folder : string
        The location of the folder into which to save the TRILEGAL file.
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
    total_objs : integer, optional
        The approximate number of objects to simulate in a TRILEGAL sightline,
        affecting how large an area to request a simulated Galactic region of.
    '''
    areaflag = 0
    triarea = 0.001
    tri_name = 'trilegal_auf_simulation'
    mag_lim = 32
    galactic_flag = True if region_frame == 'galactic' else False
    while areaflag == 0:
        get_trilegal(tri_name, ax1, ax2, folder=tri_folder, galactic=galactic_flag,
                     filterset=tri_filter_set, area=triarea, maglim=mag_lim, magnum=mag_num)
        f = open('{}/{}.dat'.format(tri_folder, tri_name), "r")
        contents = f.readlines()
        f.close()
        # Two comment lines; one at the top and one at the bottom - we add a
        # third in a moment, however
        nobjs = len(contents) - 2
        # If too few stars then increase by factor 10 and loop, or scale to give
        # about 1.5 million stars and come out of area increase loop --
        # simulations can't be more than 10 sq deg, so accept if that's as large
        # as we can go.
        if nobjs < 10000 and triarea < 10:
            triarea = min(10, triarea*10)
        else:
            triarea = min(10, triarea / nobjs * total_objs)
            areaflag = 1
        os.system('rm {}/{}.dat'.format(tri_folder, tri_name))
    get_trilegal(tri_name, ax1, ax2, folder=tri_folder, galactic=galactic_flag,
                 filterset=tri_filter_set, area=triarea, maglim=mag_lim, magnum=mag_num)
    f = open('{}/{}.dat'.format(tri_folder, tri_name), "r")
    contents = f.readlines()
    f.close()
    contents.insert(0, '#area = {} sq deg\n'.format(triarea))
    f = open('{}/{}.dat'.format(tri_folder, tri_name), "w")
    contents = "".join(contents)
    f.write(contents)
    f.close()


def calculate_local_density(a_astro, a_tot_astro, a_tot_photo, auf_folder, cat_folder,
                            density_radius, density_mag, memmap_slice_arrays):
    '''
    Calculates the number of sources above a given brightness within a specified
    radius of each source in a catalogue, to provide a local density for
    normalisation purposes.

    Parameters
    ----------
    a_astro : numpy.ndarray
        Sub-set of astrometric portion of total catalogue, for which local
        densities are to be calculated.
    a_tot_astro : numpy.ndarray
        Full astrometric catalogue, from which all potential sources above
        ``density_mag`` and coeval with ``a_astro`` sources are to be extracted.
    a_tot_photo : numpy.ndarray
        The photometry of the full catalogue, matching ``a_tot_astro``.
    auf_folder : string
        The folder designated to contain the perturbation AUF component related
        data for this catalogue.
    cat_folder : string
        The location of the catalogue files for this dataset.
    density_radius : float
        The radius, in degrees, out to which to consider the number of sources
        for the normalising density.
    density_mag : float
        The brightness, in magnitudes, above which to count sources for density
        purposes.
    memmap_slice_arrays : list of numpy.ndarray
        List of the memmap sky slice arrays, to be used in the loading of the
        rectangular sky patch.

    Returns
    -------
    count_density : numpy.ndarray
        The number of sources per square degree near to each source in
        ``a_astro`` that are above ``density_mag`` in ``a_tot_astro``.
    '''
    min_lon, max_lon = np.amin(a_astro[:, 0]), np.amax(a_astro[:, 0])
    min_lat, max_lat = np.amin(a_astro[:, 1]), np.amax(a_astro[:, 1])

    overlap_sky_cut = _load_rectangular_slice(auf_folder, '', a_tot_astro, min_lon,
                                              max_lon, min_lat, max_lat, density_radius,
                                              memmap_slice_arrays)
    cut = np.lib.format.open_memmap('{}/_temporary_slice.npy'.format(
        auf_folder), mode='w+', dtype=bool, shape=(len(a_tot_astro),))
    di = max(1, len(cut) // 20)
    for i in range(0, len(a_tot_astro), di):
        cut[i:i+di] = overlap_sky_cut[i:i+di] & (a_tot_photo[i:i+di] <= density_mag)
    a_astro_overlap_cut = a_tot_astro[cut]
    os.system('rm {}/_temporary_slice.npy'.format(auf_folder))

    counts = paf.get_density(a_astro[:, 0], a_astro[:, 1], a_astro_overlap_cut[:, 0],
                             a_astro_overlap_cut[:, 1], density_radius)
    min_lon, max_lon = np.amin(a_astro_overlap_cut[:, 0]), np.amax(a_astro_overlap_cut[:, 0])
    min_lat, max_lat = np.amin(a_astro_overlap_cut[:, 1]), np.amax(a_astro_overlap_cut[:, 1])
    circle_overlap_area = np.empty(len(a_astro), float)
    for i in range(len(a_astro)):
        circle_overlap_area[i] = get_circle_overlap_area(
            density_radius, [min_lon, max_lon], [min_lat, max_lat], a_astro[i, [0, 1]])
    count_density = counts / circle_overlap_area

    return count_density


def get_circle_overlap_area(r, x_edges, y_edges, coords):
    '''
    Calculates the overlap between a circle of given radius and rectangle
    defined by four edge coordinates.

    Parameters
    ----------
    r : float
        The radius of the circle.
    x_edges : list or numpy.ndarray
        Upper and lower limits of the rectangle.
    y_edges : list or numpy.ndarray
        Limits of the rectangle in the second orthogonal axis.
    coords : numpy.ndarray or list
        The (x, y) coordinates of the center of each circle to consider
        overlap area with rectangle for.

    Returns
    -------
    area : float
        The area of circle of radius ``r`` which overlaps the rectangle
        defined by ``x_edges`` and ``y_edges``.
    '''
    area = np.pi * r**2
    has_overlapped_edge = [0, 0, 0, 0]
    edges = np.array([x_edges[0], y_edges[0], x_edges[1], y_edges[1]])
    coords = np.array([coords[0], coords[1], coords[0], coords[1]])
    for i, (edge, coord) in enumerate(zip(edges, coords)):
        h = np.abs(coord - edge)
        if h < r:
            # The first chord integration is "free", and does not have
            # truncated limits based on overlaps; the final chord integration,
            # however, cares about truncation on both sides. The "middle two"
            # integrations only truncate to the previous side.
            a = -np.sqrt(r**2 - h**2)
            b = +np.sqrt(r**2 - h**2)

            if i == 1 and has_overlapped_edge[0]:
                a = max(a, x_edges[0] - coords[0])
            if i == 2 and has_overlapped_edge[1]:
                a = max(a, y_edges[0] - coords[1])
            if i == 3 and has_overlapped_edge[0]:
                a = max(a, x_edges[0] - coords[0])
            if i == 3 and has_overlapped_edge[2]:
                b = min(b, x_edges[1] - coords[0])

            chord_area_overlap = chord_integral_eval(b, r, h) - chord_integral_eval(a, r, h)
            has_overlapped_edge[i] = 1

            area -= chord_area_overlap

    return area


def chord_integral_eval(x, r, h):
    '''
    Calculates the indefinite integral of the distance between a given circle
    chord and the circumference of the circle, to calculate the chord area.

    Parameters
    ----------
    x : float
        The integrable coordinate, orthogonal to the line between the center
        of the circle and the chord at height ``h``.
    r : float
        The radius of the circle.
    h : float
        The height of the chord inside the circle of radius ``r``.

    Returns
    -------
    integral : float
        The result of the indefinite integral of the chord area.
    '''
    d = np.sqrt(r**2 - x**2)

    if d == 0:
        x_div_d = np.sign(x) * np.inf
    else:
        x_div_d = x / d

    integral = 0.5 * (x * d + r**2 * np.arctan(x_div_d)) - h * x
    return integral


def create_single_perturb_auf(tri_folder, filt, r, dr, rho, drho, j0s, num_trials, psf_fwhm,
                              header, density_mag, a_photo, localN, dm_max, d_mag, mag_cut):
    '''
    Creates the associated parameters for describing a single perturbation AUF
    component, for a single sky position.

    Parameters
    ----------
    tri_folder : string
        Folder where the TRILEGAL datafile is stored, and where the individual
        filter-specific perturbation AUF simulations should be saved.
    filt : float
        Filter for which to simulate the AUF component.
    r : numpy.ndarray
        Array of real-space positions.
    dr : numpy.ndarray
        Array of the bin sizes of each ``r`` position.
    rho : numpy.ndarray
        Fourier-space coordinates at which to sample the fourier transformation
        of the distribution of perturbations due to blended sources.
    drho : numpy.ndarray
        Bin widths of each ``rho`` coordinate.
    j0s : numpy.ndarray
        The Bessel Function of First Kind of Zeroth Order, evaluated at all
        ``r``-``rho`` combinations.
    num_trials : integer
        The number of realisations of blended contaminant sources to draw
        when simulating perturbations of source positions.
    psf_fwhm : float
        The full-width at half maxima of the ``filt`` filter.
    header : float
        The filter name, as given by the TRILEGAL datafile, for this simulation.
    density_mag : float
        The limiting magnitude above which to consider local normalising densities,
        corresponding to the ``filt`` bandpass.
    a_photo : numpy.ndarray
        The photometry of each source for which simulated perturbations should be
        made.
    localN : numpy.ndarray
        The local normalising densities for each source.
    dm_max : float
        The maximum magnitude down to which to simulate blended objects, limiting
        the smallest possible perturbation considered.
    d_mag : float
        The interval at which to bin the magnitudes of a given set of objects,
        for the creation of the appropriate brightness/density combinations to
        simulate.
    mag_cut : numpy.ndarray or list of floats
        The magnitude offsets -- or relative fluxes -- above which to keep track of
        the fraction of objects suffering from a contaminating source.

    Returns
    -------
    count_array : numpy.ndarray
        The simulated local normalising densities that were used to simulate
        potential perturbation distributions.
    '''
    tri_name = 'trilegal_auf_simulation'
    f = open('{}/{}.dat'.format(tri_folder, tri_name), "r")
    line = f.readline()
    f.close()
    bits = line.split(' ')
    tri_area = float(bits[2])
    tri = np.genfromtxt('{}/{}.dat'.format(tri_folder, tri_name), delimiter=None,
                        names=True, comments='#', skip_header=1)

    # TODO: extend to allow a Galactic source model that doesn't depend on TRILEGAL
    tri_mags = tri[:][header]
    tri_count = np.sum(tri_mags <= density_mag) / tri_area

    minmag = d_mag * np.floor(np.amin(tri_mags)/d_mag)
    maxmag = d_mag * np.ceil(np.amax(tri_mags)/d_mag)
    hist, model_mags = np.histogram(tri_mags, bins=np.arange(minmag, maxmag+1e-10, d_mag))
    hc = np.where(hist > 3)[0]

    hist = hist[hc]
    model_mags_interval = np.diff(model_mags)[hc]
    model_mags = model_mags[hc]

    hist = hist / model_mags_interval / tri_area
    log10y_tri = np.log10(hist)

    # TODO: add the step to get density and counts of extra-galactic sources.
    gal_count = 0
    log10y_gal = -np.inf * np.ones_like(log10y_tri)

    model_count = tri_count + gal_count

    log10y = np.log10(10**log10y_tri + 10**log10y_gal)

    # Set a magnitude bin width of 0.25 mags, to avoid oversampling.
    dmag = 0.25
    mag_min = dmag * np.floor(np.amin(a_photo)/dmag)
    mag_max = dmag * np.ceil(np.amax(a_photo)/dmag)
    magbins = np.arange(mag_min, mag_max+1e-10, dmag)
    # For local densities, we want a percentage offset, given that we're in
    # logarithmic bins, accepting a log-difference maximum. This is slightly
    # lop-sided, but for 20% results in +18%/-22% limits, which is fine.
    dlogN = 0.2
    logNvals = np.log(localN)
    logN_min = dlogN * np.floor(np.amin(logNvals)/dlogN)
    logN_max = dlogN * np.ceil(np.amax(logNvals)/dlogN)
    logNbins = np.arange(logN_min, logN_max+1e-10, dlogN)

    counts, logNbins, magbins = np.histogram2d(logNvals, a_photo, bins=[logNbins, magbins])
    Ni, magi = np.where(counts > 0)
    mag_array = 0.5*(magbins[1:]+magbins[:-1])[magi]
    count_array = np.exp(0.5*(logNbins[1:]+logNbins[:-1])[Ni])

    R = 1.185 * psf_fwhm
    seed = np.random.default_rng().choice(100000, size=paf.get_random_seed_size())
    Frac, Flux, fourieroffset, offset, cumulative = paf.perturb_aufs(
        count_array, mag_array, r[:-1]+dr/2, dr, r, j0s.T,
        model_mags+model_mags_interval/2, model_mags_interval, log10y, model_count,
        int(dm_max/d_mag) * np.ones_like(count_array), mag_cut, R, num_trials, seed)
    np.save('{}/{}/frac.npy'.format(tri_folder, filt), Frac)
    np.save('{}/{}/flux.npy'.format(tri_folder, filt), Flux)
    np.save('{}/{}/offset.npy'.format(tri_folder, filt), offset)
    np.save('{}/{}/cumulative.npy'.format(tri_folder, filt), cumulative)
    np.save('{}/{}/fourier.npy'.format(tri_folder, filt), fourieroffset)
    np.save('{}/{}/N.npy'.format(tri_folder, filt), count_array)
    np.save('{}/{}/mag.npy'.format(tri_folder, filt), mag_array)

    return count_array
