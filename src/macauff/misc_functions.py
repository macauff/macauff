# Licensed under a 3-clause BSD style license - see LICENSE
'''
This module provides miscellaneous scripts, called in other parts of the cross-match
framework.
'''

import itertools
import multiprocessing
import multiprocessing.dummy as mpd
from multiprocessing import shared_memory

import numpy as np
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord, UnitSphericalRepresentation
from scipy import spatial
from scipy.optimize import minimize
from scipy.spatial import ConvexHull  # pylint: disable=no-name-in-module

from macauff.get_trilegal_wrapper import get_av_infinity

# pylint: disable=import-error,no-name-in-module
from macauff.misc_functions_fortran import misc_functions_fortran as mff

__all__ = []


def create_auf_params_grid(perturb_auf_outputs, auf_pointings, filt_names, array_name,
                           arraylengths, len_first_axis=None):
    '''
    Minor function to offload the creation of a 3-D or 4-D array from a series
    of 2-D arrays.

    Parameters
    ----------
    perturb_auf_outputs : dictionary
        Dictionary of outputs from series of pointing-filter AUF simulations.
    auf_pointings : numpy.ndarray
        Two-dimensional array with the sky coordinates of each pointing used
        in the perturbation AUF component creation.
    filt_names : list or numpy.ndarray
        List of ordered filters for the given catalogue.
    array_name : string
        The name of the individually-saved arrays, one per sub-folder, to turn
        into a 3-D or 4-D array.
    arraylengths : numpy.ndarray
        Array containing length of the density-magnitude combinations in each
        sky/filter combination.
    len_first_axis : integer, optional
        Length of the initial axis of the 4-D array. If not provided or is
        ``None``, final array is assumed to be 3-D instead.

    Returns
    -------
    grid : numpy.ndarray
        The populated grid of ``array_name`` individual 1-D arrays.
    '''
    longestnm = np.amax(arraylengths)
    if len_first_axis is None:
        grid = np.full(fill_value=-1, dtype=float, order='F',
                       shape=(longestnm, len(filt_names), len(auf_pointings)))
    else:
        grid = np.full(fill_value=-1, dtype=float, order='F',
                       shape=(len_first_axis, longestnm, len(filt_names), len(auf_pointings)))
    for j, auf_pointing in enumerate(auf_pointings):
        ax1, ax2 = auf_pointing
        for i, filt in enumerate(filt_names):
            perturb_auf_combo = f'{ax1}-{ax2}-{filt}'
            single_array = perturb_auf_outputs[perturb_auf_combo][array_name]
            if len_first_axis is None:
                grid[:arraylengths[i, j], i, j] = single_array
            else:
                grid[:, :arraylengths[i, j], i, j] = single_array

    return grid


def _load_rectangular_slice(a, lon1, lon2, lat1, lat2, padding):
    '''
    Loads all sources in a catalogue within a given separation of a rectangle
    in sky coordinates, allowing for the search for all sources within a given
    radius of sources inside the rectangle.

    Parameters
    ----------
    a : numpy.ndarray
        Full astrometric catalogue from which the subset of sources within
        ``padding`` distance of the sky rectangle are to be drawn.
    lon1 : float
        Lower limit on on-sky rectangle, in given sky coordinates, in degrees.
    lon2 : float
        Upper limit on sky region to slice sources from ``a``.
    lat1 : float
        Lower limit on second orthogonal sky coordinate defining rectangle.
    lat2 : float
        Upper sky rectangle coordinate of the second axis.
    padding : float
        The sky separation, in degrees, to find all sources within a distance
        of in ``a``.

    Returns
    -------
    sky_cut : numpy.ndarray
        Boolean array, indicating whether each source in ``a`` is within ``padding``
        of the rectangle defined by ``lon1``, ``lon2``, ``lat1``, and ``lat2``.
    '''
    lon_shift = 180 - (lon2 + lon1)/2

    sky_cut = (_lon_cut(a, lon1, padding, 'greater', lon_shift) &
               _lon_cut(a, lon2, padding, 'lesser', lon_shift) &
               _lat_cut(a, lat1, padding, 'greater') & _lat_cut(a, lat2, padding, 'lesser'))

    return sky_cut


def _lon_cut(a, lon, padding, inequality, lon_shift):
    '''
    Function to calculate the longitude inequality criterion for astrometric
    sources relative to a rectangle defining boundary limits.

    Parameters
    ----------
    a : numpy.ndarray
        The main astrometric catalogue to be sliced.
    lon : float
        Longitude at which to cut sources, either above or below, in degrees.
    padding : float
        Maximum allowed sky separation the "wrong" side of ``lon``, to allow
        for an increase in sky box size to ensure all overlaps are caught in
        ``get_overlap_indices``.
    inequality : string, ``greater`` or ``lesser``
        Flag to determine whether a source is either above or below the
        given ``lon`` value.
    lon_shift : float
        Value by which to "move" longitudes to avoid meridian overflow issues.

    Returns
    -------
    sky_cut : numpy.ndarray
        Boolean array indicating whether the objects in ``a`` are left or right
        of the given ``lon`` value.
    '''
    b = a[:, 0] + lon_shift
    b[b > 360] = b[b > 360] - 360
    b[b < 0] = b[b < 0] + 360
    # While longitudes in the data are 358, 359, 0/360, 1, 2, longitude
    # cutout values should go -2, -1, 0, 1, 2, and hence we ought to be able
    # to avoid the 360-wrap here.
    new_lon = lon + lon_shift
    if inequality == 'lesser':
        # Allow for small floating point rounding errors in comparison.
        inequal_lon_cut = b <= new_lon + 1e-6
    else:
        inequal_lon_cut = b >= new_lon - 1e-6
    # To check whether a source should be included in this slice or not if the
    # "padding" factor is non-zero, add an extra caveat to check whether
    # Haversine great-circle distance is less than the padding factor. For
    # constant latitude this reduces to
    # r = 2 arcsin(|cos(lat) * sin(delta-lon/2)|).
    if padding > 0:
        hav_dist_const_lat = np.degrees(2 * np.arcsin(np.abs(np.cos(np.radians(a[:, 1])) *
                                        np.sin(np.radians((a[:, 0] - lon)/2)))))
        sky_cut = (hav_dist_const_lat <= padding) | inequal_lon_cut
    # However, in both zero and non-zero padding factor cases, we always require
    # the source to be above or below the longitude.
    else:
        sky_cut = inequal_lon_cut

    return sky_cut


def _lat_cut(a, lat, padding, inequality):
    '''
    Function to calculate the latitude inequality criterion for astrometric
    sources relative to a rectangle defining boundary limits.

    Parameters
    ----------
    a : numpy.ndarray
        The main astrometric catalogue to be sliced.
    lat : float
        Latitude at which to cut sources, either above or below, in degrees.
    padding : float
        Maximum allowed sky separation the "wrong" side of ``lat``, to allow
        for an increase in sky box size to ensure all overlaps are caught in
        ``get_overlap_indices``.
    inequality : string, ``greater`` or ``lesser``
        Flag to determine whether a source is either above or below the
        given ``lat`` value.

    Returns
    -------
    sky_cut : numpy.ndarray
        Boolean array indicating whether ``a`` sources are above or below the
        given ``lat`` value.
    '''

    # The "padding" factor is easier to handle for constant longitude in the
    # Haversine formula, being a straight comparison of delta-lat, and thus we
    # can simply move the required latitude padding factor to within the
    # latitude comparison.
    if padding > 0:
        if inequality == 'lesser':
            # Allow for small floating point rounding errors in comparison.
            sky_cut = a[:, 1] <= lat + padding + 1e-6
        else:
            sky_cut = a[:, 1] >= lat - padding - 1e-6
    else:
        if inequality == 'lesser':
            sky_cut = a[:, 1] <= lat + 1e-6
        else:
            sky_cut = a[:, 1] >= lat - 1e-6

    return sky_cut


def min_max_lon(a):
    """
    Returns the minimum and maximum longitude of a set of sky coordinates,
    accounting for 0 degrees being the same as 360 degrees and hence
    358 deg and -2 deg being equal, effectively re-wrapping longitude to be
    between -pi and +pi radians in cases where data exist either side of the
    boundary.

    Parameters
    ----------
    a : numpy.ndarray
        1-D array of longitudes, which are limited to between 0 and 360 degrees.

    Returns
    -------
    min_lon : float
        The longitude furthest to the "left" of 0 degrees.
    max_lon : float
        The longitude furthest around the other way from 0 degrees, with the
        largest longitude in the first quadrant of the Galaxy.
    """
    # TODO: can this be simplified with a lon_shift like lon_cut above?  pylint: disable=fixme
    min_lon, max_lon = np.amin(a), np.amax(a)
    if min_lon <= 1 and max_lon >= 359 and np.any(np.abs(a - 180) < 1):
        # If there is data both either side of 0/360 and at 180 degrees,
        # return the entire longitudinal circle as the limits.
        return 0, 360
    if min_lon <= 1 and max_lon >= 359:
        # If there's no data around the anti-longitude but data either
        # side of zero degrees exists, return the [-pi, +pi] wrapped
        # values.
        min_lon = np.amin(a[a > 180] - 360)
        max_lon = np.amax(a[a < 180])
        return min_lon, max_lon
    # Otherwise, the limits are inside [0, 360] and should be returned
    # as the "normal" minimum and maximum values.
    return min_lon, max_lon


# pylint: disable=unused-argument
def find_model_counts_corrections(data_loghist, data_dloghist, data_bin_mids, tri_hist,
                                  gal_dns, tri_mag_mids):
    '''
    Derivation of two-parameter corrections to differential source counts, fitting
    both stellar and galaxy components with separate scaling factors.

    Parameters
    ----------
    data_loghist : numpy.ndarray
        Logarithmic differential source counts.
    data_dloghist : numpy.ndarray
        Uncertainties in log-counts from ``data_loghist``.
    data_bin_mids : numpy.ndarray
        Magnitudes of each bin corresponding to ``data_loghist``.
    tri_hist : numpy.ndarray
        Stellar model linear differential source counts.
    gal_dns : numpy.ndarray
        Galaxy model linear differential source counts.
    tri_mag_mids : numpy.ndarray
        Model magnitudes for each element of ``tri_hist`` and/or ``gal_dns``.

    Returns
    -------
    res.x : numpy.ndarray
        Value of least-squares fit scaling factors for stellar and galaxy
        components of the differential source counts, as fit to the
        input data.
    '''
    def lst_sq(p, y, o, t, g):
        '''
        Function evaluating the least-squares minimisation, and Jacobian,
        of a fit to model and data differential source counts.

        Parameters
        ----------
        p : list
            ``a`` and ``b``, scaling factors for star and galaxy components
            of the source counts.
        y : numpy.ndarray
            Data log-counts.
        o : numpy.ndarray
            Uncertainties corresponding to ``y``.
        t : numpy.ndarray
            Log-counts of stellar component of source counts.
        g : numpy.ndarray
            Log-counts of galaxy component of source counts.

        Returns
        -------
        float
            Chi-squared of the model source counts fit to the data.
        list of floats
            Gradient of the chi-squared, differentiated with respect to
            the stellar and galaxy correction factors.
        '''
        a, b = p
        f = np.log10(10**t * a + 10**g * b)
        dfda = 10**t / (np.log(10) * (10**t * a + 10**g * b))
        dfdb = 10**g / (np.log(10) * (10**t * a + 10**g * b))
        dchida = np.sum(-2 * (y - f) / o**2 * dfda)
        dchidb = np.sum(-2 * (y - f) / o**2 * dfdb)
        return np.sum((y - f)**2 / o**2), [dchida, dchidb]

    q = tri_hist > 0
    tri_log_hist_at_data = np.interp(data_bin_mids, tri_mag_mids[q],
                                     np.log10(tri_hist[q]), left=np.nan, right=np.nan)
    q = gal_dns > 0
    gal_log_hist_at_data = np.interp(data_bin_mids, tri_mag_mids[q],
                                     np.log10(gal_dns[q]), left=np.nan, right=np.nan)
    q = ~np.isnan(tri_log_hist_at_data) & ~np.isnan(gal_log_hist_at_data)
    res = minimize(lst_sq, args=(data_loghist[q], np.ones_like(data_loghist[q]),
                                 tri_log_hist_at_data[q], gal_log_hist_at_data[q]), x0=[1, 1], jac=True,
                   method='L-BFGS-B', options={'ftol': 1e-12}, bounds=[(0.01, None), (0.01, None)])
    return res.x


def convex_hull_area(x, y, return_hull=False):
    """
    Function to calculate the convex hull vertices and the area enclosed by
    those verticies.

    Parameters
    ----------
    x : list or numpy.ndarray
        Coordinates in the longitudinal, or RA, sky axis, in degrees.
    y : list or numpy.ndarray
        Latitudinal, or Declination, sky axis coordinates, in degrees.
    return_hull : boolean, optional
        Boolean flag indicating whether to return the coordinates of the points
        defining the convex hull, or just return the area.

    Returns
    -------
    area : float
        The sky area, in square degrees, of the convex hull enclosing the
        points.
    """
    # min_max_lon returns negative minimum longitude for cases where area sits
    # either side of the 0/360 degree boundary but doesn't fully wrap around
    # the entire longitudinal ring.
    x_shift = 180 - np.sum(min_max_lon(x))/2
    # A quirk of min_max_lon is that if data straddle the 0/360 boundary, but
    # are mostly on the 360-side, then we get a negative min_max_lon result and
    # an x_shift larger than 180, instead of an x_shift in the desired range
    # [-180, 180]. To fix this we can check for x_shift > 180 and subtract 360.
    # If we don't, then we risk moving objects at e.g. an RA of 359 degrees
    # (min_max_lon would say this is -1 degrees) aronud by +181 degrees to 540
    # degrees, rather than backwards to 180 degrees.
    if x_shift > 180:
        x_shift -= 360
    new_x = x + x_shift
    new_x[new_x > 360] = new_x[new_x > 360] - 360
    new_x[new_x < 0] = new_x[new_x < 0] + 360
    # Pad cos-delta factor to avoid precisely collapsing to a singularity at
    # either pole.
    new_x_cosd = new_x * (np.cos(np.radians(y)) + 1e-10)
    hull = ConvexHull(np.array([new_x_cosd, y]).T)
    area = shoelace_formula_area(new_x_cosd[hull.vertices], y[hull.vertices])

    if return_hull:
        return area, np.array([new_x[hull.vertices], y[hull.vertices]]).T, x_shift
    return area


def shoelace_formula_area(x, y):
    """
    Performs the calculation of an area defined by a set of vertices as per the
    Shoelace formula.

    Parameters
    ----------
    x : list or numpy.ndarray
        Coordinates in the longitudinal, or RA, sky axis, in degrees.
    y : list or numpy.ndarray
        Latitudinal, or Declination, sky axis coordinates, in degrees.

    Returns
    -------
    area : float
        The sky area, in square degrees, of the region defined by the set of
        vertices.
    """
    area = abs(0.5 * np.array(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))

    return area


def generate_avs_inside_hull(hull_points, hull_x_shift, coord_system):
    """
    Sample V-band exctinction across an arbitrarily-shaped region, using points
    the area's convex hull.

    Parameters
    ----------
    hull_points : numpy.ndarray
        Array of shape ``(N, 2)`` of the coordinates defining the region's
        external region as a set of coordinate polygons, following
        `~macauff.convex_hull_area`.
    hull_x_shift : float
        The longitudinal shift in coordinates used by the hull to avoid 0-360
        degree wraparound issues.
    coord_system : string, "equatorial" or "galactic"
        Frame in which the coordinates are defined, used to sample the on-sky
        V-band extinction in the right coordinate system.

    Returns
    -------
    avs : numpy.ndarray
        V-band extinctions within the convex region, sampled sufficiently highly
        to give good differential extinction measurements.
    """
    not_enough_points = True
    n_dim = 7
    ax1_min, ax1_max = np.array([np.amin(hull_points[:, 0]), np.amax(hull_points[:, 0])]) - hull_x_shift
    ax2_min, ax2_max = np.array([np.amin(hull_points[:, 1]), np.amax(hull_points[:, 1])])
    while not_enough_points:
        ax1s = np.linspace(ax1_min, ax1_max, n_dim)
        ax2s = np.linspace(ax2_min, ax2_max, n_dim)
        ax1s, ax2s = np.meshgrid(ax1s, ax2s, indexing='xy')
        ax1s, ax2s = ax1s.flatten(), ax2s.flatten()
        points_in_area = np.array([coord_inside_convex_hull(
            [ax1 + hull_x_shift, ax2], hull_points) for ax1, ax2 in zip(ax1s, ax2s)])
        if np.sum(points_in_area) >= 30:
            not_enough_points = False
        else:
            n_dim += 1
        ax1s, ax2s = ax1s[points_in_area], ax2s[points_in_area]

    avs = np.empty(len(ax1s), float)
    for j, (ax1, ax2) in enumerate(zip(ax1s, ax2s)):
        if coord_system == 'equatorial':
            c = SkyCoord(ra=ax1, dec=ax2, unit='deg', frame='icrs')
            l, b = c.galactic.l.degree, c.galactic.b.degree
        else:
            l, b = ax1, ax2
        av = get_av_infinity(l, b, frame='galactic')[0]
        avs[j] = av

    return avs


def coord_inside_convex_hull(p, hull):
    """
    Given a set of convex hull points from `~macauff.convex_hull_area`, determine
    if a single coordinate is within or outside the area defined by the points,
    using the sum of the oriented angles between the point and each set of
    neighbouring hull coordinates.

    Parameters
    ----------
    p : list or numpy.ndarray
        The coordinates of the point to determine the location of. Should be a
        two-element list or array, first longitudinal then latitudinal
        coordinates.
    hull : numpy.ndarray
        Array of hull points, from `~macauff.convex_hull_area`. Should be shape
        ``(N, 2)``, with pairs of longitudinal coordinates in the first element
        of the second axis and latitudinal points in the second element.

    Returns
    -------
    boolean
        ``True`` if ``p`` is inside the ``hull`` polygon, ``False`` otherwise.

    Notes
    -----
    As ``hull`` returns from `~macauff.convex_hull_area` with an ``x_shift``
    translation to avoid 0-360 degree wraparound effects, the same shift must
    be applied to ``p`` prior to passing to this function.
    """
    hull = np.vstack((hull, hull[0]))
    i, j = np.arange(len(hull)-1), np.arange(1, len(hull))
    x1, y1 = hull[i, 0] - p[0], hull[i, 1] - p[1]
    x2, y2 = hull[j, 0] - p[0], hull[j, 1] - p[1]

    dot_prod, cross_prod = x1*x2 + y1*y2, x1*y2 - x2*y1
    theta = np.arctan2(cross_prod, dot_prod)
    sum_of_angles = np.abs(np.sum(theta))

    if sum_of_angles < np.pi:
        return False
    return True


def create_densities(b, minmag, maxmag, hull, hull_x_shift, search_radius, n_pool, mag_ind, ax1_ind, ax2_ind,
                     coord_system, unique_shared_id):
    """
    Generate local normalising densities for all sources in catalogue "b".

    Parameters
    ----------
    b : numpy.ndarray
        Catalogue of the sources for which astrometric corrections should be
        determined.
    minmag : float
        Bright limiting magnitude, fainter than which objects are used when
        determining the number of nearby sources for density purposes.
    maxmag : float
        Faintest magnitude within which to determine the density of catalogue
        ``b`` objects.
    hull : numpy.ndarray
        Array of shape ``(N, 2)``, giving the ``(ax1, ax2)`` coordinates for
        each of the ``N`` polygon points defining the convex hull of the
        region in which the objects are contained.
    hull_x_shift : float
        Amount by which ``hull`` points were shifted in longitude during
        area calculation, to avoid 0/360 wraparound issues, and the amount
        by which coordinates should be moved to mirror "new" coord system.
    search_radius : float
        Radius, in degrees, around which to calculate the density of objects.
        Smaller values will allow for more fluctuations and handle smaller scale
        variation better, but be subject to low-number statistics.
    n_pool : integer
        Number of parallel threads to run when calculating densities via
        ``multiprocessing``.
    mag_ind : integer
        Index in ``b`` where the magnitude being used is stored.
    ax1_ind : integer
        Index of ``b`` for the longitudinal coordinate column.
    ax2_ind : integer
        ``b`` index for the latitude data.
    coord_system : string
        Determines whether we are in equatorial or galactic coordinates for
        separation considerations.
    unique_shared_id : string
        ID to distinguish ``SharedNumpyArray`` memory across parallelisations.

    Returns
    -------
    narray : numpy.ndarray
        The density of objects within ``search_radius`` degrees of each object
        in catalogue ``b``.

    """
    overlap_number = calculate_overlap_counts(b, b, minmag, maxmag, search_radius, n_pool, mag_ind, ax1_ind,
                                              ax2_ind, coord_system, 'len', unique_shared_id)

    seed = np.random.default_rng().choice(100000, size=(mff.get_random_seed_size(), len(b)))

    area = mff.get_circle_area_overlap(
        b[:, ax1_ind] + hull_x_shift, b[:, ax2_ind], search_radius,
        np.append(hull[:, 0], hull[0, 0]), np.append(hull[:, 1], hull[0, 1]), seed)

    narray = overlap_number / area

    return narray


def make_pool(n_pool):
    """
    Create a multiprocessing pool, either of real processes or threads.

    Parameters
    ----------
    n_pool : integer
        Number of parallel threads to run when calculating densities via
        ``multiprocessing``. If 1, use threads instead of processes.

    Returns
    -------
    pool : multiprocessing.Pool or multiprocessing.dummy.Pool
        The created pool of processes or threads.
    """
    if n_pool == 1:
        return mpd.Pool(n_pool)
    return multiprocessing.Pool(n_pool)


def calculate_overlap_counts(a, b, minmag, maxmag, search_radius, n_pool, mag_ind, ax1_ind, ax2_ind,
                             coord_system, len_or_inds, unique_shared_id):
    """
    Calculate number of overlapping objects in catalogue "b" that are within
    an optionally-set dynamic range and also within a given radius of each
    object in catalogue "a".

    Parameters
    ----------
    a : numpy.ndarray
        Catalogue of sources for which number of opposing catalogue overlap
        counts should be calculated.
    b : numpy.ndarray
        Catalogue to search for overlapping ``a`` objects in.
    minmag : float
        Bright limiting magnitude, fainter than which objects are used when
        determining the number of nearby sources for density purposes.
    maxmag : float
        Faintest magnitude within which to determine the density of catalogue
        ``b`` objects.
    search_radius : float
        Radius, in degrees, around which to calculate the density of objects.
        Smaller values will allow for more fluctuations and handle smaller scale
        variation better, but be subject to low-number statistics.
    n_pool : integer
        Number of parallel threads to run when calculating densities via
        ``multiprocessing``.
    mag_ind : integer
        Index in ``b`` where the magnitude being used is stored. If `mag_ind`
        is NaN, then skip magnitude cut.
    ax1_ind : integer
        Index of ``b`` for the longitudinal coordinate column.
    ax2_ind : integer
        ``b`` index for the latitude data.
    coord_system : string
        Determines whether we are in equatorial or galactic coordinates for
        separation considerations.
    len_or_inds : string, 'len' or 'array'
        Flag for whether to return the number of overlaps ('len') or list of
        each array index for overlaps ('array') for each primary object.
    unique_shared_id : string
        ID to distinguish ``SharedNumpyArray`` memory across parallelisations.

    Returns
    -------
    numpy.ndarray
        The number of catalogue b objects near each catalogue a object, or the
        indices of all objects near each catalogue a source.
    """
    def _get_cart_kdt(coord):
        """
        Convenience function to create a KDTree of a set of sky coordinates,
        represented in Cartesian space on the unit sphere.

        Parameters
        ----------
        coord : ~`astropy.coordinates.SkyCoord`
            The `astropy` object containing all of the objects coordinates,
            as represented as Cartesian (x, y, z) coordinates on the unit sphere.

        Returns
        -------
        kdt : ~`scipy.spatial.KDTree`
            The KDTree for ``coord`` evaluated with Cartesian coordinates.
        """
        # Largely based on astropy.coordinates._get_cartesian_kdtree.
        KDTree = spatial.KDTree
        cartxyz = coord.cartesian.xyz
        flatxyz = cartxyz.reshape((3, np.prod(cartxyz.shape) // 3))
        kdt = KDTree(flatxyz.value.T, compact_nodes=False, balanced_tree=False)
        return kdt

    if ~np.isnan(mag_ind):
        cutmag = (b[:, mag_ind] >= minmag) & (b[:, mag_ind] <= maxmag)
    else:
        cutmag = np.ones(len(b), bool)

    if coord_system == 'galactic':
        full_cat = SkyCoord(l=a[:, ax1_ind], b=a[:, ax2_ind], unit='deg', frame='galactic')
        mag_cut_cat = SkyCoord(l=b[cutmag, ax1_ind], b=b[cutmag, ax2_ind], unit='deg',
                               frame='galactic')
    else:
        full_cat = SkyCoord(ra=a[:, ax1_ind], dec=a[:, ax2_ind], unit='deg', frame='icrs')
        mag_cut_cat = SkyCoord(ra=b[cutmag, ax1_ind], dec=b[cutmag, ax2_ind], unit='deg',
                               frame='icrs')

    full_urepr = full_cat.data.represent_as(UnitSphericalRepresentation)
    full_ucoords = full_cat.realize_frame(full_urepr)
    del full_urepr

    mag_cut_urepr = mag_cut_cat.data.represent_as(UnitSphericalRepresentation)
    mag_cut_ucoords = mag_cut_cat.realize_frame(mag_cut_urepr)
    mag_cut_kdt = _get_cart_kdt(mag_cut_ucoords)
    del mag_cut_urepr, mag_cut_ucoords

    # pylint: disable-next=no-member
    r = (2 * np.sin(Angle(search_radius * u.degree) / 2.0)).value
    if len_or_inds == 'len':
        overlap_number = np.empty(len(a), int)
    else:
        overlap_inds = [0] * len(a)

    counter = np.arange(0, len(a))
    iter_group = zip(counter, full_ucoords, itertools.repeat([mag_cut_kdt, r, len_or_inds]))
    with make_pool(n_pool) as pool:
        for stuff in pool.imap_unordered(ball_point_query, iter_group, chunksize=len(a)//n_pool):
            i, result = stuff
            if len_or_inds == 'len':
                overlap_number[i] = result
            else:
                overlap_inds[i] = result

    pool.join()

    if len_or_inds == 'len':
        return overlap_number
    return overlap_inds


def ball_point_query(iterable):
    """
    Wrapper function to distribute calculation of the number of neighbours
    around a particular sky coordinate via KDTree query.

    Parameters
    ----------
    iterable : list
        List of variables passed through ``multiprocessing``, including index
        into object having its neighbours determined, the Spherical Cartesian
        representation of objects to search for neighbours around, the KDTree
        containing all potential neighbours, the Cartesian angle
        representing the maximum on-sky separation, and the length-or-array
        flag.

    Returns
    -------
    i : integer
        The index of the object whose neighbour count was calculated.
    integer or numpy.ndarray
        Either number of neighbours in ``mag_cut_kdt`` within ``r`` of
        ``full_ucoords[i]``, or the indices into ``mag_cut_kdt`` that are
        within the specified range.
    """
    i, full_ucoord, (mag_cut_kdt, r, len_or_inds) = iterable
    # query_ball_point returns the neighbours of x (full_ucoords) around self
    # (mag_cut_kdt) within r.
    kdt_query = mag_cut_kdt.query_ball_point(full_ucoord.cartesian.xyz, r, return_sorted=True)
    if len_or_inds == 'len':
        return i, len(kdt_query)
    return i, kdt_query


def _make_regions_points(region_type, region_points, chunk_id):
    '''
    Wrapper function for the creation of "region" coordinate tuples,
    given either a set of rectangular points or a list of coordinates.

    Parameters
    ----------
    region_type : list of string and string
        String containing the kind of system the region pointings are in.
        Should be "rectangle", regularly sampled points in the two sky
        coordinates, or "points", individually specified sky coordinates,
        and the name into which to save the variable in the class.
    region_points : list of string and list
        String containing the evaluation points. If ``region_type`` is
        "rectangle", should be six values, the start and stop values and
        number of entries of the respective sky coordinates; and if
        ``region_type`` is "points", ``region_points`` should be tuples
        of the form ``(a, b)`` separated by whitespace, as well as the
        class variable name for storage.
    chunk_id : string
        Unique identifier for particular sub-region being loaded, used to
        inform of errors.

    Returns
    -------
    points : numpy.ndarray
        An array of shape (N, 2), with each second-axis pair being a single
        sky coordinate for each of the N pointings.
    '''
    rt = region_type[1].lower()
    if rt == 'rectangle':
        try:
            a = region_points[1]
            a = [float(point) for point in a]
        except ValueError as exc:
            raise ValueError(f"{region_points[0]} should be 6 numbers separated by spaces in chunk "
                             f"{chunk_id}.") from exc
        if len(a) == 6:
            if not a[2].is_integer() or not a[5].is_integer():
                raise ValueError("Number of steps between start and stop values for "
                                 f"{region_points[0]} should be integers in chunk {chunk_id}.")
            ax1_p = np.linspace(a[0], a[1], int(a[2]))
            ax2_p = np.linspace(a[3], a[4], int(a[5]))
            points = np.stack(np.meshgrid(ax1_p, ax2_p), -1).reshape(-1, 2)
        else:
            raise ValueError(f"{region_points[0]} should be 6 numbers separated by spaces in chunk "
                             f"{chunk_id}.")
    elif rt == 'points':
        try:
            points = np.array(region_points[1], dtype=float)
        except ValueError as exc:
            raise ValueError(f"{region_points[0]} should be a list of two-element lists "
                             f"'[[a, b], [c, d]]', separated by a comma in chunk {chunk_id}.") from exc

    return points  # pylint: disable=possibly-used-before-assignment


class SharedNumpyArray:
    '''
    Wraps a numpy array so that it can be shared quickly among processes,
    avoiding unnecessary copying and (de)serializing. Resource originally
    from https://e-dorigatti.github.io/python/2020/06/19/
    multiprocessing-large-objects.html.
    '''
    def __init__(self, array, name='123'):
        '''
        Creates the shared memory and copies the array.
        '''
        try:
            shm = shared_memory.SharedMemory(name=name)
            shm.unlink()
        except FileNotFoundError:
            pass
        self._shared = shared_memory.SharedMemory(name=name, create=True, size=array.nbytes)

        self._dtype, self._shape = array.dtype, array.shape

        res = np.ndarray(self._shape, dtype=self._dtype, buffer=self._shared.buf)
        res[:] = array[:]

    def read(self):
        '''
        Reads the array from the shared memory without unnecessary copying.
        '''
        return np.ndarray(self._shape, self._dtype, buffer=self._shared.buf)

    def copy(self):
        '''
        Returns a new copy of the array stored in shared memory.
        '''
        return np.copy(self.read())

    def unlink(self):
        '''
        Releases the allocated memory. Call when finished using the data,
        or when the data was copied somewhere else.
        '''
        self._shared.close()
        self._shared.unlink()
