# Licensed under a 3-clause BSD style license - see LICENSE
'''
This module provides miscellaneous scripts, called in other parts of the cross-match
framework.
'''

import numpy as np
from multiprocessing import shared_memory
from scipy.optimize import minimize

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


def load_small_ref_auf_grid(modrefind, perturb_auf_outputs, file_name_prefixes):
    '''
    Function to create reference index arrays out of larger arrays, based on
    the mappings from the original reference index array into a larger grid,
    such that the corresponding cutout reference index now maps onto the smaller
    cutout 4-D array.

    Parameters
    ----------
    modrefind : numpy.ndarray
        The reference index array that maps into saved array ``fourier_grid``
        for each source in the given catalogue.
    perturb_auf_outputs : dictionary
        Saved results from the cross-matches' extra AUF component simulations.
    file_name_prefixes : list
        Prefixes of the files stored in ``auf_folder_path`` -- the parts before
        "_grid" -- to be loaded as sub-arrays and returned.

    Returns
    -------
    small_grids : list of numpy.ndarray
        Small cutouts of ``*_grid`` files defined by ``file_name_prefixes``,
        containing only the appropriate indices for AUF pointing, filter, etc.
    modrefindsmall : numpy.ndarray
        The corresponding mappings for each source onto ``fouriergrid``, such
        that each source still points to the correct entry that it did in
        ``fourier_grid``.
    '''
    nmuniqueind, nmnewind = np.unique(modrefind[0, :], return_inverse=True)
    filtuniqueind, filtnewind = np.unique(modrefind[1, :], return_inverse=True)
    axuniqueind, axnewind = np.unique(modrefind[2, :], return_inverse=True)

    x, y, z = np.meshgrid(nmuniqueind, filtuniqueind, axuniqueind, indexing='ij')

    small_grids = []
    for name in file_name_prefixes:
        if len(perturb_auf_outputs[f'{name}_grid'].shape) == 4:
            small_grids.append(np.asfortranarray(
                perturb_auf_outputs[f'{name}_grid'][:, x, y, z]))
        else:
            small_grids.append(np.asfortranarray(
                perturb_auf_outputs[f'{name}_grid'][x, y, z]))
    modrefindsmall = np.empty((3, modrefind.shape[1]), int, order='F')
    del modrefind
    modrefindsmall[0, :] = nmnewind
    modrefindsmall[1, :] = filtnewind
    modrefindsmall[2, :] = axnewind

    return small_grids, modrefindsmall


def hav_dist_constant_lat(x_lon, x_lat, lon):
    '''
    Computes the Haversine formula in the limit that sky separation is only
    determined by longitudinal separation (i.e., delta-lat is zero).

    Parameters
    ----------
    x_lon : float
        Sky coordinate of the source in question, in degrees.
    x_lat : float
        Orthogonal sky coordinate of the source, in degrees.
    lon : float
        Longitudinal sky coordinate to calculate the "horizontal" sky separation
        of the source to.

    Returns
    -------
    dist : float
        Horizontal sky separation between source and given ``lon``, in degrees.
    '''

    dist = np.degrees(2 * np.arcsin(np.abs(np.cos(np.radians(x_lat)) *
                                           np.sin(np.radians((x_lon - lon)/2)))))

    return dist


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
        ``get_max_overlap`` or ``get_max_indices``.
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
        sky_cut = (hav_dist_constant_lat(a[:, 0], a[:, 1], lon) <= padding) | inequal_lon_cut
    # However, in both zero and non-zero padding factor cases, we always require
    # the source to be above or below the longitude for sky_cut_1 and sky_cut_2
    # in load_fourier_grid_cutouts, respectively.
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
        ``get_max_overlap`` or ``get_max_indices``.
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
