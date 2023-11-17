# Licensed under a 3-clause BSD style license - see LICENSE
'''
This module provides miscellaneous scripts, called in other parts of the cross-match
framework.
'''

import numpy as np

__all__ = []

# "Anemic" class for holding outputs to pass into later pipeline stages
# Takes any number of arguments in constructor, each becoming an attribute
# TODO: Move to own file?
class StageData:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def create_auf_params_grid(auf_folder_path, auf_pointings, filt_names, array_name,
                           arraylengths, len_first_axis=None):
    '''
    Minor function to offload the creation of a 3-D or 4-D array from a series
    of 2-D arrays.

    Parameters
    ----------
    auf_folder_path : string
        Location of the top-level folder in which all fourier grids are saved.
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
    '''
    longestNm = np.amax(arraylengths)
    if len_first_axis is None:
        grid = np.lib.format.open_memmap('{}/{}_grid.npy'.format(
            auf_folder_path, array_name), mode='w+', dtype=float, shape=(
            longestNm, len(filt_names), len(auf_pointings)), fortran_order=True)
        grid[:, :, :] = -1
    else:
        grid = np.lib.format.open_memmap('{}/{}_grid.npy'.format(
            auf_folder_path, array_name), mode='w+', dtype=float, shape=(
            len_first_axis, longestNm, len(filt_names), len(auf_pointings)), fortran_order=True)
        grid[:, :, :, :] = -1
    for j in range(0, len(auf_pointings)):
        ax1, ax2 = auf_pointings[j]
        for i in range(0, len(filt_names)):
            filt = filt_names[i]
            single_array = np.load('{}/{}/{}/{}/{}.npy'.format(auf_folder_path,
                                   ax1, ax2, filt, array_name))
            if len_first_axis is None:
                grid[:arraylengths[i, j], i, j] = single_array
            else:
                grid[:, :arraylengths[i, j], i, j] = single_array

    del longestNm, grid


def load_small_ref_auf_grid(modrefind, auf_folder_path, file_name_prefixes):
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
    auf_folder_path : string
        Location of the folder in which ``fourier_grid`` is stored.
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
        if len(np.load('{}/{}_grid.npy'.format(auf_folder_path, name), mmap_mode='r').shape) == 4:
            small_grids.append(np.asfortranarray(np.load('{}/{}_grid.npy'.format(
                auf_folder_path, name), mmap_mode='r')[:, x, y, z]))
        else:
            small_grids.append(np.asfortranarray(np.load('{}/{}_grid.npy'.format(
                auf_folder_path, name), mmap_mode='r')[x, y, z]))
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


def map_large_index_to_small_index(inds, length, folder):
    inds_unique_flat = np.unique(inds[inds > -1])
    map_array = np.zeros(dtype=int, shape=(length,))
    map_array[:] = -1
    map_array[inds_unique_flat] = np.arange(0, len(inds_unique_flat), dtype=int)
    inds_map = np.asfortranarray(map_array[inds.flatten()].reshape(inds.shape))
    del map_array

    return inds_map, inds_unique_flat


def _load_single_sky_slice(folder_path, cat_name, ind, sky_inds):
    '''
    Function to, in a memmap-friendly way, return a sub-set of the nearest sky
    indices of a given catalogue.

    Parameters
    ----------
    folder_path : string
        Folder in which to store the temporary memmap file.
    cat_name : string
        String defining whether this function was called on catalogue "a" or "b".
    ind : float
        The value of the sky indices, as defined in ``distribute_sky_indices``,
        to return a sub-set of the larger catalogue. This value represents
        the index of a given on-sky position, used to construct the "counterpart"
        and "field" likelihoods.
    sky_inds : numpy.ndarray
        The given catalogue's ``distribute_sky_indices`` values, to compare
        with ``ind``.

    Returns
    -------
    sky_cut : numpy.ndarray
        A boolean array, indicating whether each element in ``sky_inds`` matches
        ``ind`` or not.
    '''
    sky_cut = np.zeros(dtype=bool, shape=(len(sky_inds),))

    di = max(1, len(sky_inds) // 20)

    for i in range(0, len(sky_inds), di):
        sky_cut[i:i+di] = sky_inds[i:i+di] == ind

    return sky_cut


def _create_rectangular_slice_arrays(folder_path, cat_name, len_a):
    '''
    Create temporary sky slice memmap arrays for parts of the cross-match
    process to use.

    Parameters
    ----------
    folder_path : string
        Location of where to store memmap arrays.
    cat_name : string
        Unique indicator of which catalogue these arrays are for.
    len_a : integer
        The length of the catalogue in question, allowing for a one-to-one
        mapping of sky slice per source.
    '''
    np.lib.format.open_memmap('{}/{}_temporary_sky_slice_1.npy'.format(
        folder_path, cat_name), mode='w+', dtype=bool, shape=(len_a,))
    np.lib.format.open_memmap('{}/{}_temporary_sky_slice_2.npy'.format(
        folder_path, cat_name), mode='w+', dtype=bool, shape=(len_a,))
    np.lib.format.open_memmap('{}/{}_temporary_sky_slice_3.npy'.format(
        folder_path, cat_name), mode='w+', dtype=bool, shape=(len_a,))
    np.lib.format.open_memmap('{}/{}_temporary_sky_slice_4.npy'.format(
        folder_path, cat_name), mode='w+', dtype=bool, shape=(len_a,))
    np.lib.format.open_memmap('{}/{}_temporary_sky_slice_combined.npy'.format(
        folder_path, cat_name), mode='w+', dtype=bool, shape=(len_a,))

    return


def _load_rectangular_slice(folder_path, cat_name, a, lon1, lon2, lat1, lat2, padding,
                            memmap_arrays):
    '''
    Loads all sources in a catalogue within a given separation of a rectangle
    in sky coordinates, allowing for the search for all sources within a given
    radius of sources inside the rectangle.

    Parameters
    ----------
    folder_path : string
        Location of where the memmap files used in the slicing of the
        catalogue are stored.
    cat_name : string
        Indication of whether we are loading catalogue "a" or catalogue "b",
        for separation within a given folder.
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
    memmap_arrays : list of numpy.ndarray
        The list of temporary arrays to use for memory-friendly sky coordinate
        slicing.

    Returns
    -------
    sky_cut : numpy.ndarray
        Boolean array, indicating whether each source in ``a`` is within ``padding``
        of the rectangle defined by ``lon1``, ``lon2``, ``lat1``, and ``lat2``.
    '''

    # Slice the memmapped catalogue, with a memmapped slicing array to
    # preserve memory.
    sky_cut_1, sky_cut_2, sky_cut_3, sky_cut_4, sky_cut = memmap_arrays

    di = max(1, len(a) // 20)
    # Iterate over each small slice of the larger array, checking for upper
    # and lower longitude, then latitude, criterion matching.
    lon_shift = 180 - (lon2 + lon1)/2
    for i in range(0, len(a), di):
        _lon_cut(i, a, di, lon1, padding, sky_cut_1, 'greater', lon_shift)
    for i in range(0, len(a), di):
        _lon_cut(i, a, di, lon2, padding, sky_cut_2, 'lesser', lon_shift)

    for i in range(0, len(a), di):
        _lat_cut(i, a, di, lat1, padding, sky_cut_3, 'greater')
    for i in range(0, len(a), di):
        _lat_cut(i, a, di, lat2, padding, sky_cut_4, 'lesser')

    for i in range(0, len(a), di):
        sky_cut[i:i+di] = (sky_cut_1[i:i+di] & sky_cut_2[i:i+di] &
                           sky_cut_3[i:i+di] & sky_cut_4[i:i+di])

    return sky_cut


def _lon_cut(i, a, di, lon, padding, sky_cut, inequality, lon_shift):
    '''
    Function to calculate the longitude inequality criterion for astrometric
    sources relative to a rectangle defining boundary limits.

    Parameters
    ----------
    i : integer
        Index into ``sky_cut`` for slicing.
    a : numpy.ndarray
        The main astrometric catalogue to be sliced, loaded into memmap.
    di : integer
        Index stride value, for slicing.
    lon : float
        Longitude at which to cut sources, either above or below, in degrees.
    padding : float
        Maximum allowed sky separation the "wrong" side of ``lon``, to allow
        for an increase in sky box size to ensure all overlaps are caught in
        ``get_max_overlap`` or ``get_max_indices``.
    sky_cut : numpy.ndarray
        Array into which to store boolean flags for whether source meets the
        sky position criterion.
    inequality : string, ``greater`` or ``lesser``
        Flag to determine whether a source is either above or below the
        given ``lon`` value.
    lon_shift : float
        Value by which to "move" longitudes to avoid meridian overflow issues.
    '''
    b = a[i:i+di, 0] + lon_shift
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
        sky_cut[i:i+di] = (hav_dist_constant_lat(a[i:i+di, 0], a[i:i+di, 1], lon) <=
                           padding) | inequal_lon_cut
    # However, in both zero and non-zero padding factor cases, we always require
    # the source to be above or below the longitude for sky_cut_1 and sky_cut_2
    # in load_fourier_grid_cutouts, respectively.
    else:
        sky_cut[i:i+di] = inequal_lon_cut


def _lat_cut(i, a, di, lat, padding, sky_cut, inequality):
    '''
    Function to calculate the latitude inequality criterion for astrometric
    sources relative to a rectangle defining boundary limits.

    Parameters
    ----------
    i : integer
        Index into ``sky_cut`` for slicing.
    a : numpy.ndarray
        The main astrometric catalogue to be sliced, loaded into memmap.
    di : integer
        Index stride value, for slicing.
    lat : float
        Latitude at which to cut sources, either above or below, in degrees.
    padding : float
        Maximum allowed sky separation the "wrong" side of ``lat``, to allow
        for an increase in sky box size to ensure all overlaps are caught in
        ``get_max_overlap`` or ``get_max_indices``.
    sky_cut : numpy.ndarray
        Array into which to store boolean flags for whether source meets the
        sky position criterion.
    inequality : string, ``greater`` or ``lesser``
        Flag to determine whether a source is either above or below the
        given ``lat`` value.
    '''

    # The "padding" factor is easier to handle for constant longitude in the
    # Haversine formula, being a straight comparison of delta-lat, and thus we
    # can simply move the required latitude padding factor to within the
    # latitude comparison.
    if padding > 0:
        if inequality == 'lesser':
            # Allow for small floating point rounding errors in comparison.
            sky_cut[i:i+di] = a[i:i+di, 1] <= lat + padding + 1e-6
        else:
            sky_cut[i:i+di] = a[i:i+di, 1] >= lat - padding - 1e-6
    else:
        if inequality == 'lesser':
            sky_cut[i:i+di] = a[i:i+di, 1] <= lat + 1e-6
        else:
            sky_cut[i:i+di] = a[i:i+di, 1] >= lat - 1e-6


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
    # TODO: can this be simplified with a lon_shift like lon_cut above?
    min_lon, max_lon = np.amin(a), np.amax(a)
    if min_lon <= 1 and max_lon >= 359 and np.any(np.abs(a - 180) < 1):
        # If there is data both either side of 0/360 and at 180 degrees,
        # return the entire longitudinal circle as the limits.
        return 0, 360
    elif min_lon <= 1 and max_lon >= 359:
        # If there's no data around the anti-longitude but data either
        # side of zero degrees exists, return the [-pi, +pi] wrapped
        # values.
        min_lon = np.amin(a[a > 180] - 360)
        max_lon = np.amax(a[a < 180])
        return min_lon, max_lon
    else:
        # Otherwise, the limits are inside [0, 360] and should be returned
        # as the "normal" minimum and maximum values.
        return min_lon, max_lon
