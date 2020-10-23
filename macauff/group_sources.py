# Licensed under a 3-clause BSD style license - see LICENSE
'''
This module provides the framework for grouping sources from two photometric catalogues into
distinct "islands" of sources, along with calculating whether they are within overlap for
various photometric integral purposes.
'''

import sys
import operator
import numpy as np

from .misc_functions import create_cumulative_offsets_grid
# from .group_sources_fortran import group_sources_fortran as gsf

__all__ = ['make_island_groupings']


def make_island_groupings(max_sep, ax_lims, int_fracs, a_filt_names, b_filt_names, a_title,
                          b_title, r):
    '''
    Function to handle the creation of "islands" of astrometrically coeval
    sources, and identify which overlap to some probability based on their
    combined AUFs.
    '''
    print("Creating catalogue islands and overlaps...")
    sys.stdout.flush()

    print("Calculating maximum overlap...")
    sys.stdout.flush()

    # Create the 4-D grids that house the perturbation AUF cumulative integrals.
    create_cumulative_offsets_grid(a_auf_folder_path, a_auf_pointings, a_filt_names, r)
    create_cumulative_offsets_grid(b_auf_folder_path, b_auf_pointings, b_filt_names, r)

    # The initial step to create island groupings is to find the largest number
    # of overlaps for a single source, to minimise the size of the array of
    # overlap indices. To do so, we load small-ish chunks of the sky, with
    # padding in one catalogue to ensure all pairings can be found, and total
    # the number of overlaps for each object across all sky slices.

    ax1_loops = np.linspace(ax_lims[0], ax_lims[1], 11)
    # Force the sub-division of the sky area in question to be 100 chunks, or
    # one square degree chunks, whichever is larger in area.
    if ax1_loops[1] - ax1_loops[0] < 1:
        ax1_loops = np.arange(ax_lims[0], ax_lims[1]+1e-10, 1)
    ax2_loops = np.linspace(ax_lims[2], ax_lims[3], 11)
    if ax2_loops[1] - ax2_loops[0] < 1:
        ax2_loops = np.arange(ax_lims[2], ax_lims[3]+1e-10, 1)

    # Load the astrometry of each catalogue for slicing.
    a_full = np.load('{}/con_cat_astro.npy'.format(a_cat_folder_path), mmap_mode='r')
    b_full = np.load('{}/con_cat_astro.npy'.format(b_cat_folder_path), mmap_mode='r')
    for ax1_start, ax1_end in zip(ax1_loops[:-1], ax1_loops[1:]):
        for ax2_start, ax2_end in zip(ax2_loops[:-1], ax2_loops[1:]):
            pass


def _load_cumulative_grid_cutouts(a, sky_rect_coords, joint_folder_path, cat_folder_path,
                                  auf_folder_path, padding):
    '''
    Function to load a sub-set of a given catalogue's astrometry, slicing it
    in a given sky coordinate rectangle, and load the appropriate sub-array
    of the perturbation AUF's cumulative PDF.
    '''

    lon1, lon2, lat1, lat2 = sky_rect_coords
    # Slice the memmapped catalogue, with a memmapped slicing array to
    # preserve memory.
    sky_cut_1 = np.lib.format.open_memmap('{}/temporary_sky_slice_1.npy'.format(
        joint_folder_path), mode='w+', dtype=np.bool, shape=(len(a),))
    sky_cut_2 = np.lib.format.open_memmap('{}/temporary_sky_slice_2.npy'.format(
        joint_folder_path), mode='w+', dtype=np.bool, shape=(len(a),))
    sky_cut_3 = np.lib.format.open_memmap('{}/temporary_sky_slice_3.npy'.format(
        joint_folder_path), mode='w+', dtype=np.bool, shape=(len(a),))
    sky_cut_4 = np.lib.format.open_memmap('{}/temporary_sky_slice_4.npy'.format(
        joint_folder_path), mode='w+', dtype=np.bool, shape=(len(a),))
    sky_cut = np.lib.format.open_memmap('{}/temporary_sky_slice_combined.npy'.format(
        joint_folder_path), mode='w+', dtype=np.bool, shape=(len(a),))

    di = len(a) // 20
    # Iterate over each small slice of the larger array, checking for upper
    # and longer longitude criterion matching.
    for i in range(0, len(a), di):
        _lon_cut(i, a.shape[0], di, lon1, padding, joint_folder_path, operator.ge, '1')
    for i in range(0, len(a), di):
        _lon_cut(i, a.shape[0], di, lon2, padding, joint_folder_path, operator.le, '2')

    for i in range(0, len(a), di):
        _lat_cut(i, a.shape[0], di, lat1, padding, joint_folder_path, operator.ge, '3')
    for i in range(0, len(a), di):
        _lat_cut(i, a.shape[0], di, lat2, padding, joint_folder_path, operator.le, '4')

    for i in range(0, len(a), di):
        sky_cut[i:i+di] = (sky_cut_1[i:i+di] & sky_cut_2[i:i+di] &
                           sky_cut_3[i:i+di] & sky_cut_4[i:i+di])

    a_cutout = np.load('{}/con_cat_astro.npy'.format(cat_folder_path), mmap_mode='r')[sky_cut]

    modrefind = np.load('{}/modelrefinds.npy'.format(auf_folder_path), mmap_mode='r')[:, sky_cut]

    nmuniqueind, nmnewind = np.unique(modrefind[0, :], return_inverse=True)
    filtuniqueind, filtnewind = np.unique(modrefind[1, :], return_inverse=True)
    axuniqueind, axnewind = np.unique(modrefind[2, :], return_inverse=True)
    cumulatoffgrids = np.asfortranarray(np.load('{}/cumulative_grid.npy'.format(
                                        auf_folder_path), mmap_mode='r')[:, :, :,
                                        axuniqueind][:, :, filtuniqueind, :][:,
                                        nmuniqueind, :, :])
    modrefindsmall = np.empty((3, modrefind.shape[1]), int, order='F')
    del modrefind
    modrefindsmall[0, :] = nmnewind
    modrefindsmall[1, :] = filtnewind
    modrefindsmall[2, :] = axnewind

    return a_cutout, cumulatoffgrids, modrefindsmall


def _lon_cut(i, lena, di, lon, padding, joint_folder_path, inequality, num):
    '''
    Function to calculate the longitude inequality criterion for astrometric
    sources relative to a rectangle defining boundary limits.
    '''
    a = np.lib.format.open_memmap('con_cat_astro.npy', mode='r', dtype=float, shape=(lena, 3))
    sky_cut = np.lib.format.open_memmap('{}/temporary_sky_slice_{}.npy'.format(
        joint_folder_path, num), mode='r+', dtype=np.bool, shape=(len(a),))
    # To check whether a source should be included in this slice or not if the
    # "padding" factor is non-zero, add an extra caveat to check whether
    # Haversine great-circle distance is less than the padding factor. For
    # constant latitude this reduces to
    # r = 2 arcsin(|cos(lat) * sin(delta-lon/2)|).
    # However, in both zero and non-zero padding factor cases, we always require
    # the source to be above or below the longitude for sky_cut_1 and sky_cut_2
    # in load_cumulative_grid_cutouts, respectively.
    if padding > 0:
        sky_cut[i:i+di] = (np.degrees(2 * np.arcsin(np.abs(np.cos(np.radians(a[i:i+di, 1])) *
                           np.sin(np.radians((a[i:i+di, 0] - lon)/2))))) <=
                           padding) | inequality(a[i:i+di, 0], lon)
    else:
        sky_cut[i:i+di] = inequality(a[i:i+di, 0], lon)


def _lat_cut(i, lena, di, lat, padding, joint_folder_path, inequality, num):
    '''
    Function to calculate the latitude inequality criterion for astrometric
    sources relative to a rectangle defining boundary limits.
    '''
    a = np.lib.format.open_memmap('con_cat_astro.npy', mode='r', dtype=float, shape=(lena, 3))
    sky_cut = np.lib.format.open_memmap('{}/temporary_sky_slice_{}.npy'.format(
        joint_folder_path, num), mode='r+', dtype=np.bool, shape=(len(a),))
    # The "padding" factor is easier to handle for constant longitude in the
    # Haversine formula, being a straight comparison of delta-lat, and thus we
    # can simply move the required latitude padding factor to within the
    # latitude comparison.
    if padding > 0:
        sky_cut[i:i+di] = inequality(a[i:i+di, 1] + padding, lat)
    else:
        sky_cut[i:i+di] = inequality(a[i:i+di, 1], lat)
