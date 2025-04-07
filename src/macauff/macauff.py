# Licensed under a 3-clause BSD style license - see LICENSE
'''
This module provides the lower-level framework for performing catalogue-catalogue cross-matches,
running an already-established catalogue-catalogue association determination.
'''

import datetime
import os
import sys

import numpy as np

# pylint: disable=import-error,no-name-in-module
from macauff.group_sources_fortran import group_sources_fortran as gsf
from macauff.misc_functions import coord_inside_convex_hull
from macauff.misc_functions_fortran import misc_functions_fortran as mff

# pylint: enable=import-error,no-name-in-module


class Macauff():
    '''
    Class to perform the catalogue-catalogue association determination, on two
    datasets which are already pre-processed and set up for cross-matching.

    Parameters
    ----------
    cm : Class
        Input "IO" class, with the necessary parameters and datasets configured
        for cross-matching.
    '''

    def __init__(self, cm):
        self.cm = cm

    def __call__(self):
        '''
        Call each step in a cross-match in turn.
        '''

        os.system(f'rm -r {self.cm.joint_folder_path}/*')
        # The first step is to create the perturbation AUF components, if needed.
        self.create_perturb_auf()

        # Once AUF components are assembled, we now group sources based on
        # convolved AUF integration lengths, to get "common overlap" sources
        # and merge such overlaps into distinct "islands" of sources to match.
        self.group_sources()

        # The third step in this process is to, to some level, calculate the
        # photometry-related information necessary for the cross-match.
        self.calculate_phot_like()

        # The final stage of the cross-match process is that of putting together
        # the previous stages, and calculating the cross-match probabilities.
        self.pair_sources()

    def create_perturb_auf(self):
        '''
        Function wrapping the main perturbation AUF component creation routines.
        '''

        # Magnitude offsets corresponding to relative fluxes of perturbing sources; here
        # dm of 2.5 is 10% relative flux and dm = 5 corresponds to 1% relative flux. Used
        # to inform the fraction of simulations with a contaminant above these relative
        # fluxes.
        # pylint: disable-next=fixme
        # TODO: allow as user input.
        self.cm.delta_mag_cuts = np.array([2.5, 5])

        # pylint: disable-next=fixme
        # TODO: allow as user input.
        self.cm.gal_cmau_array = np.empty((5, 2, 4), float)
        # See Wilson (2022, RNAAS, 6, 60) for the meanings of the variables c, m,
        # a, and u. For each of M*/phi*/alpha/P/Q, for blue+red galaxies, 2-4
        # variables are derived as a function of wavelength, or Q(P).
        self.cm.gal_cmau_array[0, :, :] = [[-24.286513, 1.141760, 2.655846, np.nan],
                                           [-23.192520, 1.778718, 1.668292, np.nan]]
        self.cm.gal_cmau_array[1, :, :] = [[0.001487, 2.918841, 0.000510, np.nan],
                                           [0.000560, 7.691261, 0.003330, -0.065565]]
        self.cm.gal_cmau_array[2, :, :] = [[-1.257761, 0.021362, np.nan, np.nan],
                                           [-0.309077, -0.067411, np.nan, np.nan]]
        self.cm.gal_cmau_array[3, :, :] = [[-0.302018, 0.034203, np.nan, np.nan],
                                           [-0.713062, 0.233366, np.nan, np.nan]]
        self.cm.gal_cmau_array[4, :, :] = [[1.233627, -0.322347, np.nan, np.nan],
                                           [1.068926, -0.385984, np.nan, np.nan]]
        self.cm.gal_alpha0 = [[2.079, 3.524, 1.917, 1.992, 2.536], [2.461, 2.358, 2.568, 2.268, 2.402]]
        self.cm.gal_alpha1 = [[2.265, 3.862, 1.921, 1.685, 2.480], [2.410, 2.340, 2.200, 2.540, 2.464]]
        self.cm.gal_alphaweight = [[3.47e+09, 3.31e+06, 2.13e+09, 1.64e+10, 1.01e+09],
                                   [3.84e+09, 1.57e+06, 3.91e+08, 4.66e+10, 3.03e+07]]

        t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f'{t} Rank {self.cm.rank}, chunk {self.cm.chunk_id}: Creating empirical perturbation AUFs '
              'for catalogue "a"...')
        sys.stdout.flush()
        if self.cm.j0s is None:
            self.cm.j0s = mff.calc_j0(self.cm.rho[:-1]+self.cm.drho/2, self.cm.r[:-1]+self.cm.dr/2)
        self.cm.a_modelrefinds, self.cm.a_perturb_auf_outputs = self.cm.perturb_auf_func(self.cm, 'a')

        t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f'{t} Rank {self.cm.rank}, chunk {self.cm.chunk_id}: Creating empirical perturbation AUFs '
              'for catalogue "b"...')
        sys.stdout.flush()
        if self.cm.j0s is None:
            self.cm.j0s = mff.calc_j0(self.cm.rho[:-1]+self.cm.drho/2, self.cm.r[:-1]+self.cm.dr/2)
        self.cm.b_modelrefinds, self.cm.b_perturb_auf_outputs = self.cm.perturb_auf_func(self.cm, 'b')

    def group_sources(self):
        '''
        Function to handle the creation of catalogue "islands" and potential
        astrometrically related sources across the two catalogues.
        '''

        t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f'{t} Rank {self.cm.rank}, chunk {self.cm.chunk_id}: Creating catalogue islands and '
              'overlaps...')
        sys.stdout.flush()
        if self.cm.j1s is None:
            self.cm.j1s = gsf.calc_j1s(self.cm.rho[:-1]+self.cm.drho/2, self.cm.r[:-1]+self.cm.dr/2)
        self.cm.group_func(self.cm)

    def calculate_phot_like(self):
        '''
        Create the photometric likelihood information used in the cross-match
        process.
        '''

        t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f'{t} Rank {self.cm.rank}, chunk {self.cm.chunk_id}: Creating photometric priors '
              'and likelihoods...')
        sys.stdout.flush()
        self._calculate_cf_areas()
        self.cm.phot_like_func(self.cm)

    def _calculate_cf_areas(self):
        '''
        Convenience function to calculate the area around each individual
        sky coordinate where it is defined as having the smallest on-sky
        separation.
        '''
        def _calc_cf_area_call(min_lon, max_lon, min_lat, max_lat, cf_areas, n_lon, n_lat, recursion_level):
            '''
            Recursive function to calculate the area within a convex hull
            nearest to each ``cm.a_hull_point`` and ``cm.b_hull_point`` overlap
            region.

            Parameters
            ----------
            min_lon : float
                Lowest longitude to generate grid of points to check for
                overlap with convex hulls and include in respective area.
            max_lon : float
                Largest longitude to generate search grid for.
            min_lat : float
                Minimum latitude to check area overlap for, to generate
                array with ``max_lat``.
            max_lat : float
                Maximum latitude for which to check hull overlap.
            cf_areas : numpy.ndarray
                Array of areas of each coordinate in ``cm.cf_region_points``,
                for which each lon-lat coordinate is considered and recursively
                added.
            n_lon : int
                Number of longitudinal points to generate between ``min_lon``
                and ``max_lon``.
            n_lat : int
                Number of latitudinal points to search between ``min_lat`` and
                ``max_lat``.
            recursion_level : int
                Flag to indicate how many times we've "zoomed" in on the grid
                region, depending on whether our grid box lies completely in,
                completely out, or partially overlapping either convex hull.
                Once we've zoomed in sufficiently, calculate brute force area
                overlaps; otherwise zoom in again.

            Returns
            -------
            cf_areas : numpy.ndarray
                Return the same ``cf_areas`` array, updated with new grid
                searches.
            '''
            # If we've hit what we want to call the bottom of the recursion, at
            # 0.001-deg scale, then we can brute force our way through the area
            # computation, but otherwise do things more carefully.
            if recursion_level == 2:
                test_lons = np.linspace(min_lon, max_lon, n_lon)
                test_lons = test_lons[:-1] + np.diff(test_lons)/2
                test_lats = np.linspace(min_lat, max_lat, n_lat)
                test_lats = test_lats[:-1] + np.diff(test_lats)/2

                points_in_area = (
                    np.array([coord_inside_convex_hull([ax1 + self.cm.a_hull_x_shift, ax2],
                              self.cm.a_hull_points) for ax1 in test_lons for ax2 in test_lats]) &
                    np.array([coord_inside_convex_hull([ax1 + self.cm.b_hull_x_shift, ax2],
                              self.cm.b_hull_points) for ax1 in test_lons for ax2 in test_lats]))

                test_coords = np.array([[a, b] for a in test_lons for b in test_lats])

                inds = mff.find_nearest_point(
                    test_coords[:, 0], test_coords[:, 1], self.cm.cf_region_points[:, 0],
                    self.cm.cf_region_points[:, 1])

                for k, (ind, point_in_area) in enumerate(zip(inds, points_in_area)):
                    if point_in_area:
                        cf_areas[ind] += np.cos(np.radians(test_coords[k, 1])) * (
                            test_lons[1] - test_lons[0]) * (test_lats[1] - test_lats[0])

                return cf_areas

            test_lons = np.linspace(min_lon, max_lon, n_lon)
            test_lats = np.linspace(min_lat, max_lat, n_lat)

            points_in_area = (
                np.array([coord_inside_convex_hull([ax1 + self.cm.a_hull_x_shift, ax2], self.cm.a_hull_points)
                          for ax1 in test_lons for ax2 in test_lats]) &
                np.array([coord_inside_convex_hull([ax1 + self.cm.b_hull_x_shift, ax2], self.cm.b_hull_points)
                          for ax1 in test_lons for ax2 in test_lats])).reshape(
                len(test_lons), len(test_lats))

            for i in range(len(test_lons)-1):
                for j in range(len(test_lats)-1):
                    n_points_in_area = np.sum(points_in_area[i:i+2, j:j+2])
                    # If no coarse grid points are in the area, job done, no need
                    # to zoom in any further. Otherwise bifurcate based on if all
                    # points are in the hull or not.
                    if n_points_in_area == 4:
                        # If we can skip straight from the 0.1-deg bin to the end,
                        # we have 100 bins to split into; otherwise it's just 10.
                        if recursion_level == 0:
                            small_n_lon = 101
                            small_n_lat = 101
                        else:
                            small_n_lon = 11
                            small_n_lat = 11
                        # Make sure to take the centre of grid points now,
                        # rather than the corners.
                        small_test_lons = np.linspace(test_lons[i], test_lons[i+1], small_n_lon)
                        small_test_lons = small_test_lons[:-1] + np.diff(small_test_lons)/2
                        small_test_lats = np.linspace(test_lats[j], test_lats[j+1], small_n_lat)
                        small_test_lats = small_test_lats[:-1] + np.diff(small_test_lats)/2

                        test_coords = np.array([[a, b] for a in small_test_lons for b in small_test_lats])
                        small_inds = mff.find_nearest_point(
                            test_coords[:, 0], test_coords[:, 1], self.cm.cf_region_points[:, 0],
                            self.cm.cf_region_points[:, 1])

                        for k, ind in enumerate(small_inds):
                            cf_areas[ind] += np.cos(
                                np.radians(test_coords[k, 1])) * (small_test_lons[1] - small_test_lons[0]) * (
                                small_test_lats[1] - small_test_lats[0])
                    elif n_points_in_area > 0:
                        # Start the whole thing again, just 10x smaller,
                        # returning the recursively updated area grid.
                        cf_areas = _calc_cf_area_call(
                            test_lons[i], test_lons[i+1], test_lats[j], test_lats[j+1], cf_areas,
                            n_lon=11, n_lat=11, recursion_level=recursion_level+1)

            return cf_areas

        t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f'{t} Rank {self.cm.rank}, chunk {self.cm.chunk_id}: Calculating photometric region areas...')
        sys.stdout.flush()

        min_lon = max(np.amin(self.cm.a_hull_points[:, 0] - self.cm.a_hull_x_shift),
                      np.amin(self.cm.b_hull_points[:, 0] - self.cm.a_hull_x_shift))
        max_lon = min(np.amax(self.cm.a_hull_points[:, 0] - self.cm.b_hull_x_shift),
                      np.amax(self.cm.b_hull_points[:, 0] - self.cm.b_hull_x_shift))
        min_lat = max(np.amin(self.cm.a_hull_points[:, 1]), np.amin(self.cm.b_hull_points[:, 1]))
        max_lat = min(np.amax(self.cm.a_hull_points[:, 1]), np.amax(self.cm.b_hull_points[:, 1]))

        n_lon = int(np.ceil((max_lon - min_lon) / 0.1)) + 1
        n_lat = int(np.ceil((max_lat - min_lat) / 0.1)) + 1

        cf_areas = np.zeros((len(self.cm.cf_region_points)), float)
        self.cm.cf_areas = _calc_cf_area_call(min_lon, max_lon, min_lat, max_lat, cf_areas,
                                              n_lon=n_lon, n_lat=n_lat, recursion_level=0)

    def pair_sources(self):
        '''
        Assign sources in the two catalogues as either counterparts to one another
        or singly detected "field" sources.
        '''

        t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f'{t} Rank {self.cm.rank}, chunk {self.cm.chunk_id}: Determining counterparts...')
        sys.stdout.flush()
        self.cm.count_pair_func(self.cm)

        if self.cm.include_phot_like and self.cm.with_and_without_photometry:
            self.cm.count_pair_func(self.cm, force_no_phot_like=True)
