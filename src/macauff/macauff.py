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
        os.system(f'rm -rf {self.cm.joint_folder_path}/reject/*')
        self.cm.group_sources_data = self.cm.group_func(self.cm)

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
        self.cm.phot_like_data = self.cm.phot_like_func(self.cm)

    def _calculate_cf_areas(self):
        '''
        Convenience function to calculate the area around each
        ``cross_match_extent`` sky coordinate where it is defined as having the
        smallest on-sky separation.
        '''
        t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f'{t} Rank {self.cm.rank}, chunk {self.cm.chunk_id}: Calculating photometric region areas...')
        dlon, dlat = 0.01, 0.01
        test_lons = np.arange(self.cm.cross_match_extent[0], self.cm.cross_match_extent[1], dlon)
        test_lats = np.arange(self.cm.cross_match_extent[2], self.cm.cross_match_extent[3], dlat)

        test_coords = np.array([[a, b] for a in test_lons for b in test_lats])

        inds = mff.find_nearest_point(test_coords[:, 0], test_coords[:, 1],
                                      self.cm.cf_region_points[:, 0], self.cm.cf_region_points[:, 1])

        cf_areas = np.zeros((len(self.cm.cf_region_points)), float)

        # Unit area of a sphere is cos(theta) dtheta dphi if theta goes from -90
        # to +90 degrees (sin(theta) for 0 to 180 degrees). Note, however, that
        # dtheta and dphi have to be in radians, so we have to convert the entire
        # thing from degrees and re-convert at the end. Hence:
        for i, ind in enumerate(inds):
            theta = np.radians(test_coords[i, 1])
            dtheta, dphi = dlat / 180 * np.pi, dlon / 180 * np.pi
            # Remember to convert back to square degrees:
            cf_areas[ind] += (np.cos(theta) * dtheta * dphi) * (180 / np.pi)**2

        self.cm.cf_areas = cf_areas

    def pair_sources(self):
        '''
        Assign sources in the two catalogues as either counterparts to one another
        or singly detected "field" sources.
        '''

        t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f'{t} Rank {self.cm.rank}, chunk {self.cm.chunk_id}: Determining counterparts...')
        sys.stdout.flush()
        os.system(f'rm -r {self.cm.joint_folder_path}/pairing/*')
        self.cm.count_pair_func(self.cm)
