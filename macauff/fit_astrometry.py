# Licensed under a 3-clause BSD style license - see LICENSE
'''
Module for calculating corrections to astrometric uncertainties of photometric catalogues, by
fitting their AUFs and centroid uncertainties across ensembles of matches between well-understood
catalogue and one for which precisions are less well known.
'''

import numpy as np
from numpy.lib.format import open_memmap
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
import multiprocessing
import itertools
import os
from scipy.optimize import minimize
import matplotlib.gridspec as gridspec
from astropy.coordinates import SkyCoord, match_coordinates_sky, Angle, UnitSphericalRepresentation
from astropy import units as u
from scipy import spatial
from scipy.stats import chi2
import shutil
# Assume that usetex = False only applies for tests where no TeX is installed
# at all, instead of users having half-installed TeX, dvipng et al. somewhere.
usetex = not not shutil.which("tex")
if usetex:
    plt.rcParams.update({"text.usetex": True, "text.latex.preamble": r"\usepackage{amsmath}"})

from .galaxy_counts import create_galaxy_counts
from .perturbation_auf import (download_trilegal_simulation, _calculate_magnitude_offsets,
                               make_tri_counts)
from .misc_functions import min_max_lon
from .misc_functions_fortran import misc_functions_fortran as mff
from .perturbation_auf_fortran import perturbation_auf_fortran as paf
from .get_trilegal_wrapper import get_AV_infinity


__all__ = ['AstrometricCorrections']


class AstrometricCorrections:
    """
    Class to calculate any potential corrections to quoted astrometric
    precisions in photometric catalogues, based on reliable cross-matching
    to a well-understood second dataset.
    """
    def __init__(self, psf_fwhm, numtrials, nn_radius, dens_search_radius, save_folder, trifolder,
                 triname, maglim_f, magnum, tri_num_faint, trifilterset, trifiltname,
                 gal_wav_micron, gal_ab_offset, gal_filtname, gal_alav, dm, dd_params, l_cut,
                 ax1_mids, ax2_mids, ax_dimension, mag_array, mag_slice, sig_slice, n_pool,
                 npy_or_csv, coord_or_chunk, pos_and_err_indices, mag_indices, mag_unc_indices,
                 mag_names, best_mag_index, coord_system, pregenerate_cutouts, cutout_area=None,
                 cutout_height=None, single_sided_auf=True, chunks=None):
        """
        Initialisation of AstrometricCorrections, accepting inputs required for
        the running of the optimisation and parameterisation of astrometry of
        a photometric catalogue, benchmarked against a catalogue of much higher
        astrometric resolution and precision, such as Gaia or the Hubble Source
        Catalog.

        Parameters
        ----------
        psf_fwhm : float
            The full-width at half-maximum of the Point Spread Function, used to
            determine the size of the PSF for perturber placement purposes.
        numtrials : integer
            Number of simulations to run when deriving pertubation statistics.
        nn_radius : float
            Size of nearest-neighbour search for construction of intermediate
            cross-match distributions, in arcseconds.
        dens_search_radius : float
            Radius out to which to search around objects internal to a
            catalogue, to determine the local normalising density for each
            source, in arcseconds.
        save_folder : string
            Absolute or relative filepath of folder into which to store
            temporary and generated outputs from the fitting process.
        trifolder : string
            Filepath of the location into which to save TRILEGAL simulations.
        triname : string
            Name to give TRILEGAL simulations when downloaded. Is required to have
            two format ``{}`` options in string, for unique ax1-ax2 sightline
            combination downloads. Should not contain any file types, just the
            name of the file up to, but not including, the final ".".
        maglim_f : float
            Magnitude in the ``magnum`` filter down to which sources should be
            drawn for the "faint" sample.
        magnum : float
            Zero-indexed column number of the chosen filter limiting magnitude.
        tri_num_faint : integer
            Approximate number of objects to simulate in the chosen filter for
            TRILEGAL simulations.
        trifilterset : string
            Name of the TRILEGAL filter set for which to generate simulations.
        trifiltname : string
            Name of the specific filter to generate perturbation AUF component in.
        gal_wav_micron : float
            Wavelength, in microns, of the ``trifiltname`` filter, for use in
            simulating galaxy counts.
        gal_ab_offset : float
            The offset between the ``trifiltname`` filter zero point and the
            AB magnitude offset.
        gal_filtname : string
            Name of the filter in the ``speclite`` compound naming convention.
        gal_alav : float
            Differential reddening vector of the given filter.
        dm : float
            Bin spacing for magnitude histograms of TRILEGAL simulations.
        dd_params : numpy.ndarray
            Array, of shape ``(5, X, 2)``, containing the parameterisation of
            the skew-normal used to construct background-dominated PSF
            perturbations.
        l_cut : numpy.ndarray
            Array of shape ``(3,)`` containing the cuts between the different
            regimes of background-dominated PSF perturbation.
        ax1_mids : numpy.ndarray
            Array of longitudes (e.g. RA or l) used to center regions used to
            determine astrometric corrections across the sky. Depending on
            ``ax_correction``, either the unique values that with ``ax2_mids``
            form a rectangle, or a unique ax-ax combination with a corresponding
            ``ax2_mid``.
        ax2_mids : numpy.ndarray
            Array of latitudes (Dec/b) defining regions for calculating
            astrometric corrections. Either unique rectangle values to combine
            with ``ax1_mids`` or unique ``ax1_mids``-``ax2_mids`` pairs, one
            per entry.
        ax_dimension : integer, either ``1`` or ``2``
            If ``1`` then ``ax1_mids`` and ``ax2_mids`` form unique sides of a
            rectangle when combined in a grid, or if ``2`` each
            ``ax1_mids``-``ax2_mids`` combination is a unique ax-ax pairing used
            as given.
        mag_array : numpy.ndarray
            List of magnitudes in the ``trifiltname`` filter to derive astrometry
            for.
        mag_slice : numpy.ndarray
            Widths of interval at which to take slices of magnitudes for deriving
            astrometric properties. Each ``mag_slice`` maps elementwise to each
            ``mag_array``, and hence they should be the same shape.
        sig_slice : numpy.ndarray
            Interval widths of quoted astrometric uncertainty to use when
            isolating individual sets of objects for AUF derivation. Length
            should match ``mag_array``.
        n_pool : integer
            The maximum number of threads to use when calling
            ``multiprocessing``.
        npy_or_csv : string, either ``npy`` or ``csv``
            Indicator as to whether the small chunks of sky to be loaded for
            each sightline's evaluation are in binary ``numpy`` format or saved
            to disk as a comma-separated values file.
        coord_or_chunk : string, either ``coord`` or ``chunk``
            String indicating whether intermediate files should be saved with
            filenames that are unique by two coordinates (l/b or RA/Dec) or
            some kind of singular "chunk" number. Output filenames would then
            need to follow ``'file_{}{}'`` or ``'file_{}'`` formatting
            respectively.
        pos_and_err_indices : list of list or numpy.ndarray of integers
            In order, the indices within catalogue "a" and then "b" respecetively
            of either the .npy or .csv file of the longitudinal (e.g. RA or l),
            latitudinal (Dec or b), and *singular*, circular astrometric
            precision array. Coordinates should be in degrees while precision
            should be in the same units as ``sig_slice`` and those of the
            nearest-neighbour distances, likely arcseconds. For example,
            ``[[0, 1, 2], [6, 3, 0]]`` where catalogue "a" has its coordinates
            in the first three columns in RA/Dec/Err order, while catalogue "b"
            has its coordinates in a more random order.
        mag_indices : list or numpy.ndarray
            In appropriate order, as expected by e.g. `~macauff.CrossMatch` inputs
            and `~macauff.make_perturb_aufs`, list the indexes of each magnitude
            column within either the ``.npy`` or ``.csv`` file loaded for each
            sub-catalogue sightline. These should be zero-indexed.
        mag_unc_indices : list or numpy.ndarray
            For each ``mag_indices`` entry, the corresponding magnitude
            uncertainty index in the catalogue.
        mag_names : list or numpy.ndarray of strings
            Names of each ``mag_indices`` magnitude.
        best_mag_index : integer
            Index into ``mag_indices`` of the preferred magnitude to use to
            construct astrometric scaling relations from. Should generally
            be the one with the most coverage across all detections in a
            catalogue, or the one with the most precise fluxes.
        coord_system : string, "equatorial" or "galactic"
            Identifier of which coordinate system the data are in. Both datasets
            must be in the same system, which can either be RA/Dec (equatorial)
            or l/b (galactic) coordinates.
        pregenerate_cutouts : boolean
            Indicates whether sightline catalogues must have been pre-made,
            or whether they can be generated by ``AstrometricCorrections``
            using specified lines-of-sight and cutout areas and heights.
        cutout_area : float, optional
            The size, in square degrees, of the regions used to simulate
            AUFs and determine astrometric corrections. Required if
            ``pregenerate_cutouts`` is ``False``.
        cutout_height : float, optional
            The latitudinal height of the rectangular regions used in
            calculating astrometric corrections. Required if
            ``pregenerate_cutouts`` is ``False``.
        single_sided_auf : boolean, optional
            Flag indicating whether the AUF of catalogue "a" can be ignored
            when considering match statistics, or if astrometric corrections
            are being constructed from matches to a catalogue that also suffers
            significant non-noise-based astrometric uncertainty.
        chunks = list or numpy.ndarray of strings, optional
            List of IDs for each unique set of data if ``coord_or_chunk`` is
            ``chunk``. In this case, ``ax_dimension`` must be ``2`` and each
            ``chunk`` must correspond to its ``ax1_mids``-``ax2_mids`` coordinate.
        """
        if single_sided_auf is not True:
            raise ValueError("single_sided_auf must be True.")
        if not (ax_dimension == 1 or ax_dimension == 2):
            raise ValueError("ax_dimension must either be '1' or '2'.")
        if npy_or_csv != "npy" and npy_or_csv != "csv":
            raise ValueError("npy_or_csv must either be 'npy' or 'csv'.")
        if coord_or_chunk != "coord" and coord_or_chunk != "chunk":
            raise ValueError("coord_or_chunk must either be 'coord' or 'chunk'.")
        if coord_or_chunk == "chunk" and chunks is None:
            raise ValueError("chunks must be provided if coord_or_chunk is 'chunk'.")
        if coord_or_chunk == "chunk" and ax_dimension == 1:
            raise ValueError("ax_dimension must be 2, and ax1-ax2 pairings provided for each chunk "
                             "in chunks if coord_or_chunk is 'chunk'.")
        if coord_or_chunk == "chunk" and (len(ax1_mids) != len(chunks) or
                                          len(ax2_mids) != len(chunks)):
            raise ValueError("ax1_mids, ax2_mids, and chunks must all be the same length if "
                             "coord_or_chunk is 'chunk'.")
        if not (coord_system == "equatorial" or coord_system == "galactic"):
            raise ValueError("coord_system must either be 'equatorial' or 'galactic'.")
        if pregenerate_cutouts is not True and pregenerate_cutouts is not False:
            raise ValueError("pregenerate_cutouts should either be 'True' or 'False'.")
        if not pregenerate_cutouts and cutout_area is None:
            raise ValueError("cutout_area must be given if pregenerate_cutouts is 'False'.")
        if not pregenerate_cutouts and cutout_height is None:
            raise ValueError("cutout_height must be given if pregenerate_cutouts is 'False'.")
        self.psf_fwhm = psf_fwhm
        self.numtrials = numtrials
        self.nn_radius = nn_radius
        self.dens_search_radius = dens_search_radius
        # Currently hard-coded as it isn't useful for this work, but is required
        # for calling the perturbation AUF algorithms.
        self.dmcut = [2.5]

        self.R = 1.185 * self.psf_fwhm
        self.psfsig = self.psf_fwhm / (2 * np.sqrt(2 * np.log(2)))
        self.r, self.rho = np.linspace(0, self.R, 10000), np.linspace(0, 100, 9999)
        self.dr, self.drho = np.diff(self.r), np.diff(self.rho)

        self.save_folder = save_folder

        self.trifolder = trifolder
        self.triname = triname
        self.maglim_f = maglim_f
        self.magnum = magnum
        self.tri_num_faint = tri_num_faint
        self.trifilterset = trifilterset
        self.trifiltname = trifiltname
        self.gal_wav_micron = gal_wav_micron
        self.gal_ab_offset = gal_ab_offset
        self.gal_filtname = gal_filtname
        self.gal_alav = gal_alav

        self.dm = dm

        self.dd_params = dd_params
        self.l_cut = l_cut

        self.ax1_mids = ax1_mids
        self.ax2_mids = ax2_mids
        self.ax_dimension = ax_dimension

        self.pregenerate_cutouts = pregenerate_cutouts
        if not self.pregenerate_cutouts:
            self.cutout_area = cutout_area
            self.cutout_height = cutout_height

        self.mag_array = np.array(mag_array)
        self.mag_slice = np.array(mag_slice)
        self.sig_slice = np.array(sig_slice)

        self.npy_or_csv = npy_or_csv
        self.coord_or_chunk = coord_or_chunk
        self.chunks = chunks

        self.coord_system = coord_system

        if npy_or_csv == 'npy':
            self.pos_and_err_indices = pos_and_err_indices
            self.mag_indices = mag_indices
            self.mag_unc_indices = mag_unc_indices
            self.mag_names = mag_names
            self.best_mag_index = best_mag_index
        else:
            # np.loadtxt will load in pos_and_err or pos_and_err, mag_ind,
            # mag_unc_ind order for the two catalogues. Each will effectively
            # change its ordering, since we then load [0] for pos_and_err[0][0],
            # etc. for all options. These need saving for np.loadtxt but
            # also for obtaining the correct column in the resulting sub-set of
            # the loaded csv file.
            self.a_cols = np.array(pos_and_err_indices[0])
            self.b_cols = np.concatenate((pos_and_err_indices[1], mag_indices, mag_unc_indices))

            self.pos_and_err_indices = [
                [np.argmin(np.abs(q - self.a_cols)) for q in pos_and_err_indices[0]],
                [np.argmin(np.abs(q - self.b_cols)) for q in pos_and_err_indices[1]]]
            self.mag_indices = [np.argmin(np.abs(q - self.b_cols)) for q in mag_indices]
            self.mag_unc_indices = [np.argmin(np.abs(q - self.b_cols)) for q in mag_unc_indices]

            self.mag_names = mag_names
            self.best_mag_index = best_mag_index

        self.n_pool = n_pool

        self.j0s = mff.calc_j0(self.rho[:-1]+self.drho/2, self.r[:-1]+self.dr/2)

        self.n_mag_cols = np.ceil(np.sqrt(len(self.mag_array))).astype(int)
        self.n_mag_rows = np.ceil(len(self.mag_array) / self.n_mag_cols).astype(int)

        self.n_filt_cols = np.ceil(np.sqrt(len(self.mag_indices))).astype(int)
        self.n_filt_rows = np.ceil(len(self.mag_indices) / self.n_filt_cols).astype(int)

        for folder in [self.save_folder, '{}/npy'.format(self.save_folder),
                       '{}/pdf'.format(self.save_folder)]:
            if not os.path.exists(folder):
                os.makedirs(folder)

    def __call__(self, a_cat_name, b_cat_name, a_cat_func=None, b_cat_func=None, tri_download=True,
                 overwrite_all_sightlines=False, make_plots=False, make_summary_plot=True):
        """
        Call function for the correction calculation process.

        Parameters
        ----------
        a_cat_name : string
            Name of the catalogue "a" filename, pre-generated or saved by
            ``a_cat_func``. Must accept one or two formats via Python string
            formatting (e.g. ``'a_string_{}'``) that represent ``chunk``, or
            ``ax1_mid`` and ``ax2_mid``, depending on ``coord_or_chunk``.
        b_cat_name : string
            Name of the catalogue "b" filename. Must accept the same string
            formatting as ``a_cat_name``.
        a_cat_func : callable, optional
            Function used to generate reduced catalogue table for catalogue "a".
            Must be given if ``pregenerate_cutouts`` is ``False``.
        b_cat_func : callable
            Function used to generate reduced catalogue table for catalogue "b".
            Must be given if ``pregenerate_cutouts`` is ``False``.
        tri_download : boolean, optional
            Flag determining if TRILEGAL simulations should be re-downloaded
            if they already exist on disk.
        overwrite_all_sightlines : boolean
            Flag for whether to create a totally fresh run of astrometric
            corrections, regardless of whether ``abc_array`` or ``m_sigs_array``
            are saved on disk. Defaults to ``False``.
        make_plots : boolean, optional
            Determines if intermediate figures are generated in the process
            of deriving astrometric corrections.
        make_summary_plot : boolean, optional
            If ``True`` then final summary plot is created, even if
            ``make_plots`` is ``False`` and intermediate plots are not created.
        """
        if a_cat_func is None and not self.pregenerate_cutouts:
            raise ValueError("a_cat_func must be given if pregenerate_cutouts is 'False'.")
        if b_cat_func is None and not self.pregenerate_cutouts:
            raise ValueError("b_cat_func must be given if pregenerate_cutouts is 'False'.")
        self.a_cat_func = a_cat_func
        self.b_cat_func = b_cat_func
        self.a_cat_name = a_cat_name
        self.b_cat_name = b_cat_name

        self.tri_download = tri_download

        self.make_plots = make_plots
        self.make_summary_plot = make_summary_plot

        self.make_ax_coords()

        if not self.pregenerate_cutouts:
            self.make_catalogue_cutouts()

        # Making coords/cutouts happens for all sightlines, and then we
        # loop through each individually:
        if self.coord_or_chunk == 'coord':
            zip_list = (self.ax1_mids, self.ax2_mids, self.ax1_mins, self.ax1_maxs,
                        self.ax2_mins, self.ax2_maxs)
        else:
            zip_list = (self.ax1_mids, self.ax2_mids, self.ax1_mins, self.ax1_maxs, self.ax2_mins,
                        self.ax2_maxs, self.chunks)

        if (overwrite_all_sightlines or
                not os.path.isfile('{}/npy/snr_mag_params.npy'.format(self.save_folder)) or
                not os.path.isfile('{}/npy/m_sigs_array.npy'.format(self.save_folder)) or
                not os.path.isfile('{}/npy/n_sigs_array.npy'.format(self.save_folder))):
            abc_array = open_memmap('{}/npy/snr_mag_params.npy'.format(self.save_folder), mode='w+',
                                    dtype=float,
                                    shape=(len(self.mag_indices), len(self.ax1_mids), 5))
            abc_array[:, :, :] = -1
            m_sigs = open_memmap('{}/npy/m_sigs_array.npy'.format(self.save_folder), mode='w+',
                                 dtype=float, shape=(len(self.ax1_mids),))
            m_sigs[:] = -1
            n_sigs = open_memmap('{}/npy/n_sigs_array.npy'.format(self.save_folder), mode='w+',
                                 dtype=float, shape=(len(self.ax1_mids),))
            n_sigs[:] = -1
        else:
            abc_array = open_memmap('{}/npy/snr_mag_params.npy'.format(self.save_folder), mode='r+')
            m_sigs = open_memmap('{}/npy/m_sigs_array.npy'.format(self.save_folder), mode='r+')
            n_sigs = open_memmap('{}/npy/n_sigs_array.npy'.format(self.save_folder), mode='r+')

        if self.make_summary_plot:
            self.gs = self.make_gridspec('12312', 2, 2, 0.8, 10)
            self.ax_b = plt.subplot(self.gs[0])

            self.ylims = [999, 0]

            self.cols = ['k', 'r', 'b', 'g', 'c', 'm', 'orange', 'brown', 'purple', 'grey', 'olive',
                         'cornflowerblue', 'deeppink', 'maroon', 'palevioletred', 'teal', 'crimson',
                         'chocolate', 'darksalmon', 'steelblue', 'slateblue', 'tan', 'yellowgreen',
                         'silver']

        for index_, list_of_things in enumerate(zip(*zip_list)):
            if not (m_sigs[index_] == -1 and n_sigs[index_] == -1):
                continue
            print('Running astrometry fits for sightline {}/{}...'.format(
                index_+1, len(self.ax1_mids)))

            if self.coord_or_chunk == 'coord':
                ax1_mid, ax2_mid, _, _, _, _ = list_of_things
                cat_args = (ax1_mid, ax2_mid)
                file_name = '{}_{}'.format(ax1_mid, ax2_mid)
            else:
                ax1_mid, ax2_mid, _, _, _, _, chunk = list_of_things
                cat_args = (chunk,)
                file_name = '{}'.format(chunk)
            self.list_of_things = list_of_things
            self.cat_args = cat_args
            self.file_name = file_name

            self.a = self.load_catalogue('a', self.cat_args)
            self.b = self.load_catalogue('b', self.cat_args)

            self.a_array, self.b_array, self.c_array = self.make_snr_model()
            abc_array[:, index_, 0] = self.a_array
            abc_array[:, index_, 1] = self.b_array
            abc_array[:, index_, 2] = self.c_array
            abc_array[:, index_, 3] = ax1_mid
            abc_array[:, index_, 4] = ax2_mid

            self.make_star_galaxy_counts()
            if self.make_plots:
                self.plot_star_galaxy_counts()
            self.calculate_local_densities_and_nearest_neighbours()
            self.simulate_aufs()
            self.create_auf_pdfs()
            self.fit_uncertainty()
            self.plot_fits_calculate_chi_sq()
            m_sig, n_sig = self.make_ma_fits_snr_h_plot()
            m_sigs[index_] = m_sig
            n_sigs[index_] = n_sig
            if self.make_summary_plot:
                plt.figure('12312')
                c = self.cols[index_ % len(self.cols)]
                self.ax_b.errorbar(self.avg_sig[~self.skip_flags, 0],
                                   self.fit_sigs[~self.skip_flags, 0],
                                   linestyle='None', c=c, marker='.')
                self.ylims[0] = min(self.ylims[0], np.amin(self.fit_sigs[:, 0]))
                self.ylims[1] = max(self.ylims[1], np.amax(self.fit_sigs[:, 0]))

        self.m_sigs, self.n_sigs = m_sigs, n_sigs
        if self.make_summary_plot:
            plt.figure('12312')
            x_array = np.linspace(0, self.ax_b.get_xlim()[1], 100)
            self.ax_b.plot(x_array, x_array, 'g:')
            self.ax_b.set_ylim(0.95 * self.ylims[0], 1.05 * self.ylims[1])
            if usetex:
                self.ax_b.set_xlabel(r'Input astrometric $\sigma$ / "')
                self.ax_b.set_ylabel(r'Fit astrometric $\sigma$ / "')
            else:
                self.ax_b.set_xlabel(r'Input astrometric sigma / "')
                self.ax_b.set_ylabel(r'Fit astrometric sigma / "')
            self.finalise_summary_plot()

    def make_ax_coords(self, check_b_only=False):
        """
        Derive the unique ax1-ax2 combinations used in fitting astrometry, and
        calculate corner coordinates based on the size of the box and its
        central coordinates.

        Parameters
        ==========
        check_b_only : boolean, optional
            Overloadable flag for cases where we can ignore catalogue 'a' and
            only need to check for catalogue 'b' existing if pregenerate_cutouts
            is True.
        """
        # If ax1 and ax2 are given as one-dimensional arrays, we need to
        # propagate those into two-dimensional grids first. Otherwise we
        # can skip this step.
        if self.ax_dimension == 1:
            self.ax1_mids_ = np.copy(self.ax1_mids)
            self.ax2_mids_ = np.copy(self.ax2_mids)
            self.ax1_mids, self.ax2_mids = np.meshgrid(self.ax1_mids_, self.ax2_mids_)
            self.ax1_mids, self.ax2_mids = self.ax1_mids.flatten(), self.ax2_mids.flatten()
            self.ax1_grid_length = len(self.ax1_mids_)
            self.ax2_grid_length = len(self.ax2_mids_)
        else:
            self.ax1_grid_length = np.ceil(np.sqrt(len(self.ax1_mids))).astype(int)
            self.ax2_grid_length = np.ceil(len(self.ax1_mids) / self.ax1_grid_length).astype(int)

        self.ax1_mins, self.ax1_maxs = np.empty_like(self.ax1_mids), np.empty_like(self.ax1_mids)
        self.ax2_mins, self.ax2_maxs = np.empty_like(self.ax1_mids), np.empty_like(self.ax1_mids)
        # Force constant box height, but allow longitude to float to make sure
        # that we get good area coverage as the cos-delta factor increases
        # towards the poles.
        if not self.pregenerate_cutouts:
            # If we don't force pre-generated sightlines then we can generate
            # corners based on requested size and height, and latitude.
            for i, (ax1_mid, ax2_mid) in enumerate(zip(self.ax1_mids, self.ax2_mids)):
                self.ax2_mins[i] = ax2_mid-self.cutout_height/2
                self.ax2_maxs[i] = ax2_mid+self.cutout_height/2

                lat_integral = (np.sin(np.radians(self.ax2_maxs[i])) -
                                np.sin(np.radians(self.ax2_mins[i]))) * 180/np.pi

                if 360 * lat_integral < self.cutout_area:
                    # If sufficiently high in latitude, we ought to be able to take
                    # a full latitudinal slice around the entire sphere.
                    delta_lon = 180
                else:
                    delta_lon = np.around(0.5 * self.cutout_area / lat_integral, decimals=1)
                # Handle wrap-around longitude maths naively by forcing 0/360 as the
                # minimum/maximum allowed limits of each box.
                # TODO: once the rest of AstrometricCorrections handles the wraparound
                # logic, relax this requirement.
                if ax1_mid - delta_lon < 0:
                    self.ax1_mins[i] = 0
                    self.ax1_maxs[i] = 2 * delta_lon
                elif ax1_mid + delta_lon > 360:
                    self.ax1_maxs[i] = 360
                    self.ax1_mins[i] = 360 - 2 * delta_lon
                else:
                    self.ax1_mins[i] = ax1_mid - delta_lon
                    self.ax1_maxs[i] = ax1_mid + delta_lon
        else:
            # If we've definitely already made all of the cutouts, the corners
            # are preset.
            if self.coord_or_chunk == 'coord':
                zip_list = (self.ax1_mids, self.ax2_mids)
            else:
                zip_list = (self.chunks,)

            for i, list_of_things in enumerate(zip(*zip_list)):
                if self.coord_or_chunk == 'coord':
                    ax1_mid, ax2_mid = list_of_things
                    cat_args = (ax1_mid, ax2_mid)
                else:
                    chunk, = list_of_things
                    cat_args = (chunk,)
                if not check_b_only:
                    if not os.path.isfile(self.a_cat_name.format(*cat_args)):
                        raise ValueError("If pregenerate_cutouts is 'True' all files must "
                                         "exist already, but {} does not.".format(
                                             self.a_cat_name.format(*cat_args)))
                if not os.path.isfile(self.b_cat_name.format(*cat_args)):
                    raise ValueError("If pregenerate_cutouts is 'True' all files must "
                                     "exist already, but {} does not.".format(
                                         self.b_cat_name.format(*cat_args)))
                # Check for both files, but assume they are the same size
                # for ax1_min et al. purposes.
                b = self.load_catalogue('b', cat_args)
                # Handle "minimum" longitude on the interval [-pi, +pi], such that
                # if data straddles the 0-360 boundary we don't return 0 and 360
                # for min and max values.
                self.ax1_mins[i], self.ax1_maxs[i] = min_max_lon(
                    b[:, self.pos_and_err_indices[1][0]])
                self.ax2_mins[i] = np.amin(b[:, self.pos_and_err_indices[1][1]])
                self.ax2_maxs[i] = np.amax(b[:, self.pos_and_err_indices[1][1]])

        np.save('{}/npy/ax1_mids.npy'.format(self.save_folder), self.ax1_mids)
        np.save('{}/npy/ax2_mids.npy'.format(self.save_folder), self.ax2_mids)

    def make_catalogue_cutouts(self):
        """
        Generate cutout catalogues for regions as defined by corner ax1-ax2
        coordinates.
        """
        if self.coord_or_chunk == 'coord':
            zip_list = (self.ax1_mids, self.ax2_mids, self.ax1_mins, self.ax1_maxs,
                        self.ax2_mins, self.ax2_maxs)
        else:
            zip_list = (self.chunks, self.ax1_mins, self.ax1_maxs, self.ax2_mins, self.ax2_maxs)
        for index_, list_of_things in enumerate(zip(*zip_list)):
            print('Creating catalogue cutouts... {}/{}'.format(index_+1, len(self.ax1_mids)),
                  end='\r')

            if self.coord_or_chunk == 'coord':
                ax1_mid, ax2_mid, ax1_min, ax1_max, ax2_min, ax2_max = list_of_things
            else:
                chunk, ax1_min, ax1_max, ax2_min, ax2_max = list_of_things

            if self.coord_or_chunk == 'coord':
                cat_args = (ax1_mid, ax2_mid)
            else:
                cat_args = (chunk,)
            if not os.path.isfile(self.a_cat_name.format(*cat_args)):
                self.a_cat_func(ax1_min, ax1_max, ax2_min, ax2_max, *cat_args)
            if not os.path.isfile(self.b_cat_name.format(*cat_args)):
                self.b_cat_func(ax1_min, ax1_max, ax2_min, ax2_max, *cat_args)

        print('')

    def make_snr_model(self):
        r"""
        Calculates the relationship between source magnitude and its typical
        signal-to-noise ratio, based on the ensemble scaling relations of a
        given region.

        We treat SNR as being related to flux S with
        :math:`\frac{S}{\sqrt{cS + b + (aS)^2}}`, with ``a``, ``b``, and ``c``
        setting the scaling with systematic uncertainty, background flux,
        and photon noise respectively. We then use
        :math:`S = 10^{-M/2.5}`, and hence typical magnitude-SNR
        relations are completely described by ``a``, ``b``, and ``c``.
        """

        print("Making SNR model...")
        a_array = np.ones(len(self.mag_indices), float) * np.nan
        b_array = np.ones(len(self.mag_indices), float) * np.nan
        c_array = np.ones(len(self.mag_indices), float) * np.nan
        if self.coord_or_chunk == 'coord':
            ax1_mid, ax2_mid, _, _, _, _ = self.list_of_things
        else:
            ax1_mid, ax2_mid, _, _, _, _, _ = self.list_of_things
        if self.make_plots:
            gs = self.make_gridspec('2', self.n_filt_cols, self.n_filt_rows, 0.8, 8)
        for j in range(len(self.mag_indices)):
            (res, s_bins, s_d_snr_med,
             s_d_snr_dmed, snr_med, snr_dmed) = self.fit_snr_model(j)

            a, b, c = 10**res.x
            a_array[j] = a
            b_array[j] = b
            c_array[j] = c

            if self.make_plots:
                q = ~np.isnan(s_d_snr_med)
                _x = np.linspace(s_bins[0], s_bins[-1], 10000)

                ax = plt.subplot(gs[j])
                ax.plot(_x, np.log10(np.sqrt(c * 10**_x + b + (a * 10**_x)**2)),
                        'r-', zorder=5)

                ax.errorbar((s_bins[:-1]+np.diff(s_bins)/2)[q], s_d_snr_med[q], fmt='k.',
                            yerr=s_d_snr_dmed[q], zorder=3)

                ax1_name = 'l' if self.coord_system == 'galactic' else 'RA'
                ax2_name = 'b' if self.coord_system == 'galactic' else 'Dec'
                ax.set_title('{} = {}, {} = {}, {}\na = {:.2e}, b = {:.2e}, c = {:.2e}'
                             .format(ax1_name, ax1_mid, ax2_name, ax2_mid, self.mag_names[j],
                                     a, b, c),
                             fontsize=28)

                if usetex:
                    ax.set_xlabel('log$_{10}$(S)')
                    ax.set_ylabel('log$_{10}$(S / SNR)')
                else:
                    ax.set_xlabel('log10(S)')
                    ax.set_ylabel('log10(S / SNR)')

                ax1 = ax.twinx()
                ax1.errorbar((s_bins[:-1]+np.diff(s_bins)/2)[q], snr_med[q], fmt='b.',
                             yerr=snr_dmed[q], zorder=3)
                ax1.plot(_x, np.log10(10**_x / np.sqrt(c * 10**_x + b + (a * 10**_x)**2)),
                         'r--', zorder=5)
                if usetex:
                    ax1.set_ylabel('log$_{10}$(SNR)', color='b')
                else:
                    ax1.set_ylabel('log10(SNR)', color='b')

            if self.make_plots:
                plt.tight_layout()
                plt.savefig('{}/pdf/s_vs_snr_{}.pdf'.format(self.save_folder, self.file_name))
                plt.close()

        return a_array, b_array, c_array

    def make_gridspec(self, name, y, x, ratio, z, **kwargs):
        """
        Convenience function to generate a matplotlib canvas and populate it
        with a gridspec grid.

        Parameters
        ----------
        name : string
            Unique identifier for the figure returned.
        y : int
            Number of rows in the resulting grid of axes.
        x : int
            Number of columns in the figure grid.
        ratio : float
            Aspect ratio of axis, with ``0.5`` being half as tall as it is
            wide.
        z : float
            Width of a single axis in real figure units, likely inches.

        Returns
        -------
        gs : ~`matplotlib.gridspec.GridSpec`
            The created gridspec instance.
        """
        plt.figure(name, figsize=(z*x, z*ratio*y))
        gs = gridspec.GridSpec(y, x, **kwargs)

        return gs

    def fit_snr_model(self, j):
        """
        Function to derive the scaling relation between magnitude via flux
        and SNR.

        Parameters
        ----------
        j : integer
            Index into ``self.mag_indices``.

        Returns
        -------
        res : ~scipy.optimize.OptimizeResult`
            Contains the results of the minimisation, as determined by `scipy`.
        s_bins : numpy.ndarray
            Log-flux bins used to construct ``s_d_snr_med`` and ``snr_med``.
        s_d_snr_med : numpy.ndarray
            Median ratio of flux to signal-to-noise ratio in each ``s_bins``
            bin.
        s_d_snr_dmed : numpy.ndarray
            Standard deviation of S/SNR in each ``s_bins`` bin.
        snr_med : numpy.ndarray
            Median SNR for each ``s_bins`` bin.
        snr_dmed : numpy.ndarray
            Standard deviation of SNR for all objects in a given ``s_bins`` bin.
        """
        def fit_snr_sqrt(p, x, y):
            """
            Minimisation function for determining magnitude-SNR scaling.

            Determined via log:math:`_{10}`(S/SNR), through a standard
            least-squares function and its derivatives with respect to
            ``a``, ``b``, and ``c``.

            Parameters
            ----------
            p : list
                Contains the current minimisation iteration values of
                ``a``, ``b``, and ``c``.
            x : numpy.ndarray
                Array of log-flux values.
            y : numpy.ndarray
                Array of log-"noise" values, log:math:`_{10}`(S/SNR).

            Returns
            -------
            numpy.ndarray
                The least-squares sum of the data-model residuals, and
                their derivative with respect to each component in ``p``.
            """
            a, b, c = p
            g = 10**c * 10**x + 10**b + (10**a * 10**x)**2
            f = np.log10(np.sqrt(g))
            dfdg = 1/(g * np.log(100))
            dgda = 2 * np.log(10) * 10**(2*x + 2*a)
            dgdb = np.log(10) * 10**b
            dgdc = 1 * np.log(10) * 10**(x + c)
            dfda = dgda * dfdg
            dfdb = dgdb * dfdg
            dfdc = dgdc * dfdg
            return np.sum((f - y)**2), np.array([np.sum(2 * (f - y) * i)
                                                 for i in [dfda, dfdb, dfdc]])

        s = 10**(-1/2.5 * self.b[:, self.mag_indices[j]])
        # Based on a naive dm = 2.5 log10((S+N)/S).
        snr = 1 / (10**(self.b[:, self.mag_unc_indices[j]] / 2.5) - 1)

        q = ~np.isnan(s) & ~np.isnan(snr) & (snr > 2)
        s, snr = s[q], snr[q]
        s_perc = np.percentile(s, [0.1, 99.9])
        s_d_snr_perc = np.percentile(s/snr, [0.1, 99.9])
        q = ((s > s_perc[0]) & (s < s_perc[1]) & (s/snr > s_d_snr_perc[0]) &
             (s/snr < s_d_snr_perc[1]))
        s, snr = s[q], snr[q]

        s_bins = np.linspace(np.amin(np.log10(s)), np.amax(np.log10(s)), 400)

        s_d_snr_med, _, _ = binned_statistic(np.log10(s), np.log10(s/snr),
                                             statistic='median', bins=s_bins)
        s_d_snr_dmed, _, _ = binned_statistic(np.log10(s), np.log10(s/snr),
                                              statistic='std', bins=s_bins)

        snr_med, _, _ = binned_statistic(np.log10(s), np.log10(snr),
                                         statistic='median', bins=s_bins)
        snr_dmed, _, _ = binned_statistic(np.log10(s), np.log10(snr), statistic='std', bins=s_bins)

        q = ~np.isnan(s_d_snr_med)

        # Find initial guesses, based on [S/SNR](S) = sqrt(c * S + b + (a * S)^2)
        # or SNR(S) = S / sqrt(c * S + b + (a * S)^2).
        # First, get the systematic inverse-SNR, in the regime where a dominates
        # and SNR(S) ~ 1/a, but remember that parameters are passed as log10(a).
        a_guess = np.log10(np.mean(1/10**snr_med[q][-15:]))
        # Can also directly get b, the background noise term, from S/SNR in the
        # limit that S is small, and S/SNR ~ sqrt(b).
        b_guess = np.log10(np.mean((10**s_d_snr_med[q][:15])**2))
        # Then take average of the difference between the a+b model and the full
        # model to get the photon-noise, sqrt(S)-scaling final parameter.
        half_model = 10**b_guess + (10**a_guess * 10**((s_bins[:-1]+np.diff(s_bins)/2)[q]))
        # If (S/SNR)^2 = c * S + b + (a * S)^2, then c ~ (S/SNR)^2 / half_model:
        c_guess = np.log10(np.mean((10**s_d_snr_med[q])**2 / half_model))

        res = minimize(fit_snr_sqrt, args=((s_bins[:-1]+np.diff(s_bins)/2)[q],
                       s_d_snr_med[q]), x0=[a_guess, b_guess, c_guess], jac=True,
                       method='SLSQP')

        return res, s_bins, s_d_snr_med, s_d_snr_dmed, snr_med, snr_dmed

    def make_star_galaxy_counts(self):
        """
        Generate differential source counts for each cutout region, simulating
        both stars and galaxies.
        """
        # TODO: load from CrossMatch in some fashion to avoid errors from
        # updating one but not the other.
        self.gal_cmau_array = np.empty((5, 2, 4), float)
        # See Wilson (2022, RNAAS, 6, 60) for the meanings of the variables c, m,
        # a, and u. For each of M*/phi*/alpha/P/Q, for blue+red galaxies, 2-4
        # variables are derived as a function of wavelength, or Q(P).
        self.gal_cmau_array[0, :, :] = [[-24.286513, 1.141760, 2.655846, np.nan],
                                        [-23.192520, 1.778718, 1.668292, np.nan]]
        self.gal_cmau_array[1, :, :] = [[0.001487, 2.918841, 0.000510, np.nan],
                                        [0.000560, 7.691261, 0.003330, -0.065565]]
        self.gal_cmau_array[2, :, :] = [[-1.257761, 0.021362, np.nan, np.nan],
                                        [-0.309077, -0.067411, np.nan, np.nan]]
        self.gal_cmau_array[3, :, :] = [[-0.302018, 0.034203, np.nan, np.nan],
                                        [-0.713062, 0.233366, np.nan, np.nan]]
        self.gal_cmau_array[4, :, :] = [[1.233627, -0.322347, np.nan, np.nan],
                                        [1.068926, -0.385984, np.nan, np.nan]]
        self.gal_alpha0 = [[2.079, 3.524, 1.917, 1.992, 2.536], [2.461, 2.358, 2.568, 2.268, 2.402]]
        self.gal_alpha1 = [[2.265, 3.862, 1.921, 1.685, 2.480], [2.410, 2.340, 2.200, 2.540, 2.464]]
        self.gal_alphaweight = [[3.47e+09, 3.31e+06, 2.13e+09, 1.64e+10, 1.01e+09],
                                [3.84e+09, 1.57e+06, 3.91e+08, 4.66e+10, 3.03e+07]]

        print('Creating simulated star+galaxy counts...')
        if self.coord_or_chunk == 'coord':
            ax1_mid, ax2_mid, ax1_min, ax1_max, ax2_min, ax2_max = self.list_of_things
        else:
            ax1_mid, ax2_mid, ax1_min, ax1_max, ax2_min, ax2_max, _ = self.list_of_things

        mag_ind = self.mag_indices[self.best_mag_index]
        b_mag_data = self.b[~np.isnan(self.b[:, mag_ind]), mag_ind]

        hist_mag, bins = np.histogram(b_mag_data, bins='auto')
        minmag = bins[0]
        # Ensure that we're only counting sources for normalisation purposes
        # down to the completeness turnover.
        # TODO: make the half-mag offset flexible, passing from CrossMatch and/or
        # directly into AstrometricCorrections.
        maxmag = bins[:-1][np.argmax(hist_mag)] - 0.5

        if (self.tri_download or not
                os.path.isfile('{}/{}_faint.dat'.format(
                    self.trifolder, self.triname.format(ax1_mid, ax2_mid)))):
            # Have to check for the existence of the folder as well as just
            # the lack of file. However, we can't guarantee that self.triname
            # doesn't contain folder sub-structure (e.g.
            # trifolder/ax1/ax2/file_faint.dat) and hence check generically.
            base_auf_folder = os.path.split('{}/{}_faint.dat'.format(
                self.trifolder, self.triname.format(ax1_mid, ax2_mid)))[0]
            if not os.path.exists(base_auf_folder):
                os.makedirs(base_auf_folder, exist_ok=True)

            rect_area = (ax1_max - (ax1_min)) * (
                np.sin(np.radians(ax2_max)) - np.sin(np.radians(ax2_min))) * 180/np.pi

            data_bright_dens = np.sum(~np.isnan(b_mag_data) & (b_mag_data <= maxmag)) / rect_area
            # TODO: un-hardcode min_bright_tri_number
            min_bright_tri_number = 1000
            min_area = min_bright_tri_number / data_bright_dens

            download_trilegal_simulation('.', self.trifilterset, ax1_mid, ax2_mid, self.magnum,
                                         self.coord_system, self.maglim_f, min_area, AV=1,
                                         sigma_AV=0, total_objs=self.tri_num_faint)
            os.system('mv {}/trilegal_auf_simulation.dat {}/{}_faint.dat'
                      .format(self.trifolder, self.trifolder,
                              self.triname.format(ax1_mid, ax2_mid)))

        ax1s = np.linspace(ax1_min, ax1_max, 7)
        ax2s = np.linspace(ax2_min, ax2_max, 7)
        avs = np.empty((len(ax1s), len(ax2s)), float)
        for j, ax1 in enumerate(ax1s):
            for k, ax2 in enumerate(ax2s):
                if self.coord_system == 'equatorial':
                    c = SkyCoord(ra=ax1, dec=ax2, unit='deg', frame='icrs')
                    l, b = c.galactic.l.degree, c.galactic.b.degree
                else:
                    l, b = ax1, ax2
                AV = get_AV_infinity(l, b, frame='galactic')[0]
                avs[j, k] = AV
        avs = avs.flatten()
        tri_hist, tri_mags, _, dtri_mags, tri_uncert, _ = make_tri_counts(
            self.trifolder, self.triname.format(ax1_mid, ax2_mid), self.trifiltname, self.dm,
            np.amin(b_mag_data), maxmag, al_av=self.gal_alav, av_grid=avs)

        gal_dNs = create_galaxy_counts(
            self.gal_cmau_array, tri_mags+dtri_mags/2, np.linspace(0, 4, 41),
            self.gal_wav_micron, self.gal_alpha0, self.gal_alpha1, self.gal_alphaweight,
            self.gal_ab_offset, self.gal_filtname, self.gal_alav*avs)

        log10y = np.log10(tri_hist + gal_dNs)
        new_uncert = np.sqrt(tri_uncert**2 + (0.05*gal_dNs)**2)
        dlog10y = 1/np.log(10) * new_uncert / (tri_hist + gal_dNs)

        mag_slice = (tri_mags >= minmag) & (tri_mags+dtri_mags <= maxmag)
        N_norm = np.sum(10**log10y[mag_slice] * dtri_mags[mag_slice])
        self.log10y, self.dlog10y = log10y, dlog10y
        self.tri_hist, self.tri_mags, self.dtri_mags = tri_hist, tri_mags, dtri_mags
        self.tri_uncert, self.gal_dNs = tri_uncert, gal_dNs
        self.minmag, self.maxmag, self.N_norm = minmag, maxmag, N_norm

    def plot_star_galaxy_counts(self):
        """
        Plotting routine to display data and model differential source counts,
        for verification purposes.
        """
        gs = self.make_gridspec('123123', 1, 1, 0.8, 15)
        print('Plotting data and model counts...')

        if self.coord_or_chunk == 'coord':
            ax1_mid, ax2_mid, ax1_min, ax1_max, ax2_min, ax2_max = self.list_of_things
        else:
            ax1_mid, ax2_mid, ax1_min, ax1_max, ax2_min, ax2_max, _ = self.list_of_things

        # Unit area is cos(t) dt dx for 0 <= t <= 90deg, 0 <= x <= 360 deg,
        # integrated between ax2_min < t < ax2_max, ax1_min < x < ax1_max, converted
        # to degrees.
        rect_area = (ax1_max - (ax1_min)) * (
            np.sin(np.radians(ax2_max)) - np.sin(np.radians(ax2_min))) * 180/np.pi

        mag_ind = self.mag_indices[self.best_mag_index]
        data_mags = self.b[~np.isnan(self.b[:, mag_ind]), mag_ind]
        # Correction to model is the ratio of data counts per unit area
        # to model source density.
        correction = np.sum((data_mags >= self.minmag) &
                            (data_mags <= self.maxmag)) / rect_area / self.N_norm

        ax = plt.subplot(gs[0])
        ax1_name = 'l' if self.coord_system == 'galactic' else 'RA'
        ax2_name = 'b' if self.coord_system == 'galactic' else 'Dec'
        ax.set_title('{} = {}, {} = {}'.format(ax1_name, ax1_mid, ax2_name, ax2_mid))
        ax.errorbar(self.tri_mags+self.dtri_mags/2, self.log10y + np.log10(correction),
                    yerr=self.dlog10y, c='k', marker='.', zorder=1, ls='None')

        data_hist, data_bins = np.histogram(data_mags, bins='auto')
        d_hc = np.where(data_hist > 3)[0]
        data_hist = data_hist[d_hc]
        data_dbins = np.diff(data_bins)[d_hc]
        data_bins = data_bins[d_hc]

        data_uncert = np.sqrt(data_hist) / data_dbins / rect_area
        data_hist = data_hist / data_dbins / rect_area
        data_loghist = np.log10(data_hist)
        data_dloghist = 1/np.log(10) * data_uncert / data_hist
        ax.errorbar(data_bins+data_dbins/2, data_loghist, yerr=data_dloghist, c='r',
                    marker='.', zorder=1, ls='None')

        lims = ax.get_ylim()
        ax.plot(self.tri_mags+self.dtri_mags/2, np.log10(self.tri_hist) + np.log10(correction),
                'b--')
        ax.plot(self.tri_mags+self.dtri_mags/2, np.log10(self.gal_dNs) + np.log10(correction), 'b:')
        ax.set_ylim(*lims)

        ax.set_xlabel('Magnitude')
        if usetex:
            ax.set_ylabel(r'log$_{10}\left(\mathrm{D}\ /\ \mathrm{mag}^{-1}\,'
                          r'\mathrm{deg}^{-2}\right)$')
        else:
            ax.set_ylabel(r'log10(D / mag^-1 deg^-2)')

        plt.figure('123123')
        plt.tight_layout()
        plt.savefig('{}/pdf/counts_comparison_{}.pdf'.format(self.save_folder, self.file_name))
        plt.close()

    def calculate_local_densities_and_nearest_neighbours(self):
        """
        Calculate local normalising catalogue densities and catalogue-catalogue
        nearest neighbour match pairings for each cutout region.
        """
        print('Creating local densities and nearest neighbour matches...')

        if self.coord_or_chunk == 'coord':
            ax1_mid, ax2_mid, ax1_min, ax1_max, ax2_min, ax2_max = self.list_of_things
        else:
            ax1_mid, ax2_mid, ax1_min, ax1_max, ax2_min, ax2_max, _ = self.list_of_things

        lon_slice = np.linspace(ax1_min, ax1_max, int(np.floor((ax1_max-ax1_min)*2 + 1)))
        lat_slice = np.linspace(ax2_min, ax2_max, int(np.floor((ax2_max-ax2_min)*2 + 1)))

        Narray = create_densities(
            ax1_mid, ax2_mid, self.b, self.minmag, self.maxmag, lon_slice, lat_slice, ax1_min,
            ax1_max, ax2_min, ax2_max, self.dens_search_radius, self.n_pool, self.save_folder,
            self.mag_indices[self.best_mag_index], self.pos_and_err_indices[1][0],
            self.pos_and_err_indices[1][1], self.coord_system)

        _, bmatch, dists = create_distances(
            self.a, self.b, ax1_mid, ax2_mid, self.nn_radius, self.save_folder,
            self.pos_and_err_indices[0][0], self.pos_and_err_indices[0][1],
            self.pos_and_err_indices[1][0], self.pos_and_err_indices[1][1], self.coord_system)

        # TODO: extend to 3-D search around N-m-sig to find as many good
        # enough bins as possible, instead of only keeping one N-sig bin
        # per magnitude?
        _h, _b = np.histogram(Narray[bmatch], bins='auto')
        modeN = (_b[:-1]+np.diff(_b)/2)[np.argmax(_h)]
        dN = 0.05*modeN

        self.Narray, self.dists, self.bmatch = Narray, dists, bmatch
        self.modeN, self.dN = modeN, dN

    def simulate_aufs(self):
        """
        Simulate unresolved blended contaminants for each magnitude-sightline
        combination, for both aperture photometry and background-dominated PSF
        algorithms.
        """
        print('Creating AUF simulations...')

        a = self.a_array[self.best_mag_index]
        b = self.b_array[self.best_mag_index]
        c = self.c_array[self.best_mag_index]
        B = 0.05
        # Self-consistent, non-zeropointed "flux", based on the relation
        # given in make_snr_model.
        flux = 10**(-1/2.5 * self.mag_array)
        snr = flux / np.sqrt(c * 10**flux + b + (a * 10**flux)**2)
        dm_max = _calculate_magnitude_offsets(
            self.modeN*np.ones_like(self.mag_array), self.mag_array, B, snr, self.tri_mags,
            self.log10y, self.dtri_mags, self.R, self.N_norm)

        seed = np.random.default_rng().choice(100000, size=(paf.get_random_seed_size(),
                                                            len(self.mag_array)))
        _, _, four_off_fw, _, _ = \
            paf.perturb_aufs(
                self.modeN*np.ones_like(self.mag_array), self.mag_array, self.r[:-1]+self.dr/2,
                self.dr, self.r, self.j0s.T, self.tri_mags+self.dtri_mags/2, self.dtri_mags,
                self.log10y, self.N_norm, (dm_max/self.dm).astype(int), self.dmcut, self.R,
                self.psfsig, self.numtrials, seed, self.dd_params, self.l_cut, 'fw')

        seed = np.random.default_rng().choice(100000, size=(paf.get_random_seed_size(),
                                                            len(self.mag_array)))
        _, _, four_off_ps, _, _ = \
            paf.perturb_aufs(
                self.modeN*np.ones_like(self.mag_array), self.mag_array, self.r[:-1]+self.dr/2,
                self.dr, self.r, self.j0s.T, self.tri_mags+self.dtri_mags/2, self.dtri_mags,
                self.log10y, self.N_norm, (dm_max/self.dm).astype(int), self.dmcut, self.R,
                self.psfsig, self.numtrials, seed, self.dd_params, self.l_cut, 'psf')

        self.four_off_fw, self.four_off_ps = four_off_fw, four_off_ps

    def create_auf_pdfs(self):
        """
        Using perturbation AUF simulations, generate probability density functions
        of perturbation distance for all cutout regions, as well as recording key
        statistics such as average magnitude or SNR.
        """
        print('Creating catalogue AUF probability densities...')
        b_matches = self.b[self.bmatch]

        skip_flags = np.zeros_like(self.mag_array, dtype=bool)

        pdfs, pdf_uncerts, q_pdfs, pdf_bins = [], [], [], []

        avg_sig = np.empty((len(self.mag_array), 3), float)
        avg_snr = np.empty((len(self.mag_array), 3), float)
        avg_mag = np.empty((len(self.mag_array), 3), float)

        mag_ind = self.mag_indices[self.best_mag_index]
        for i in range(len(self.mag_array)):
            mag_cut = ((b_matches[:, mag_ind] <= self.mag_array[i]+self.mag_slice[i]) &
                       (b_matches[:, mag_ind] >= self.mag_array[i]-self.mag_slice[i]))
            if np.sum(mag_cut) == 0:
                skip_flags[i] = 1
                pdfs.append([-1])
                pdf_uncerts.append([-1])
                q_pdfs.append([-1])
                pdf_bins.append([-1])
                continue
            sig = np.percentile(b_matches[mag_cut, self.pos_and_err_indices[1][2]], 50)
            sig_cut = ((b_matches[:, self.pos_and_err_indices[1][2]] <= sig+self.sig_slice[i]) &
                       (b_matches[:, self.pos_and_err_indices[1][2]] >= sig-self.sig_slice[i]))
            N_cut = (self.Narray[self.bmatch] >= self.modeN-self.dN) & (
                self.Narray[self.bmatch] <= self.modeN+self.dN)

            final_slice = sig_cut & mag_cut & N_cut & (self.dists <= 20*sig)
            final_dists = self.dists[final_slice]
            if len(final_dists) < 100:
                skip_flags[i] = 1
                pdfs.append([-1])
                pdf_uncerts.append([-1])
                q_pdfs.append([-1])
                pdf_bins.append([-1])
                continue

            bm = b_matches[final_slice]
            snr = 1 / (10**(bm[:, self.mag_unc_indices[self.best_mag_index]] / 2.5) - 1)
            avg_snr[i, 0] = np.median(snr)
            avg_snr[i, [1, 2]] = np.abs(np.percentile(snr, [16, 84]) - np.median(snr))
            avg_mag[i, 0] = np.median(bm[:, mag_ind])
            avg_mag[i, [1, 2]] = np.abs(np.percentile(bm[:, mag_ind], [16, 84]) -
                                        np.median(bm[:, mag_ind]))
            avg_sig[i, 0] = np.median(bm[:, self.pos_and_err_indices[1][2]])
            avg_sig[i, [1, 2]] = np.abs(np.percentile(bm[:, self.pos_and_err_indices[1][2]],
                                                      [16, 84]) -
                                        np.median(bm[:, self.pos_and_err_indices[1][2]]))

            h, bins = np.histogram(final_dists, bins='auto')
            num = np.sum(h)
            pdf = h / np.diff(bins) / num
            pdf_uncert = np.sqrt(h) / np.diff(bins) / num
            q_pdf = h > 3

            pdfs.append(pdf)
            pdf_uncerts.append(pdf_uncert)
            q_pdfs.append(q_pdf)
            pdf_bins.append(bins)

        self.avg_snr, self.avg_mag, self.avg_sig = avg_snr, avg_mag, avg_sig
        self.pdfs, self.pdf_uncerts = pdfs, pdf_uncerts
        self.q_pdfs, self.pdf_bins = q_pdfs, pdf_bins
        self.skip_flags = skip_flags

    def fit_uncertainty(self):
        """
        For each magnitude-sightline combination, fit for the empirical centroid
        uncertainty describing the distribution of match separations.
        """
        print('Creating joint H/sig fits...')

        fit_sigs = np.zeros((len(self.mag_array), 2), float)

        resses = [0]*len(self.mag_array)
        # Use the lower of the number of magnitues to run in parallel and the
        # maximum specified number of threads to call.
        n_pool = min(self.n_pool, len(self.mag_array))
        pool = multiprocessing.Pool(n_pool)
        counter = np.arange(0, len(self.mag_array))
        iter_group = zip(counter, itertools.repeat([
            self.pdfs, self.pdf_uncerts, self.rho, self.drho, self.r, self.dr, self.four_off_fw,
            self.four_off_ps, self.q_pdfs, self.pdf_bins, self.avg_sig[:, 0], self.avg_snr[:, 0],
            self.j0s, self.modeN/3600**2, self.a_array[self.best_mag_index]]))
        for s in pool.imap_unordered(self.fit_auf, iter_group,
                                     chunksize=max(1, len(self.mag_array) // n_pool)):
            res, i = s
            resses[i] = res
            if res == -1:
                self.skip_flags[i] = 1
            else:
                fit_sig, _ = resses[i].x
                cov = resses[i].hess_inv.todense()
                fit_sigs[i, 0] = fit_sig / 10
                fit_sigs[i, 1] = np.sqrt(cov[0, 0]) / 10

        pool.close()
        pool.join()

        self.resses, self.fit_sigs = resses, fit_sigs

    def fit_auf(self, iterable):
        """
        Determine the best-fit false match fraction and astrometric
        uncertainty for the set of matches at a particular brightness
        and sky coordinates.

        Parameters
        ----------
        iterable : list
            List of parameters passed through from ``multiprocessing``.
            Includes index, histogram of match separations and uncertainty,
            perturbation AUF PDF histogram and uncertainty, real- and
            fourier-space bins and spacing, quoted astrometric precision,
            average SNR, J0 evaluated at each ``r``-``rho`` combination for
            use in fourier transformations, average catalogue density, and
            systematic inverse-SNR.

        Returns
        -------
        res : ~`scipy.optimize.OptimizeResult`
            The result of the `scipy` minimisation routine, containing the
            minimum function value, parameter values at minimum, local
            derivatives, etc.
        i : integer
            Index into the magnitude in ``mag_array`` currently being fit for.
        """
        def blend_frac(p, y, dy, rho, drho, r, dr, four_combined, q, bins, j0s, N):
            """
            Minimisation routine for fitting false match fraction and
            empirical astrometric uncertainty for a given distribution of
            match separations.

            Parameters
            ----------
            p : list
                Astrometric uncertainty (scaled by a factor 10) and false
                match fraction of this iteration of the minimisation
                process.
            y : numpy.ndarray
                Array of data points, a PDF of the distribution of
                nearest neighbour separations between the two catalogues.
            dy : numpy.ndarray
                Uncertainty in the value of each element of ``y``.
            rho : numpy.ndarray
                Fourier-space bin edges, used in converting fourier-space
                convolutions back to real space.
            drho : numpy.ndarray
                Bin widths of each element of ``rho``.
            r : numpy.ndarray
                Real space bin edges for conversion from fourier
                space to real space at.
            dr : numpy.ndarray
                Widths of each bin in ``r``.
            four_combined : numpy.ndarray
                Weighted average of the fourier-space representations of the
                two versions of the perturbation AUF component.
            q : numpy.ndarray
                Array of boolean values, indicating whether the data bin
                should be used or not due to low-number statistics.
            bins : numpy.ndarray
                Bin edges of the separations that form the histogram
                of ``y``.
            j0s : numpy.ndarray
                Bessell function of first kind of zeroth order, evaluated
                at each ``r``-``rho`` combination, used in the inverse
                fourier transformation from fourier space to real space.
            N : float
                Density of the catalogue, used in evaluating the
                distribution of false positive nearest neighbours due to
                randomly placed non-matches.

            Returns
            -------
            log_like : numpy.ndarray
                The negative of the log-likelihood, half of the chi-squared
                of the data-model uncertainty-normalised residuals.
            """
            o, nnf = p

            o /= 10

            four_gauss = np.exp(-2 * np.pi**2 * (rho[:-1]+drho/2)**2 * o**2)

            convolve_hist = paf.fourier_transform(four_combined*four_gauss, rho[:-1]+drho/2,
                                                  drho, j0s)

            convolve_hist_dr = convolve_hist * np.pi * (r[1:]**2 - r[:-1]**2) / dr

            int_convhist = np.sum(convolve_hist_dr * dr)

            nn_model = 2 * np.pi * (r[:-1]+dr/2) * N * np.exp(-np.pi * (r[:-1]+dr/2)**2 * N) / (
                1 - np.exp(-np.pi * bins[1:][q][-1]**2 * N))

            tot_model = nnf * nn_model + (1 - nnf) * convolve_hist_dr

            modely, _, _ = binned_statistic(r[:-1]+dr/2, tot_model, bins=bins)

            int_modely = np.sum(modely[q] / int_convhist * np.diff(bins)[q])

            log_like = 0.5 * np.sum((y[q] - modely[q]/int_convhist / int_modely)**2 / dy[q]**2)

            return log_like

        i, (pdfs, pdf_uncerts, rho, drho, r, dr, f_fw, f_ps, q_pdfs, pdf_bins,
            sigs, snrs, j0s, N, _a) = iterable
        (pdf, pdf_uncert, q_pdf, pdf_bin, sig,
         snr) = pdfs[i], pdf_uncerts[i], q_pdfs[i], pdf_bins[i], sigs[i], snrs[i]
        if pdf[0] == -1:
            return -1, i

        H = 1 - np.sqrt(1 - min(1, _a**2 * snr**2))

        four_combined = H * f_fw[:, i] + (1 - H) * f_ps[:, i]

        res = minimize(blend_frac, x0=[sig*10, 0],
                       args=(pdf, pdf_uncert, rho, drho, r, dr,
                             four_combined, q_pdf, pdf_bin, j0s, N),
                       method='L-BFGS-B', options={'ftol': 1e-12},
                       bounds=[(0, None), (0, 1)])

        if np.all(res.x == [sig*10, 0]):
            res = -1

        return res, i

    def plot_fits_calculate_chi_sq(self):
        """
        Calculate chi-squared value and create verification plots showing the
        quality of the fits.
        """
        x2s = np.ones((len(self.mag_array), 2), float) * np.nan

        if self.make_plots:
            print('Creating individual AUF figures and calculating goodness-of-fits...')
        else:
            print('Calculating goodness-of-fits...')

        if self.make_plots:
            # Grid just big enough square to cover mag_array entries.
            gs1 = self.make_gridspec('34242b', self.n_mag_rows, self.n_mag_cols, 0.8, 15)
            ax1s = [plt.subplot(gs1[i]) for i in range(len(self.mag_array))]

        b_matches = self.b[self.bmatch]

        for i in range(len(self.mag_array)):
            if self.make_plots:
                ax = ax1s[i]
            if self.skip_flags[i]:
                continue
            pdf, pdf_uncert, q_pdf, pdf_bin = (
                self.pdfs[i], self.pdf_uncerts[i], self.q_pdfs[i], self.pdf_bins[i])
            if self.make_plots:
                ax.errorbar((pdf_bin[:-1]+np.diff(pdf_bin)/2)[q_pdf], pdf[q_pdf],
                            yerr=pdf_uncert[q_pdf], c='k', marker='.', zorder=1, ls='None')

            mag_ind = self.mag_indices[self.best_mag_index]
            pos_err_ind = self.pos_and_err_indices[1][2]
            mag_cut = ((b_matches[:, mag_ind] <= self.mag_array[i]+self.mag_slice[i]) &
                       (b_matches[:, mag_ind] >= self.mag_array[i]-self.mag_slice[i]))
            bsig = np.percentile(b_matches[mag_cut, pos_err_ind], 50)
            sig_cut = ((b_matches[:, pos_err_ind] <= bsig+self.sig_slice[i]) &
                       (b_matches[:, pos_err_ind] >= bsig-self.sig_slice[i]))
            N_cut = (self.Narray[self.bmatch] >= self.modeN-self.dN) & (
                self.Narray[self.bmatch] <= self.modeN+self.dN)
            final_slice = sig_cut & mag_cut & N_cut & (self.dists <= 20*bsig)
            bm = b_matches[final_slice]

            _N = np.percentile(self.Narray[self.bmatch][final_slice], 50)/3600**2
            fit_sig, nn_frac = self.resses[i].x
            fit_sig /= 10

            H = 1 - np.sqrt(1 - min(1,
                                    self.a_array[self.best_mag_index]**2 * self.avg_snr[i, 0]**2))

            if self.make_plots:
                ax = ax1s[i]
                if usetex:
                    labels_list = [r'H/$\sigma_\mathrm{fit}$', r'1/$\sigma_\mathrm{fit}$',
                                   r'0/$\sigma_\mathrm{fit}$', r'H/$\sigma_\mathrm{quoted}$',
                                   r'1/$\sigma_\mathrm{quoted}$', r'0/$\sigma_\mathrm{quoted}$']
                else:
                    labels_list = [r'H/sigma_fit', r'1/sigma_fit', r'0/sigma_fit',
                                   r'H/sigma_quoted', r'1/sigma_quoted', r'0/sigma_quoted']
            for j, (sig, _H, ls, lab) in enumerate(zip(
                    [fit_sig, fit_sig, fit_sig, bsig, bsig, bsig], [H, 1, 0, H, 1, 0],
                    ['r-', 'r-.', 'r:', 'k-', 'k-.', 'k:'], labels_list)):
                if not self.make_plots and j != 0:
                    continue
                four_gauss = np.exp(-2 * np.pi**2 * (self.rho[:-1]+self.drho/2)**2 * sig**2)

                four_hist = _H * self.four_off_fw[:, i] + (1 - _H) * self.four_off_ps[:, i]
                convolve_hist = paf.fourier_transform(
                    four_hist*four_gauss, self.rho[:-1]+self.drho/2, self.drho, self.j0s)

                convolve_hist_dr = convolve_hist * np.pi * (
                    self.r[1:]**2 - self.r[:-1]**2) / self.dr

                int_convhist = np.sum(convolve_hist_dr * self.dr)

                nn_model = (2 * np.pi * (self.r[:-1]+self.dr/2) *
                            _N * np.exp(-np.pi * (self.r[:-1]+self.dr/2)**2 * _N) / (
                            1 - np.exp(-np.pi * pdf_bin[1:][q_pdf][-1]**2 * _N)))

                tot_model = nn_frac * nn_model + (1 - nn_frac) * convolve_hist_dr

                modely, _, _ = binned_statistic(self.r[:-1]+self.dr/2, tot_model, bins=pdf_bin)

                int_modely = np.sum(modely[q_pdf] / int_convhist * np.diff(pdf_bin)[q_pdf])

                if j == 0:
                    x2s[i, 0] = np.sum((pdf[q_pdf] - modely[q_pdf] / int_convhist / int_modely)**2 /
                                       pdf_uncert[q_pdf]**2)
                    # Fit for sig and false match fraction.
                    x2s[i, 1] = np.sum(q_pdf) - 2

                if self.make_plots:
                    ax.plot((pdf_bin[:-1]+np.diff(pdf_bin)/2)[q_pdf],
                            modely[q_pdf]/int_convhist / int_modely, ls, label=lab)

            if self.make_plots:
                r_arr = np.linspace(0, pdf_bin[1:][q_pdf][-1], 1000)
                # NN distribution is 2 pi r N exp(-pi r^2 N),
                # with N converted from sq deg and r in arcseconds;
                # integral to R is 1 - exp(-pi R^2 N)
                nn_dist = 2 * np.pi * r_arr * _N * np.exp(-np.pi * r_arr**2 * _N) / (
                    1 - np.exp(-np.pi * r_arr[-1]**2 * _N))
                ax.plot(r_arr, nn_dist, c='g', ls='-')

                cov = self.resses[i].hess_inv.todense()
                if usetex:
                    ax.set_title(r'mag = {}, H = {:.2f}, $\sigma$ = {:.3f}$\pm${:.3f} ({:.3f})"; '
                                 r'$F$ = {:.2f}; SNR = {:.2f}$^{{+{:.2f}}}_{{-{:.2f}}}$; '
                                 r'N = {}'.format(self.mag_array[i], H, fit_sig,
                                                  np.sqrt(cov[0, 0])/10, bsig, nn_frac,
                                                  self.avg_snr[i, 0], self.avg_snr[i, 2],
                                                  self.avg_snr[i, 1], len(bm)), fontsize=22)
                else:
                    ax.set_title(r'mag = {}, H = {:.2f}, sigma = {:.3f}+/-{:.3f} ({:.3f})"; '
                                 r'F = {:.2f}; SNR = {:.2f}+{:.2f}/-{:.2f}; '
                                 r'N = {}'.format(self.mag_array[i], H, fit_sig,
                                                  np.sqrt(cov[0, 0])/10, bsig, nn_frac,
                                                  self.avg_snr[i, 0], self.avg_snr[i, 2],
                                                  self.avg_snr[i, 1], len(bm)), fontsize=22)
                ax.legend(fontsize=15)
                ax.set_xlabel('Radius / arcsecond')
                if usetex:
                    ax.set_ylabel('PDF / arcsecond$^{-1}$')
                else:
                    ax.set_ylabel('PDF / arcsecond^-1')
        if self.make_plots:
            plt.tight_layout()
            plt.savefig('{}/pdf/auf_fits_{}.pdf'.format(self.save_folder, self.file_name))
            plt.close()

        self.x2s = x2s

    def make_ma_fits_snr_h_plot(self):
        """
        Create best-fit uncertainty-quoted uncertainty fits for each sightline,
        and optionally make final summary plot showing the astrometric
        correction across all sightlines, detailing the quality of the fits,
        and showing the values for the equations detailing the astrometric
        corrections across all sightline values.
        """
        def quad_sig_fit(p, x, y, o):
            """
            Calculate the chi-squared of the residuals between derived
            astrometric precisions and the model relation, for all quoted
            astrometric uncertainties across all magnitude bins for a given
            sightline.

            Parameters
            ----------
            p : list
                List holding the scaling and systematic values describing
                the astrometric corrections.
            x : numpy.ndarray
                Array of quoted average astrometric precisions.
            y : numpy.ndarray
                Array of fit astrometric precisions.
            o : numpy.ndarray
                Uncertainty on data points ``y``.

            Returns
            -------
            numpy.ndarray
                The chi-squared value of the sum of data-model astrometric
                precision residuals, normalised by the data uncertainties.
            """
            m, n = p
            modely = np.sqrt((m * x)**2 + n**2)

            return np.sum((y - modely)**2 / o**2)

        if not self.make_plots:
            print("Creating sig-sig relations...")
        else:
            print("Creating sig-sig relations and figure...")
            gs_s = self.make_gridspec('123123b', 1, 1, 0.8, 15)

        if self.coord_or_chunk == 'coord':
            ax1_mid, ax2_mid, _, _, _, _ = self.list_of_things
        else:
            ax1_mid, ax2_mid, _, _, _, _, _ = self.list_of_things

        if self.make_plots:
            ax1 = plt.subplot(gs_s[0])
            ax1.errorbar(self.avg_sig[~self.skip_flags, 0], self.fit_sigs[~self.skip_flags, 0],
                         xerr=self.avg_sig[~self.skip_flags, 1:].T,
                         yerr=self.fit_sigs[~self.skip_flags, 1],
                         linestyle='None', c='k', marker='.')
            ax1.set_ylim(0.95*np.amin(self.fit_sigs[~self.skip_flags, 0]),
                         1.05*np.amax(self.fit_sigs[~self.skip_flags, 0]))

        res_sig = minimize(quad_sig_fit, x0=[1, 0], args=(self.avg_sig[~self.skip_flags, 0],
                           self.fit_sigs[~self.skip_flags, 0], self.fit_sigs[~self.skip_flags, 1]),
                           method='L-BFGS-B', options={'ftol': 1e-12},
                           bounds=[(0, None), (0, None)])

        m_sig, n_sig = res_sig.x

        if self.make_plots:
            x_array = np.linspace(0, ax1.get_xlim()[1], 100)
            ax1.plot(x_array, x_array, 'g:', label='y=x')
            ax1.plot(x_array, np.sqrt((m_sig*x_array)**2 + n_sig**2), 'r-.',
                     alpha=0.8, label='Fit ma')

            if usetex:
                ax1.set_xlabel(r'Input astrometric $\sigma$ / "')
                ax1.set_ylabel(r'Fit astrometric $\sigma$ / "')
            else:
                ax1.set_xlabel(r'Input astrometric sigma / "')
                ax1.set_ylabel(r'Fit astrometric sigma / "')
            ax1_name = 'l' if self.coord_system == 'galactic' else 'RA'
            ax2_name = 'b' if self.coord_system == 'galactic' else 'Dec'
            ax1.set_title('{} = {}, {} = {}\nm = {:.2f}, a = {:.2f}'.format(
                          ax1_name, ax1_mid, ax2_name, ax2_mid, m_sig, n_sig))
            ax1.legend()
            plt.figure('123123b')
            plt.tight_layout()
            plt.savefig('{}/pdf/sig_fit_comparisons_{}.pdf'.format(
                self.save_folder, self.file_name))
            plt.close()

        return m_sig, n_sig

    def finalise_summary_plot(self):
        """
        After running all of the sightlines' fits, generate a final
        summary plot of the sig-sig relations, quality of fits, and
        resulting "m" and "n" scaling parameters.
        """
        ax_d = plt.subplot(self.gs[1])
        q = ~np.isnan(self.x2s[:, 0])
        chi_sqs, dofs = self.x2s[:, 0][q].flatten(), self.x2s[:, 1][q].flatten()
        chi_sq_cdf = np.empty(np.sum(q), float)
        for i, (chi_sq, dof) in enumerate(zip(chi_sqs, dofs)):
            chi_sq_cdf[i] = chi2.cdf(chi_sq, dof)
        # Under the hypothesis all CDFs are "true" we should expect
        # the distribution of CDFs to be linear with fraction of the way
        # through the sorted list of CDFs -- that is, the CDF ranked 30%
        # in order should be ~0.3.
        q_sort = np.argsort(chi_sq_cdf)
        filter_log_nans = chi_sq_cdf[q_sort] < 1
        true_hypothesis_cdf_dist = (np.arange(1, len(chi_sq_cdf)+1, 1) - 0.5) / len(chi_sq_cdf)
        ax_d.plot(np.log10(1 - chi_sq_cdf[q_sort][filter_log_nans]),
                  true_hypothesis_cdf_dist[filter_log_nans], 'k.')
        ax_d.plot(np.log10(1 - true_hypothesis_cdf_dist), true_hypothesis_cdf_dist, 'r--')
        if usetex:
            ax_d.set_xlabel(r'$\log_{10}(1 - \mathrm{CDF})$')
        else:
            ax_d.set_xlabel(r'log10(1 - CDF)')
        ax_d.set_ylabel('Fraction')

        for i, (f, label) in enumerate(zip([self.m_sigs, self.n_sigs], ['m', 'n'])):
            ax = plt.subplot(self.gs[i+2])
            img = ax.scatter(self.ax1_mids, self.ax2_mids, c=f, cmap='viridis')
            c = plt.colorbar(img, ax=ax, use_gridspec=True)
            c.ax.set_ylabel(label)
            ax1_name = 'l' if self.coord_system == 'galactic' else 'RA'
            ax2_name = 'b' if self.coord_system == 'galactic' else 'Dec'
            ax.set_xlabel('{} / deg'.format(ax1_name))
            ax.set_ylabel('{} / deg'.format(ax2_name))

        plt.tight_layout()
        plt.savefig('{}/pdf/sig_h_stats.pdf'.format(self.save_folder))
        plt.close()

    def load_catalogue(self, cat_type, sub_cat_id):
        """
        Load specific sightline's catalogue, accounting for catalogue "a"
        vs "b", filetype, and method by which sightlines are divided.

        Parameters
        ----------
        cat_type : string, "a" or "b"
            Identifier for which of the two catalogues to load.
        sub_cat_id : list
            Contains the variables to format the name of the catalogue with,
            in the sense of ``string.format(x, y, ...)``. Should either contain
            ax1 and ax2, or chunk ID, depending on ``coord_or_chunk``.

        Returns
        -------
        x : numpy.ndarray
            The catalogue's dataset.
        """
        name = (self.a_cat_name.format(*sub_cat_id) if cat_type == 'a' else
                self.b_cat_name.format(*sub_cat_id))
        if self.npy_or_csv == 'npy':
            x = np.load(name)
        else:
            cols = self.a_cols if cat_type == 'a' else self.b_cols
            x = np.loadtxt(name, delimiter=',', usecols=cols)

        return x


def create_densities(ax1_mid, ax2_mid, b, minmag, maxmag, lon_slice, lat_slice, ax1_min, ax1_max,
                     ax2_min, ax2_max, search_radius, n_pool, save_folder, mag_ind, ax1_ind,
                     ax2_ind, coord_system):
    """
    Generate local normalising densities for all sources in catalogue "b".

    Parameters
    ----------
    ax1_mid : float
        Longitude of the center of the cutout region.
    ax2_mid : float
        Latitude of the middle of the cutout region.
    b : numpy.ndarray
        Catalogue of the sources for which astrometric corrections should be
        determined.
    minmag : float
        Bright limiting magnitude, fainter than which objects are used when
        determining the number of nearby sources for density purposes.
    maxmag : float
        Faintest magnitude within which to determine the density of catalogue
        ``b`` objects.
    lon_slice : numpy.ndarray
        Array of intermediate longitude values, between ``ax1_min`` and
        ``ax1_max``, used to load smaller sub-sets of the catalogue ``b``.
    lat_slice : numpy.ndarray
        An array of intermediate latitude values, between ``ax2_min`` and
        ``ax2_max``, used to load smaller sub-sets of the catalogue ``b``.
    ax1_min : float
        The minimum longitude of the box of the cutout region.
    ax1_max : float
        The maximum longitude of the box of the cutout region.
    ax2_min : float
        The minimum latitude of the box of the cutout region.
    ax2_max : float
        The maximum latitude of the box of the cutout region.
    search_radius : float
        Radius, in arcseconds, around which to calculate the density of objects.
        Smaller values will allow for more fluctuations and handle smaller scale
        variation better, but be subject to low-number statistics.
    n_pool : integer
        Number of parallel threads to run when calculating densities via
        ``multiprocessing``.
    save_folder : string
        Location on disk into which to save densities.
    mag_ind : integer
        Index in ``b`` where the magnitude being used is stored.
    ax1_ind : integer
        Index of ``b`` for the longitudinal coordinate column.
    ax2_ind : integer
        ``b`` index for the latitude data.
    coord_system : string
        Determines whether we are in equatorial or galactic coordinates for
        separation considerations.

    Returns
    -------
    Narray : numpy.ndarray
        The density of objects within ``search_radius`` arcseconds of each object
        in catalogue ``b``.

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

    if not os.path.isfile('{}/npy/narray_sky_{}_{}.npy'.format(save_folder, ax1_mid, ax2_mid)):
        cutmag = (b[:, mag_ind] >= minmag) & (b[:, mag_ind] <= maxmag)

        if coord_system == 'galactic':
            full_cat = SkyCoord(l=b[:, ax1_ind], b=b[:, ax2_ind], unit='deg', frame='galactic')
            mag_cut_cat = SkyCoord(l=b[cutmag, ax1_ind], b=b[cutmag, ax2_ind], unit='deg',
                                   frame='galactic')
        else:
            full_cat = SkyCoord(ra=b[:, ax1_ind], dec=b[:, ax2_ind], unit='deg', frame='icrs')
            mag_cut_cat = SkyCoord(ra=b[cutmag, ax1_ind], dec=b[cutmag, ax2_ind], unit='deg',
                                   frame='icrs')

        full_urepr = full_cat.data.represent_as(UnitSphericalRepresentation)
        full_ucoords = full_cat.realize_frame(full_urepr)

        mag_cut_urepr = mag_cut_cat.data.represent_as(UnitSphericalRepresentation)
        mag_cut_ucoords = mag_cut_cat.realize_frame(mag_cut_urepr)
        mag_cut_kdt = _get_cart_kdt(mag_cut_ucoords)

        r = (2 * np.sin(Angle(search_radius * u.arcsecond) / 2.0)).value
        overlap_number = np.empty(len(b), int)

        pool = multiprocessing.Pool(n_pool)
        counter = np.arange(0, len(b))
        iter_group = zip(counter, itertools.repeat([full_ucoords, mag_cut_kdt, r]))
        for stuff in pool.imap_unordered(ball_point_query, iter_group, chunksize=len(b)//n_pool):
            i, len_query = stuff
            overlap_number[i] = len_query

        pool.close()
        pool.join()

        area = paf.get_circle_area_overlap(b[:, ax1_ind], b[:, ax2_ind], search_radius/3600,
                                           ax1_min, ax1_max, ax2_min, ax2_max)

        Narray = overlap_number / area

        np.save('{}/npy/narray_sky_{}_{}.npy'.format(save_folder, ax1_mid, ax2_mid), Narray)
    Narray = np.load('{}/npy/narray_sky_{}_{}.npy'.format(save_folder, ax1_mid, ax2_mid))

    return Narray


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
        containing all potential neighbours, and the Cartesian angle
        representing the maximum on-sky separation.

    Returns
    -------
    i : integer
        The index of the object whose neighbour count was calculated.
    integer
        The number of neighbours in ``mag_cut_kdt`` within ``r`` of
        ``full_ucoords[i]``.
    """
    i, (full_ucoords, mag_cut_kdt, r) = iterable
    # query_ball_point returns the neighbours of x (full_ucoords) around self
    # (mag_cut_kdt) within r.
    kdt_query = mag_cut_kdt.query_ball_point(full_ucoords[i].cartesian.xyz, r)
    return i, len(kdt_query)


def create_distances(a, b, ax1_mid, ax2_mid, nn_radius, save_folder, a_ax1_ind, a_ax2_ind,
                     b_ax1_ind, b_ax2_ind, coord_system):
    """
    Calculate nearest neighbour matches between two catalogues.

    Parameters
    ----------
    a : numpy.ndarray
        Array containing catalogue "a"'s sources. Must have astrometry as its
        first and second columns.
    b : numpy.ndarray
        Catalogue "b"'s object array. Longitude and latitude must be its first
        two axes respectively.
    ax1_mid : float
        Center of the cutout region in longitude.
    ax2_mid : float
        Latitude of the cutout region's central coordinate.
    nn_radius : float
        Maximum match radius within which to consider potential counterpart
        assignments, in arcseconds.
    save_folder : string
        Location on disk where matches should be saved.
    a_ax1_ind : integer
        Index into ``a`` of the longitude data.
    a_ax2_ind : integer
        ``a`` index for latitude column.
    b_ax1_ind : integer
        Longitude index in the ``b`` dataset.
    b_ax2_ind : integer
        Index into ``b`` that holds the latitude data.
    coord_system : string
        Sets coordinate system to equatorial or galactic.

    Returns
    -------
    amatch : numpy.ndarray
        Indices in catalogue ``a`` that have a match to a source in catalogue
        ``b`` within ``nn_radius`` that the opposite object has as its nearest
        neighbour as well. ``amatch[i]`` pairs with ``bmatch[i]`` for all ``i``.
    bmatch : numpy.ndarray
        The indices into catalogue ``b`` that have a pair in catalogue ``a``,
        in an elementwise match between ``amatch`` and ``bmatch``.
    dists : numpy.ndarray
        The separations between each nearest neighbour match, in arcseconds.
    """
    if (not os.path.isfile('{}/npy/a_matchind_{}_{}.npy'.format(
            save_folder, ax1_mid, ax2_mid)) or
            not os.path.isfile('{}/npy/b_matchind_{}_{}.npy'.format(
                save_folder, ax1_mid, ax2_mid)) or
            not os.path.isfile('{}/npy/ab_dists_{}_{}.npy'.format(save_folder, ax1_mid, ax2_mid))):
        if coord_system == 'galactic':
            ac = SkyCoord(l=a[:, a_ax1_ind], b=a[:, a_ax2_ind], unit='deg', frame='galactic')
            bc = SkyCoord(l=b[:, b_ax1_ind], b=b[:, b_ax2_ind], unit='deg', frame='galactic')
        else:
            ac = SkyCoord(ra=a[:, a_ax1_ind], dec=a[:, a_ax2_ind], unit='deg', frame='icrs')
            bc = SkyCoord(ra=b[:, b_ax1_ind], dec=b[:, b_ax2_ind], unit='deg', frame='icrs')
        amatchind, adists, _ = match_coordinates_sky(ac, bc)
        bmatchind, bdists, _ = match_coordinates_sky(bc, ac)
        # Since match_coordinates_sky doesn't set a maximum cutoff, we manually
        # ensure only matches within nn_radius return by setting larger matches
        # to unique negative indices, so that found_match_slice and
        # found_match_slice2 fail for those later.
        q = adists.arcsecond > nn_radius
        amatchind[q] = -1
        q = bdists.arcsecond > nn_radius
        bmatchind[q] = -2
        adists = adists.arcsecond

        _ainds = np.arange(0, len(a), dtype=int)
        found_match_slice = amatchind >= 0
        found_match_slice2 = _ainds[found_match_slice] == bmatchind[amatchind[found_match_slice]]

        # The indices should swap here. amatchind is the catalogue b indices
        # for each catalogue a object, but we really just care about which
        # catalogue b objects are matched.
        bmatch = amatchind[found_match_slice][found_match_slice2]
        amatch = _ainds[found_match_slice][found_match_slice2]
        dists = adists[found_match_slice][found_match_slice2]

        np.save('{}/npy/a_matchind_{}_{}.npy'.format(save_folder, ax1_mid, ax2_mid), amatch)
        np.save('{}/npy/b_matchind_{}_{}.npy'.format(save_folder, ax1_mid, ax2_mid), bmatch)
        np.save('{}/npy/ab_dists_{}_{}.npy'.format(save_folder, ax1_mid, ax2_mid), dists)
    amatch = np.load('{}/npy/a_matchind_{}_{}.npy'.format(save_folder, ax1_mid, ax2_mid))
    bmatch = np.load('{}/npy/b_matchind_{}_{}.npy'.format(save_folder, ax1_mid, ax2_mid))
    dists = np.load('{}/npy/ab_dists_{}_{}.npy'.format(save_folder, ax1_mid, ax2_mid))

    return amatch, bmatch, dists


class SNRMagnitudeRelationship(AstrometricCorrections):
    """
    Class to derive signal-to-noise ratio-magnitude relationships for sources
    in photometric catalogues.
    """
    def __init__(self, save_folder, ax1_mids, ax2_mids, ax_dimension, npy_or_csv, coord_or_chunk,
                 pos_and_err_indices, mag_indices, mag_unc_indices, mag_names, coord_system,
                 chunks=None):
        """
        Initialisation of AstrometricCorrections, accepting inputs required for
        the running of the optimisation and parameterisation of astrometry of
        a photometric catalogue, benchmarked against a catalogue of much higher
        astrometric resolution and precision, such as Gaia or the Hubble Source
        Catalog.

        Parameters
        ----------
        save_folder : string
            Absolute or relative filepath of folder into which to store
            temporary and generated outputs from the fitting process.
        ax1_mids : numpy.ndarray
            Array of longitudes (e.g. RA or l) used to center regions used to
            determine astrometric corrections across the sky. Depending on
            ``ax_correction``, either the unique values that with ``ax2_mids``
            form a rectangle, or a unique ax-ax combination with a corresponding
            ``ax2_mid``.
        ax2_mids : numpy.ndarray
            Array of latitudes (Dec/b) defining regions for calculating
            astrometric corrections. Either unique rectangle values to combine
            with ``ax1_mids`` or unique ``ax1_mids``-``ax2_mids`` pairs, one
            per entry.
        ax_dimension : integer, either ``1`` or ``2``
            If ``1`` then ``ax1_mids`` and ``ax2_mids`` form unique sides of a
            rectangle when combined in a grid, or if ``2`` each
            ``ax1_mids``-``ax2_mids`` combination is a unique ax-ax pairing used
            as given.
        npy_or_csv : string, either ``npy`` or ``csv``
            Indicator as to whether the small chunks of sky to be loaded for
            each sightline's evaluation are in binary ``numpy`` format or saved
            to disk as a comma-separated values file.
        coord_or_chunk : string, either ``coord`` or ``chunk``
            String indicating whether intermediate files should be saved with
            filenames that are unique by two coordinates (l/b or RA/Dec) or
            some kind of singular "chunk" number. Output filenames would then
            need to follow ``'file_{}{}'`` or ``'file_{}'`` formatting
            respectively.
        pos_and_err_indices : list or numpy.ndarray of integers
            In order, the indices within the catalogue, either a .npy or .csv
            file, of the longitudinal (e.g. RA or l), latitudinal (Dec or b),
            and *singular*, circular astrometric precision array. Coordinates
            should be in degrees while precision should be in the same units as
            ``sig_slice`` and those of the nearest-neighbour distances, likely
            arcseconds. For example, ``[0, 1, 2]`` or ``[6, 3, 0]`` where the
            first example has its coordinates in the first three columns in
            RA/Dec/Err order, while the second has its coordinates in a more
            random order.
        mag_indices : list or numpy.ndarray
            In appropriate order, as expected by e.g. `~macauff.CrossMatch` inputs
            and `~macauff.make_perturb_aufs`, list the indexes of each magnitude
            column within either the ``.npy`` or ``.csv`` file loaded for each
            sub-catalogue sightline. These should be zero-indexed.
        mag_unc_indices : list or numpy.ndarray
            For each ``mag_indices`` entry, the corresponding magnitude
            uncertainty index in the catalogue.
        mag_names : list or numpy.ndarray of strings
            Names of each ``mag_indices`` magnitude.
        coord_system : string, "equatorial" or "galactic"
            Identifier of which coordinate system the data are in. Both datasets
            must be in the same system, which can either be RA/Dec (equatorial)
            or l/b (galactic) coordinates.
        chunks = list or numpy.ndarray of strings, optional
            List of IDs for each unique set of data if ``coord_or_chunk`` is
            ``chunk``. In this case, ``ax_dimension`` must be ``2`` and each
            ``chunk`` must correspond to its ``ax1_mids``-``ax2_mids`` coordinate.
        """
        if not (ax_dimension == 1 or ax_dimension == 2):
            raise ValueError("ax_dimension must either be '1' or '2'.")
        if npy_or_csv != "npy" and npy_or_csv != "csv":
            raise ValueError("npy_or_csv must either be 'npy' or 'csv'.")
        if coord_or_chunk != "coord" and coord_or_chunk != "chunk":
            raise ValueError("coord_or_chunk must either be 'coord' or 'chunk'.")
        if coord_or_chunk == "chunk" and chunks is None:
            raise ValueError("chunks must be provided if coord_or_chunk is 'chunk'.")
        if coord_or_chunk == "chunk" and ax_dimension == 1:
            raise ValueError("ax_dimension must be 2, and ax1-ax2 pairings provided for each chunk "
                             "in chunks if coord_or_chunk is 'chunk'.")
        if coord_or_chunk == "chunk" and (len(ax1_mids) != len(chunks) or
                                          len(ax2_mids) != len(chunks)):
            raise ValueError("ax1_mids, ax2_mids, and chunks must all be the same length if "
                             "coord_or_chunk is 'chunk'.")
        if not (coord_system == "equatorial" or coord_system == "galactic"):
            raise ValueError("coord_system must either be 'equatorial' or 'galactic'.")

        self.save_folder = save_folder

        self.ax1_mids = ax1_mids
        self.ax2_mids = ax2_mids
        self.ax_dimension = ax_dimension

        self.npy_or_csv = npy_or_csv
        self.coord_or_chunk = coord_or_chunk
        self.chunks = chunks

        self.coord_system = coord_system

        if npy_or_csv == 'npy':
            self.pos_and_err_indices = [None, pos_and_err_indices]
            self.mag_indices = mag_indices
            self.mag_unc_indices = mag_unc_indices
            self.mag_names = mag_names
        else:
            # np.genfromtxt will load in pos_and_err or pos_and_err, mag_ind,
            # mag_unc_ind order for the two catalogues. Each will effectively
            # change its ordering, since we then load [0] for pos_and_err[0][0],
            # etc. for all options. These need saving for np.genfromtxt but
            # also for obtaining the correct column in the resulting sub-set of
            # the loaded csv file.
            self.b_cols = np.concatenate((pos_and_err_indices, mag_indices, mag_unc_indices))

            self.pos_and_err_indices = [
                None, [np.argmin(np.abs(q - self.b_cols)) for q in pos_and_err_indices]]

            self.mag_indices = [np.argmin(np.abs(q - self.b_cols)) for q in mag_indices]
            self.mag_unc_indices = [np.argmin(np.abs(q - self.b_cols)) for q in mag_unc_indices]

            self.mag_names = mag_names

        self.n_filt_cols = np.ceil(np.sqrt(len(self.mag_indices))).astype(int)
        self.n_filt_rows = np.ceil(len(self.mag_indices) / self.n_filt_cols).astype(int)

        for folder in [self.save_folder, '{}/npy'.format(self.save_folder),
                       '{}/pdf'.format(self.save_folder)]:
            if not os.path.exists(folder):
                os.makedirs(folder)

    def __call__(self, b_cat_name, overwrite_all_sightlines=False, make_plots=False):
        """
        Call function for the correction calculation process.

        Parameters
        ----------
        a_cat_name : string
            Name of the catalogue "b" filename, pre-generated. Must accept one
            or two formats via Python string formatting (e.g. ``'a_string_{}'``)
            that represent ``chunk``, or ``ax1_mid`` and ``ax2_mid``, depending
            on ``coord_or_chunk``.
        overwrite_all_sightlines : boolean
            Flag for whether to create a totally fresh run of astrometric
            corrections, regardless of whether ``abc_array`` is saved on disk.
            Defaults to ``False``.
        make_plots : boolean, optional
            Determines if intermediate figures are generated in the process
            of deriving astrometric corrections.
        """
        self.b_cat_name = b_cat_name
        self.make_plots = make_plots

        # Force pregenerate_cutouts for super purposes.
        self.pregenerate_cutouts = True
        self.make_ax_coords(check_b_only=True)

        # Making coords/cutouts happens for all sightlines, and then we
        # loop through each individually:
        if self.coord_or_chunk == 'coord':
            zip_list = (self.ax1_mids, self.ax2_mids, self.ax1_mins, self.ax1_maxs,
                        self.ax2_mins, self.ax2_maxs)
        else:
            zip_list = (self.ax1_mids, self.ax2_mids, self.ax1_mins, self.ax1_maxs, self.ax2_mins,
                        self.ax2_maxs, self.chunks)

        if (overwrite_all_sightlines or
                not os.path.isfile('{}/npy/snr_mag_params.npy'.format(self.save_folder))):
            abc_array = open_memmap('{}/npy/snr_mag_params.npy'.format(self.save_folder), mode='w+',
                                    dtype=float,
                                    shape=(len(self.mag_indices), len(self.ax1_mids), 5))
            abc_array[:, :, :] = -1
        else:
            abc_array = open_memmap('{}/npy/snr_mag_params.npy'.format(self.save_folder), mode='r+')

        for index_, list_of_things in enumerate(zip(*zip_list)):
            if not np.all(abc_array[:, index_, :] == [-1, -1, -1, -1, -1]):
                continue
            print('Running SNR-mag fits for sightline {}/{}...'.format(
                index_+1, len(self.ax1_mids)))

            if self.coord_or_chunk == 'coord':
                ax1_mid, ax2_mid, _, _, _, _ = list_of_things
                cat_args = (ax1_mid, ax2_mid)
                file_name = '{}_{}'.format(ax1_mid, ax2_mid)
            else:
                ax1_mid, ax2_mid, _, _, _, _, chunk = list_of_things
                cat_args = (chunk,)
                file_name = '{}'.format(chunk)
            self.list_of_things = list_of_things
            self.cat_args = cat_args
            self.file_name = file_name

            self.b = self.load_catalogue('b', self.cat_args)

            self.a_array, self.b_array, self.c_array = self.make_snr_model()
            abc_array[:, index_, 0] = self.a_array
            abc_array[:, index_, 1] = self.b_array
            abc_array[:, index_, 2] = self.c_array
            abc_array[:, index_, 3] = ax1_mid
            abc_array[:, index_, 4] = ax2_mid
