# Licensed under a 3-clause BSD style license - see LICENSE
'''
Module for calculating corrections to astrometric uncertainties of photometric catalogues, by
fitting their AUFs and centroid uncertainties across ensembles of matches between well-understood
catalogue and one for which precisions are less well known.
'''

import numpy as np
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
from .misc_functions_fortran import misc_functions_fortran as mff
from .perturbation_auf_fortran import perturbation_auf_fortran as paf


__all__ = ['AstrometricCorrections']


class AstrometricCorrections:
    """
    Class to calculate any potential corrections to quoted astrometric
    precisions in photometric catalogues, based on reliable cross-matching
    to a well-understood second dataset.
    """
    def __init__(self, psf_fwhm, numtrials, nn_radius, dens_search_radius, save_folder, trifolder,
                 triname, maglim_b, maglim_f, magnum, trifilterset, trifiltname,
                 gal_wav_micron, gal_ab_offset, gal_filtname, gal_alav, bright_mag, dm, dd_params,
                 l_cut, lmids, bmids, lb_dimension, cutout_area, cutout_height, mag_array,
                 mag_slice, sig_slice, n_pool, npy_or_csv, coord_or_chunk, pos_and_err_indices,
                 mag_indices, mag_unc_indices, mag_names, best_mag_index, single_sided_auf=True,
                 chunks=None):
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
            Name to give TRILEGAL simulations when downloaded. Will have
            suffix appended to the end, for unique l-b sightline combination
            downloads.
        maglim_b : float
            Magnitude in the ``magnum`` filter down to which sources should be
            drawn for the "bright" sample.
        maglim_f : float
            Magnitude in the ``magnum`` filter down to which sources should be
            drawn for the "faint" sample.
        magnum : float
            Zero-indexed column number of the chosen filter limiting magnitude.
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
        bright_mag : float
            Limiting magnitude down to which to calculate normalising density
            of sources.
        dm : float
            Bin spacing for magnitude histograms of TRILEGAL simulations.
        dd_params : numpy.ndarray
            Array, of shape ``(5, X, 2)``, containing the parameterisation of
            the skew-normal used to construct background-dominated PSF
            perturbations.
        l_cut : numpy.ndarray
            Array of shape ``(3,)`` containing the cuts between the different
            regimes of background-dominated PSF perturbation.
        lmids : numpy.ndarray
            Array of Galactic longitudes used to center regions used to determine
            astrometric corrections across the sky. Depending on
            ``lb_correction``, either the unique values that with ``bmids`` form
            a rectangle, or a unique l-b combination with a corresponding
            ``bmid``.
        bmids : numpy.ndarray
            Array of Galactic latitudes defining regions for calculating
            astrometric corrections. Either unique rectangle values to combine
            with ``lmids`` or unique ``lmids``-``bmids`` pairs, one per entry.
        lb_dimension : integer, either ``1`` or ``2``
            If ``1`` then ``lmids`` and ``bmids`` form unique sides of a
            rectangle when combined in a grid, or if ``2`` each
            ``lmids``-``bmids`` combination is a unique l-b pairing used
            as given.
        cutout_area : float
            The size, in square degrees, of the regions used to simulate
            AUFs and determine astrometric corrections.
        cutout_height : float
            The latitudinal height of the rectangular regions used in
            calculating astrometric corrections.
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
        single_sided_auf : boolean, optional
            Flag indicating whether the AUF of catalogue "a" can be ignored
            when considering match statistics, or if astrometric corrections
            are being constructed from matches to a catalogue that also suffers
            significant non-noise-based astrometric uncertainty.
        chunks = list or numpy.ndarray of strings, optional
            List of IDs for each unique set of data if ``coord_or_chunk`` is
            ``chunk``. In this case, ``lb_dimension`` must be ``2`` and each
            ``chunk`` must correspond to its ``lmids``-``bmids`` coordinate.
        """
        if single_sided_auf is not True:
            raise ValueError("single_sided_auf must be True.")
        if not (lb_dimension == 1 or lb_dimension == 2):
            raise ValueError("lb_dimension must either be '1' or '2'.")
        if npy_or_csv != "npy" and npy_or_csv != "csv":
            raise ValueError("npy_or_csv must either be 'npy' or 'csv'.")
        if coord_or_chunk != "coord" and coord_or_chunk != "chunk":
            raise ValueError("coord_or_chunk must either be 'coord' or 'chunk'.")
        if coord_or_chunk == "chunk" and chunks is None:
            raise ValueError("chunks must be provided if coord_or_chunk is 'chunk'.")
        if coord_or_chunk == "chunk" and lb_dimension == 1:
            raise ValueError("lb_dimension must be 2, and l-b pairings provided for each chunk "
                             "in chunks if coord_or_chunk is 'chunk'.")
        if coord_or_chunk == "chunk" and (len(lmids) != len(chunks) or len(bmids) != len(chunks)):
            raise ValueError("lmids, bmids, and chunks must all be the same length if "
                             "coord_or_chunk is 'chunk'.")
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
        self.triname = triname + '_{}_{}'
        self.maglim_b = maglim_b
        self.maglim_f = maglim_f
        self.magnum = magnum
        self.trifilterset = trifilterset
        self.trifiltname = trifiltname
        self.gal_wav_micron = gal_wav_micron
        self.gal_ab_offset = gal_ab_offset
        self.gal_filtname = gal_filtname
        self.gal_alav = gal_alav

        self.bright_mag = bright_mag
        self.dm = dm

        self.dd_params = dd_params
        self.l_cut = l_cut

        self.lmids = lmids
        self.bmids = bmids
        self.lb_dimension = lb_dimension
        self.cutout_area = cutout_area
        self.cutout_height = cutout_height

        self.mag_array = np.array(mag_array)
        self.mag_slice = np.array(mag_slice)
        self.sig_slice = np.array(sig_slice)

        self.npy_or_csv = npy_or_csv
        self.coord_or_chunk = coord_or_chunk
        self.chunks = chunks

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

        for folder in [self.save_folder, '{}/npy'.format(self.save_folder),
                       '{}/pdf'.format(self.save_folder)]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        self.make_lb()

    def make_lb(self):
        """
        Derive the unique l-b combinations used in fitting astrometry, and
        calculate corner coordinates based on the size of the box and its
        central coordinates.
        """
        # If l and b are given as one-dimensional arrays, we need to propagate
        # those into two-dimensional grids first. Otherwise we can skip this
        # step.
        if self.lb_dimension == 1:
            self.lmids_ = np.copy(self.lmids)
            self.bmids_ = np.copy(self.bmids)
            self.lmids, self.bmids = np.meshgrid(self.lmids_, self.bmids_)
            self.lmids, self.bmids = self.lmids.flatten(), self.bmids.flatten()
            self.l_grid_length = len(self.lmids_)
            self.b_grid_length = len(self.bmids_)
        else:
            self.l_grid_length = np.ceil(np.sqrt(len(self.lmids))).astype(int)
            self.b_grid_length = np.ceil(len(self.lmids) / self.l_grid_length).astype(int)

        # Force constant box height, but allow longitude to float to make sure
        # that we get good area coverage as the cos-delta factor increases
        # towards the poles.

        self.lmins, self.lmaxs = np.empty_like(self.lmids), np.empty_like(self.lmids)
        self.bmins, self.bmaxs = np.empty_like(self.lmids), np.empty_like(self.lmids)
        for i, (lmid, bmid) in enumerate(zip(self.lmids, self.bmids)):
            self.bmins[i], self.bmaxs[i] = bmid-self.cutout_height/2, bmid+self.cutout_height/2

            lat_integral = (np.sin(np.radians(self.bmaxs[i])) -
                            np.sin(np.radians(self.bmins[i]))) * 180/np.pi

            if 360 * lat_integral < self.cutout_area:
                # If sufficiently high in latitude, assuming Galactic coordinates,
                # we ought to be able to take a full latitudinal slice around
                # the entire sphere.
                delta_lon = 180
            else:
                delta_lon = np.around(0.5 * self.cutout_area / lat_integral, decimals=1)
            # Handle wrap-around longitude maths naively by forcing 0/360 as the
            # minimum/maximum allowed limits of each box.
            if lmid - delta_lon < 0:
                self.lmins[i] = 0
                self.lmaxs[i] = 2 * delta_lon
            elif lmid + delta_lon > 360:
                self.lmaxs[i] = 360
                self.lmins[i] = 360 - 2 * delta_lon
            else:
                self.lmins[i] = lmid - delta_lon
                self.lmaxs[i] = lmid + delta_lon

        np.save('{}/npy/lmids.npy'.format(self.save_folder), self.lmids)
        np.save('{}/npy/bmids.npy'.format(self.save_folder), self.bmids)

    def __call__(self, a_cat_func, b_cat_func, a_cat_name, b_cat_name, snr_model_recreate=True,
                 cat_recreate=True, count_recreate=True, tri_download=True, dens_recreate=True,
                 nn_recreate=True, auf_sim_recreate=True, auf_pdf_recreate=True,
                 h_o_fit_recreate=True, fit_x2s_recreate=True, make_plots=False,
                 make_summary_plot=True):
        """
        Call function for the correction calculation process.

        Parameters
        ----------
        a_cat_func : callable
            Function used to generate reduced catalogue table for catalogue "a".
        b_cat_func : callable
            Function used to generate reduced catalogue table for catalogue "b".
        a_cat_name : string
            Name of the catalogue "a" filename, as saved by ``a_cat_func``. Must
            accept four formats via Python string formatting (e.g.
            ``'a_string_{}'``) that represent ``lmin``, ``lmax``, ``bmin``, and
            ``bmax`` respectively.
        b_cat_name : string
            Name of the catalogue "b" filename created by ``b_cat_func``. Must
            accept astometric coordinate box corners within string formatting.
        snr_model_recreate : boolean, optional
            If ``True`` magnitude-SNR relations are re-calculated even if outputs
            exist on disk.
        cat_recreate : boolean, optional
            Flag indicating whether to re-make reduced catalogue tables if they
            already exist on the disk.
        count_recreate : boolean, optional
            Determines whether to recreate the combined star-galaxy differential
            magnitude counts for each sightline if they exist.
        tri_download : boolean, optional
            Flag determining if TRILEGAL simulations should be re-downloaded
            if they already exist on disk.
        dens_recreate : boolean, optional
            Controls whether the local normalising density is re-calculated for
            sources again, even if outputs are saved.
        nn_recreate : boolean, optional
            Re-calculate nearest neighbour matches if ``True``.
        auf_sim_recreate : boolean, optional
            Controls if perturbations due to unresolved, blended contaminant
            objects are re-simulated even if outputs exist on disk.
        auf_pdf_recreate : boolean, optional
            Flag controlling whether probability density functions are
            calculated based on previously derived AUF simulations, or
            if previous PDFs are loaded from disk.
        h_o_fit_recreate : boolean, optional
            If ``True`` empirical astrometric uncertainties are re-derived even
            if previously calculated.
        fit_x2s_recreate : boolean, optional
            Flag controlling whether chi-squared statistics are calculated for
            derived fits, or if previously determined results are loaded.
        make_plots : boolean, optional
            Determines if intermediate figures are generated in the process
            of deriving astrometric corrections.
        make_summary_plot : boolean, optional
            If ``True`` then final summary plot is created, even if
            ``make_plots`` is ``False`` and intermediate plots are not created.
        """
        self.a_cat_func = a_cat_func
        self.b_cat_func = b_cat_func
        self.a_cat_name = a_cat_name
        self.b_cat_name = b_cat_name

        self.cat_recreate = cat_recreate
        self.count_recreate = count_recreate
        self.tri_download = tri_download
        self.dens_recreate = dens_recreate
        self.nn_recreate = nn_recreate
        self.auf_sim_recreate = auf_sim_recreate
        self.auf_pdf_recreate = auf_pdf_recreate
        self.snr_model_recreate = snr_model_recreate
        self.h_o_fit_recreate = h_o_fit_recreate
        self.fit_x2s_recreate = fit_x2s_recreate

        self.make_plots = make_plots
        self.make_summary_plot = make_summary_plot

        self.make_catalogue_cutouts()

        self.make_snr_model()

        self.make_star_galaxy_counts()
        if self.make_plots:
            self.plot_star_galaxy_counts()
        self.calculate_local_densities_and_nearest_neighbours()
        self.simulate_aufs()
        self.create_auf_pdfs()
        self.fit_uncertainty()
        if self.make_plots or (self.fit_x2s_recreate or not
                               os.path.isfile('{}/npy/fit_x2s.npy'.format(self.save_folder))):
            self.plot_fits_calculate_chi_sq()
        self.make_ma_fits_snr_h_plot()

    def make_catalogue_cutouts(self):
        """
        Generate cutout catalogues for regions as defined by corner l-b
        coordinates.
        """
        if self.coord_or_chunk == 'coord':
            zip_list = (self.lmids, self.bmids, self.lmins, self.lmaxs, self.bmins, self.bmaxs)
        else:
            zip_list = (self.chunks, self.lmins, self.lmaxs, self.bmins, self.bmaxs)
        for index_, list_of_things in enumerate(zip(*zip_list)):
            print('Creating catalogue cutouts... {}/{}'.format(index_+1, len(self.lmids)), end='\r')

            if self.coord_or_chunk == 'coord':
                lmid, bmid, lmin, lmax, bmin, bmax = list_of_things
            else:
                chunk, lmin, lmax, bmin, bmax = list_of_things

            if self.coord_or_chunk == 'coord':
                cat_args = (lmid, bmid)
            else:
                cat_args = (chunk,)
            if (not os.path.isfile(self.a_cat_name.format(*cat_args)) or
                    self.cat_recreate):
                self.a_cat_func(lmin, lmax, bmin, bmax, *cat_args)
            if (not os.path.isfile(self.b_cat_name.format(*cat_args)) or
                    self.cat_recreate):
                self.b_cat_func(lmin, lmax, bmin, bmax, *cat_args)

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
        if (self.snr_model_recreate or not
                os.path.isfile('{}/npy/snr_model.npy'.format(self.save_folder))):
            print("Making SNR model...")
            abc_array = np.empty((len(self.mag_indices), len(self.lmids), 3), float)
            for j in range(len(self.mag_indices)):
                if self.make_plots:
                    gs = self.make_gridspec('2', self.b_grid_length, self.l_grid_length, 0.8, 8)

                pool = multiprocessing.Pool(self.n_pool)
                counter = np.arange(0, len(self.lmids))
                if self.coord_or_chunk == 'coord':
                    iter_group = zip(counter, self.lmids, self.bmids, itertools.repeat(j))
                else:
                    iter_group = zip(counter, self.chunks, itertools.repeat(j))

                for results in pool.imap_unordered(self.fit_snr_model, iter_group, chunksize=2):
                    i, res, s_bins, s_d_snr_med, s_d_snr_dmed, snr_med, snr_dmed = results
                    a, b, c = 10**res.x
                    abc_array[j, i, 0] = a
                    abc_array[j, i, 1] = b
                    abc_array[j, i, 2] = c

                    lmid, bmid = self.lmids[i], self.bmids[i]

                    if self.make_plots:
                        q = ~np.isnan(s_d_snr_med)
                        _x = np.linspace(s_bins[0], s_bins[-1], 10000)

                        ax = plt.subplot(gs[i])
                        ax.plot(_x, np.log10(np.sqrt(c * 10**_x + b + (a * 10**_x)**2)),
                                'r-', zorder=5)

                        ax.errorbar((s_bins[:-1]+np.diff(s_bins)/2)[q], s_d_snr_med[q], fmt='k.',
                                    yerr=s_d_snr_dmed[q], zorder=3)

                        ax.set_title('l = {}, b = {}\na = {:.2e}, b = {:.2e}, c = {:.2e}'
                                     .format(lmid, bmid, a, b, c), fontsize=28)

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

                pool.close()
                pool.join()

                if self.make_plots:
                    plt.tight_layout()
                    plt.savefig('{}/pdf/s_vs_snr_{}.pdf'.format(self.save_folder,
                                                                self.mag_names[j]))
                    plt.close()

            np.save('{}/npy/snr_model.npy'.format(self.save_folder), abc_array)

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

    def fit_snr_model(self, iterable):
        """
        Function to derive the scaling relation between magnitude via flux
        and SNR.

        Parameters
        ----------
        iterable : list
            List of input parameters as passed through ``multiprocessing``.
            Includes central and corner coordinates of region, and the
            name of the catalogue on disk.

        Returns
        -------
        i : integer
            Index into ``self.lmids``.
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

        if self.coord_or_chunk == 'coord':
            i, lmid, bmid, j = iterable
            b = self.load_catalogue('b', (lmid, bmid))
        else:
            i, chunk, j = iterable
            b = self.load_catalogue('b', (chunk,))

        # TODO: un-hardcode magnitude/uncertainty/coordinate column numbers?
        s = 10**(-1/2.5 * b[:, self.mag_indices[j]])
        # Based on a naive dm = 2.5 log10((S+N)/S).
        snr = 1 / (10**(b[:, self.mag_unc_indices[j]] / 2.5) - 1)

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

        return i, res, s_bins, s_d_snr_med, s_d_snr_dmed, snr_med, snr_dmed

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

        if self.coord_or_chunk == 'coord':
            zip_list = (self.lmids, self.bmids)
        else:
            zip_list = (self.lmids, self.bmids, self.chunks)
        for index_, list_of_things in enumerate(zip(*zip_list)):
            print('Creating simulated star+galaxy counts... {}/{}'.format(
                  index_+1, len(self.lmids)), end='\r')
            if self.coord_or_chunk == 'coord':
                lmid, bmid = list_of_things
                cat_args = (lmid, bmid)
                file_name = '{}_{}'.format(lmid, bmid)
            else:
                lmid, bmid, chunk = list_of_things
                cat_args = (chunk,)
                file_name = '{}'.format(chunk)
            if not self.count_recreate and \
                    os.path.isfile('{}/npy/sim_counts_{}.npz'.format(
                                   self.save_folder, file_name)):
                continue

            num_bright, num_faint = 90000, 0.75e6
            if (self.tri_download or not
                    os.path.isfile('{}/{}_faint.dat'.format(self.trifolder,
                                                            self.triname.format(lmid, bmid)))):
                download_trilegal_simulation('.', self.trifilterset, lmid, bmid, self.magnum,
                                             'galactic', self.maglim_f, total_objs=num_faint)
                os.system('mv trilegal_auf_simulation.dat {}/{}_faint.dat'
                          .format(self.trifolder, self.triname.format(lmid, bmid)))
            if (self.tri_download or not
                    os.path.isfile('{}/{}_bright.dat'.format(self.trifolder,
                                                             self.triname.format(lmid, bmid)))):
                download_trilegal_simulation('.', self.trifilterset, lmid, bmid, self.magnum,
                                             'galactic', self.maglim_b, total_objs=num_bright)
                os.system('mv trilegal_auf_simulation.dat {}/{}_bright.dat'
                          .format(self.trifolder, self.triname.format(lmid, bmid)))

            tri_hist, tri_mags, _, dtri_mags, tri_uncert, tri_av = make_tri_counts(
                self.trifolder, self.triname.format(lmid, bmid), self.trifiltname, self.dm)
            gal_dNs = create_galaxy_counts(
                self.gal_cmau_array, tri_mags+dtri_mags/2, np.linspace(0, 4, 41),
                self.gal_wav_micron, self.gal_alpha0, self.gal_alpha1, self.gal_alphaweight,
                self.gal_ab_offset, self.gal_filtname, self.gal_alav*tri_av)

            log10y = np.log10(tri_hist + gal_dNs)
            new_uncert = np.sqrt(tri_uncert**2 + (0.05*gal_dNs)**2)
            dlog10y = 1/np.log(10) * new_uncert / (tri_hist + gal_dNs)

            b = self.load_catalogue('b', cat_args)
            mag_ind = self.mag_indices[self.best_mag_index]
            hist_mag, bins = np.histogram(b[~np.isnan(b[:, mag_ind]), mag_ind], bins='auto')
            minmag = bins[0]
            # Ensure that we're only counting sources for normalisation purposes
            # down to specified bright_mag or the completeness turnover,
            # whichever is brighter.
            maxmag = min(self.bright_mag, bins[:-1][np.argmax(hist_mag)])

            mag_slice = (tri_mags >= minmag) & (tri_mags+dtri_mags <= maxmag)
            N_norm = np.sum(10**log10y[mag_slice] * dtri_mags[mag_slice])
            np.savez('{}/npy/sim_counts_{}.npz'.format(
                     self.save_folder, file_name), log10y, dlog10y, tri_hist, tri_mags,
                     dtri_mags, tri_uncert, [tri_av], gal_dNs, [minmag], [maxmag], [N_norm])
        print('')

    def plot_star_galaxy_counts(self):
        """
        Plotting routine to display data and model differential source counts,
        for verification purposes.
        """
        if (self.count_recreate or self.cat_recreate or not
                os.path.isfile('{}/pdf/counts_comparison.pdf'.format(self.save_folder))):
            gs = self.make_gridspec('123123', self.b_grid_length, self.l_grid_length, 0.8, 15)
            if self.coord_or_chunk == 'coord':
                zip_list = (self.lmids, self.bmids, self.lmins, self.lmaxs, self.bmins, self.bmaxs)
            else:
                zip_list = (self.lmids, self.bmids, self.lmins, self.lmaxs, self.bmins,
                            self.bmaxs, self.chunks)
            for index_, list_of_things in enumerate(zip(*zip_list)):
                print('Plotting data and model counts... {}/{}'.format(
                      index_+1, len(self.lmids)), end='\r')

                if self.coord_or_chunk == 'coord':
                    lmid, bmid, lmin, lmax, bmin, bmax = list_of_things
                    cat_args = (lmid, bmid)
                    file_name = '{}_{}'.format(lmid, bmid)
                else:
                    lmid, bmid, lmin, lmax, bmin, bmax, chunk = list_of_things
                    cat_args = (chunk,)
                    file_name = '{}'.format(chunk)

                b = self.load_catalogue('b', cat_args)
                npyfilez = np.load('{}/npy/sim_counts_{}.npz'.format(
                                   self.save_folder, file_name))
                (log10y, dlog10y, tri_hist, tri_mags, dtri_mags, tri_uncert, [tri_av], gal_dNs,
                 [minmag], [maxmag], [N_norm]) = [npyfilez['arr_{}'.format(ii)] for ii in range(11)]

                # Unit area is cos(t) dt dx for 0 <= t <= 90deg, 0 <= x <= 360 deg,
                # integrated between bmin < t < bmax, lmin < x < lmax, converted
                # to degrees.
                rect_area = (lmax - (lmin)) * (
                    np.sin(np.radians(bmax)) - np.sin(np.radians(bmin))) * 180/np.pi

                mag_ind = self.mag_indices[self.best_mag_index]
                data_mags = b[~np.isnan(b[:, mag_ind]), mag_ind]
                # Correction to model is the ratio of data counts per unit area
                # to model source density.
                correction = np.sum((data_mags >= minmag) &
                                    (data_mags <= maxmag)) / rect_area / N_norm

                ax = plt.subplot(gs[index_])
                ax.set_title('l = {}, b = {}'.format(lmid, bmid))
                ax.errorbar(tri_mags+dtri_mags/2, log10y + np.log10(correction), yerr=dlog10y,
                            c='k', marker='.', zorder=1, ls='None')

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
                ax.plot(tri_mags+dtri_mags/2, np.log10(tri_hist) + np.log10(correction), 'b--')
                ax.plot(tri_mags+dtri_mags/2, np.log10(gal_dNs) + np.log10(correction), 'b:')
                ax.set_ylim(*lims)

                ax.set_xlabel('Magnitude')
                if usetex:
                    ax.set_ylabel(r'log$_{10}\left(\mathrm{D}\ /\ \mathrm{mag}^{-1}\,'
                                  r'\mathrm{deg}^{-2}\right)$')
                else:
                    ax.set_ylabel(r'log10(D / mag^-1 deg^-2)')
            print('')
            plt.figure('123123')
            plt.tight_layout()
            plt.savefig('{}/pdf/counts_comparison.pdf'.format(self.save_folder))
            plt.close()

    def calculate_local_densities_and_nearest_neighbours(self):
        """
        Calculate local normalising catalogue densities and catalogue-catalogue
        nearest neighbour match pairings for each cutout region.
        """
        if self.coord_or_chunk == 'coord':
            zip_list = (self.lmids, self.bmids, self.lmins, self.lmaxs, self.bmins, self.bmaxs)
        else:
            zip_list = (self.lmids, self.bmids, self.lmins, self.lmaxs, self.bmins,
                        self.bmaxs, self.chunks)
        for index_, list_of_things in enumerate(zip(*zip_list)):
            print('Creating local densities and nearest neighbour matches... {}/{}'.format(
                index_+1, len(self.lmids)), end='\r')

            if self.coord_or_chunk == 'coord':
                lmid, bmid, lmin, lmax, bmin, bmax = list_of_things
                cat_args = (lmid, bmid)
                file_name = '{}_{}'.format(lmid, bmid)
            else:
                lmid, bmid, lmin, lmax, bmin, bmax, chunk = list_of_things
                cat_args = (chunk,)
                file_name = '{}'.format(chunk)

            if (os.path.isfile('{}/npy/local_dens_nn_matches_{}.npz'.format(
                               self.save_folder, file_name)) and not
                    self.dens_recreate and not self.nn_recreate):
                continue

            a = self.load_catalogue('a', cat_args)
            b = self.load_catalogue('b', cat_args)
            npyfilez = np.load('{}/npy/sim_counts_{}.npz'.format(self.save_folder, file_name))
            _, _, _, _, _, _, _, _, [minmag], [maxmag], _ = \
                [npyfilez['arr_{}'.format(ii)] for ii in range(11)]

            lon_slice = np.linspace(lmin, lmax, int(np.floor((lmax-lmin)*2 + 1)))
            lat_slice = np.linspace(bmin, bmax, int(np.floor((bmax-bmin)*2 + 1)))

            Narray = create_densities(
                lmid, bmid, b, minmag, maxmag, lon_slice, lat_slice, lmin, lmax, bmin, bmax,
                self.dens_search_radius, self.n_pool, self.dens_recreate, self.save_folder,
                self.mag_indices[self.best_mag_index], self.pos_and_err_indices[1][0],
                self.pos_and_err_indices[1][1])

            _, bmatch, dists = create_distances(
                a, b, lmid, bmid, self.nn_radius, self.nn_recreate, self.save_folder,
                self.pos_and_err_indices[0][0], self.pos_and_err_indices[0][1],
                self.pos_and_err_indices[1][0], self.pos_and_err_indices[1][1])

            # TODO: extend to 3-D search around N-m-sig to find as many good
            # enough bins as possible, instead of only keeping one N-sig bin
            # per magnitude?
            _h, _b = np.histogram(Narray[bmatch], bins='auto')
            modeN = (_b[:-1]+np.diff(_b)/2)[np.argmax(_h)]
            dN = 0.05*modeN

            np.savez('{}/npy/local_dens_nn_matches_{}.npz'.format(
                     self.save_folder, file_name), Narray, dists, bmatch, [modeN], [dN])
        print('')

    def simulate_aufs(self):
        """
        Simulate unresolved blended contaminants for each magnitude-sightline
        combination, for both aperture photometry and background-dominated PSF
        algorithms.
        """
        abc_array = np.load('{}/npy/snr_model.npy'.format(self.save_folder))
        if self.coord_or_chunk == 'coord':
            zip_list = (self.lmids, self.bmids)
        else:
            zip_list = (self.chunks,)
        for index_, list_of_things in enumerate(zip(*zip_list)):
            print('Creating AUF simulations... {}/{}'.format(index_+1, len(self.lmids)), end='\r')
            if self.coord_or_chunk == 'coord':
                lmid, bmid = list_of_things
                file_name = '{}_{}'.format(lmid, bmid)
            else:
                chunk, = list_of_things
                file_name = '{}'.format(chunk)

            if not self.auf_sim_recreate and os.path.isfile(
                    '{}/npy/four_auf_{}.npz'.format(
                    self.save_folder, file_name)):
                continue

            npyfilez = np.load('{}/npy/local_dens_nn_matches_{}.npz'.format(
                self.save_folder, file_name))
            _, _, _, [modeN], _ = [npyfilez['arr_{}'.format(ii)] for ii in range(5)]
            npyfilez = np.load('{}/npy/sim_counts_{}.npz'.format(
                               self.save_folder, file_name))
            log10y, _, _, tri_mags, dtri_mags, _, _, _, _, _, [N_norm] = \
                [npyfilez['arr_{}'.format(ii)] for ii in range(11)]

            a, b, c = abc_array[self.best_mag_index, index_]
            B = 0.05
            # Self-consistent, non-zeropointed "flux", based on the relation
            # given in make_snr_model.
            flux = 10**(-1/2.5 * self.mag_array)
            snr = flux / np.sqrt(c * 10**flux + b + (a * 10**flux)**2)
            dm_max = _calculate_magnitude_offsets(
                modeN*np.ones_like(self.mag_array), self.mag_array, B, snr, tri_mags, log10y,
                dtri_mags, self.R, N_norm)

            seed = np.random.default_rng().choice(100000, size=(paf.get_random_seed_size(),
                                                                len(self.mag_array)))
            _, _, four_off_fw, _, _ = \
                paf.perturb_aufs(
                    modeN*np.ones_like(self.mag_array), self.mag_array, self.r[:-1]+self.dr/2,
                    self.dr, self.r, self.j0s.T, tri_mags+dtri_mags/2, dtri_mags, log10y, N_norm,
                    (dm_max/self.dm).astype(int), self.dmcut, self.R, self.psfsig,
                    self.numtrials, seed, self.dd_params, self.l_cut, 'fw')

            seed = np.random.default_rng().choice(100000, size=(paf.get_random_seed_size(),
                                                                len(self.mag_array)))
            _, _, four_off_ps, _, _ = \
                paf.perturb_aufs(
                    modeN*np.ones_like(self.mag_array), self.mag_array, self.r[:-1]+self.dr/2,
                    self.dr, self.r, self.j0s.T, tri_mags+dtri_mags/2, dtri_mags, log10y, N_norm,
                    (dm_max/self.dm).astype(int), self.dmcut, self.R, self.psfsig,
                    self.numtrials, seed, self.dd_params, self.l_cut, 'psf')

            np.savez('{}/npy/four_auf_{}.npz'.format(
                     self.save_folder, file_name), four_off_fw, four_off_ps)
        print('')

    def create_auf_pdfs(self):
        """
        Using perturbation AUF simulations, generate probability density functions
        of perturbation distance for all cutout regions, as well as recording key
        statistics such as average magnitude or SNR.
        """
        if self.coord_or_chunk == 'coord':
            zip_list = (self.lmids, self.bmids)
        else:
            zip_list = (self.chunks,)
        for index_, list_of_things in enumerate(zip(*zip_list)):
            print('Creating catalogue AUF probability densities... {}/{}'.format(
                  index_+1, len(self.lmids)), end='\r')
            if self.coord_or_chunk == 'coord':
                lmid, bmid = list_of_things
                cat_args = (lmid, bmid)
                file_name = '{}_{}'.format(lmid, bmid)
            else:
                chunk, = list_of_things
                cat_args = (chunk,)
                file_name = '{}'.format(chunk)
            if not self.auf_pdf_recreate and os.path.isfile(
                    '{}/npy/auf_pdf_{}.npz'.format(self.save_folder, file_name)):
                continue

            b = self.load_catalogue('b', cat_args)
            npyfilez = np.load('{}/npy/local_dens_nn_matches_{}.npz'.format(
                               self.save_folder, file_name))
            Narray, dists, bmatch, [modeN], [dN] = [npyfilez['arr_{}'.format(ii)] for ii in
                                                    range(5)]

            b_matches = b[bmatch]

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
                N_cut = (Narray[bmatch] >= modeN-dN) & (Narray[bmatch] <= modeN+dN)

                final_slice = sig_cut & mag_cut & N_cut & (dists <= 20*sig)
                final_dists = dists[final_slice]
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

            pdfs, pdf_uncerts, q_pdfs, pdf_bins = (
                np.array(pdfs, dtype=object), np.array(pdf_uncerts, dtype=object),
                np.array(q_pdfs, dtype=object), np.array(pdf_bins, dtype=object))

            np.savez('{}/npy/auf_pdf_{}.npz'.format(self.save_folder, file_name),
                     avg_snr, avg_mag, avg_sig, pdfs, pdf_uncerts, q_pdfs, pdf_bins, skip_flags)

        print('')

    def fit_uncertainty(self):
        """
        For each magnitude-sightline combination, fit for the empirical centroid
        uncertainty describing the distribution of match separations.
        """
        a_array = np.load('{}/npy/snr_model.npy'.format(self.save_folder))[
            self.best_mag_index, :, 0]
        if self.coord_or_chunk == 'coord':
            zip_list = (self.lmids, self.bmids, a_array)
        else:
            zip_list = (self.chunks, a_array)
        for index_, list_of_things in enumerate(zip(*zip_list)):
            print('Creating joint H/sig fits... {}/{}'.format(index_+1, len(self.lmids)), end='\r')
            if self.coord_or_chunk == 'coord':
                lmid, bmid, _a = list_of_things
                file_name = '{}_{}'.format(lmid, bmid)
            else:
                chunk, _a = list_of_things
                file_name = '{}'.format(chunk)
            if not self.h_o_fit_recreate and os.path.isfile(
                    '{}/npy/h_o_fit_res_{}.npz'.format(
                    self.save_folder, file_name)):
                continue

            fit_sigs = np.zeros((len(self.mag_array), 2), float)
            fit_nnf = np.zeros((len(self.mag_array), 2), float)

            npyfilez = np.load('{}/npy/auf_pdf_{}.npz'.format(
                               self.save_folder, file_name), allow_pickle=True)
            avg_snr, avg_mag, avg_sig, pdfs, pdf_uncerts, q_pdfs, pdf_bins, skip_flags = \
                [npyfilez['arr_{}'.format(ii)] for ii in range(8)]
            npyfilez = np.load('{}/npy/four_auf_{}.npz'.format(
                               self.save_folder, file_name))
            four_off_fw, four_off_ps = [npyfilez['arr_{}'.format(ii)] for ii in range(2)]

            npyfilez = np.load('{}/npy/local_dens_nn_matches_{}.npz'.format(self.save_folder,
                               file_name))
            _, _, _, [modeN], _ = [npyfilez['arr_{}'.format(ii)] for ii in range(5)]

            resses = [0]*len(self.mag_array)
            # Use the lower of the number of magnitues to run in parallel and the
            # maximum specified number of threads to call.
            n_pool = min(self.n_pool, len(self.mag_array))
            pool = multiprocessing.Pool(n_pool)
            counter = np.arange(0, len(self.mag_array))
            iter_group = zip(counter, itertools.repeat([pdfs, pdf_uncerts, self.rho, self.drho,
                                                        self.r, self.dr, four_off_fw, four_off_ps,
                                                        q_pdfs, pdf_bins, avg_sig[:, 0],
                                                        avg_snr[:, 0], self.j0s,
                                                        modeN/3600**2, _a]))
            for s in pool.imap_unordered(self.fit_auf, iter_group,
                                         chunksize=max(1, len(self.mag_array) // n_pool)):
                res, i = s
                resses[i] = res
                if res == -1:
                    skip_flags[i] = 1
                else:
                    fit_sig, nn_frac = resses[i].x
                    cov = resses[i].hess_inv.todense()
                    fit_sigs[i, 0] = fit_sig / 10
                    fit_sigs[i, 1] = np.sqrt(cov[0, 0]) / 10

                    fit_nnf[i, 0] = nn_frac
                    fit_nnf[i, 1] = np.sqrt(cov[1, 1])

            pool.close()
            pool.join()

            resses = np.array(resses, dtype=object)

            np.savez('{}/npy/h_o_fit_res_{}.npz'.format(
                     self.save_folder, file_name), resses, fit_sigs, skip_flags)

        print('')

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

    def make_plot_calc_chisq(self, iterable):
        """
        Make an individual sightline's verification plot, or calculate the
        goodness-of-fit of its best-fitting model.

        Parameters
        ----------
        iterable : list
            List of parameters passed through from ``multiprocessing``. Includes
            the index, central and corner coordinate, systematic signal-to-noise
            ratio, and flag indicating whether or not to fit for chi-squareds.

        Returns
        -------
        index : integer
            Index into ``lmids`` for the run. Only returned if ``fit_x2_flag``
            is ``True``.
        x2s : numpy.ndarray
            The chi-squared goodness-of-fit results from all fits performed
            in this sightline. Only returned if ``fit_x2_flag`` is ``True``.
        """
        if self.coord_or_chunk == 'coord':
            index, lmid, bmid, _a, (fit_x2_flag) = iterable
            cat_args = (lmid, bmid)
            file_name = '{}_{}'.format(lmid, bmid)
        else:
            index, chunk, _a, (fit_x2_flag) = iterable
            cat_args = (chunk,)
            file_name = '{}'.format(chunk)
        if self.make_plots:
            # Grid just big enough square to cover mag_array entries.
            gs1 = self.make_gridspec('34242b', self.n_mag_rows, self.n_mag_cols, 0.8, 15)
            ax1s = [plt.subplot(gs1[i]) for i in range(len(self.mag_array))]

        npyfilez = np.load('{}/npy/auf_pdf_{}.npz'.format(
                           self.save_folder, file_name), allow_pickle=True)
        avg_snr, avg_mag, avg_sig, pdfs, pdf_uncerts, q_pdfs, pdf_bins, _ = \
            [npyfilez['arr_{}'.format(ii)] for ii in range(8)]

        npyfilez = np.load('{}/npy/four_auf_{}.npz'.format(
                           self.save_folder, file_name))
        four_off_fw, four_off_ps = [npyfilez['arr_{}'.format(ii)] for ii in range(2)]

        b = self.load_catalogue('b', cat_args)
        npyfilez = np.load('{}/npy/local_dens_nn_matches_{}.npz'.format(
                           self.save_folder, file_name))
        Narray, dists, bmatch, [modeN], [dN] = [npyfilez['arr_{}'.format(ii)] for ii in
                                                range(5)]

        b_matches = b[bmatch]

        npyfilez = np.load('{}/npy/h_o_fit_res_{}.npz'.format(
                           self.save_folder, file_name), allow_pickle=True)
        resses, _, skip_flags = [npyfilez['arr_{}'.format(ii)] for ii in range(3)]

        if fit_x2_flag:
            x2s = np.ones((len(self.mag_array), 2), float) * np.nan

        for i in range(len(self.mag_array)):
            if self.make_plots:
                ax = ax1s[i]
            if skip_flags[i]:
                continue
            pdf, pdf_uncert, q_pdf, pdf_bin = (pdfs[i], pdf_uncerts[i], q_pdfs[i], pdf_bins[i])
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
            N_cut = (Narray[bmatch] >= modeN-dN) & (Narray[bmatch] <= modeN+dN)
            final_slice = sig_cut & mag_cut & N_cut & (dists <= 20*bsig)
            bm = b_matches[final_slice]

            _N = np.percentile(Narray[bmatch][final_slice], 50)/3600**2
            fit_sig, nn_frac = resses[i].x
            fit_sig /= 10

            H = 1 - np.sqrt(1 - min(1, _a**2 * avg_snr[i, 0]**2))

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

                four_hist = _H * four_off_fw[:, i] + (1 - _H) * four_off_ps[:, i]
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

                if fit_x2_flag and j == 0:
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

                cov = resses[i].hess_inv.todense()
                if usetex:
                    ax.set_title(r'mag = {}, H = {:.2f}, $\sigma$ = {:.3f}$\pm${:.3f} ({:.3f})"; '
                                 r'$F$ = {:.2f}; SNR = {:.2f}$^{{+{:.2f}}}_{{-{:.2f}}}$; '
                                 r'N = {}'.format(self.mag_array[i], H, fit_sig,
                                                  np.sqrt(cov[0, 0])/10, bsig, nn_frac,
                                                  avg_snr[i, 0], avg_snr[i, 2], avg_snr[i, 1],
                                                  len(bm)), fontsize=22)
                else:
                    ax.set_title(r'mag = {}, H = {:.2f}, sigma = {:.3f}+/-{:.3f} ({:.3f})"; '
                                 r'F = {:.2f}; SNR = {:.2f}+{:.2f}/-{:.2f}; '
                                 r'N = {}'.format(self.mag_array[i], H, fit_sig,
                                                  np.sqrt(cov[0, 0])/10, bsig, nn_frac,
                                                  avg_snr[i, 0], avg_snr[i, 2], avg_snr[i, 1],
                                                  len(bm)), fontsize=22)
                ax.legend(fontsize=15)
                ax.set_xlabel('Radius / arcsecond')
                if usetex:
                    ax.set_ylabel('PDF / arcsecond$^{-1}$')
                else:
                    ax.set_ylabel('PDF / arcsecond^-1')
        if self.make_plots:
            plt.tight_layout()
            plt.savefig('{}/pdf/auf_fits_{}.pdf'.format(self.save_folder, file_name))
            plt.close()

        if fit_x2_flag:
            return index, x2s

    def plot_fits_calculate_chi_sq(self):
        """
        Calculate chi-squared values for each magnitude-sightline combination,
        and create verification plots showing the quality of the fits.
        """
        fit_x2_flag = (self.fit_x2s_recreate or not
                       os.path.isfile('{}/npy/fit_x2s.npy'.format(self.save_folder)))
        if fit_x2_flag:
            x2s = np.ones((len(self.lmids), len(self.mag_array), 2), float) * np.nan

        a_array = np.load('{}/npy/snr_model.npy'.format(self.save_folder))[
            self.best_mag_index, :, 0]
        if self.make_plots and fit_x2_flag:
            print('Creating individual AUF figures and calculating goodness-of-fits...')
        elif self.make_plots:
            print('Creating individual AUF figures ...')
        elif fit_x2_flag:
            print('Calculating goodness-of-fits...')
        pool = multiprocessing.Pool(self.n_pool)
        counter = np.arange(0, len(self.lmids))
        if self.coord_or_chunk == 'coord':
            iter_group = zip(counter, self.lmids, self.bmids, a_array,
                             itertools.repeat(fit_x2_flag))
        else:
            iter_group = zip(counter, self.chunks, a_array, itertools.repeat(fit_x2_flag))
        for results in pool.imap_unordered(self.make_plot_calc_chisq, iter_group,
                                           chunksize=len(self.lmids)//self.n_pool):
            if fit_x2_flag:
                index, chi_sq = results
                x2s[index, :, :] = chi_sq

        pool.close()
        pool.join()

        if fit_x2_flag:
            np.save('{}/npy/fit_x2s.npy'.format(self.save_folder), np.array(x2s))

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

        x2s = np.load('{}/npy/fit_x2s.npy'.format(self.save_folder))

        if not (self.make_plots or self.make_summary_plot):
            print("Creating sig-sig relations...")
        else:
            print("Creating sig-sig relations and SNR-H summary figure...")
            gs_s = self.make_gridspec('123123b', self.b_grid_length, self.l_grid_length, 0.8, 15)
            gs = self.make_gridspec('12312', 2, 2, 0.8, 10)
            ax_b = plt.subplot(gs[0])
            ax_d = plt.subplot(gs[1])

            ylims = [999, 0]

            cols = ['k', 'r', 'b', 'g', 'c', 'm', 'orange', 'brown', 'purple', 'grey', 'olive',
                    'cornflowerblue', 'deeppink', 'maroon', 'palevioletred', 'teal', 'crimson',
                    'chocolate', 'darksalmon', 'steelblue', 'slateblue', 'tan', 'yellowgreen',
                    'silver']

        m_sigs = np.empty_like(self.lmids)
        n_sigs = np.empty_like(self.lmids)

        if self.coord_or_chunk == 'coord':
            zip_list = (self.lmids, self.bmids)
        else:
            zip_list = (self.lmids, self.bmids, self.chunks)

        for i, list_of_things in enumerate(zip(*zip_list)):
            if self.coord_or_chunk == 'coord':
                lmid, bmid = list_of_things
                file_name = '{}_{}'.format(lmid, bmid)
            else:
                lmid, bmid, chunk = list_of_things
                file_name = '{}'.format(chunk)
            npyfilez = np.load('{}/npy/auf_pdf_{}.npz'.format(
                               self.save_folder, file_name), allow_pickle=True)
            _, _, data_sigs, _, _, _, _, _ = \
                [npyfilez['arr_{}'.format(ii)] for ii in range(8)]

            npyfilez = np.load('{}/npy/h_o_fit_res_{}.npz'.format(
                               self.save_folder, file_name), allow_pickle=True)
            _, fit_sigs, skip_flags = [npyfilez['arr_{}'.format(ii)] for ii in range(3)]

            if self.make_plots or self.make_summary_plot:
                c = cols[i % len(cols)]

                plt.figure('123123b')
                ax1 = plt.subplot(gs_s[i])
                ax1.errorbar(data_sigs[~skip_flags, 0], fit_sigs[~skip_flags, 0],
                             xerr=data_sigs[~skip_flags, 1:].T, yerr=fit_sigs[~skip_flags, 1],
                             linestyle='None', c='k', marker='.')
                ax1.set_ylim(0.95*np.amin(fit_sigs[~skip_flags, 0]),
                             1.05*np.amax(fit_sigs[~skip_flags, 0]))

            res_sig = minimize(quad_sig_fit, x0=[1, 0], args=(data_sigs[~skip_flags, 0],
                               fit_sigs[~skip_flags, 0], fit_sigs[~skip_flags, 1]),
                               method='L-BFGS-B', options={'ftol': 1e-12},
                               bounds=[(0, None), (0, None)])

            m_sig, n_sig = res_sig.x

            m_sigs[i] = m_sig
            n_sigs[i] = n_sig

            if self.make_plots or self.make_summary_plot:
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
                ax1.set_title('l = {}, b = {}\nm = {:.2f}, a = {:.2f}'.format(
                              lmid, bmid, m_sig, n_sig))
                ax1.legend()
                plt.figure('12312')
                ax_b.errorbar(data_sigs[~skip_flags, 0], fit_sigs[~skip_flags, 0],
                              linestyle='None', c=c, marker='.')
                ylims[0] = min(ylims[0], np.amin(fit_sigs[:, 0]))
                ylims[1] = max(ylims[1], np.amax(fit_sigs[:, 0]))

        np.save('{}/npy/m_sigs_array.npy'.format(self.save_folder), m_sigs)
        np.save('{}/npy/n_sigs_array.npy'.format(self.save_folder), n_sigs)

        if self.make_plots or self.make_summary_plot:
            plt.figure('123123b')
            plt.tight_layout()
            plt.savefig('{}/pdf/sig_fit_comparisons.pdf'.format(self.save_folder))
            plt.close()
            plt.figure('12312')

            q = ~np.isnan(x2s[:, :, 0])
            chi_sqs, dofs = x2s[:, :, 0][q].flatten(), x2s[:, :, 1][q].flatten()
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

            x_array = np.linspace(0, ax_b.get_xlim()[1], 100)
            ax_b.plot(x_array, x_array, 'g:')
            ax_b.set_ylim(0.95 * ylims[0], 1.05 * ylims[1])
            if usetex:
                ax_b.set_xlabel(r'Input astrometric $\sigma$ / "')
                ax_b.set_ylabel(r'Fit astrometric $\sigma$ / "')
            else:
                ax_b.set_xlabel(r'Input astrometric sigma / "')
                ax_b.set_ylabel(r'Fit astrometric sigma / "')

            for i, (f, label) in enumerate(zip([m_sigs, n_sigs], ['m', 'n'])):
                ax = plt.subplot(gs[i+2])
                img = ax.scatter(self.lmids, self.bmids, c=f, cmap='viridis')
                c = plt.colorbar(img, ax=ax, use_gridspec=True)
                c.ax.set_ylabel(label)
                ax.set_xlabel('l / deg')
                ax.set_ylabel('b / deg')

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
            l and b, or chunk ID, depending on ``coord_or_chunk``.

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


def create_densities(lmid, bmid, b, minmag, maxmag, lon_slice, lat_slice, lmin, lmax, bmin,
                     bmax, search_radius, n_pool, dens_recreate, save_folder,
                     mag_ind, l_ind, b_ind):
    """
    Generate local normalising densities for all sources in catalogue "b".

    Parameters
    ----------
    lmid : float
        Longitude of the center of the cutout region.
    bmid : float
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
        Array of intermediate longitude values, between ``lmin`` and
        ``lmax``, used to load smaller sub-sets of the catalogue ``b``.
    lat_slice : numpy.ndarray
        An array of intermediate latitude values, between ``bmin`` and
        ``bmax``, used to load smaller sub-sets of the catalogue ``b``.
    lmin : float
        The minimum longitude of the box of the cutout region.
    lmax : float
        The maximum longitude of the box of the cutout region.
    bmin : float
        The minimum latitude of the box of the cutout region.
    bmax : float
        The maximum latitude of the box of the cutout region.
    search_radius : float
        Radius, in arcseconds, around which to calculate the density of objects.
        Smaller values will allow for more fluctuations and handle smaller scale
        variation better, but be subject to low-number statistics.
    n_pool : integer
        Number of parallel threads to run when calculating densities via
        ``multiprocessing``.
    dens_recreate : boolean
        Flag indicating whether or not to run density calculations if output
        file is already on disk.
    save_folder : string
        Location on disk into which to save densities.
    mag_ind : integer
        Index in ``b`` where the magnitude being used is stored.
    l_ind : integer
        Index of ``b`` for the longitudinal coordinate column.
    b_ind : integer
        ``b`` index for the galactic latitude data.

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

    if dens_recreate or not os.path.isfile('{}/npy/narray_sky_{}_{}.npy'.format(
                                           save_folder, lmid, bmid)):
        cutmag = (b[:, mag_ind] >= minmag) & (b[:, mag_ind] <= maxmag)

        full_cat = SkyCoord(l=b[:, l_ind], b=b[:, b_ind], unit='deg', frame='galactic')
        mag_cut_cat = SkyCoord(l=b[cutmag, l_ind], b=b[cutmag, b_ind], unit='deg', frame='galactic')

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

        area = paf.get_circle_area_overlap(b[:, l_ind], b[:, b_ind], search_radius/3600,
                                           lmin, lmax, bmin, bmax)

        Narray = overlap_number / area

        np.save('{}/npy/narray_sky_{}_{}.npy'.format(save_folder, lmid, bmid), Narray)
    Narray = np.load('{}/npy/narray_sky_{}_{}.npy'.format(save_folder, lmid, bmid))

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


def create_distances(a, b, lmid, bmid, nn_radius, match_recreate, save_folder, a_l_ind, a_b_ind,
                     b_l_ind, b_b_ind):
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
    lmid : float
        Center of the cutout region in longitude.
    bmid : float
        Latitude of the cutout region's central coordinate.
    nn_radius : float
        Maximum match radius within which to consider potential counterpart
        assignments, in arcseconds.
    match_recreate : boolean
        Flag determining whether we re-run the neighbour finding process or
        if previous runs on disk can be used instead.
    save_folder : string
        Location on disk where matches should be saved.
    a_l_ind : integer
        Index into ``a`` of the galactic longitude data.
    a_b_ind : integer
        ``a`` index for galactic latitude column.
    b_l_ind : integer
        Longitude index in the ``b`` dataset.
    b_b_ind : integer
        Index into ``b`` that holds the galactic latitude data.

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
    if (match_recreate or
            not os.path.isfile('{}/npy/a_matchind_{}_{}.npy'.format(save_folder, lmid, bmid)) or
            not os.path.isfile('{}/npy/b_matchind_{}_{}.npy'.format(save_folder, lmid, bmid)) or
            not os.path.isfile('{}/npy/ab_dists_{}_{}.npy'.format(save_folder, lmid, bmid))):
        # TODO: make flexible if we relax requirement of Galactic coordinates.
        ac = SkyCoord(l=a[:, a_l_ind], b=a[:, a_b_ind], unit='deg', frame='galactic')
        bc = SkyCoord(l=b[:, b_l_ind], b=b[:, b_b_ind], unit='deg', frame='galactic')
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

        np.save('{}/npy/a_matchind_{}_{}.npy'.format(save_folder, lmid, bmid), amatch)
        np.save('{}/npy/b_matchind_{}_{}.npy'.format(save_folder, lmid, bmid), bmatch)
        np.save('{}/npy/ab_dists_{}_{}.npy'.format(save_folder, lmid, bmid), dists)
    amatch = np.load('{}/npy/a_matchind_{}_{}.npy'.format(save_folder, lmid, bmid))
    bmatch = np.load('{}/npy/b_matchind_{}_{}.npy'.format(save_folder, lmid, bmid))
    dists = np.load('{}/npy/ab_dists_{}_{}.npy'.format(save_folder, lmid, bmid))

    return amatch, bmatch, dists
