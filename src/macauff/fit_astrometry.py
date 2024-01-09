# Licensed under a 3-clause BSD style license - see LICENSE
'''
Module for calculating corrections to astrometric uncertainties of photometric catalogues, by
fitting their AUFs and centroid uncertainties across ensembles of matches between well-understood
catalogue and one for which precisions are less well known.
'''
# pylint: disable=too-many-lines
# pylint: disable=duplicate-code

import itertools
import multiprocessing
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord, UnitSphericalRepresentation, match_coordinates_sky
from matplotlib import gridspec
from numpy.lib.format import open_memmap
from scipy import spatial
from scipy.optimize import minimize
from scipy.stats import binned_statistic, chi2  # pylint: disable=unused-import

# Assume that usetex = False only applies for tests where no TeX is installed
# at all, instead of users having half-installed TeX, dvipng et al. somewhere.
usetex = not not shutil.which("tex")  # pylint: disable=unnecessary-negation
if usetex:
    plt.rcParams.update({"text.usetex": True, "text.latex.preamble": r"\usepackage{amsmath}"})

# pylint: disable=wrong-import-position,import-error,no-name-in-module
from macauff.galaxy_counts import create_galaxy_counts
from macauff.get_trilegal_wrapper import get_av_infinity
from macauff.misc_functions import min_max_lon
from macauff.misc_functions_fortran import misc_functions_fortran as mff
from macauff.perturbation_auf import (
    _calculate_magnitude_offsets,
    download_trilegal_simulation,
    make_tri_counts,
)
from macauff.perturbation_auf_fortran import perturbation_auf_fortran as paf

# pylint: enable=wrong-import-position,import-error,no-name-in-module

__all__ = ['AstrometricCorrections']


class AstrometricCorrections:  # pylint: disable=too-many-instance-attributes
    """
    Class to calculate any potential corrections to quoted astrometric
    precisions in photometric catalogues, based on reliable cross-matching
    to a well-understood second dataset.
    """
    # pylint: disable-next=too-many-locals,too-many-arguments
    def __init__(self, psf_fwhm, numtrials, nn_radius, dens_search_radius, save_folder,
                 gal_wav_micron, gal_ab_offset, gal_filtname, gal_alav, dm, dd_params, l_cut,
                 ax1_mids, ax2_mids, ax_dimension, mag_array, mag_slice, sig_slice, n_pool,
                 npy_or_csv, coord_or_chunk, pos_and_err_indices, mag_indices, mag_unc_indices,
                 mag_names, best_mag_index, coord_system, pregenerate_cutouts, n_r, n_rho, max_rho,
                 trifolder=None, triname=None, maglim_f=None, magnum=None, tri_num_faint=None,
                 trifilterset=None, trifiltname=None, tri_hist=None, tri_mags=None, dtri_mags=None,
                 tri_uncert=None, use_photometric_uncertainties=False, cutout_area=None, cutout_height=None,
                 single_sided_auf=True, chunks=None, return_nm=False):
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
            source, in degrees.
        save_folder : string
            Absolute or relative filepath of folder into which to store
            temporary and generated outputs from the fitting process.
        gal_wav_micron : float
            Wavelength, in microns, of the chosen filter, for use in
            simulating galaxy counts.
        gal_ab_offset : float
            The offset between the filter zero point and the AB magnitude
            offset.
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
            List of magnitudes in the filter to derive astrometry for.
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
        pregenerate_cutouts : boolean or None
            Indicates whether sightline catalogues must have been pre-made,
            or whether they can be generated by ``AstrometricCorrections``
            using specified lines-of-sight and cutout areas and heights. If
            ``None`` then in-memory catalogues must be passed in the call
            signature of ``AstrometricCorrections``.
        n_r : integer
            Number of elements to generate for real-space evaluation of Bessel
            functions, used in the convolution of AUF PDfs.
        n_rho : integer
            Number of elements to generate for fourier-space Bessel function
            evaluation, used in the convolution of AUF PDfs.
        max_rho : float
            Largest fourier-space value to integrate functions to during the
            convolution of AUF PDFs.
        trifolder : string, optional
            Filepath of the location into which to save TRILEGAL simulations. If
            provided ``tri_hist``, ``tri_mags``, ``dtri_mags``, and ``tri_uncert``
            must be ``None``, and ``triname``, ``maglim_f``, ``magnum``,
            ``tri_num_faint``, ``trifilterset``, and ``trifiltname`` must be
            given.
        triname : string, optional
            Name to give TRILEGAL simulations when downloaded. Is required to have
            two format ``{}`` options in string, for unique ax1-ax2 sightline
            combination downloads. Should not contain any file types, just the
            name of the file up to, but not including, the final ".". Should be
            ``None`` if ``tri_hist`` et al. are provided.
        maglim_f : float, optional
            Magnitude in the ``magnum`` filter down to which sources should be
            drawn for the "faint" sample. Should be ``None`` if ``tri_hist``
            et al. are provided.
        magnum : float, optional
            Zero-indexed column number of the chosen filter limiting magnitude.
            Should be ``None`` if ``tri_hist`` et al. are provided.
        tri_num_faint : integer, optional
            Approximate number of objects to simulate in the chosen filter for
            TRILEGAL simulations. Should be ``None`` if ``tri_hist`` et al.
            are provided.
        trifilterset : string, optional
            Name of the TRILEGAL filter set for which to generate simulations.
            Should be ``None`` if ``tri_hist`` et al. are provided.
        trifiltname : string, optional
            Name of the specific filter to generate perturbation AUF component in.
            Should be ``None`` if ``tri_hist`` et al. are provided.
        tri_hist : numpy.ndarray or None, optional
            If given, array of differential source densities, per square degree
            per magnitude, in the given filter, as computed by
            `~macauff.make_tri_counts`. Must be provided if ``trifolder`` is
            ``None``, else must itself be ``None``.
        tri_mags : numpy.ndarray, optional
            Left-hand magnitude bin edges for each bin in ``tri_hist``. Must be
            given if ``tri_hist`` is provided, else ``None``.
        dtri_mags : numpy.ndarray, optional
            Magnitude bin widths for each ``tri_mags`` bin.  Must be given if
            ``tri_hist`` is provided, else ``None``.
        tri_uncert : numpy.ndarray, optional
            Differential source count uncertainties, as derived by
            `~macauff.make_tri_counts`.  Must be given if ``tri_hist`` is
            provided, else ``None``.
        use_photometric_uncertainties : boolean, optional
            Flag for whether or not to use the photometric uncertainties instead
            of astrometric uncertainties when deriving astrometric uncertainties
            as a function of input precision. Defaults to False (use astrometric
            uncertainties).
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
        return_nm : boolean, optional
            Flag for whether the output correction arrays ``m`` and ``n`` should
            be saved to disk (``False``) or returned by the function (``True``).
        """
        if single_sided_auf is not True:
            raise ValueError("single_sided_auf must be True.")
        if ax_dimension not in (1, 2):
            raise ValueError("ax_dimension must either be '1' or '2'.")
        if npy_or_csv not in ("npy", "csv"):
            raise ValueError("npy_or_csv must either be 'npy' or 'csv'.")
        if coord_or_chunk not in ("coord", "chunk"):
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
        if coord_system not in ("equatorial", "galactic"):
            raise ValueError("coord_system must either be 'equatorial' or 'galactic'.")
        if (pregenerate_cutouts is not None and pregenerate_cutouts is not True and
                pregenerate_cutouts is not False):
            raise ValueError("pregenerate_cutouts should either be 'None', 'True' or 'False'.")
        if pregenerate_cutouts is False and cutout_area is None:
            raise ValueError("cutout_area must be given if pregenerate_cutouts is 'False'.")
        if pregenerate_cutouts is False and cutout_height is None:
            raise ValueError("cutout_height must be given if pregenerate_cutouts is 'False'.")
        if use_photometric_uncertainties is not True and use_photometric_uncertainties is not False:
            raise ValueError("use_photometric_uncertainties must either be True or False.")
        if return_nm is not True and return_nm is not False:
            raise ValueError("return_nm must either be True or False.")
        self.return_nm = return_nm
        self.psf_fwhm = psf_fwhm
        self.numtrials = numtrials
        self.nn_radius = nn_radius
        self.dens_search_radius = dens_search_radius
        # Currently hard-coded as it isn't useful for this work, but is required
        # for calling the perturbation AUF algorithms.
        self.dmcut = [2.5]

        self.psf_radius = 1.185 * self.psf_fwhm
        self.psfsig = self.psf_fwhm / (2 * np.sqrt(2 * np.log(2)))
        self.r, self.rho = np.linspace(0, self.psf_radius, n_r), np.linspace(0, max_rho, n_rho)
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

        self.tri_hist = tri_hist
        self.tri_mags = tri_mags
        self.dtri_mags = dtri_mags
        self.tri_uncert = tri_uncert

        self.dm = dm

        self.dd_params = dd_params
        self.l_cut = l_cut

        self.ax1_mids = ax1_mids
        self.ax2_mids = ax2_mids
        self.ax_dimension = ax_dimension

        self.pregenerate_cutouts = pregenerate_cutouts
        if self.pregenerate_cutouts is not None:
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
            # np.genfromtxt will load in pos_and_err or pos_and_err, mag_ind,
            # mag_unc_ind order for the two catalogues. Each will effectively
            # change its ordering, since we then load [0] for pos_and_err[0][0],
            # etc. for all options. These need saving for np.genfromtxt but
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

        self.use_photometric_uncertainties = use_photometric_uncertainties

        self.j0s = mff.calc_j0(self.rho[:-1]+self.drho/2, self.r[:-1]+self.dr/2)

        self.n_mag_cols = np.ceil(np.sqrt(len(self.mag_array))).astype(int)
        self.n_mag_rows = np.ceil(len(self.mag_array) / self.n_mag_cols).astype(int)

        self.n_filt_cols = np.ceil(np.sqrt(len(self.mag_indices))).astype(int)
        self.n_filt_rows = np.ceil(len(self.mag_indices) / self.n_filt_cols).astype(int)

        for folder in [self.save_folder, f'{self.save_folder}/npy', f'{self.save_folder}/pdf']:
            if not (return_nm and 'npy' in folder):
                if not os.path.exists(folder):
                    os.makedirs(folder)

    # pylint: disable-next=too-many-statements
    def __call__(self, a_cat=None, b_cat=None, a_cat_name=None, b_cat_name=None, a_cat_func=None,
                 b_cat_func=None, tri_download=True, overwrite_all_sightlines=False, make_plots=False,
                 make_summary_plot=True):
        """
        Call function for the correction calculation process.

        Parameters
        ----------
        a_cat : numpy.ndarray or list of numpy.ndarray, optional
            A pre-loaded, or list of pre-loaded, reference catalogue for the
            aiding in the determination the astrometric uncertainties for
            ``b_cat``. Should be provided if ``pregenerate_cutouts`` is
            ``None``, and must be itself ``None`` if ``a_cat_name`` is given.
        b_cat : numpy.ndarray or list of numpy.ndarray, optional
            A pre-loaded, or list of pre-loaded, catalogue for which to
            determination the astrometric uncertainties. Should be provided
            if ``pregenerate_cutouts`` is ``None``, and must be itself ``None``
            if ``a_cat_name`` is given.
        a_cat_name : string, optional
            Name of the catalogue "a" filename, pre-generated or saved by
            ``a_cat_func``. Must accept one or two formats via Python string
            formatting (e.g. ``'a_string_{}'``) that represent ``chunk``, or
            ``ax1_mid`` and ``ax2_mid``, depending on ``coord_or_chunk``. Must
            be given if ``a_cat`` is ``None``.
        b_cat_name : string, optional
            Name of the catalogue "b" filename. Must accept the same string
            formatting as ``a_cat_name``. Must be given if ``b_cat`` is ``None``.
        a_cat_func : callable, optional
            Function used to generate reduced catalogue table for catalogue "a".
            Must be given if ``pregenerate_cutouts`` is ``False``.
        b_cat_func : callable
            Function used to generate reduced catalogue table for catalogue "b".
            Must be given if ``pregenerate_cutouts`` is ``False``.
        tri_download : boolean or None, optional
            Flag determining if TRILEGAL simulations should be re-downloaded
            if they already exist on disk. Should be ``None`` if ``tri_hist``
            et al. were given in initialisation.
        overwrite_all_sightlines : boolean, optional
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
        if (a_cat is None and b_cat is not None) or (a_cat is not None and b_cat is None):
            raise ValueError("a_cat and b_cat must either both be None or both not be None.")
        if (a_cat_name is None and b_cat_name is not None) or (a_cat_name is not None and b_cat_name is None):
            raise ValueError("a_cat_name and b_cat_name must either both be None or both not be None.")
        if self.pregenerate_cutouts is not None and a_cat is not None:
            raise ValueError("pregenerate_cutouts must be None if a_cat is not None.")
        if a_cat_func is not None and a_cat is not None:
            raise ValueError("a_cat_func must be None if a_cat is not None.")
        if b_cat_func is not None and b_cat is not None:
            raise ValueError("b_cat_func must be None if b_cat is not None.")
        if self.pregenerate_cutouts is None and a_cat is None:
            raise ValueError("a_cat must not be None if pregenerate_cutouts is None.")
        if (a_cat is not None and a_cat_name is not None) or (a_cat is None and a_cat_name is None):
            raise ValueError("a_cat and a_cat_name must not both be None or both not be None.")
        if a_cat_func is None and self.pregenerate_cutouts is False:
            raise ValueError("a_cat_func must be given if pregenerate_cutouts is 'False'.")
        if b_cat_func is None and self.pregenerate_cutouts is False:
            raise ValueError("b_cat_func must be given if pregenerate_cutouts is 'False'.")
        if tri_download not in (None, True, False):
            raise ValueError("tri_download must either be True, False, or None.")
        if self.trifolder is not None and tri_download not in (True, False):
            raise ValueError("tri_download must either be True or False if trifolder given.")
        if self.tri_hist is not None and tri_download is not None:
            raise ValueError("tri_download must be None if tri_hist is given.")
        self.a_cat_func = a_cat_func
        self.b_cat_func = b_cat_func
        self.a_cat_name = a_cat_name
        self.b_cat_name = b_cat_name
        if isinstance(a_cat, np.ndarray):
            self.a_cat = [a_cat]
        else:
            self.a_cat = a_cat
        if isinstance(b_cat, np.ndarray):
            self.b_cat = [b_cat]
        else:
            self.b_cat = b_cat

        self.tri_download = tri_download

        self.make_plots = make_plots
        self.make_summary_plot = make_summary_plot

        self.make_ax_coords()

        if self.pregenerate_cutouts is False:
            self.make_catalogue_cutouts()

        # Making coords/cutouts happens for all sightlines, and then we
        # loop through each individually:
        if self.coord_or_chunk == 'coord':
            zip_list = (self.ax1_mids, self.ax2_mids, self.ax1_mins, self.ax1_maxs,
                        self.ax2_mins, self.ax2_maxs)
        else:
            zip_list = (self.ax1_mids, self.ax2_mids, self.ax1_mins, self.ax1_maxs, self.ax2_mins,
                        self.ax2_maxs, self.chunks)

        if (self.return_nm or overwrite_all_sightlines or
                not os.path.isfile(f'{self.save_folder}/npy/snr_mag_params.npy') or
                not os.path.isfile(f'{self.save_folder}/npy/m_sigs_array.npy') or
                not os.path.isfile(f'{self.save_folder}/npy/n_sigs_array.npy')):
            if self.return_nm:
                abc_array = np.empty(dtype=float, shape=(len(self.mag_indices), len(self.ax1_mids), 5))
            else:
                abc_array = open_memmap(f'{self.save_folder}/npy/snr_mag_params.npy', mode='w+',
                                        dtype=float,
                                        shape=(len(self.mag_indices), len(self.ax1_mids), 5))
            abc_array[:, :, :] = -1
            if not self.return_nm:
                m_sigs = open_memmap(f'{self.save_folder}/npy/m_sigs_array.npy', mode='w+',
                                     dtype=float, shape=(len(self.ax1_mids),))
            else:
                m_sigs = np.empty(dtype=float, shape=(len(self.ax1_mids),))
            m_sigs[:] = -1
            if not self.return_nm:
                n_sigs = open_memmap(f'{self.save_folder}/npy/n_sigs_array.npy', mode='w+',
                                     dtype=float, shape=(len(self.ax1_mids),))
            else:
                n_sigs = np.empty(dtype=float, shape=(len(self.ax1_mids),))
            n_sigs[:] = -1
        else:
            abc_array = open_memmap(f'{self.save_folder}/npy/snr_mag_params.npy', mode='r+')
            m_sigs = open_memmap(f'{self.save_folder}/npy/m_sigs_array.npy', mode='r+')
            n_sigs = open_memmap(f'{self.save_folder}/npy/n_sigs_array.npy', mode='r+')

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
            print(f'Running astrometry fits for sightline {index_+1}/{len(self.ax1_mids)}...')

            if self.coord_or_chunk == 'coord':
                ax1_mid, ax2_mid, _, _, _, _ = list_of_things
                cat_args = (ax1_mid, ax2_mid)
                file_name = f'{ax1_mid}_{ax2_mid}'
            else:
                ax1_mid, ax2_mid, _, _, _, _, chunk = list_of_things
                cat_args = (chunk,)
                file_name = f'{chunk}'
            self.list_of_things = list_of_things
            self.cat_args = cat_args
            self.file_name = file_name

            if self.pregenerate_cutouts is None:
                self.a = self.a_cat[index_]
                self.b = self.b_cat[index_]
            else:
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
            m_sig, n_sig = self.fit_uncertainty()
            self.plot_fits_calculate_chi_sq()
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
                if not self.use_photometric_uncertainties:
                    self.ax_b.set_xlabel(r'Input astrometric $\sigma$ / "')
                else:
                    self.ax_b.set_xlabel(r'Input photometric $\sigma$ / "')
                self.ax_b.set_ylabel(r'Fit astrometric $\sigma$ / "')
            else:
                if not self.use_photometric_uncertainties:
                    self.ax_b.set_xlabel(r'Input astrometric sigma / "')
                else:
                    self.ax_b.set_xlabel(r'Input photometric sigma / "')
                self.ax_b.set_ylabel(r'Fit astrometric sigma / "')
            self.finalise_summary_plot()

        if self.return_nm:
            return m_sigs, n_sigs, abc_array
        return None

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
        if self.pregenerate_cutouts is False:
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
                # pylint: disable-next=fixme
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
                if self.pregenerate_cutouts is not None:
                    if not check_b_only:
                        if not os.path.isfile(self.a_cat_name.format(*cat_args)):
                            raise ValueError("If pregenerate_cutouts is 'True' all files must "
                                             f"exist already, but {self.a_cat_name.format(*cat_args)} "
                                             "does not.")
                    if not os.path.isfile(self.b_cat_name.format(*cat_args)):
                        raise ValueError("If pregenerate_cutouts is 'True' all files must "
                                         f"exist already, but {self.b_cat_name.format(*cat_args)} does not.")
                # Check for both files, but assume they are the same size
                # for ax1_min et al. purposes.
                if self.pregenerate_cutouts is None:
                    b = self.b_cat[i]
                else:
                    b = self.load_catalogue('b', cat_args)
                # Handle "minimum" longitude on the interval [-pi, +pi], such that
                # if data straddles the 0-360 boundary we don't return 0 and 360
                # for min and max values.
                self.ax1_mins[i], self.ax1_maxs[i] = min_max_lon(
                    b[:, self.pos_and_err_indices[1][0]])
                self.ax2_mins[i] = np.amin(b[:, self.pos_and_err_indices[1][1]])
                self.ax2_maxs[i] = np.amax(b[:, self.pos_and_err_indices[1][1]])

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
            print(f'Creating catalogue cutouts... {index_+1}/{len(self.ax1_mids)}', end='\r')

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
                ax.set_title(f'{ax1_name} = {ax1_mid}, {ax2_name} = {ax2_mid}, '
                             f'{self.mag_names[j]}\na = {a:.2e}, b = {b:.2e}, c = {c:.2e}', fontsize=15)

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
            plt.savefig(f'{self.save_folder}/pdf/s_vs_snr_{self.file_name}.pdf')
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
        # pylint: disable-next=fixme
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
        # pylint: disable-next=fixme
        # TODO: make the half-mag offset flexible, passing from CrossMatch and/or
        # directly into AstrometricCorrections.
        maxmag = bins[:-1][np.argmax(hist_mag)] - 0.5

        if (self.trifolder is not None and (
                self.tri_download or not
                os.path.isfile(f'{self.trifolder}/{self.triname.format(ax1_mid, ax2_mid)}_faint.dat'))):
            # Have to check for the existence of the folder as well as just
            # the lack of file. However, we can't guarantee that self.triname
            # doesn't contain folder sub-structure (e.g.
            # trifolder/ax1/ax2/file_faint.dat) and hence check generically.
            base_auf_folder = os.path.split(f'{self.trifolder}/'
                                            f'{self.triname.format(ax1_mid, ax2_mid)}_faint.dat')[0]
            if not os.path.exists(base_auf_folder):
                os.makedirs(base_auf_folder, exist_ok=True)

            rect_area = (ax1_max - (ax1_min)) * (
                np.sin(np.radians(ax2_max)) - np.sin(np.radians(ax2_min))) * 180/np.pi

            data_bright_dens = np.sum(~np.isnan(b_mag_data) & (b_mag_data <= maxmag)) / rect_area
            # pylint: disable-next=fixme
            # TODO: un-hardcode min_bright_tri_number
            min_bright_tri_number = 1000
            min_area = min_bright_tri_number / data_bright_dens

            download_trilegal_simulation(self.trifolder, self.trifilterset, ax1_mid, ax2_mid,
                                         self.magnum, self.coord_system, self.maglim_f, min_area,
                                         av=1, sigma_av=0, total_objs=self.tri_num_faint)
            os.system(f'mv {self.trifolder}/trilegal_auf_simulation.dat '
                      f'{self.trifolder}/{self.triname.format(ax1_mid, ax2_mid)}_faint.dat')

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
                av = get_av_infinity(l, b, frame='galactic')[0]
                avs[j, k] = av
        avs = avs.flatten()

        if self.trifolder is not None:
            tri_hist, tri_mags, _, dtri_mags, tri_uncert, _ = make_tri_counts(
                self.trifolder, self.triname.format(ax1_mid, ax2_mid), self.trifiltname, self.dm,
                np.amin(b_mag_data), maxmag, al_av=self.gal_alav, av_grid=avs)
        else:
            tri_hist, tri_mags = self.tri_hist, self.tri_mags
            dtri_mags, tri_uncert = self.dtri_mags, self.tri_uncert

        gal_dns = create_galaxy_counts(
            self.gal_cmau_array, tri_mags+dtri_mags/2, np.linspace(0, 4, 41),
            self.gal_wav_micron, self.gal_alpha0, self.gal_alpha1, self.gal_alphaweight,
            self.gal_ab_offset, self.gal_filtname, self.gal_alav*avs)

        log10y = np.log10(tri_hist + gal_dns)
        new_uncert = np.sqrt(tri_uncert**2 + (0.05*gal_dns)**2)
        dlog10y = 1/np.log(10) * new_uncert / (tri_hist + gal_dns)

        mag_slice = (tri_mags >= minmag) & (tri_mags+dtri_mags <= maxmag)
        n_norm = np.sum(10**log10y[mag_slice] * dtri_mags[mag_slice])
        self.log10y, self.dlog10y = log10y, dlog10y
        self.tri_hist, self.tri_mags, self.dtri_mags = tri_hist, tri_mags, dtri_mags
        self.tri_uncert, self.gal_dns = tri_uncert, gal_dns
        self.minmag, self.maxmag, self.n_norm = minmag, maxmag, n_norm

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
                            (data_mags <= self.maxmag)) / rect_area / self.n_norm

        ax = plt.subplot(gs[0])
        ax1_name = 'l' if self.coord_system == 'galactic' else 'RA'
        ax2_name = 'b' if self.coord_system == 'galactic' else 'Dec'
        ax.set_title(f'{ax1_name} = {ax1_mid}, {ax2_name} = {ax2_mid}')
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
        q = self.tri_hist > 0
        ax.plot((self.tri_mags+self.dtri_mags/2)[q], np.log10(self.tri_hist[q]) +
                np.log10(correction), 'b--')
        q = self.gal_dns > 0
        ax.plot((self.tri_mags+self.dtri_mags/2)[q], np.log10(self.gal_dns[q]) +
                np.log10(correction), 'b:')
        ax.set_ylim(*lims)

        ax.set_xlabel('Magnitude')
        if usetex:
            ax.set_ylabel(r'log$_{10}\left(\mathrm{D}\ /\ \mathrm{mag}^{-1}\,'
                          r'\mathrm{deg}^{-2}\right)$')
        else:
            ax.set_ylabel(r'log10(D / mag^-1 deg^-2)')

        plt.figure('123123')
        plt.tight_layout()
        plt.savefig(f'{self.save_folder}/pdf/counts_comparison_{self.file_name}.pdf')
        plt.close()

    def calculate_local_densities_and_nearest_neighbours(self):
        """
        Calculate local normalising catalogue densities and catalogue-catalogue
        nearest neighbour match pairings for each cutout region.
        """
        print('Creating local densities and nearest neighbour matches...')

        if self.coord_or_chunk == 'coord':
            _, _, ax1_min, ax1_max, ax2_min, ax2_max = self.list_of_things
        else:
            _, _, ax1_min, ax1_max, ax2_min, ax2_max, _ = self.list_of_things

        narray = create_densities(
            self.b, self.minmag, self.maxmag, ax1_min, ax1_max, ax2_min, ax2_max,
            self.dens_search_radius, self.n_pool, self.mag_indices[self.best_mag_index],
            self.pos_and_err_indices[1][0], self.pos_and_err_indices[1][1], self.coord_system)

        _, bmatch, dists = create_distances(
            self.a, self.b, self.nn_radius, self.pos_and_err_indices[0][0], self.pos_and_err_indices[0][1],
            self.pos_and_err_indices[1][0], self.pos_and_err_indices[1][1], self.coord_system)

        # pylint: disable-next=fixme
        # TODO: extend to 3-D search around N-m-sig to find as many good
        # enough bins as possible, instead of only keeping one N-sig bin
        # per magnitude?
        _h, _b = np.histogram(narray[bmatch], bins='auto')
        moden = (_b[:-1]+np.diff(_b)/2)[np.argmax(_h)]
        dn = 0.05*moden

        self.narray, self.dists, self.bmatch = narray, dists, bmatch
        self.moden, self.dn = moden, dn

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
        b_ratio = 0.05
        # Self-consistent, non-zeropointed "flux", based on the relation
        # given in make_snr_model.
        flux = 10**(-1/2.5 * self.mag_array)
        snr = flux / np.sqrt(c * 10**flux + b + (a * 10**flux)**2)
        dm_max = _calculate_magnitude_offsets(
            self.moden*np.ones_like(self.mag_array), self.mag_array, b_ratio, snr, self.tri_mags,
            self.log10y, self.dtri_mags, self.psf_radius, self.n_norm)

        seed = np.random.default_rng().choice(100000, size=(paf.get_random_seed_size(),
                                                            len(self.mag_array)))
        _, _, four_off_fw, _, _ = \
            paf.perturb_aufs(
                self.moden*np.ones_like(self.mag_array), self.mag_array, self.r[:-1]+self.dr/2,
                self.dr, self.r, self.j0s.T, self.tri_mags+self.dtri_mags/2, self.dtri_mags,
                self.log10y, self.n_norm, (dm_max/self.dm).astype(int), self.dmcut, self.psf_radius,
                self.psfsig, self.numtrials, seed, self.dd_params, self.l_cut, 'fw')

        seed = np.random.default_rng().choice(100000, size=(paf.get_random_seed_size(),
                                                            len(self.mag_array)))
        _, _, four_off_ps, _, _ = \
            paf.perturb_aufs(
                self.moden*np.ones_like(self.mag_array), self.mag_array, self.r[:-1]+self.dr/2,
                self.dr, self.r, self.j0s.T, self.tri_mags+self.dtri_mags/2, self.dtri_mags,
                self.log10y, self.n_norm, (dm_max/self.dm).astype(int), self.dmcut, self.psf_radius,
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
        for i, mag in enumerate(self.mag_array):
            mag_cut = ((b_matches[:, mag_ind] <= mag+self.mag_slice[i]) &
                       (b_matches[:, mag_ind] >= mag-self.mag_slice[i]))
            if np.sum(mag_cut) == 0:
                skip_flags[i] = 1
                pdfs.append([-1])
                pdf_uncerts.append([-1])
                q_pdfs.append([-1])
                pdf_bins.append([-1])
                continue
            if not self.use_photometric_uncertainties:
                sig = np.percentile(b_matches[mag_cut, self.pos_and_err_indices[1][2]], 50)
                sig_cut = ((b_matches[:, self.pos_and_err_indices[1][2]] <= sig+self.sig_slice[i]) &
                           (b_matches[:, self.pos_and_err_indices[1][2]] >= sig-self.sig_slice[i]))
            else:
                sig = np.percentile(b_matches[mag_cut, self.mag_unc_indices[self.best_mag_index]],
                                    50)
                sig_cut = ((b_matches[:, self.mag_unc_indices[self.best_mag_index]] <=
                            sig+self.sig_slice[i]) &
                           (b_matches[:, self.mag_unc_indices[self.best_mag_index]] >=
                            sig-self.sig_slice[i]))
            n_cut = (self.narray[self.bmatch] >= self.moden-self.dn) & (
                self.narray[self.bmatch] <= self.moden+self.dn)

            # Since we expect the astrometric/photometric scaling to be roughly
            # a factor FWHM/(2 * sqrt(2 * ln(2))), i.e. the sigma of a
            # Gaussian-ish PSF, scale the "20-sigma" cut accordingly:
            if not self.use_photometric_uncertainties:
                final_slice = sig_cut & mag_cut & n_cut & (self.dists <= 20*sig)
            else:
                final_slice = sig_cut & mag_cut & n_cut & (self.dists <= 20*self.psfsig*sig)
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
            if not self.use_photometric_uncertainties:
                avg_sig[i, 0] = np.median(bm[:, self.pos_and_err_indices[1][2]])
                avg_sig[i, [1, 2]] = np.abs(np.percentile(bm[:, self.pos_and_err_indices[1][2]],
                                                          [16, 84]) -
                                            np.median(bm[:, self.pos_and_err_indices[1][2]]))
            else:
                avg_sig[i, 0] = np.median(bm[:, self.mag_unc_indices[self.best_mag_index]])
                avg_sig[i, [1, 2]] = np.abs(
                    np.percentile(bm[:, self.mag_unc_indices[self.best_mag_index]], [16, 84]) -
                    np.median(bm[:, self.mag_unc_indices[self.best_mag_index]]))

            h, bins = np.histogram(final_dists, bins='auto')
            num = np.sum(h)
            pdf = h / np.diff(bins) / num
            pdf_uncert = np.sqrt(h) / np.diff(bins) / num
            # To avoid binned_statistic NaNing when the data are beyond the
            # edge of the model, we want to limit our bins to the maximum r value.
            q_pdf = (h > 3) & (bins[1:] <= self.r[-1])

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
        For all magnitudes for eachsightline, fit for the empirical centroid
        uncertainty relationship that describes the distribution of match
        separations.

        Returns
        -------
        m : float
            The multiplicative factor for converting old uncertainties to new
            astrometric precision.
        n : float
            Quadratically added constant uncertainty to add to scaling between
            input and output uncertainties.
        """

        print('Creating joint H/sig fits...')

        self.fit_sigs = np.zeros((len(self.mag_array), 2), float)

        res = minimize(self.fit_auf_joint, x0=[1, 0.005], method='L-BFGS-B',
                       options={'ftol': 1e-12}, bounds=[(0, None), (0, None)])
        self.res = res

        cov = res.hess_inv.todense()
        m, n = res.x
        om, on = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])
        for i, pdf in enumerate(self.pdfs):
            if pdf[0] == -1:
                self.skip_flags[i] = 1
                continue
            sig_orig = self.avg_sig[i, 0]
            new_sig = np.sqrt((m * sig_orig)**2 + n**2)
            self.fit_sigs[i, 0] = new_sig
            # sig_sig = sqrt((dsig_dm * sig_m)**2 + (dsig_dn * sig_n)**2);
            # sig = sqrt((m * sig_orig)**2 + n)
            do_dm = m * sig_orig**2 / np.sqrt((m * sig_orig)**2 + n**2)
            do_dn = n / np.sqrt((m * sig_orig)**2 + n**2)
            # Bother propagating standard deviation in sig_orig?
            self.fit_sigs[i, 1] = np.sqrt((do_dm * om)**2 + (do_dn * on)**2)

        return m, n

    def fit_auf_joint(self, p):
        """
        Fits all magnitude slices for a single sightline, modelling
        empirically derived centroid uncertainties in all cases in addition
        to fixed uncertainty contributions.

        Parameters
        ----------
        p : list
            List containing the two scaling values to be fit for.

        Returns
        -------
        chi_sq : float
            Chi-squared of the fit, to be minimised in the wrapping function call.
        """
        if self.coord_or_chunk == 'coord':
            _, _, ax1_min, ax1_max, ax2_min, ax2_max = self.list_of_things
        else:
            _, _, ax1_min, ax1_max, ax2_min, ax2_max, _ = self.list_of_things
        m, n = p

        chi_sq = 0
        ndof = -2
        for i, pdf in enumerate(self.pdfs):
            if self.pdfs[i][0] == -1:
                continue
            (y, dy, q, bins, sig,
             snr) = (pdf, self.pdf_uncerts[i], self.q_pdfs[i], self.pdf_bins[i],
                     self.avg_sig[i, 0], self.avg_snr[i, 0])

            h = 1 - np.sqrt(1 - min(1, self.a_array[self.best_mag_index]**2 * snr**2))
            four_combined = h * self.four_off_fw[:, i] + (1 - h) * self.four_off_ps[:, i]

            o = np.sqrt((m * sig)**2 + n**2)

            four_gauss = np.exp(-2 * np.pi**2 * (self.rho[:-1]+self.drho/2)**2 * o**2)

            convolve_hist = paf.fourier_transform(
                four_combined*four_gauss, self.rho[:-1]+self.drho/2, self.drho, self.j0s)

            convolve_hist_dr = convolve_hist * np.pi * (self.r[1:]**2 - self.r[:-1]**2) / self.dr

            # We have two components to fit to our data: first, some fraction F
            # of the objects do not have a counterpart and therefore follow
            # a nearest-neighbour distribution, n(r), while 1-F of the objects follow
            # an NN-contaminated counterpart distribution, m(r).
            # Here n(r) = 2 pi r rho exp(-pi r^2 rho) and
            # N(r) = \int_0^r n(r') dr' = 1 - exp(-pi r^2 rho).
            # We then have to consider two contributions to our match distribution;
            # the match could either be a counterpart if there's no NN inside of it,
            # or an NN if there's no counterpart inside of it, and hence
            # m(r) = c(r)[1 - N(r)] + n(r)[1 - C(r)], with c(r) the empirical
            # distribution of counterparts and C(r) = \int_0^r c(r') dr'.
            # Finally, the combination y(r) = F n(r) + (1 - F) m(r).

            # For a symmetric nearest neighbour distribution we need to use the
            # combined density of sources -- this is derived from consideration
            # of the probability of no NN object inside r being the chance that
            # NEITHER source has an asymmetric nearest neighbour inside r, or
            # F(r) = (1 - \int a(r) dr) * (1 - \int b(r) dr), where the integrals
            # of a and b, being of the form 2 pi r N exp(-pi r^2 N) are
            # 1 - exp(-pi r^2 N), and hence the survival integrals multiplied
            # together form N' = N_a + N_b, and then you use f(r) = dF/dr.
            # Unit area is cos(t) dt dx for 0 <= t <= 90deg, 0 <= x <= 360 deg,
            # integrated between ax2_min < t < ax2_max, ax1_min < x < ax1_max, converted
            # to degrees.
            rect_area = (ax1_max - (ax1_min)) * (
                np.sin(np.radians(ax2_max)) - np.sin(np.radians(ax2_min))) * 180/np.pi
            avg_a_dens = len(self.a) / rect_area
            density = (self.moden + avg_a_dens) / 3600**2
            nn_model = 2 * np.pi * (self.r[:-1]+self.dr/2) * density * np.exp(
                -np.pi * (self.r[:-1]+self.dr/2)**2 * density)

            m_conv_plus_nn = (convolve_hist_dr * (np.exp(-np.pi * (self.r[:-1]+self.dr/2)**2 *
                              density)) + nn_model * np.cumsum(convolve_hist_dr * self.dr))
            # Normalise m(r) so that it integrates to unity inside R.
            qq = self.r[1:] <= bins[-1]
            int_m_c_p_nn = np.sum(m_conv_plus_nn[qq] * self.dr[qq])
            m_conv_plus_nn /= int_m_c_p_nn

            # Normalise n(r) so it integrates to unity inside R as well.
            reduced_nn_model, _, _ = binned_statistic(
                self.r[:-1]+self.dr/2,
                nn_model / (1 - np.exp(-np.pi * bins[1:][q][-1]**2 * density)), bins=bins)
            reduced_m_conv_plus_nn, _, _ = binned_statistic(self.r[:-1]+self.dr/2,
                                                            m_conv_plus_nn, bins=bins)

            # For the subset of data in the single magnitude slice we need
            # a nearest neighbour fraction, which we can fit for on-the-fly
            # for each slice separately (for the same m/n). Here we have
            # x2 = sum_i ((y_i - f_i(F))**2 / o_yi**2),
            # f_i(F) = F a_i + (1-F) b_i, giving
            # F = [sum_i (y_i - b_i)*(a_i - b_i) / i_yi**2] /
            #      [sum_i (a_i - b_i)**2 / i_yi**2];
            # here a <-> n and b <-> m, and hence
            a, b = reduced_nn_model, reduced_m_conv_plus_nn
            nnf = (np.sum((y[q] - b[q])*(a[q] - b[q]) / dy[q]**2) /
                   np.sum((a[q] - b[q])**2 / dy[q]**2))
            # Since this should be parabolic in F, cap at the bounded limits.
            nnf = min(1, max(0, nnf))

            tot_model = nnf * nn_model / (
                1 - np.exp(-np.pi * bins[1:][q][-1]**2 * density)) + (
                1 - nnf) * m_conv_plus_nn

            modely, _, _ = binned_statistic(self.r[:-1]+self.dr/2, tot_model, bins=bins)

            chi_sq += np.sum((y[q] - modely[q])**2 / dy[q]**2)
            ndof += np.sum(q)

        return chi_sq

    def plot_fits_calculate_chi_sq(self):  # pylint: disable=too-many-locals
        """
        Calculate chi-squared value and create verification plots showing the
        quality of the fits.
        """
        if self.coord_or_chunk == 'coord':
            _, _, ax1_min, ax1_max, ax2_min, ax2_max = self.list_of_things
        else:
            _, _, ax1_min, ax1_max, ax2_min, ax2_max, _ = self.list_of_things
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

        for i, mag in enumerate(self.mag_array):
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
            mag_cut = ((b_matches[:, mag_ind] <= mag+self.mag_slice[i]) &
                       (b_matches[:, mag_ind] >= mag-self.mag_slice[i]))
            if not self.use_photometric_uncertainties:
                bsig = np.percentile(b_matches[mag_cut, pos_err_ind], 50)
                sig_cut = ((b_matches[:, pos_err_ind] <= bsig+self.sig_slice[i]) &
                           (b_matches[:, pos_err_ind] >= bsig-self.sig_slice[i]))
            else:
                bsig = np.percentile(b_matches[mag_cut, self.mag_unc_indices[self.best_mag_index]],
                                     50)
                sig_cut = ((b_matches[:, self.mag_unc_indices[self.best_mag_index]] <=
                            bsig+self.sig_slice[i]) &
                           (b_matches[:, self.mag_unc_indices[self.best_mag_index]] >=
                            bsig-self.sig_slice[i]))
            n_cut = (self.narray[self.bmatch] >= self.moden-self.dn) & (
                self.narray[self.bmatch] <= self.moden+self.dn)
            if not self.use_photometric_uncertainties:
                final_slice = sig_cut & mag_cut & n_cut & (self.dists <= 20*bsig)
            else:
                final_slice = sig_cut & mag_cut & n_cut & (self.dists <= 20*self.psfsig*bsig)
            bm = b_matches[final_slice]

            rect_area = (ax1_max - (ax1_min)) * (
                np.sin(np.radians(ax2_max)) - np.sin(np.radians(ax2_min))) * 180/np.pi
            avg_a_dens = len(self.a) / rect_area
            density = (np.percentile(self.narray[self.bmatch][final_slice], 50) +
                       avg_a_dens) / 3600**2
            nn_model = 2 * np.pi * (self.r[:-1]+self.dr/2) * density * np.exp(
                -np.pi * (self.r[:-1]+self.dr/2)**2 * density)

            fit_sig = self.fit_sigs[i, 0]
            h = 1 - np.sqrt(1 - min(1,
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
            for j, (sig, _h, ls, lab) in enumerate(zip(
                    [fit_sig, fit_sig, fit_sig, bsig, bsig, bsig], [h, 1, 0, h, 1, 0],
                    ['r-', 'r-.', 'r:', 'k-', 'k-.', 'k:'], labels_list)):
                if not self.make_plots and j != 0:
                    continue
                four_gauss = np.exp(-2 * np.pi**2 * (self.rho[:-1]+self.drho/2)**2 * sig**2)

                four_hist = _h * self.four_off_fw[:, i] + (1 - _h) * self.four_off_ps[:, i]
                convolve_hist = paf.fourier_transform(
                    four_hist*four_gauss, self.rho[:-1]+self.drho/2, self.drho, self.j0s)

                convolve_hist_dr = convolve_hist * np.pi * (
                    self.r[1:]**2 - self.r[:-1]**2) / self.dr

                m_conv_plus_nn = (convolve_hist_dr * (np.exp(-np.pi * (self.r[:-1]+self.dr/2)**2 *
                                  density)) + nn_model * np.cumsum(convolve_hist_dr * self.dr))

                # Normalise m(r) so that it integrates to unity inside R.
                qq = self.r[1:] <= pdf_bin[-1]
                int_m_c_p_nn = np.sum(m_conv_plus_nn[qq] * self.dr[qq])
                m_conv_plus_nn /= int_m_c_p_nn

                if j == 0:
                    # Normalise NN distribution so it also integrates to 1 inside R.
                    reduced_nn_model, _, _ = binned_statistic(
                        self.r[:-1]+self.dr/2,
                        nn_model / (1 - np.exp(-np.pi * pdf_bin[1:][q_pdf][-1]**2 * density)),
                        bins=pdf_bin)
                    reduced_m_conv_plus_nn, _, _ = binned_statistic(self.r[:-1]+self.dr/2,
                                                                    m_conv_plus_nn, bins=pdf_bin)

                    a, b = reduced_nn_model, reduced_m_conv_plus_nn
                    nn_frac = (np.sum((pdf[q_pdf] - b[q_pdf])*(a[q_pdf] - b[q_pdf]) /
                                      pdf_uncert[q_pdf]**2) /
                               np.sum((a[q_pdf] - b[q_pdf])**2 / pdf_uncert[q_pdf]**2))
                    nn_frac = min(1, max(0, nn_frac))

                tot_model = nn_frac * nn_model / (
                    1 - np.exp(-np.pi * pdf_bin[1:][q_pdf][-1]**2 * density)) + (
                    1 - nn_frac) * m_conv_plus_nn

                modely, _, _ = binned_statistic(self.r[:-1]+self.dr/2, tot_model, bins=pdf_bin)

                if j == 0:
                    x2s[i, 0] = np.sum((pdf[q_pdf] - modely[q_pdf])**2 / pdf_uncert[q_pdf]**2)
                    # We fit the ensemble for m and n simultaneously, but ignore
                    # that here since we can't define DOF properly for part of
                    # the chi-squared.
                    x2s[i, 1] = np.sum(q_pdf)

                if self.make_plots:
                    ax.plot((pdf_bin[:-1]+np.diff(pdf_bin)/2)[q_pdf], modely[q_pdf], ls, label=lab)

                    if j == 0:
                        _q = (self.r[:-1]+self.dr/2) <= (pdf_bin[:-1]+np.diff(pdf_bin)/2)[q_pdf][-1]
                        ax.plot((self.r[:-1]+self.dr/2)[_q], m_conv_plus_nn[_q], 'g--')
                        ax.plot((self.r[:-1]+self.dr/2)[_q],
                                (convolve_hist_dr[_q] * (np.exp(-np.pi *
                                 (self.r[:-1]+self.dr/2)[_q]**2 * density))) / int_m_c_p_nn, 'g-.')
                        ax.plot((self.r[:-1]+self.dr/2)[_q],
                                (nn_model[_q] * np.cumsum(convolve_hist_dr * self.dr)[_q]) /
                                int_m_c_p_nn, 'g:')

            if self.make_plots:
                r_arr = np.linspace(0, pdf_bin[1:][q_pdf][-1], 1000)
                # NN distribution is 2 pi r N exp(-pi r^2 N),
                # with N converted from sq deg and r in arcseconds;
                # integral to R is 1 - exp(-pi R^2 N)
                nn_dist = 2 * np.pi * r_arr * density * np.exp(-np.pi * r_arr**2 * density)
                ax.plot(r_arr, nn_dist / (1 - np.exp(-np.pi * r_arr[-1]**2 * density)),
                        c='g', ls='-')

                o_sig = self.fit_sigs[i, 1]
                if usetex:
                    ax.set_title(rf'mag = {mag}, H = {h:.2f}, '
                                 rf'$\sigma$ = {fit_sig:.3f}$\pm${o_sig:.3f} ({bsig:.3f})"; '
                                 rf'$F$ = {nn_frac:.2f}; SNR = {self.avg_snr[i, 0]:.2f}'
                                 rf'$^{{+{self.avg_snr[i, 2]:.2f}}}_{{-{self.avg_snr[i, 1]:.2f}}}$; '
                                 rf'N = {len(bm)}', fontsize=22)
                else:
                    ax.set_title(rf'mag = {mag}, H = {h:.2f}, '
                                 rf'sigma = {fit_sig:.3f}+/-{o_sig:.3f} ({bsig:.3f})"; '
                                 rf'F = {nn_frac:.2f}; SNR = {self.avg_snr[i, 0]:.2f}'
                                 rf'+{self.avg_snr[i, 2]:.2f}/-{self.avg_snr[i, 1]:.2f}; '
                                 rf'N = {len(bm)}', fontsize=22)
                ax.legend(fontsize=15)
                ax.set_xlabel('Radius / arcsecond')
                if usetex:
                    ax.set_ylabel('PDF / arcsecond$^{-1}$')
                else:
                    ax.set_ylabel('PDF / arcsecond^-1')
        if self.make_plots:
            plt.tight_layout()
            plt.savefig(f'{self.save_folder}/pdf/auf_fits_{self.file_name}.pdf')
            plt.close()

        self.x2s = x2s

    def finalise_summary_plot(self):
        """
        After running all of the sightlines' fits, generate a final
        summary plot of the sig-sig relations, quality of fits, and
        resulting "m" and "n" scaling parameters.
        """
        # pylint: disable-next=fixme
        # TODO: put chi-squred distribution comparison back, but that needs
        # self.x2s saving out somehow so that we can skip to the end and just
        # run finalise_summary_plot on re-runs when we don't need to fit any
        # astrometry, and currently that isn't possible.

        # ax_d = plt.subplot(self.gs[1])
        # q = ~np.isnan(self.x2s[:, 0])
        # chi_sqs, dofs = self.x2s[:, 0][q].flatten(), self.x2s[:, 1][q].flatten()
        # chi_sq_cdf = np.empty(np.sum(q), float)
        # for i, (chi_sq, dof) in enumerate(zip(chi_sqs, dofs)):
        #     chi_sq_cdf[i] = chi2.cdf(chi_sq, dof)
        # # Under the hypothesis all CDFs are "true" we should expect
        # # the distribution of CDFs to be linear with fraction of the way
        # # through the sorted list of CDFs -- that is, the CDF ranked 30%
        # # in order should be ~0.3.
        # q_sort = np.argsort(chi_sq_cdf)
        # filter_log_nans = chi_sq_cdf[q_sort] < 1
        # true_hypothesis_cdf_dist = (np.arange(1, len(chi_sq_cdf)+1, 1) - 0.5) / len(chi_sq_cdf)
        # ax_d.plot(np.log10(1 - chi_sq_cdf[q_sort][filter_log_nans]),
        #           true_hypothesis_cdf_dist[filter_log_nans], 'k.')
        # ax_d.plot(np.log10(1 - true_hypothesis_cdf_dist), true_hypothesis_cdf_dist, 'r--')
        # if usetex:
        #     ax_d.set_xlabel(r'$\log_{10}(1 - \mathrm{CDF})$')
        # else:
        #     ax_d.set_xlabel(r'log10(1 - CDF)')
        # ax_d.set_ylabel('Fraction')

        for i, (f, label) in enumerate(zip([self.m_sigs, self.n_sigs], ['m', 'n'])):
            ax = plt.subplot(self.gs[i+2])
            img = ax.scatter(self.ax1_mids, self.ax2_mids, c=f, cmap='viridis')
            c = plt.colorbar(img, ax=ax, use_gridspec=True)
            c.ax.set_ylabel(label)
            ax1_name = 'l' if self.coord_system == 'galactic' else 'RA'
            ax2_name = 'b' if self.coord_system == 'galactic' else 'Dec'
            ax.set_xlabel(f'{ax1_name} / deg')
            ax.set_ylabel(f'{ax2_name} / deg')

        plt.tight_layout()
        plt.savefig(f'{self.save_folder}/pdf/sig_h_stats.pdf')
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
            x = np.genfromtxt(name, delimiter=',', usecols=cols)

        return x


def create_densities(b, minmag, maxmag, ax1_min, ax1_max, ax2_min, ax2_max, search_radius,
                     n_pool, mag_ind, ax1_ind, ax2_ind, coord_system):
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
    ax1_min : float
        The minimum longitude of the box of the cutout region.
    ax1_max : float
        The maximum longitude of the box of the cutout region.
    ax2_min : float
        The minimum latitude of the box of the cutout region.
    ax2_max : float
        The maximum latitude of the box of the cutout region.
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

    Returns
    -------
    narray : numpy.ndarray
        The density of objects within ``search_radius`` degrees of each object
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

    r = (2 * np.sin(Angle(search_radius * u.degree) / 2.0)).value
    overlap_number = np.empty(len(b), int)

    counter = np.arange(0, len(b))
    iter_group = zip(counter, itertools.repeat([full_ucoords, mag_cut_kdt, r]))
    with multiprocessing.Pool(n_pool) as pool:
        for stuff in pool.imap_unordered(ball_point_query, iter_group, chunksize=len(b)//n_pool):
            i, len_query = stuff
            overlap_number[i] = len_query

    pool.join()

    area = paf.get_circle_area_overlap(b[:, ax1_ind], b[:, ax2_ind], search_radius,
                                       ax1_min, ax1_max, ax2_min, ax2_max)

    narray = overlap_number / area

    return narray


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


def create_distances(a, b, nn_radius, a_ax1_ind, a_ax2_ind, b_ax1_ind, b_ax2_ind, coord_system):
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
    nn_radius : float
        Maximum match radius within which to consider potential counterpart
        assignments, in arcseconds.
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

    return amatch, bmatch, dists


class SNRMagnitudeRelationship(AstrometricCorrections):  # pylint: disable=too-many-instance-attributes
    """
    Class to derive signal-to-noise ratio-magnitude relationships for sources
    in photometric catalogues.
    """
    # pylint: disable-next=super-init-not-called
    def __init__(self, save_folder, ax1_mids, ax2_mids, ax_dimension, npy_or_csv, coord_or_chunk,
                 pos_and_err_indices, mag_indices, mag_unc_indices, mag_names, coord_system,
                 chunks=None, return_nm=False):
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
        return_nm : boolean, optional
            Flag for whether the output correction arrays ``m`` and ``n`` should
            be saved to disk (``False``) or returned by the function (``True``).
        """
        if ax_dimension not in (1, 2):
            raise ValueError("ax_dimension must either be '1' or '2'.")
        if npy_or_csv not in ("npy", "csv"):
            raise ValueError("npy_or_csv must either be 'npy' or 'csv'.")
        if coord_or_chunk not in ("coord", "chunk"):
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
        if coord_system not in ("equatorial", "galactic"):
            raise ValueError("coord_system must either be 'equatorial' or 'galactic'.")
        if return_nm is not True and return_nm is not False:
            raise ValueError("return_nm must either be True or False.")

        self.save_folder = save_folder

        self.ax1_mids = ax1_mids
        self.ax2_mids = ax2_mids
        self.ax_dimension = ax_dimension

        self.npy_or_csv = npy_or_csv
        self.coord_or_chunk = coord_or_chunk
        self.chunks = chunks

        self.coord_system = coord_system

        self.return_nm = return_nm

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

        for folder in [self.save_folder, f'{self.save_folder}/npy', f'{self.save_folder}/pdf']:
            if not os.path.exists(folder):
                os.makedirs(folder)

    # pylint: disable-next=signature-differs
    def __call__(self, b_cat=None, b_cat_name=None, overwrite_all_sightlines=False, make_plots=False):
        """
        Call function for the correction calculation process.

        Parameters
        ----------
        b_cat : numpy.ndarray or list of numpy.ndarray, optional
            A pre-loaded, or list of pre-loaded, catalogue for which to
            determination the SNR-mag relation. Must be ``None`` if
            ``b_cat_name`` is given.
        b_cat_name : string, optional
            Name of the catalogue "b" filename, pre-generated. Must accept one
            or two formats via Python string formatting (e.g. ``'b_string_{}'``)
            that represent ``chunk``, or ``ax1_mid`` and ``ax2_mid``, depending
            on ``coord_or_chunk``.
        overwrite_all_sightlines : boolean, optional
            Flag for whether to create a totally fresh run of astrometric
            corrections, regardless of whether ``abc_array`` is saved on disk.
            Defaults to ``False``.
        make_plots : boolean, optional
            Determines if intermediate figures are generated in the process
            of deriving astrometric corrections.
        """
        if (b_cat is None and b_cat_name is None):
            raise ValueError("One of b_cat and b_cat_name must be given.")
        if (b_cat is not None and b_cat_name is not None):
            raise ValueError("Only one of b_cat and b_cat_name should be given.")
        if isinstance(b_cat, np.ndarray):
            self.b_cat = [b_cat]
        else:
            self.b_cat = b_cat
        self.b_cat_name = b_cat_name
        self.make_plots = make_plots

        # Force pregenerate_cutouts for super purposes.
        if self.b_cat is None:
            self.pregenerate_cutouts = True
        else:
            self.pregenerate_cutouts = None
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
                not os.path.isfile(f'{self.save_folder}/npy/snr_mag_params.npy')):
            abc_array = open_memmap(f'{self.save_folder}/npy/snr_mag_params.npy', mode='w+',
                                    dtype=float,
                                    shape=(len(self.mag_indices), len(self.ax1_mids), 5))
            abc_array[:, :, :] = -1
        else:
            abc_array = open_memmap(f'{self.save_folder}/npy/snr_mag_params.npy', mode='r+')

        for index_, list_of_things in enumerate(zip(*zip_list)):
            if not np.all(abc_array[:, index_, :] == [-1, -1, -1, -1, -1]):
                continue
            print(f'Running SNR-mag fits for sightline {index_+1}/{len(self.ax1_mids)}...')

            if self.coord_or_chunk == 'coord':
                ax1_mid, ax2_mid, _, _, _, _ = list_of_things
                cat_args = (ax1_mid, ax2_mid)
                file_name = f'{ax1_mid}_{ax2_mid}'
            else:
                ax1_mid, ax2_mid, _, _, _, _, chunk = list_of_things
                cat_args = (chunk,)
                file_name = f'{chunk}'
            self.list_of_things = list_of_things
            self.cat_args = cat_args
            self.file_name = file_name

            if self.b_cat is None:
                self.b = self.load_catalogue('b', self.cat_args)
            else:
                self.b = self.b_cat[index_]

            self.a_array, self.b_array, self.c_array = self.make_snr_model()
            abc_array[:, index_, 0] = self.a_array
            abc_array[:, index_, 1] = self.b_array
            abc_array[:, index_, 2] = self.c_array
            abc_array[:, index_, 3] = ax1_mid
            abc_array[:, index_, 4] = ax2_mid

        if self.return_nm:
            return abc_array
        return None
