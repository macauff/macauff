# Licensed under a 3-clause BSD style license - see LICENSE
'''
Module for calculating corrections to astrometric uncertainties of photometric catalogues, by
fitting their AUFs and centroid uncertainties across ensembles of matches between well-understood
catalogue and one for which precisions are less well known.
'''
# pylint: disable=too-many-lines
# pylint: disable=duplicate-code

import datetime
import os
import shutil
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord, match_coordinates_sky
from matplotlib import gridspec
from numpy.lib.format import open_memmap
from scipy.optimize import minimize
from scipy.special import factorial
from scipy.stats import binned_statistic, poisson

# Assume that usetex = False only applies for tests where no TeX is installed
# at all, instead of users having half-installed TeX, dvipng et al. somewhere.
usetex = not not shutil.which("tex")  # pylint: disable=unneeded-not
if usetex:
    plt.rcParams.update({"text.usetex": True, "text.latex.preamble": r"\usepackage{amsmath}"})

# pylint: disable=wrong-import-position,import-error,no-name-in-module
from macauff.galaxy_counts import create_galaxy_counts
from macauff.misc_functions import (
    convex_hull_area,
    create_densities,
    find_model_counts_corrections,
    generate_avs_inside_hull,
)
from macauff.misc_functions_fortran import misc_functions_fortran as mff
from macauff.parse_catalogue import apply_proper_motion
from macauff.perturbation_auf import (
    _calculate_magnitude_offsets,
    download_trilegal_simulation,
    make_tri_counts,
)
from macauff.perturbation_auf_fortran import perturbation_auf_fortran as paf

# pylint: enable=wrong-import-position,import-error,no-name-in-module

__all__ = ['AstrometricCorrections']


def derive_astrometric_corrections(self, which):
    """
    Wrapper to set various parameters and call AstrometricCorrections,
    for either catalogue "a" or "b".

    Parameters
    ----------
    which : string
        Either 'a' or 'b', indicating which side catalogue to fit.
    """
    # Generate from current data: just need the singular chunk mid-points
    # and to leave all other parameters as they are.
    if len(getattr(self, f'{which}_auf_region_points')) > 1:
        warnings.warn(f"{which}_auf_region_points contains more than one AUF sampling point, but "
                      f"{which}_correct_astrometry is True. Check results carefully.")
    ax1_mids = np.array([getattr(self, f'{which}_auf_region_points')[0, 0]])
    ax2_mids = np.array([getattr(self, f'{which}_auf_region_points')[0, 1]])
    ax_dimension = 2
    a_npy_or_csv = 'csv'
    a_coord_or_chunk = 'chunk'
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{t} Rank {self.rank}, chunk {self.chunk_id}: Calculating catalogue 'a' "
          "uncertainty corrections...")
    apply_pm = (getattr(self, f'{which}_apply_proper_motion') or
                getattr(self, f'{which}_ref_apply_proper_motion'))
    if apply_pm:
        pm_cols, pm_ref_epoch_or_index = [None, None], [None, None]
        pm_move_to_epoch = self.move_to_epoch
        if getattr(self, f'{which}_apply_proper_motion'):
            pm_cols[0] = getattr(self, f'{which}_pm_indices')
            pm_ref_epoch_or_index[0] = getattr(self, f'{which}_ref_epoch_or_index')
        if getattr(self, f'{which}_ref_apply_proper_motion'):
            pm_cols[1] = getattr(self, f'{which}_ref_pm_indices')
            pm_ref_epoch_or_index[1] = getattr(self, f'{which}_ref_ref_epoch_or_index')
    else:
        pm_cols, pm_ref_epoch_or_index, pm_move_to_epoch = None, None, None
    ac = AstrometricCorrections(
        getattr(self, f'{which}_psf_fwhms'), self.num_trials, getattr(self, f'{which}_nn_radius'),
        getattr(self, f'{which}_dens_dist'), getattr(self, f'{which}_correct_astro_save_folder'),
        getattr(self, f'{which}_gal_wavs'), getattr(self, f'{which}_gal_aboffsets'),
        getattr(self, f'{which}_gal_filternames'), getattr(self, f'{which}_gal_al_avs'), self.d_mag,
        getattr(self, 'dd_params'), getattr(self, 'l_cut'), ax1_mids, ax2_mids,
        ax_dimension, getattr(self, f'{which}_correct_mag_array'),
        getattr(self, f'{which}_correct_mag_slice'), getattr(self, f'{which}_correct_sig_slice'), self.n_pool,
        a_npy_or_csv, a_coord_or_chunk, getattr(self, f'{which}_pos_and_err_indices'),
        getattr(self, f'{which}_mag_indices'), getattr(self, f'{which}_snr_indices'),
        getattr(self, f'{which}_filt_names'), getattr(self, f'{which}_correct_astro_mag_indices_index'),
        getattr(self, f'{which}_auf_region_frame'), getattr(self, f'{which}_saturation_magnitudes'),
        trifilepath=getattr(self, f'{which}_auf_file_path'),
        maglim_f=getattr(self, f'{which}_tri_maglim_faint'), magnum=getattr(self, f'{which}_tri_filt_num'),
        tri_num_faint=getattr(self, f'{which}_tri_num_faint'),
        trifilterset=getattr(self, f'{which}_tri_set_name'),
        trifiltnames=getattr(self, f'{which}_tri_filt_names'),
        tri_dens_cube=getattr(self, f'{which}_tri_dens_cube'),
        tri_dens_array=getattr(self, f'{which}_tri_dens_array'),
        use_photometric_uncertainties=getattr(self, f'{which}_use_photometric_uncertainties'),
        pregenerate_cutouts=True, chunks=[self.chunk_id], n_r=self.real_hankel_points,
        n_rho=self.four_hankel_points, max_rho=self.four_max_rho,
        mn_fit_type=getattr(self, f'{which}_mn_fit_type'), apply_proper_motion_flag=apply_pm,
        pm_indices=pm_cols, pm_ref_epoch_or_index=pm_ref_epoch_or_index, pm_move_to_epoch=pm_move_to_epoch)
    ac(a_cat_name=getattr(self, f'{which}_ref_cat_csv_file_path'),
       b_cat_name=getattr(self, f'{which}_cat_csv_file_path'),
       tri_download=getattr(self, f'{which}_download_tri'), make_plots=True, overwrite_all_sightlines=True,
       seeing_ranges=getattr(self, f'{which}_seeing_ranges'))


# pylint: disable=too-many-instance-attributes,too-many-statements,too-many-locals
# pylint: disable=too-many-arguments,too-many-positional-arguments
class AstrometricCorrections:
    """
    Class to calculate any potential corrections to quoted astrometric
    precisions in photometric catalogues, based on reliable cross-matching
    to a well-understood second dataset.
    """
    def __init__(self, psf_fwhms, numtrials, nn_radius, dens_search_radius, save_folder,
                 gal_wavs_micron, gal_ab_offsets, gal_filtnames, gal_alavs, dm, dd_params, l_cut,
                 ax1_mids, ax2_mids, ax_dimension, mag_arrays, mag_slices, sig_slices, n_pool,
                 npy_or_csv, coord_or_chunk, pos_and_err_indices, mag_indices, snr_indices,
                 mag_names, correct_astro_mag_indices_index, coord_system, saturation_magnitudes,
                 pregenerate_cutouts, n_r, n_rho, max_rho, mn_fit_type, trifilepath=None, maglim_f=None,
                 magnum=None, tri_num_faint=None, trifilterset=None, trifiltnames=None, tri_dens_cube=None,
                 tri_dens_array=None, use_photometric_uncertainties=False, cutout_area=None,
                 cutout_height=None, single_sided_auf=True, chunks=None, return_nm=False,
                 apply_proper_motion_flag=False, pm_indices=None, pm_ref_epoch_or_index=None,
                 pm_move_to_epoch=None):
        """
        Initialisation of AstrometricCorrections, accepting inputs required for
        the running of the optimisation and parameterisation of astrometry of
        a photometric catalogue, benchmarked against a catalogue of much higher
        astrometric resolution and precision, such as Gaia or the Hubble Source
        Catalog.

        Parameters
        ----------
        psf_fwhms : list or numpy.ndarray of floats
            The full-widths at half-maximum of the Point Spread Functions, used to
            determine the sizes of the PSF for perturber placement purposes.
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
        gal_wavs_micron : list or numpy.ndarray of floats
            Wavelength, in microns, of the chosen filters, for use in
            simulating galaxy counts.
        gal_ab_offsets : list or numpy.ndarray of floats
            The offsets between the filter zero points and the AB magnitude
            offset.
        gal_filtnames : list or numpy.ndarray of strings
            Names of the filter in the ``speclite`` compound naming convention.
        gal_alavs : list or numpy.ndarray of floats
            Differential reddening vector of the given filters.
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
        mag_arrays : list of numpy.ndarrays or numpy.ndarray
            List of lists of magnitudes in the filter to derive astrometry for,
            with each element of ``mag_arrays[i]`` a list or array of magnitudes
            at which to take cuts in the corresponding ``mag_names[i]`` filter.
        mag_slices : list of numpy.ndarrays or numpy.ndarray
            Widths of interval at which to take slices of magnitudes for deriving
            astrometric properties. Each ``mag_slices[i]`` maps elementwise to each
            ``mag_array[i]``, and hence they should be the same shape, with the two
            lists or arrays meeting ``len(mag_arrays) == len(mag_slices)``.
        sig_slices : list of numpy.ndarrays or numpy.ndarray
            Interval widths of quoted astrometric uncertainty to use when
            isolating individual sets of objects for AUF derivation. Length
            should match ``mag_array``. List must have the same number of elements
            as ``mag_arrays``, with each list-element agreeing in length as well;
            if a numpy array, then
            ``np.all(mag_arrays.shape == sig_slices.shape)`` must be true.
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
            In order, the indices within catalogue "b" and then "a" respecetively
            of either the .npy or .csv file of the longitudinal (e.g. RA or l),
            latitudinal (Dec or b), and *singular*, circular astrometric
            precision array. Coordinates should be in degrees while precision
            should be in the same units as ``sig_slice`` and those of the
            nearest-neighbour distances, likely arcseconds. Note that coordinates
            are reversed between calogues: for example, ``[[6, 3, 0], [0, 1, 2]]``
            is the case where catalogue "a" has its coordinates in the first
            three columns in RA/Dec/Err order, while catalogue "b" has its
            coordinates in a more random order. If ``use_photometric_uncertainties``
            is ``True`` then the third index of the first triplet (i.e., the
            index for the to-be-fit catalogue's uncertainties) should be that of
            a photometric uncertainty rather than an astrometric uncertainty.
        mag_indices : list or numpy.ndarray
            In appropriate order, as expected by e.g. `~macauff.CrossMatch` inputs
            and `~macauff.make_perturb_aufs`, list the indexes of each magnitude
            column within either the ``.npy`` or ``.csv`` file loaded for each
            sub-catalogue sightline. These should be zero-indexed.
        snr_indices : list or numpy.ndarray
            For each ``mag_indices`` entry, the corresponding signal-to-noise
            ratio index in the catalogue.
        mag_names : list or numpy.ndarray of strings
            Names of each ``mag_indices`` magnitude.
        correct_astro_mag_indices_index : integer
            Index into ``mag_indices`` of the preferred magnitude to use to
            construct astrometric scaling relations from. Should generally
            be the one with the most coverage across all detections in a
            catalogue, or the one with the most precise fluxes.
        coord_system : string, "equatorial" or "galactic"
            Identifier of which coordinate system the data are in. Both datasets
            must be in the same system, which can either be RA/Dec (equatorial)
            or l/b (galactic) coordinates.
        saturation_magnitudes : list or numpy.ndarray
            List of magnitudes brighter than which the given survey suffers
            saturation, each element matching the filter of ``mag_indices``.
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
        mn_fit_type : string
            Determines whether we perform quadratic or linear scaling for
            hyper-parameter fits to data-driven vs quoted astrometric
            uncertainties. Must either be "quadratic" or "linear."
        trifilepath : string, optional
            Filepath of the location into which to save TRILEGAL simulations. If
            provided ``tri_dens_cube`` and ``tri_dens_array`` must be
            ``None``, and ``maglim_fs``, ``magnums``, ``tri_num_faints``,
            ``trifilterset``, and ``trifiltnames`` must be given. Must contain
            two format ``{}`` options in string, for unique ax1-ax2 sightline
            combination downloads.
        maglim_f : float, optional
            Magnitude in the ``magnum`` filter down to which sources should be
            drawn for the "faint" sample. Should be ``None`` if ``tri_dens_cube``
            and ``tri_dens_array`` are provided.
        magnum : float, optional
            Zero-indexed column number of the chosen filter's limiting magnitude.
            Should be ``None`` if ``tri_dens_cube`` and ``tri_dens_array`` are
            provided.
        tri_num_faint : integer, optional
            Approximate number of objects to simulate in the chosen filter for
            TRILEGAL simulations. Should be ``None`` if ``tri_dens_cube`` and
            ``tri_dens_array`` are provided.
        trifilterset : string, optional
            Name of the TRILEGAL filter set for which to generate simulations.
            Should be ``None`` if ``tri_dens_cube`` and ``tri_dens_array`` are
            provided.
        trifiltnames : list of string, optional
            Name of the specific filters to generate perturbation AUF component in.
            Should be ``None`` if ``tri_dens_cube`` and ``tri_dens_array`` are
            provided.
        tri_dens_cube : numpy.ndarray or None, optional
            If given, array of differential source densities, per square degree
            per magnitude, along with magnitude bins and intervals, for a set
            of given filters and sky positions, as computed by
            `~macauff.make_tri_counts`. Must be provided if ``trifilepath`` is
            ``None``, else must itself be ``None``.
        tri_dens_array : numpy.ndarray or None, optional
            Corresponding sky-coordinate array to extract the relevant column
            from ``tri_dens_cube``. Must be given if ``tri_dens_cub`` is
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
        apply_proper_motion_flag : boolean, optional
            Flag to indicate if either dataset has proper motions that must be
            applied to its positions before determining astrometric corrections.
            Defaults to False, in which case neither dataset will check for, or
            load, proper motion-based data.
        pm_indices : list of list or numpy.ndarray of integers, optional
            If given, must contain a list of two elements, each of which contains
            the two orthogonal sky-axis proper motions' indices for the given
            dataset, in the same reference frame as the positions to be loaded
            along with positions, SNRs, photometry, as necessary. As this is
            on a per-catalogue basis, if either (or both) dataset has no motion
            information, ``None`` must be given, with the first element of the
            list being, to be consistent with ``pos_and_err_indices``, the "b",
            to-be-fit, catalogue, with the second element the "a", "truth" dataset.
            We could have e.g. ``[None, [4, 5]]`` for the case where we apply
            motion information to our truth catalogue, with proper motion
            information being stored in the 5th and 6th columns.
        pm_ref_epoch_or_index : list of strings or integers, optional
            If provided, necessary when ``pm_indices`` is passed, each element,
            consistent with ``pm_indices`` catalogue ordering, must either be
            a single string, valid for loading into astropy's Time function,
            representing a common date of observation for all sources in the file
            or a single integer, loading from the dataset the astropy Time-valid
            strings for each object individually as a column of the input file.
            Again, if no motions are to be applied for an individual catalogue
            but ``apply_proper_motion_flag`` is ``True``, then ``None`` must be
            given for that list element; for the above example we could either
            supply ``[None, 6]`` or ``[None, 'J2000']`` to load per-source
            epochs from our file or supply a fixed observation date respectively.
        pm_move_to_epoch : string, optional
            If ``pm_indices`` is provided this must be given, containing a
            single, astropy Time valid, string, the final epoch to apply proper
            motions of all objects to. This must be a single string, rather than
            a per-catalogue list, to ensure that a common epoch is used when
            proper motions are to be applied to both datasets.
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
        if mn_fit_type not in ("quadratic", "linear"):
            raise ValueError("mn_fit_type must either be 'quadratic' or 'linear'.")
        if apply_proper_motion_flag and np.any([q is None for q in
                                               [pm_indices, pm_ref_epoch_or_index, pm_move_to_epoch]]):
            raise ValueError("apply_proper_motion_flag cannot be True without supplying pm_indices, "
                             "pm_ref_epoch_or_index, and pm_move_to_epoch.")
        self.return_nm = return_nm
        self.psf_fwhms = psf_fwhms
        self.numtrials = numtrials
        self.nn_radius = nn_radius
        self.dens_search_radius = dens_search_radius
        # Currently hard-coded as it isn't useful for this work, but is required
        # for calling the perturbation AUF algorithms.
        self.dmcut = [2.5]

        self.n_r, self.max_rho, self.n_rho = n_r, max_rho, n_rho

        self.save_folder = save_folder

        self.trifilepath = trifilepath
        self.maglim_f = maglim_f
        self.magnum = magnum
        self.tri_num_faint = tri_num_faint
        self.trifilterset = trifilterset
        self.trifiltnames = trifiltnames
        self.gal_wavs_micron = gal_wavs_micron
        self.gal_ab_offsets = gal_ab_offsets
        self.gal_filtnames = gal_filtnames
        self.gal_alavs = gal_alavs

        self.tri_dens_cube = tri_dens_cube
        self.tri_dens_array = tri_dens_array

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

        self.mag_arrays = [np.array(x) for x in mag_arrays]
        self.mag_slices = [np.array(x) for x in mag_slices]
        self.sig_slices = [np.array(x) for x in sig_slices]

        self.saturation_magnitudes = np.array(saturation_magnitudes)

        self.npy_or_csv = npy_or_csv
        self.coord_or_chunk = coord_or_chunk
        self.chunks = chunks

        self.coord_system = coord_system

        self.mn_fit_type = mn_fit_type

        self.apply_proper_motion_flag = apply_proper_motion_flag
        self.pm_move_to_epoch = pm_move_to_epoch

        if npy_or_csv == 'npy':
            self.pos_and_err_indices_full = pos_and_err_indices
            self.mag_indices = mag_indices
            self.snr_indices = snr_indices
            self.mag_names = mag_names
            self.correct_astro_mag_indices_index = correct_astro_mag_indices_index
            if apply_proper_motion_flag:
                self.pm_indices = pm_indices
                self.pm_ref_epoch_or_index = pm_ref_epoch_or_index
        else:
            # np.genfromtxt will load in pos_and_err or pos_and_err, mag_ind,
            # snr_ind order for the two catalogues. Each will effectively
            # change its ordering, since we then load [0] for pos_and_err[0][0],
            # etc. for all options. These need saving for np.genfromtxt but
            # also for obtaining the correct column in the resulting sub-set of
            # the loaded csv file.
            self.a_cols = np.array(pos_and_err_indices[1])
            self.b_cols = np.concatenate((pos_and_err_indices[0], mag_indices, snr_indices))
            if apply_proper_motion_flag:
                if pm_indices[0] is not None:
                    self.b_cols = np.concatenate((self.b_cols, pm_indices[0]))
                    if isinstance(pm_ref_epoch_or_index[0], int):
                        self.b_cols = np.concatenate((self.b_cols, pm_ref_epoch_or_index[0]))
                if pm_indices[1] is not None:
                    self.a_cols = np.concatenate((self.a_cols, pm_indices[1]))
                    if isinstance(pm_ref_epoch_or_index[1], int):
                        self.a_cols = np.concatenate((self.a_cols, pm_ref_epoch_or_index[1]))
            self.pos_and_err_indices_full = [
                [np.argmin(np.abs(q - self.b_cols)) for q in pos_and_err_indices[0]],
                [np.argmin(np.abs(q - self.a_cols)) for q in pos_and_err_indices[1]]]
            self.mag_indices = [np.argmin(np.abs(q - self.b_cols)) for q in mag_indices]
            self.snr_indices = [np.argmin(np.abs(q - self.b_cols)) for q in snr_indices]

            self.mag_names = mag_names
            self.correct_astro_mag_indices_index = correct_astro_mag_indices_index

            if apply_proper_motion_flag:
                self.pm_indices = [None, None]
                self.pm_ref_epoch_or_index = [None, None]
                if pm_indices[0] is not None:
                    self.pm_indices[0] = [np.argmin(np.abs(q - self.b_cols)) for q in pm_indices[0]]
                    if isinstance(pm_ref_epoch_or_index[0], int):
                        self.pm_ref_epoch_or_index[0] = np.argmin(np.abs(pm_ref_epoch_or_index[0] -
                                                                         self.b_cols))
                    else:
                        self.pm_ref_epoch_or_index[0] = pm_ref_epoch_or_index[0]
                if pm_indices[1] is not None:
                    self.pm_indices[1] = [np.argmin(np.abs(q - self.a_cols)) for q in pm_indices[1]]
                    if isinstance(pm_ref_epoch_or_index[1], int):
                        self.pm_ref_epoch_or_index[1] = np.argmin(np.abs(pm_ref_epoch_or_index[1] -
                                                                         self.a_cols))
                    else:
                        self.pm_ref_epoch_or_index[1] = pm_ref_epoch_or_index[1]

        self.n_pool = n_pool

        self.use_photometric_uncertainties = use_photometric_uncertainties

        self.n_filt_cols = np.ceil(np.sqrt(len(self.mag_indices))).astype(int)
        self.n_filt_rows = np.ceil(len(self.mag_indices) / self.n_filt_cols).astype(int)

        for folder in [self.save_folder, f'{self.save_folder}/npy', f'{self.save_folder}/pdf']:
            if not (return_nm and 'npy' in folder):
                if not os.path.exists(folder):
                    os.makedirs(folder)

    # pylint: disable-next=too-many-statements,too-many-branches
    def __call__(self, a_cat=None, b_cat=None, a_cat_name=None, b_cat_name=None, a_cat_func=None,
                 b_cat_func=None, tri_download=True, overwrite_all_sightlines=False, make_plots=False,
                 make_summary_plot=True, seeing_ranges=None, single_or_repeat="single",
                 repeat_unique_visits_list=None):
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
            if they already exist on disk. Should be ``None`` if
            ``tri_dens_cube`` was given in initialisation.
        overwrite_all_sightlines : boolean, optional
            Flag for whether to create a totally fresh run of astrometric
            corrections, regardless of whether``mn_sigs_array`` is saved on
            disk. Defaults to ``False``.
        make_plots : boolean, optional
            Determines if intermediate figures are generated in the process
            of deriving astrometric corrections.
        make_summary_plot : boolean, optional
            If ``True`` then final summary plot is created, even if
            ``make_plots`` is ``False`` and intermediate plots are not created.
        seeing_ranges : list or numpy.ndarray, optional
            If ``make_plots`` is True, must be a valid 1-, 2-, or 3-length list
            of floats, the seeing, in arcseconds, of observations, for plotting
            over astrometric precision vs SNR distributions.
        single_or_repeat : string, either "single" or "repeat"
            Flag indicating whether catalogue is made of a single set of
            observations or repeated "visits" to this patch of sky. If "single"
            then each row is assumed to be unique, but if "repeat" is given then
            ``repeat_unique_visits_list`` must be provided, giving the indices for
            which row comes from which unique repeat-observation of the sky
            region.
        repeat_unique_visits_list : list of list or numpy.ndarray of integers
            If ``single_or_repeat`` is "repeat", then these arrays must be
            provided, giving the ID of each sub-visit that has been merged
            into a single catalogue, for use in separating individual visits
            out for cross-matches and local normalising density calculations.
            Length must match ``a_cat`` and ``b_cat``, one set of visit indices
            per field pointing.
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
        if self.trifilepath is not None and tri_download not in (True, False):
            raise ValueError("tri_download must either be True or False if trifilepath given.")
        if tri_download is not None and self.tri_dens_cube is not None:
            raise ValueError("tri_download must be None if tri_dens_cube is given.")
        if make_plots and seeing_ranges is None:
            raise ValueError("seeing_ranges must be provided if make_plots is True.")
        if make_plots:
            seeing_ranges = np.array(seeing_ranges)
            if len(seeing_ranges) not in [1, 2, 3] or len(seeing_ranges.shape) != 1:
                raise ValueError("seeing_ranges must be a list of length 1, 2, or 3.")
            try:
                seeing_ranges = np.array([float(f) for f in seeing_ranges])
            except ValueError as exc:
                raise ValueError('seeing_ranges should be a list of floats.') from exc
            self.seeing_ranges = seeing_ranges
        else:
            self.seeing_ranges = None
        if single_or_repeat not in ["single", "repeat"]:
            raise ValueError("single_or_repeat must either be 'single' or 'repeat'.")
        # For now, limit repeat observation handling to datasets passed through
        # directly, to avoid extra work loading those.
        if (not (self.pregenerate_cutouts is None and a_cat is not None)) and single_or_repeat == "repeat":
            raise ValueError("single_or_repeat cannot be 'repeat' unless pregenerate_cutouts is None and "
                             "a_cat and b_cat are provided directly.")
        self.single_or_repeat = single_or_repeat
        if self.single_or_repeat == "repeat" and repeat_unique_visits_list is None:
            raise ValueError("repeat_unique_visits_list must be provided if single_or_repeat is 'repeat'.")
        if self.single_or_repeat == "repeat":
            self.repeat_unique_visits_list = repeat_unique_visits_list
            if isinstance(self.repeat_unique_visits_list, np.ndarray):
                self.repeat_unique_visits_list = [self.repeat_unique_visits_list]
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
            zip_list = (self.ax1_mids, self.ax2_mids)
        else:
            zip_list = (self.ax1_mids, self.ax2_mids, self.chunks)

        if self.use_photometric_uncertainties:
            shape = (len(self.ax1_mids), len(self.mag_names), 4)
        else:
            shape = (len(self.ax1_mids), 4)

        if (self.return_nm or overwrite_all_sightlines or
                not os.path.isfile(f'{self.save_folder}/npy/mn_sigs_array.npy')):
            if not self.return_nm:
                mn_sigs = open_memmap(f'{self.save_folder}/npy/mn_sigs_array.npy', mode='w+',
                                      dtype=float, shape=shape)
            else:
                mn_sigs = np.empty(dtype=float, shape=shape)
            mn_sigs[:] = -9999
        else:
            mn_sigs = open_memmap(f'{self.save_folder}/npy/mn_sigs_array.npy', mode='r+')

        if self.make_summary_plot:
            if self.use_photometric_uncertainties:
                self.gs_single_sigsig = self.make_gridspec('single_sigsig', len(self.mag_names), 2, 0.8, 5)
                self.ax_b_sing = [plt.subplot(self.gs_single_sigsig[q, 0]) for q in
                                  range(len(self.mag_names))]
                sys.stdout.flush()
                self.ax_b_log_sing = [plt.subplot(self.gs_single_sigsig[q, 1]) for q in
                                      range(len(self.mag_names))]
            else:
                self.gs_single_sigsig = self.make_gridspec('single_sigsig', 1, 2, 0.8, 5)
                self.ax_b_sing = [plt.subplot(self.gs_single_sigsig[0])]
                self.ax_b_log_sing = [plt.subplot(self.gs_single_sigsig[1])]

            self.ylims_sing = [999, 0]

            self.cols = ['k', 'r', 'b', 'g', 'c', 'm', 'orange', 'brown', 'purple', 'grey', 'olive',
                         'cornflowerblue', 'deeppink', 'maroon', 'palevioletred', 'teal', 'crimson',
                         'chocolate', 'darksalmon', 'steelblue', 'slateblue', 'tan', 'yellowgreen',
                         'silver']

            if self.use_photometric_uncertainties:
                shape = (len(self.ax1_mids), len(self.mag_names))
            else:
                shape = len(self.ax1_mids)
            self.avg_b_dens = np.empty(shape, float)

            self.mn_poisson_cdfs = np.empty(shape, object)
            self.ind_poisson_cdfs = np.empty(shape, object)

        for index_, list_of_things in enumerate(zip(*zip_list)):
            if np.all(mn_sigs[index_, :] != -9999):
                continue
            print(f'Running astrometry fits for sightline {index_+1}/{len(self.ax1_mids)}...')
            sys.stdout.flush()

            if self.coord_or_chunk == 'coord':
                ax1_mid, ax2_mid = list_of_things
                cat_args = (ax1_mid, ax2_mid)
                file_name = f'{ax1_mid}_{ax2_mid}'
            else:
                ax1_mid, ax2_mid, chunk = list_of_things
                cat_args = (chunk,)
                file_name = f'{chunk}'
            self.list_of_things = list_of_things
            self.cat_args = cat_args
            self.file_name = file_name

            if self.pregenerate_cutouts is None:
                self.a = self.a_cat[index_]
                self.b = self.b_cat[index_]
                if self.single_or_repeat == 'repeat':
                    self.repeat_unique_visits = self.repeat_unique_visits_list[index_]
            else:
                self.a = self.load_catalogue('a', self.cat_args)
                if self.apply_proper_motion_flag and self.pm_indices[1] is not None:
                    pm_r_e = self.pm_ref_epoch_or_index[1] if not isinstance(
                        self.pm_ref_epoch_or_index[1], int) else self.a[:, self.pm_ref_epoch_or_index[1]]
                    x, y = apply_proper_motion(
                        self.a[:, self.pos_and_err_indices_full[1][0]],
                        self.a[:, self.pos_and_err_indices_full[1][1]], self.a[:, self.pm_indices[1][0]],
                        self.a[:, self.pm_indices[1][1]], pm_r_e, self.pm_move_to_epoch, self.coord_system)
                    self.a[:, self.pos_and_err_indices_full[1][0]] = x
                    self.a[:, self.pos_and_err_indices_full[1][1]] = y
                self.b = self.load_catalogue('b', self.cat_args)
                if self.apply_proper_motion_flag and self.pm_indices[0] is not None:
                    pm_r_e = self.pm_ref_epoch_or_index[0] if not isinstance(
                        self.pm_ref_epoch_or_index[0], int) else self.b[:, self.pm_ref_epoch_or_index[0]]
                    x, y = apply_proper_motion(
                        self.b[:, self.pos_and_err_indices_full[0][0]],
                        self.b[:, self.pos_and_err_indices_full[0][1]], self.b[:, self.pm_indices[0][0]],
                        self.b[:, self.pm_indices[0][1]], pm_r_e, self.pm_move_to_epoch, self.coord_system)
                    self.b[:, self.pos_and_err_indices_full[0][0]] = x
                    self.b[:, self.pos_and_err_indices_full[0][1]] = y

            self.area, self.hull_points, self.hull_x_shift = convex_hull_area(
                self.b[:, self.pos_and_err_indices_full[0][0]],
                self.b[:, self.pos_and_err_indices_full[0][1]], return_hull=True)

            # At this point we need to loop to accommodate the possibility
            # that we're generating a per-band parameterisation. If
            # use_photometric_uncertainties is False this will be a single loop,
            # boringly doing nothing.
            for unc_index in range(len(self.pos_and_err_indices_full[0])-2):
                # If we aren't using photometric uncertainties then we only do
                # do a single loop through unc_index, but need to use the
                # correct_astro_mag_indices_index element of all of our
                # magnitude-related terms; however, if we are using photometry,
                # then unc_index loops as intended.
                p = unc_index if self.use_photometric_uncertainties else self.correct_astro_mag_indices_index
                self.psf_fwhm = self.psf_fwhms[p]
                self.gal_wav_micron = self.gal_wavs_micron[p]
                self.gal_ab_offset = self.gal_ab_offsets[p]
                self.gal_filtname = self.gal_filtnames[p]
                self.gal_alav = self.gal_alavs[p]
                self.mag_array = self.mag_arrays[p]
                self.mag_slice = self.mag_slices[p]
                self.sig_slice = self.sig_slices[p]
                if self.trifilepath is not None:
                    # For record keeping, trifilepath and trifilterset, magnum,
                    # maglim_f, and tri_num_faint aren't per-band, so don't get
                    # selected out of a list.
                    self.trifiltname = self.trifiltnames[p]
                else:
                    # tri_dens_cube is of shape (S, F, x, 3), and we need to
                    # determine which index into S and F we want. Assume that
                    # we put our filter indices in the "right" order, and then
                    # pull nearest-neighbour sky position.
                    sky_index = mff.find_nearest_point([ax1_mid], [ax2_mid], self.tri_dens_array[:, 0],
                                                       self.tri_dens_array[:, 1])[0]
                    self.tri_hist = self.tri_dens_cube[sky_index, p, :, 0]
                    self.tri_hist = self.tri_hist[~np.isnan(self.tri_hist)]
                    self.tri_mags = self.tri_dens_cube[sky_index, p, :, 0]
                    self.tri_mags = self.tri_mags[~np.isnan(self.tri_mags)]
                    self.dtri_mags = self.tri_dens_cube[sky_index, p, :, 0]
                    self.dtri_mags = self.dtri_mags[~np.isnan(self.dtri_mags)]

                self.psf_radius = 1.185 * self.psf_fwhm
                self.psfsig = self.psf_fwhm / (2 * np.sqrt(2 * np.log(2)))
                self.r = np.linspace(0, self.psf_radius, self.n_r)
                self.rho = np.linspace(0, self.max_rho, self.n_rho)
                self.dr, self.drho = np.diff(self.r), np.diff(self.rho)

                self.j0s = mff.calc_j0(self.rho[:-1]+self.drho/2, self.r[:-1]+self.dr/2)

                self.n_mag_cols = np.ceil(np.sqrt(len(self.mag_array))).astype(int)
                self.n_mag_rows = np.ceil(len(self.mag_array) / self.n_mag_cols).astype(int)

                self.unc_index = unc_index
                self.pos_and_err_indices = [
                    [self.pos_and_err_indices_full[0][0], self.pos_and_err_indices_full[0][1],
                     self.pos_and_err_indices_full[0][unc_index+2]], self.pos_and_err_indices_full[1]]
                if self.use_photometric_uncertainties:
                    self.magnitude_append = f'{self.mag_names[unc_index]}_'
                else:
                    self.magnitude_append = ''

                if self.use_photometric_uncertainties:
                    n_sources = np.sum(~np.isnan(self.b[:, self.pos_and_err_indices[0][2]]))
                else:
                    n_sources = len(self.b)
                if self.single_or_repeat == 'repeat':
                    # Divide the density through by the number of repeat visits,
                    # since we want the average density of the visits, not the
                    # sum of all repeated visits.
                    n_visits = len(np.unique(self.repeat_unique_visits))
                    if self.use_photometric_uncertainties:
                        self.avg_b_dens[index_, unc_index] = n_sources / self.area / n_visits
                    else:
                        self.avg_b_dens[index_] = n_sources / self.area / n_visits
                else:
                    if self.use_photometric_uncertainties:
                        self.avg_b_dens[index_, unc_index] = n_sources / self.area
                    else:
                        self.avg_b_dens[index_] = n_sources / self.area

                self.make_star_galaxy_counts()
                if self.make_plots:
                    self.plot_star_galaxy_counts()
                self.calculate_local_densities_and_nearest_neighbours()

                self.simulate_aufs()
                self.create_auf_pdfs()
                # If we don't have at least 5 unique magnitude slices to calculate,
                # reduce the number of data points per histogram to increase
                # number statistics.
                if np.sum([q[0] == -1 for q in self.pdfs]) > len(self.pdfs)-5:
                    warnings.warn("Reduced PDF histogram counts to 75.")
                    self.create_auf_pdfs(min_hist_cut=75)
                if np.sum([q[0] == -1 for q in self.pdfs]) > len(self.pdfs)-5:
                    warnings.warn("Reduced PDF histogram counts to 50.")
                    self.create_auf_pdfs(min_hist_cut=50)
                if np.sum([q[0] == -1 for q in self.pdfs]) <= len(self.pdfs)-5:
                    m_sig, n_sig = self.fit_uncertainty()
                else:
                    # Fall back to not correcting anything if data still too poor
                    # to draw any meaningful conclusions from.
                    m_sig, n_sig = 1, 0
                    self.fit_sigs = np.zeros((len(self.mag_array), 2), float)
                    self.fit_sigs[:, 0] = self.avg_sig[:, 0]
                    self.fit_sigs[:, 1] = self.avg_sig[:, 0]
                if self.use_photometric_uncertainties:
                    mn_sigs[index_, unc_index, 0] = m_sig
                    mn_sigs[index_, unc_index, 1] = n_sig
                    mn_sigs[index_, unc_index, 2] = ax1_mid
                    mn_sigs[index_, unc_index, 3] = ax2_mid
                else:
                    mn_sigs[index_, 0] = m_sig
                    mn_sigs[index_, 1] = n_sig
                    mn_sigs[index_, 2] = ax1_mid
                    mn_sigs[index_, 3] = ax2_mid
                mn_poisson_cdfs, ind_poisson_cdfs = self.plot_fits_calculate_cdfs()
                if self.make_summary_plot:
                    if self.use_photometric_uncertainties:
                        self.mn_poisson_cdfs[index_, unc_index] = mn_poisson_cdfs
                        self.ind_poisson_cdfs[index_, unc_index] = ind_poisson_cdfs
                    else:
                        self.mn_poisson_cdfs[index_] = mn_poisson_cdfs
                        self.ind_poisson_cdfs[index_] = ind_poisson_cdfs
                    if np.sum(~self.skip_flags) > 0:
                        c = self.cols[index_ % len(self.cols)]
                        plt.figure('single_sigsig')
                        self.ax_b_sing[unc_index].errorbar(self.avg_sig[~self.skip_flags, 0],
                                                           self.fit_sigs[~self.skip_flags, 1],
                                                           linestyle='None', c=c, marker='x', label=file_name)
                        self.ax_b_log_sing[unc_index].errorbar(self.avg_sig[~self.skip_flags, 0],
                                                               self.fit_sigs[~self.skip_flags, 1],
                                                               linestyle='None', c=c, marker='x')
                        self.ylims_sing[0] = min(self.ylims_sing[0],
                                                 np.amin(self.fit_sigs[~self.skip_flags, 1]))
                        self.ylims_sing[1] = max(self.ylims_sing[1],
                                                 np.amax(self.fit_sigs[~self.skip_flags, 1]))
                    self.plot_snr_mag_sig()

        self.mn_sigs = mn_sigs
        if self.make_summary_plot:
            self.finalise_summary_plots()

        if self.return_nm:
            return mn_sigs
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

        if self.pregenerate_cutouts is False:
            # If we don't force pre-generated sightlines then we can generate
            # corners based on requested size and height, and latitude. Force
            # constant box height, but allow longitude to float to make sure
            # that we get good area coverage as the cos-delta factor increases
            # towards the poles.
            self.ax1_mins, self.ax1_maxs = np.empty_like(self.ax1_mids), np.empty_like(self.ax1_mids)
            self.ax2_mins, self.ax2_maxs = np.empty_like(self.ax1_mids), np.empty_like(self.ax1_mids)
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
            sys.stdout.flush()

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
        sys.stdout.flush()

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
        sys.stdout.flush()
        if self.coord_or_chunk == 'coord':
            ax1_mid, ax2_mid = self.list_of_things
        else:
            ax1_mid, ax2_mid, _ = self.list_of_things

        p_ind = self.unc_index if self.use_photometric_uncertainties else self.correct_astro_mag_indices_index
        mag_ind = self.mag_indices[p_ind]
        b_mag_data = self.b[~np.isnan(self.b[:, mag_ind]), mag_ind]

        hist_mag, bins = np.histogram(b_mag_data, bins='auto')
        minmag = bins[0]
        # Ensure that we're only counting sources for normalisation purposes
        # down to the completeness turnover.
        # pylint: disable-next=fixme
        # TODO: make the half-mag offset flexible, passing from CrossMatch and/or
        # directly into AstrometricCorrections.
        maxmag = bins[:-1][np.argmax(hist_mag)] - 0.5

        if self.trifilepath is not None:
            x, y = os.path.splitext(self.trifilepath.format(ax1_mid, ax2_mid))
            full_file = x + '_faint' + y
        # pylint: disable-next=possibly-used-before-assignment
        if (self.trifilepath is not None and (self.tri_download or not os.path.isfile(full_file))):
            data_bright_dens = np.sum(~np.isnan(b_mag_data) & (b_mag_data <= maxmag)) / self.area
            # pylint: disable-next=fixme
            # TODO: un-hardcode min_bright_tri_number
            min_bright_tri_number = 1000
            min_area = min_bright_tri_number / data_bright_dens

            download_trilegal_simulation(full_file, self.trifilterset, ax1_mid, ax2_mid,
                                         self.magnum, self.coord_system, self.maglim_f, min_area,
                                         av=1, sigma_av=0, total_objs=self.tri_num_faint)

        avs = generate_avs_inside_hull(self.hull_points, self.hull_x_shift, self.coord_system)

        if self.trifilepath is not None:
            # Don't pass the _faint-appended filepath to make_tri_counts, since it
            # handles that itself.
            tri_hist, tri_mags, dtri_mags = make_tri_counts(
                self.trifilepath.format(ax1_mid, ax2_mid), self.trifiltname, self.dm, np.amin(b_mag_data),
                al_av=self.gal_alav, av_grid=avs)
        else:
            tri_hist, tri_mags = self.tri_hist, self.tri_mags
            dtri_mags = self.dtri_mags

        gal_dns = create_galaxy_counts(
            self.gal_cmau_array, tri_mags+dtri_mags/2, np.linspace(0, 4, 41),
            self.gal_wav_micron, self.gal_alpha0, self.gal_alpha1, self.gal_alphaweight,
            self.gal_ab_offset, self.gal_filtname, self.gal_alav*avs)

        d_hc = np.where(hist_mag > 3)[0]
        data_hist = hist_mag[d_hc]
        data_dbins = np.diff(bins)[d_hc]
        data_bins = bins[d_hc]

        data_uncert = np.sqrt(data_hist) / data_dbins / self.area
        data_hist = data_hist / data_dbins / self.area

        if self.single_or_repeat == 'repeat':
            # Divide the counts through by the number of repeat visits.
            n_visits = len(np.unique(self.repeat_unique_visits))
            data_hist = data_hist / n_visits
            data_uncert = data_uncert / np.sqrt(n_visits)

        data_loghist = np.log10(data_hist)
        data_dloghist = 1/np.log(10) * data_uncert / data_hist

        q = (data_bins <= maxmag) & (data_bins >= self.saturation_magnitudes[p_ind])
        tri_corr, gal_corr = find_model_counts_corrections(data_loghist[q], data_dloghist[q],
                                                           data_bins[q]+data_dbins[q]/2, tri_hist, gal_dns,
                                                           tri_mags+dtri_mags/2)
        log10y = np.log10(tri_hist * tri_corr + gal_dns * gal_corr)

        mag_slice = (tri_mags >= minmag) & (tri_mags+dtri_mags <= maxmag)
        n_norm = np.sum(10**log10y[mag_slice] * dtri_mags[mag_slice])
        self.log10y = log10y
        self.tri_hist, self.tri_mags, self.dtri_mags = tri_hist, tri_mags, dtri_mags
        self.gal_dns = gal_dns
        self.minmag, self.maxmag, self.n_norm = minmag, maxmag, n_norm
        self.tri_corr, self.gal_corr = tri_corr, gal_corr

    def plot_star_galaxy_counts(self):
        """
        Plotting routine to display data and model differential source counts,
        for verification purposes.
        """
        gs = self.make_gridspec('123123', 1, 1, 0.8, 5)
        print('Plotting data and model counts...')
        sys.stdout.flush()

        p_ind = self.unc_index if self.use_photometric_uncertainties else self.correct_astro_mag_indices_index
        mag_ind = self.mag_indices[p_ind]
        data_mags = self.b[~np.isnan(self.b[:, mag_ind]), mag_ind]

        ax = plt.subplot(gs[0])
        ax.errorbar(self.tri_mags+self.dtri_mags/2, self.log10y, c='k', marker='.', zorder=1, ls='None')

        data_hist, data_bins = np.histogram(data_mags, bins='auto')
        d_hc = np.where(data_hist > 3)[0]
        data_hist = data_hist[d_hc]
        data_dbins = np.diff(data_bins)[d_hc]
        data_bins = data_bins[d_hc]

        data_uncert = np.sqrt(data_hist) / data_dbins / self.area
        data_hist = data_hist / data_dbins / self.area

        if self.single_or_repeat == 'repeat':
            # Divide the counts through by the number of repeat visits.
            n_visits = len(np.unique(self.repeat_unique_visits))
            data_hist = data_hist / n_visits
            data_uncert = data_uncert / np.sqrt(n_visits)

        data_loghist = np.log10(data_hist)
        data_dloghist = 1/np.log(10) * data_uncert / data_hist
        ax.errorbar(data_bins+data_dbins/2, data_loghist, yerr=data_dloghist, c='r',
                    marker='.', zorder=1, ls='None')

        lims = ax.get_ylim()
        q = self.tri_hist > 0
        ax.plot((self.tri_mags+self.dtri_mags/2)[q], np.log10(self.tri_hist[q]) +
                np.log10(self.tri_corr), 'k--')
        q = self.gal_dns > 0
        ax.plot((self.tri_mags+self.dtri_mags/2)[q], np.log10(self.gal_dns[q]) +
                np.log10(self.gal_corr), 'k:')
        ax.set_ylim(*lims)

        ax.set_xlabel(f'{self.mag_names[self.unc_index]} / mag')
        if usetex:
            ax.set_ylabel(r'log$_{10}\left(\mathrm{D}\ /\ \mathrm{mag}^{-1}\,'
                          r'\mathrm{deg}^{-2}\right)$')
        else:
            ax.set_ylabel(r'log10(D / mag^-1 deg^-2)')

        plt.figure('123123')
        plt.tight_layout()
        plt.savefig(f'{self.save_folder}/pdf/{self.magnitude_append}counts_comparison_{self.file_name}.pdf')
        plt.close()

    def calculate_local_densities_and_nearest_neighbours(self):
        """
        Calculate local normalising catalogue densities and catalogue-catalogue
        nearest neighbour match pairings for each cutout region.
        """
        print('Creating local densities and nearest neighbour matches...')
        sys.stdout.flush()

        p_ind = self.unc_index if self.use_photometric_uncertainties else self.correct_astro_mag_indices_index
        narray = create_densities(
            self.b, self.minmag, self.maxmag, self.hull_points, self.hull_x_shift, self.dens_search_radius,
            self.n_pool, self.mag_indices[p_ind], self.pos_and_err_indices[0][0],
            self.pos_and_err_indices[0][1], self.coord_system, self.file_name)

        if self.single_or_repeat == 'repeat':
            # Divide the counts through by the number of repeat visits.
            n_visits = len(np.unique(self.repeat_unique_visits))
            narray = narray / n_visits

        if self.single_or_repeat == 'single':
            _, bmatch, dists = create_distances(
                self.a, self.b, self.nn_radius, self.pos_and_err_indices[1][0],
                self.pos_and_err_indices[1][1], self.pos_and_err_indices[0][0],
                self.pos_and_err_indices[0][1], self.coord_system)
        else:
            bmatch, dists = [], []
            for v in np.unique(self.repeat_unique_visits):
                q = self.repeat_unique_visits == v
                _, _bm, _d = create_distances(
                    self.a, self.b[q], self.nn_radius, self.pos_and_err_indices[1][0],
                    self.pos_and_err_indices[1][1], self.pos_and_err_indices[1][0],
                    self.pos_and_err_indices[0][1], self.coord_system)
                # The _bm array returned is [0, len(self.b[q])] indexed, so we
                # need to convert to indices back into self.b.
                b_match_large = np.arange(len(self.b))[q]
                if len(bmatch) == 0:
                    bmatch = b_match_large[_bm]
                    dists = _d
                else:
                    bmatch = np.concatenate((bmatch, b_match_large[_bm]))
                    dists = np.concatenate((dists, _d))

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
        sys.stdout.flush()

        b_ratio = 0.05

        p_ind = self.unc_index if self.use_photometric_uncertainties else self.correct_astro_mag_indices_index
        _snr = self.b[:, self.snr_indices[p_ind]]
        _mag = self.b[:, self.mag_indices[p_ind]]
        p = ((_snr > 0) & ~np.isnan(_snr) & np.isfinite(_snr) &
             (_mag > 0) & ~np.isnan(_mag) & np.isfinite(_mag))
        snr, _, _ = binned_statistic(_mag[p], _snr[p], statistic='median',
                                     bins=np.append(self.mag_array-self.mag_slice,
                                                    self.mag_array[-1]+self.mag_slice[-1]))
        dm_max = _calculate_magnitude_offsets(
            self.moden*np.ones_like(self.mag_array), self.mag_array, b_ratio, snr, self.tri_mags,
            self.log10y, self.dtri_mags, self.psf_radius, self.n_norm)

        seed = np.random.default_rng().choice(100000, size=(mff.get_random_seed_size(),
                                                            len(self.mag_array)))
        _, _, four_off_fw, _, _ = \
            paf.perturb_aufs(
                self.moden*np.ones_like(self.mag_array), self.mag_array, self.r[:-1]+self.dr/2,
                self.dr, self.r, self.j0s.T, self.tri_mags+self.dtri_mags/2, self.dtri_mags,
                self.log10y, self.n_norm, (dm_max/self.dm).astype(int), self.dmcut, self.psf_radius,
                self.psfsig, self.numtrials, seed, self.dd_params, self.l_cut, 'fw')

        seed = np.random.default_rng().choice(100000, size=(mff.get_random_seed_size(),
                                                            len(self.mag_array)))
        _, _, four_off_ps, _, _ = \
            paf.perturb_aufs(
                self.moden*np.ones_like(self.mag_array), self.mag_array, self.r[:-1]+self.dr/2,
                self.dr, self.r, self.j0s.T, self.tri_mags+self.dtri_mags/2, self.dtri_mags,
                self.log10y, self.n_norm, (dm_max/self.dm).astype(int), self.dmcut, self.psf_radius,
                self.psfsig, self.numtrials, seed, self.dd_params, self.l_cut, 'psf')

        self.four_off_fw, self.four_off_ps = four_off_fw, four_off_ps

    def create_auf_pdfs(self, min_hist_cut=100):
        """
        Using perturbation AUF simulations, generate probability density functions
        of perturbation distance for all cutout regions, as well as recording key
        statistics such as average magnitude or SNR.

        Parameters
        ----------
        min_hist_cut : integer, optional
            Number of data points in each magnitude-uncertainty slice to be
            considered for fitting for scaling relations.
        """
        print('Creating catalogue AUF probability densities...')
        sys.stdout.flush()
        b_matches = self.b[self.bmatch]

        skip_flags = np.zeros_like(self.mag_array, dtype=bool)

        pdfs, pdf_uncerts, q_pdfs, pdf_bins = [], [], [], []
        nums = []

        avg_sig = np.empty((len(self.mag_array), 3), float)
        avg_snr = np.empty((len(self.mag_array), 3), float)
        avg_mag = np.empty((len(self.mag_array), 3), float)

        p_ind = self.unc_index if self.use_photometric_uncertainties else self.correct_astro_mag_indices_index
        mag_ind = self.mag_indices[p_ind]
        for i, mag in enumerate(self.mag_array):
            mag_cut = ((b_matches[:, mag_ind] <= mag+self.mag_slice[i]) &
                       (b_matches[:, mag_ind] >= mag-self.mag_slice[i]))
            if np.sum(mag_cut) == 0:
                skip_flags[i] = 1
                pdfs.append([-1])
                pdf_uncerts.append([-1])
                q_pdfs.append([-1])
                pdf_bins.append([-1])
                nums.append([-1])
                continue
            sig = np.percentile(b_matches[mag_cut, self.pos_and_err_indices[0][2]], 50)
            sig_cut = ((b_matches[:, self.pos_and_err_indices[0][2]] <= sig+self.sig_slice[i]) &
                       (b_matches[:, self.pos_and_err_indices[0][2]] >= sig-self.sig_slice[i]))
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
            if len(final_dists) < min_hist_cut:
                skip_flags[i] = 1
                pdfs.append([-1])
                pdf_uncerts.append([-1])
                q_pdfs.append([-1])
                pdf_bins.append([-1])
                nums.append([-1])
                continue

            bm = b_matches[final_slice]
            p = ((bm[:, self.snr_indices[p_ind]] > 0) & ~np.isnan(bm[:, self.snr_indices[p_ind]]) &
                 np.isfinite(bm[:, self.snr_indices[p_ind]]))
            snr = bm[p, self.snr_indices[p_ind]]
            avg_snr[i, 0] = np.median(snr)
            avg_snr[i, [1, 2]] = np.abs(np.percentile(snr, [16, 84]) - np.median(snr))
            avg_mag[i, 0] = np.median(bm[:, mag_ind])
            avg_mag[i, [1, 2]] = np.abs(np.percentile(bm[:, mag_ind], [16, 84]) -
                                        np.median(bm[:, mag_ind]))
            avg_sig[i, 0] = np.median(bm[:, self.pos_and_err_indices[0][2]])
            avg_sig[i, [1, 2]] = np.abs(np.percentile(bm[:, self.pos_and_err_indices[0][2]], [16, 84]) -
                                        np.median(bm[:, self.pos_and_err_indices[0][2]]))

            h, bins = np.histogram(final_dists, bins='auto')
            num = np.sum(h)
            pdf = h / np.diff(bins) / num
            pdf_uncert = np.sqrt(h) / np.diff(bins) / num
            # To avoid binned_statistic NaNing when the data are beyond the
            # edge of the model, we want to limit our bins to the maximum r value.
            q_pdf = bins[1:] <= self.r[-1]

            pdfs.append(pdf)
            pdf_uncerts.append(pdf_uncert)
            q_pdfs.append(q_pdf)
            pdf_bins.append(bins)
            nums.append(num)

        self.nums = nums
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
        sys.stdout.flush()

        self.fit_sigs = np.zeros((len(self.mag_array), 2), float)

        res = minimize(self.fit_auf_joint, x0=[1, 0.001], method='L-BFGS-B',
                       options={'ftol': 1e-9}, bounds=[(0, None), (0, None)])
        self.res = res

        m, n = res.x
        for i, pdf in enumerate(self.pdfs):
            if pdf[0] == -1:
                self.skip_flags[i] = 1
                continue
            sig_orig = self.avg_sig[i, 0]
            if self.mn_fit_type == 'quadratic':
                new_sig = np.sqrt((m * sig_orig)**2 + n**2)
            else:
                new_sig = m * sig_orig + n
            self.fit_sigs[i, 0] = new_sig

            (y, q, bins, sig, snr, num) = (pdf, self.q_pdfs[i], self.pdf_bins[i],
                                           self.avg_sig[i, 0], self.avg_snr[i, 0], self.nums[i])
            res = minimize(self.calc_single_joint_auf, x0=[sig], args=(i, bins, y, q, num, snr),
                           method='L-BFGS-B', options={'ftol': 1e-9}, bounds=[(0, None)])

            self.fit_sigs[i, 1] = res.x[0]

        return m, n

    def fit_auf_joint(self, p):
        """
        Fits all magnitude slices for a single sightline, modelling
        empirically derived centroid uncertainties in all cases in addition
        to fixed uncertainty contributions.

        Parameters
        ----------
        p : list
            List containing the scaling values to be fit for.

        Returns
        -------
        neg_log_like : float
            Negative-log-likelihood of the fit, to be minimised in the wrapping
            function call.
        """
        m, n = p

        neg_log_like = 0
        for i, pdf in enumerate(self.pdfs):
            if self.pdfs[i][0] == -1:
                continue
            (y, q, bins, sig, snr, num) = (pdf, self.q_pdfs[i], self.pdf_bins[i],
                                           self.avg_sig[i, 0], self.avg_snr[i, 0], self.nums[i])

            if self.mn_fit_type == 'quadratic':
                o = np.sqrt((m * sig)**2 + n**2)
            else:
                o = m * sig + n

            neg_log_like += self.calc_single_joint_auf(o, i, bins, y, q, num, snr)

        return neg_log_like

    def calc_single_joint_auf(self, o, i, bins, y, q, num, snr):
        """
        Calculates the negative-log-likelihood of a single magnitude slice of
        match separations as fit with an AUF.

        Parameters
        ----------
        o : float
            The Gaussian uncertainty of the centroid component of the AUF.
        i : int
            Index to access particular magnitude slice.
        bins : numpy.ndarray
            Array of floats, bin edges of histogram of cross-match distances.
        y : numpy.ndarray
            Array of floats, counts in each bin of ``bins`` of cross-match
            distances.
        q : numpy.ndarray
            Array of booleans, flags for whether to use each ``y`` bin or not.
        num : int
            Total number of counts within each PDF in ``y``, to convert back to
            raw counts for Poisson counting statistics.
        snr: numpy.ndarray
            Array of signal-to-noise ratios, for weighting between different
            algorithms for the Perturbation component of the AUF.

        Returns
        -------
        neg_log_like_part : numpy.ndarray
            The negative-log-likelihood of dataset of separations given the AUF
            model.
        """

        p_ind = self.unc_index if self.use_photometric_uncertainties else self.correct_astro_mag_indices_index
        _snr = self.b[:, self.snr_indices[p_ind]]
        p = ((_snr > 0) & ~np.isnan(_snr) & np.isfinite(_snr))
        n_sources_inv_max_snr = 1000
        inv_max_snr = 1 / np.percentile(_snr[p][np.argsort(_snr[p])][n_sources_inv_max_snr:], 50)
        h = 1 - np.sqrt(1 - min(1, inv_max_snr**2 * snr**2))
        four_combined = h * self.four_off_fw[:, i] + (1 - h) * self.four_off_ps[:, i]

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

        avg_a_dens = len(self.a) / self.area
        # For a symmetric nearest neighbour distribution we need to use the
        # combined density of sources -- this is derived from consideration
        # of the probability of no NN object inside r being the chance that
        # NEITHER source has an asymmetric nearest neighbour inside r, or
        # F(r) = (1 - \int a(r) dr) * (1 - \int b(r) dr), where the integrals
        # of a and b, being of the form 2 pi r N exp(-pi r^2 N) are
        # 1 - exp(-pi r^2 N), and hence the survival integrals multiplied
        # together form N' = N_a + N_b, and then you use f(r) = dF/dr.
        density = (self.moden + avg_a_dens) / 3600**2
        nn_model = 2 * np.pi * (self.r[:-1]+self.dr/2) * density * np.exp(
            -np.pi * (self.r[:-1]+self.dr/2)**2 * density)

        m_conv_plus_nn = (convolve_hist_dr * (np.exp(-np.pi * (self.r[:-1]+self.dr/2)**2 *
                          density)) + nn_model * (1 - np.cumsum(convolve_hist_dr * self.dr)))

        reduced_nn_model, _, _ = binned_statistic(self.r[:-1]+self.dr/2, nn_model, bins=bins)
        reduced_m_conv_plus_nn, _, _ = binned_statistic(self.r[:-1]+self.dr/2,
                                                        m_conv_plus_nn, bins=bins)
        # Empty binned_statistic bins return as NaNs.
        _q = q & ~np.isnan(reduced_nn_model) & ~np.isnan(reduced_m_conv_plus_nn)
        # For the subset of data in the single magnitude slice we need
        # a nearest neighbour fraction, which we can fit for on-the-fly
        # for each slice separately (for the same m/n).
        _nll = 1e10
        nnf = -1

        k = y[_q] * np.diff(bins)[_q] * num
        # Ramanujan, The Lost Notebook and other Unpublished Papers, gives
        # an approximation for ln(n!) as n ln(n) - n + 1/6 ln(8 n^3 +
        # 4 n^2 + n + 1/30) + ln(pi)/2. We use this, to avoid overflowing
        # n!, above n=100
        log_fac_k = np.empty(len(k), float)
        log_fac_k[k < 100] = np.log(factorial(k[k < 100]))
        log_fac_k[k >= 100] = k[k >= 100] * np.log(k[k >= 100]) - k[k >= 100] + 1/6 * np.log(
            8 * k[k >= 100]**3 + 4 * k[k >= 100]**2 + k[k >= 100] + 1/30) + np.log(np.pi)/2
        for _nnf in np.linspace(0, 1, 101):
            modely = _nnf * reduced_nn_model + (1 - _nnf) * reduced_m_conv_plus_nn

            # Poisson is exp(-L) L**k / k!, so negative log-likelihood is
            # L - k*ln(L) + ln(k!). Have to convert from PDF to counts through
            # bin width and number of objects, forcing a non-zero rate to avoid
            # logarithmic issues.
            _l = modely[_q] * np.diff(bins)[_q] * num
            _l[_l <= 1e-10] = 1e-10
            temp_neg_log_like = np.sum(_l - k*np.log(_l) + log_fac_k)
            if temp_neg_log_like < _nll:
                nnf = _nnf
                _nll = temp_neg_log_like

        modely = nnf * reduced_nn_model + (1 - nnf) * reduced_m_conv_plus_nn

        k = y[_q] * np.diff(bins)[_q] * num
        _l = modely[_q] * np.diff(bins)[_q] * num
        _l[_l <= 1e-10] = 1e-10
        neg_log_like_part = np.sum(_l - k*np.log(_l) + log_fac_k)

        return neg_log_like_part

    def plot_fits_calculate_cdfs(self):  # pylint: disable=too-many-locals,too-many-statements
        """
        Calculate poisson CDFs and create verification plots showing the
        quality of the fits.
        """
        mn_poisson_cdfs = np.array([], float)
        ind_poisson_cdfs = np.array([], float)

        if self.make_plots:
            print('Creating individual AUF figures and calculating goodness-of-fits...')
        else:
            print('Calculating goodness-of-fits...')
        sys.stdout.flush()

        if self.make_plots:
            # Grid just big enough square to cover mag_array entries.
            gs1 = self.make_gridspec('34242b', self.n_mag_rows, self.n_mag_cols, 0.8, 5)
            ax1s = [plt.subplot(gs1[i]) for i in range(len(self.mag_array))]

        b_matches = self.b[self.bmatch]

        p_ind = self.unc_index if self.use_photometric_uncertainties else self.correct_astro_mag_indices_index
        for i, mag in enumerate(self.mag_array):
            if self.skip_flags[i]:
                continue
            if self.make_plots:
                ax = ax1s[i]
            pdf, pdf_uncert, q_pdf, pdf_bin, num_pdf = (
                self.pdfs[i], self.pdf_uncerts[i], self.q_pdfs[i], self.pdf_bins[i], self.nums[i])
            if self.make_plots:
                ax.errorbar((pdf_bin[:-1]+np.diff(pdf_bin)/2)[q_pdf], pdf[q_pdf],
                            yerr=pdf_uncert[q_pdf], c='k', marker='.', zorder=1, ls='None')

            mag_ind = self.mag_indices[p_ind]
            pos_err_ind = self.pos_and_err_indices[0][2]
            mag_cut = ((b_matches[:, mag_ind] <= mag+self.mag_slice[i]) &
                       (b_matches[:, mag_ind] >= mag-self.mag_slice[i]))
            bsig = np.percentile(b_matches[mag_cut, pos_err_ind], 50)
            sig_cut = ((b_matches[:, pos_err_ind] <= bsig+self.sig_slice[i]) &
                       (b_matches[:, pos_err_ind] >= bsig-self.sig_slice[i]))
            n_cut = (self.narray[self.bmatch] >= self.moden-self.dn) & (
                self.narray[self.bmatch] <= self.moden+self.dn)
            if not self.use_photometric_uncertainties:
                final_slice = sig_cut & mag_cut & n_cut & (self.dists <= 20*bsig)
            else:
                final_slice = sig_cut & mag_cut & n_cut & (self.dists <= 20*self.psfsig*bsig)

            avg_a_dens = len(self.a) / self.area
            density = (np.percentile(self.narray[self.bmatch][final_slice], 50) +
                       avg_a_dens) / 3600**2
            nn_model = 2 * np.pi * (self.r[:-1]+self.dr/2) * density * np.exp(
                -np.pi * (self.r[:-1]+self.dr/2)**2 * density)

            fit_sig = self.fit_sigs[i, 0]
            _snr = self.b[:, self.snr_indices[p_ind]]
            p = ((_snr > 0) & ~np.isnan(_snr) & np.isfinite(_snr))
            n_sources_inv_max_snr = 1000
            inv_max_snr = 1 / np.percentile(_snr[p][np.argsort(_snr[p])][n_sources_inv_max_snr:], 50)
            h = 1 - np.sqrt(1 - min(1, inv_max_snr**2 * self.avg_snr[i, 0]**2))

            ind_fit_sig = self.fit_sigs[i, 1]
            if self.make_plots:
                ax = ax1s[i]
            for j, (sig, _h, ls) in enumerate(zip(
                    [fit_sig, fit_sig, fit_sig, self.avg_sig[i, 0], self.avg_sig[i, 0], self.avg_sig[i, 0],
                     ind_fit_sig, ind_fit_sig, ind_fit_sig],
                    [h, 1, 0, h, 1, 0, h, 1, 0], ['r-', 'r-.', 'r:', 'k-', 'k-.', 'k:', 'c-', 'c-.', 'c:'])):
                if not self.make_plots and j != 0:
                    continue
                four_gauss = np.exp(-2 * np.pi**2 * (self.rho[:-1]+self.drho/2)**2 * sig**2)

                four_hist = _h * self.four_off_fw[:, i] + (1 - _h) * self.four_off_ps[:, i]
                convolve_hist = paf.fourier_transform(
                    four_hist*four_gauss, self.rho[:-1]+self.drho/2, self.drho, self.j0s)

                convolve_hist_dr = convolve_hist * np.pi * (
                    self.r[1:]**2 - self.r[:-1]**2) / self.dr

                m_conv_plus_nn = (convolve_hist_dr * (np.exp(-np.pi * (self.r[:-1]+self.dr/2)**2 *
                                  density)) + nn_model * (1 - np.cumsum(convolve_hist_dr * self.dr)))

                reduced_nn_model, _, _ = binned_statistic(self.r[:-1]+self.dr/2, nn_model, bins=pdf_bin)
                reduced_m_conv_plus_nn, _, _ = binned_statistic(self.r[:-1]+self.dr/2,
                                                                m_conv_plus_nn, bins=pdf_bin)

                if j in [0, 3, 6]:
                    _nll = 1e10
                    nn_frac = -1

                    k = pdf[q_pdf] * np.diff(pdf_bin)[q_pdf] * num_pdf
                    log_fac_k = np.empty(len(k), float)
                    log_fac_k[k < 100] = np.log(factorial(k[k < 100]))
                    log_fac_k[k >= 100] = k[k >= 100] * np.log(k[k >= 100]) - k[k >= 100] + 1/6 * np.log(
                        8 * k[k >= 100]**3 + 4 * k[k >= 100]**2 + k[k >= 100] + 1/30) + np.log(np.pi)/2
                    for _nnf in np.linspace(0, 1, 101):
                        modely = _nnf * reduced_nn_model + (1 - _nnf) * reduced_m_conv_plus_nn

                        _l = modely[q_pdf] * np.diff(pdf_bin)[q_pdf] * num_pdf
                        _l[_l <= 1e-10] = 1e-10
                        temp_neg_log_like = np.sum(_l - k*np.log(_l) + log_fac_k)
                        if temp_neg_log_like < _nll:
                            nn_frac = _nnf
                            _nll = temp_neg_log_like
                    if j == 0:
                        nn_frac_mn = nn_frac
                    if j == 3:
                        nn_frac_quot = nn_frac
                    if j == 6:
                        nn_frac_ind = nn_frac

                modely = nn_frac * reduced_nn_model + (1 - nn_frac) * reduced_m_conv_plus_nn

                if j in [0, 3, 6]:
                    _l = modely[q_pdf] * np.diff(pdf_bin)[q_pdf] * num_pdf
                    _l[_l <= 1e-10] = 1e-10
                    p_cdf = np.empty(len(_l), float)
                    for _i in range(len(_l)):  # pylint: disable=consider-using-enumerate
                        p_cdf[_i] = poisson.cdf(k=k[_i], mu=_l[_i])
                    if j == 0:
                        mn_poisson_cdfs = np.append(mn_poisson_cdfs, p_cdf)
                    if j == 6:
                        ind_poisson_cdfs = np.append(ind_poisson_cdfs, p_cdf)

                if self.make_plots:
                    if j in [0, 3, 6]:
                        sig_type = 'fit' if j == 0 else 'quoted' if j == 3 else 'ind'
                        sig_val = fit_sig if j == 0 else self.avg_sig[i, 0] if j == 3 else ind_fit_sig
                        f_val = nn_frac_mn if j == 0 else nn_frac_quot if j == 3 else nn_frac_ind
                        h_str = f', H = {h:.2f}' if j == 0 else ''
                        if usetex:
                            lab = rf'$\sigma_\mathrm{{{sig_type}}}$ = {sig_val:.4f}", F = {f_val:.2f}{h_str}'
                        else:
                            lab = rf'sigma_{sig_type} = {sig_val:.4f}", F = {f_val:.2f}{h_str}'
                    else:
                        lab = ''
                    modely_norm = np.sum(modely * np.diff(pdf_bin))
                    ax.plot((pdf_bin[:-1]+np.diff(pdf_bin)/2)[q_pdf], modely[q_pdf] / modely_norm,
                            ls, label=lab)

            if self.make_plots:
                ax.legend(fontsize=8)
                ax.set_xlabel('Radius / arcsecond')
                if usetex:
                    ax.set_ylabel('PDF / arcsecond$^{-1}$')
                else:
                    ax.set_ylabel('PDF / arcsecond^-1')

        if self.make_plots:
            plt.tight_layout()
            plt.savefig(f'{self.save_folder}/pdf/{self.magnitude_append}auf_fits_{self.file_name}.pdf')
            plt.close()

        return mn_poisson_cdfs, ind_poisson_cdfs

    def plot_snr_mag_sig(self):  # pylint: disable=too-many-statements
        """
        Generate 2-D histograms of SNR, quoted/fit astrometric uncertainty, and
        photometric magnitude.
        """
        print("Plotting SNR-Magnitude-Uncertainty scaling relations...")
        sys.stdout.flush()
        p_ind = self.unc_index if self.use_photometric_uncertainties else self.correct_astro_mag_indices_index
        p = ((self.b[:, self.snr_indices[p_ind]] > 0) & ~np.isnan(self.b[:, self.snr_indices[p_ind]]) &
             np.isfinite(self.b[:, self.snr_indices[p_ind]]))
        _snr = self.b[p, self.snr_indices[p_ind]]
        obj_err = self.b[p, self.pos_and_err_indices[0][2]]
        obj_mag = self.b[p, self.mag_indices[p_ind]]

        gs = self.make_gridspec('sig_vs_snr_vs_mag', 1, 3, 0.8, 6)
        mag_label = f' ({self.mag_names[self.unc_index]})' if self.use_photometric_uncertainties else ''
        ax = plt.subplot(gs[0])
        q = (obj_err < 1) & (_snr > 1) & (obj_err > 0)
        log_inv_snr = np.log10(1 / _snr[q])
        log_err = np.log10(obj_err[q])
        h, x, y = np.histogram2d(log_inv_snr, log_err, bins=(100, 101))
        ax.pcolormesh(x, y, h.T, edgecolors='face', cmap='viridis', rasterized=True)
        ylims = ax.get_ylim()
        xlims = ax.get_xlim()
        if len(self.seeing_ranges) == 1:
            ls_list = ['k-']
        elif len(self.seeing_ranges) == 2:
            ls_list = ['r--', 'k-']
        elif len(self.seeing_ranges) == 3:
            ls_list = ['r--', 'k-', 'r-']
        if not self.use_photometric_uncertainties:
            # pylint: disable-next=possibly-used-before-assignment
            for seeing, ls in zip(self.seeing_ranges, ls_list):
                log_ks = np.log10(seeing / (2 * np.sqrt(2 * np.log(2))))
                ax.plot(x, log_ks + x, ls, label=rf'seeing = {seeing:.1f} arcsec')
        else:
            ax.plot(x, x, 'k-')

        q = ~self.skip_flags
        if np.sum(q) > 0:
            man_snr = self.avg_snr[q, 0]
            man_sig_quoted = self.avg_sig[q, 0]
            # Here we want the second column of fit_sigs, the individual
            # derivations of astrometric uncertainty.
            man_sig_fit = self.fit_sigs[q, 1]
            man_log_inv_snr = np.log10(1/man_snr)
            man_log_err_fit = np.log10(man_sig_fit)
            man_log_err_quoted = np.log10(man_sig_quoted)
            for _index in range(len(man_log_inv_snr)):  # pylint: disable=consider-using-enumerate
                ax.arrow(man_log_inv_snr[_index], man_log_err_quoted[_index],
                         0, man_log_err_fit[_index]-man_log_err_quoted[_index], color='k',
                         zorder=30, head_width=0.06, head_length=0.03)

        ax.set_xlim(*xlims)
        ax.set_ylim(*ylims)
        if not self.use_photometric_uncertainties:
            ax.legend(fontsize=10)
        if usetex:
            ax.set_xlabel(r'$\log_{10}$(1 / SNR)')
        else:
            ax.set_xlabel(r'log10(1 / SNR)')
        if usetex:
            ax.set_ylabel(rf'$\log_{10}$(Quoted{mag_label} uncertainty / arcsecond)')
        else:
            ax.set_ylabel(rf'log10(Quoted{mag_label} uncertainty / arcsecond)')

        ax = plt.subplot(gs[1])
        q = (obj_err < 1) & (_snr > 1) & (obj_err > 0) & ~np.isnan(obj_mag)
        log_inv_snr = np.log10(1 / _snr[q])
        h, x, y = np.histogram2d(log_inv_snr, obj_mag[q], bins=(100, 101))
        ax.pcolormesh(x, y, h.T, edgecolors='face', cmap='viridis', rasterized=True)
        ylims = ax.get_ylim()
        xlims = ax.get_xlim()

        q = ~self.skip_flags
        if np.sum(q) > 0:
            man_snr = self.avg_snr[q, 0]
            man_mag = self.mag_array[q]
            man_log_inv_snr = np.log10(1/man_snr)
            ax.plot(man_log_inv_snr, man_mag, 'kx')

            # m = -2.5 log10(flux), SNR = flux/B, thus
            # m = -2.5 log10(B) - 2.5 log10(SNR), so
            # m = -2.5 log10(B) + 2.5 x.
            # Pin x and y (m) to the last data point for scaling.
            _i = np.argmax(man_mag)
            log_B = (man_mag[_i] - (2.5 * man_log_inv_snr[_i])) / (-2.5)  # pylint: disable=invalid-name
            ax.plot(x, 2.5 * x - 2.5 * log_B, 'k-', label='Background-dominated')

            # Now SNR = sqrt(flux), if flux_err = sqrt(flux).
            # Thus m = -2.5 log10(SNR^2) = -5 log10(SNR) = 5 x.
            # This also needs some kind of + C since we don't zeropoint, though.
            _j = np.argmin(man_mag)
            C = man_mag[_j] - (5 * man_log_inv_snr[_j])
            ax.plot(x, 5 * x + C, 'r--', label='Photon-limited')

        ax.set_xlim(*xlims)
        ax.set_ylim(*ylims)
        if np.sum(q) > 0:
            ax.legend(fontsize=10)
        if usetex:
            ax.set_xlabel(r'$\log_{10}$(1 / SNR)')
        else:
            ax.set_xlabel(r'log10(1 / SNR)')
        ax.set_xlabel(f'{self.mag_names[self.unc_index]} / mag')

        ax = plt.subplot(gs[2])
        q = (obj_err < 1) & (_snr > 1) & (obj_err > 0) & ~np.isnan(obj_mag)
        log_err = np.log10(obj_err[q])
        h, x, y = np.histogram2d(obj_mag[q], log_err, bins=(100, 101))
        ax.pcolormesh(x, y, h.T, edgecolors='face', cmap='viridis', rasterized=True)
        ylims = ax.get_ylim()
        xlims = ax.get_xlim()

        q = ~self.skip_flags
        if np.sum(q) > 0:
            man_mag = self.mag_array[q]
            man_sig_quoted = self.avg_sig[q, 0]
            # Remember, individually fit not parameterisation.
            man_sig_fit = self.fit_sigs[q, 1]
            man_log_err_fit = np.log10(man_sig_fit)
            man_log_err_quoted = np.log10(man_sig_quoted)
            for _index in range(len(man_mag)):  # pylint: disable=consider-using-enumerate
                ax.arrow(man_mag[_index], man_log_err_quoted[_index],
                         0, man_log_err_fit[_index]-man_log_err_quoted[_index], color='k',
                         zorder=30, head_width=0.06, head_length=0.03)

        ax.set_xlim(*xlims)
        ax.set_ylim(*ylims)
        ax.set_xlabel(f'{self.mag_names[self.unc_index]} / mag')
        if usetex:
            ax.set_ylabel(rf'$\log_{10}$(Quoted{mag_label} uncertainty / arcsecond)')
        else:
            ax.set_ylabel(rf'log10(Quoted{mag_label} uncertainty / arcsecond)')
        plt.tight_layout()
        plt.savefig(f'{self.save_folder}/pdf/{self.magnitude_append}'
                    f'histogram_mag_vs_sig_vs_snr_{self.file_name}.pdf')
        plt.close()

    def finalise_summary_plots(self):
        """
        After running all of the sightlines' fits, generate final
        summary plots of the sig-sig relations, quality of fits, and
        resulting "m" and "n" scaling parameters.
        """

        loop_len = len(self.mag_names) if self.use_photometric_uncertainties else 1
        plt.figure('single_sigsig')
        for unc_index in range(loop_len):
            x_array = np.linspace(0, self.ax_b_sing[unc_index].get_xlim()[1], 1000)
            self.ax_b_sing[unc_index].plot(x_array, x_array, 'g:')
            self.ax_b_sing[unc_index].set_ylim(0.95 * self.ylims_sing[0], 1.05 * self.ylims_sing[1])
            self.ax_b_log_sing[unc_index].plot(x_array, x_array, 'g:')
            self.ax_b_log_sing[unc_index].set_xscale('log')
            self.ax_b_log_sing[unc_index].set_yscale('log')

            if usetex:
                if not self.use_photometric_uncertainties:
                    self.ax_b_sing[unc_index].set_xlabel(r'Quoted astrometric $\sigma$ / arcsecond')
                    self.ax_b_log_sing[unc_index].set_xlabel(r'Quoted astrometric $\sigma$ / arcsecond')
                else:
                    self.ax_b_sing[unc_index].set_xlabel(
                        rf'Quoted photometric {self.mag_names[unc_index]} $\sigma$ / arcsecond')
                    self.ax_b_log_sing[unc_index].set_xlabel(
                        rf'Quoted photometric {self.mag_names[unc_index]} $\sigma$ / arcsecond')
                self.ax_b_sing[unc_index].set_ylabel(r'Individually-fit astrometric $\sigma$ / arcsecond')
                self.ax_b_log_sing[unc_index].set_ylabel(r'Individually-fit astrometric $\sigma$ / arcsecond')
            else:
                if not self.use_photometric_uncertainties:
                    self.ax_b_sing[unc_index].set_xlabel(r'Quoted astrometric sigma / arcsecond')
                    self.ax_b_log_sing[unc_index].set_xlabel(r'Quoted astrometric sigma / arcsecond')
                else:
                    self.ax_b_sing[unc_index].set_xlabel(
                        rf'Quoted photometric {self.mag_names[unc_index]} sigma / arcsecond')
                    self.ax_b_log_sing[unc_index].set_xlabel(
                        rf'Quoted photometric {self.mag_names[unc_index]} sigma / arcsecond')
                self.ax_b_sing[unc_index].set_ylabel(r'Individually-fit astrometric sigma / arcsecond')
                self.ax_b_log_sing[unc_index].set_ylabel(r'Individually-fit astrometric sigma / arcsecond')
        plt.savefig(f'{self.save_folder}/pdf/summary_individual_sig_vs_sig.pdf')
        plt.close()

        gs = self.make_gridspec('fits_cdf', loop_len, 2, 0.8, 6)
        for unc_index in range(loop_len):
            if self.use_photometric_uncertainties:
                _m, _i = self.mn_poisson_cdfs[:, unc_index], self.ind_poisson_cdfs[:, unc_index]
            else:
                _m, _i = self.mn_poisson_cdfs, self.ind_poisson_cdfs
            for i, (f, label) in enumerate(zip([_m, _i], ['HyperParameter', 'Individual'])):
                ax_d = plt.subplot(gs[unc_index, i])
                for j, g in enumerate(f):
                    if g is None:
                        continue
                    sys.stdout.flush()
                    q = ~np.isnan(g)
                    p_cdf = g[q]
                    # Under the hypothesis all CDFs are "true" we should expect
                    # the distribution of CDFs to be linear with fraction of the way
                    # through the sorted list of CDFs -- that is, the CDF ranked 30%
                    # in order should be ~0.3.
                    q_sort = np.argsort(p_cdf)
                    filter_log_nans = p_cdf[q_sort] < 1
                    true_hypothesis_cdf_dist = (np.arange(1, len(p_cdf)+1, 1) - 0.5) / len(p_cdf)
                    c = self.cols[j % len(self.cols)]
                    ax_d.plot(np.log10(1 - p_cdf[q_sort][filter_log_nans]),
                              true_hypothesis_cdf_dist[filter_log_nans], ls='None', c=c, marker='.')
                    ax_d.plot(np.log10(1 - true_hypothesis_cdf_dist), true_hypothesis_cdf_dist, 'r--')
                if usetex:
                    if not self.use_photometric_uncertainties:
                        ax_d.set_xlabel(rf'$\log_{{10}}(1 - \mathrm{{{label} CDF}})$')
                    else:
                        ax_d.set_xlabel(rf'$\log_{{10}}(1 - \mathrm{{{label} CDF}}) '
                                        rf'({self.mag_names[unc_index]})$')
                else:
                    if not self.use_photometric_uncertainties:
                        ax_d.set_xlabel(rf'log10(1 - {label} CDF)')
                    else:
                        ax_d.set_xlabel(rf'log10(1 - {label} CDF) ({self.mag_names[unc_index]})')
                ax_d.set_ylabel('Fraction')
        plt.savefig(f'{self.save_folder}/pdf/summary_mn_ind_cdfs.pdf')
        plt.close()

        self.gs_mn_sky = self.make_gridspec('mn_sky', 2*loop_len, 2, 0.8, 5)
        for unc_index in range(loop_len):
            if self.use_photometric_uncertainties:
                _m, _n = self.mn_sigs[:, unc_index, 0], self.mn_sigs[:, unc_index, 1]
            else:
                _m, _n = self.mn_sigs[:, 0], self.mn_sigs[:, 1]
            mag_label = f' ({self.mag_names[unc_index]})' if self.use_photometric_uncertainties else ''
            for i, (f, label) in enumerate(zip([_m, _n], [f'm{mag_label}', f'n / arcsecond{mag_label}'])):
                ax = plt.subplot(self.gs_mn_sky[2*unc_index, i])
                img = ax.scatter(self.ax1_mids, self.ax2_mids, c=f, cmap='viridis')
                c = plt.colorbar(img, ax=ax, use_gridspec=True)
                c.ax.set_ylabel(label)
                ax1_name = 'l' if self.coord_system == 'galactic' else 'RA'
                ax2_name = 'b' if self.coord_system == 'galactic' else 'Dec'
                ax.set_xlabel(f'{ax1_name} / deg')
                ax.set_ylabel(f'{ax2_name} / deg')

                xlims = ax.get_xlim()
                ylims = ax.get_ylim()

                # Plot the Galactic and Ecliptic planes.
                for _b in [-20, 0, 20]:
                    l = np.linspace(0, 360, 10000)
                    b = np.zeros_like(l) + _b
                    c = SkyCoord(l=l, b=b, unit='deg', frame='galactic')
                    ra, dec = c.icrs.ra.degree, c.icrs.dec.degree
                    q = np.argsort(ra)
                    ra, dec = ra[q], dec[q]
                    ax.plot(ra, dec, ls='-', c='k' if _b == 0 else 'grey', alpha=0.7, zorder=-1)
                for _lat in [-20, 0, 20]:
                    lon = np.linspace(0, 360, 10000)
                    lat = np.zeros_like(lon) + _lat
                    c = SkyCoord(lon=lon, lat=lat, unit='deg', frame='barycentricmeanecliptic')
                    ra, dec = c.icrs.ra.degree, c.icrs.dec.degree
                    q = np.argsort(ra)
                    ra, dec = ra[q], dec[q]
                    ax.plot(ra, dec, ls='--', c='k' if _lat == 0 else 'grey', alpha=0.7, zorder=-1)

                ax.set_xlim(*xlims)
                ax.set_ylim(*ylims)

                ax = plt.subplot(self.gs_mn_sky[2*unc_index+1, i])
                ax.plot(self.avg_b_dens[:, unc_index] if self.use_photometric_uncertainties else
                        self.avg_b_dens, f, 'k.')
                if usetex:
                    ax.set_xlabel(rf'Average density / deg$^{-2}${mag_label}')
                else:
                    ax.set_xlabel(f'Average density / deg^-2{mag_label}')
                ax.set_ylabel(label)

        plt.tight_layout()
        plt.savefig(f'{self.save_folder}/pdf/summary_mn_sky.pdf')
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


def create_distances(a, b, nn_radius, a_ax1_ind, a_ax2_ind, b_ax1_ind, b_ax2_ind, coord_system):
    """
    Calculate nearest neighbour matches between two catalogues.

    Parameters
    ----------
    a : numpy.ndarray
        Array containing catalogue "a"'s sources. Must have astrometric
        coordinates as two of its columns.
    b : numpy.ndarray
        Catalogue "b"'s object array. Longitude and latitude must be two
        columns in the array.
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
