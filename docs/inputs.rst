****************
Input Parameters
****************

This page details the various inputs expected by `~macauff.CrossMatch` via its input parameter files. These are split into two main sections: inputs that both catalogues need separately (contained in their respective catalogue parameters file) and inputs that are defined with respect to the catalogue-catalogue cross-match.

These input parameters are required to be in ``crossmatch_params*.txt`` for the `Joint Parameters`_ options, and ``cat_a_params*.txt``, ``cat_b_params*.txt`` for the `Catalogue-specific Parameters`_ options, where the asterisk indicates a wildcard -- i.e., the text files must end ``.txt`` and begin e.g. ``crossmatch_params``,  but can contain text between those two parts of the file.

Depending on the size of the match being performed, it is likely that you may want to use the MPI parallelisation option to save runtime or memory overhead, splitting your matches into "chunks." Even if your cross-match area is small enough that you only have a single file per catalogue, this is treated as a single chunk. Each individual chunk -- or your only chunk -- is required to have its own set of three input text parameter files, within folders inside ``chunks_folder_path`` as passed to `~macauff.CrossMatch`. Most of the parameters within these files will be the same across all chunks -- you'll likely always want to include the perturbation component of the AUF or include photometric likelihoods, or use the same filters across all sub-catalogue cross-matches -- but some will vary on a per-chunk basis, most notably anything that involves astrometric coordinates, like ``cross_match_extent``.

The sub-folder structure should look something like::

    /path/to/your/chunks_folder_path
                                   ├─── 2017
                                   │       ├── crossmatch_params_2017.txt
                                   │       ├── cat_a_params_2017.txt
                                   │       └── cat_b_params_2017.txt
                                   │
                                   ├─── 2018
                                   │       ├── crossmatch_params_2018.txt
                                   │       ├── cat_a_params_2018.txt
                                   │       └── cat_b_params_2018.txt

where e.g. ``crossmatch_params_2017.txt`` and ``crossmatch_params_2018.txt`` would both contain all of the "Joint Parameters" parameters necessary to run the matches in their respective chunks; again, significant numbers of these would be the same, but e.g. ``cf_region_points`` would differ.

For a simple example of how this works for a one-chunk cross-match, see :doc:`quickstart`, and for more details on the inputs and outputs from individual functions within the cross-match process check the :doc:`documentation<macauff>`.

CrossMatch Inputs
=================

As well as the parameters required that are ingested through the input parameters files, `~macauff.CrossMatch` itself requires some input arguments.

- ``chunks_folder_path``: the directory in which the folders containing parameters files are stored

- ``use_mpi``: boolean flag for whether to parallelise distribution of chunks using MPI. If ``False`` chunks will be run in serial; defaults to ``True``, but with a fallback for if the appropriate module is not available.

If you do not want to (or cannot) use MPI to distribute larger cross-match runs, with numerous chunks that will take significant compute time to run, then the first two inputs, combined with with ``use_mpi=False``, are all you need to consider. However, if you wish to use MPI then the remaining keyword arguments control its use:

- ``resume_file_path``: location of where to store the file that contains information on whether a particular chunk (e.g. ``2018`` in the example above) is finished, to avoid re-running parameters if matches have to be re-started for any reason. Optional, defaulting to having no such checkpointing capabilities.

- ``walltime``: related to ``resume_file_path``, this controls how long a singular run of the cross-matching can be performed for on the particular machine before being stopped (e.g. if being run on a shared machine). Default is no such consideration for duration of match runtimes, but otherwise expects a string in the format ``H:M:S``.

- ``end_within``: combined with ``walltime`` this avoids unnecessarily starting new chunks being run by the MPI manager if within ``end_within`` of the ``walltime`` stoppage. Also expects ``H:M:S`` string format.

- ``polling_rate``: controls the speed at which the MPI manager checks for finished jobs and distributes follow-up jobs to the worker.

Joint Parameters
================

These parameters are only provided in the single, common-parameter input file, and given as ``joint_file_path`` in `~macauff.CrossMatch`.

There are some parameters that must be given in all runs:

``joint_folder_path``, ``include_perturb_auf``, ``include_phot_like``, ``use_phot_priors``, ``cross_match_extent``, ``pos_corr_dist``, ``cf_region_type``, ``cf_region_frame``, ``cf_region_points``, ``real_hankel_points``, ``four_hankel_points``, ``four_max_rho``, ``int_fracs``, ``make_output_csv``, and ``n_pool``;

options which need to be supplied if ``make_output_csv`` is ``True``:

``output_csv_folder``, ``match_out_csv_name``, and ``nonmatch_out_csv_name``;

and those options which only need to be supplied if ``include_perturb_auf`` is ``True``:

``num_trials``, and ``d_mag``.

.. note::
    ``num_trials`` and ``d_mag`` currently need to be supplied if either ``correct_astrometry`` option in the two `Catalogue-specific Parameters`_ config files is ``True`` as well.

Common Parameter Description
----------------------------

``joint_folder_path``

The top-level folder location, into which all intermediate files and folders are placed, when created during the cross-match process. This can either be an absolute file path, or relative to the folder from which your script that called `CrossMatch()` is based.

``include_perturb_auf``

Flag for whether to include the simulated effects of blended sources on the measured astrometry in the two catalogues or not.

.. note::
    If ``include_perturb_auf`` is ``True`` then ``dustmaps`` will be called to obtain line-of-sight extinction values. You will need the SFD dustmaps to be pre-downloaded; to do so before you call the cross-match procedure you must run ``dustmaps.sfd.fetch()``; see the ``dustmaps`` documentation for more details on how to specific a particular download location.

``include_phot_like``

Flag for the inclusion of the likelihood of match or non-match based on the photometric information in the two catalogues.

``use_phot_priors``

Flag to determine whether to calculate the priors on match or non-match using the photometry (if set to ``True``) or calculate them based on a naive asymmetric density argument (``False``).

``cross_match_extent``

The maximum extent of the matching process. When not matching all-sky catalogues, these extents are used to eliminate potential matches within "island" overlap range of the edge of the data, whose potential incompleteness renders the probabilities of match derived uncertain. Must be of the form ``lower-longitude upper-longitude lower-latitude upper-latitude``; accepts four space-separated floats.

.. note::
    In cases where the boundary defining the cross-match overlaps the 0-360 boundary of the given coordinate system, the longitudes should be given relative to 0 degrees. For example, if we had a boundary that ran from 350 degrees up to 360 (0) degrees, and on to 10 degrees, ``cross_match_extent`` would have for its input longitudes ``-10 10``. Internally the software is able to handle the boundary for source coordinates, but requires the extents to be correctly input for these regions.

``pos_corr_dist``

The floating point precision number determining the maximum possible separation between two sources in opposing catalogues.

``cf_region_type``

This flag controls whether the areas in which photometry-related variables (likelihoods, priors, etc.) are calculated is determined by ``rectangle`` -- evenly spaced longitude/latitude pairings -- or ``points`` -- tuples of randomly placed coordinates.

``cf_region_frame``

This allows either ``equatorial`` or ``galactic`` frame coordinates to be used in the match process.

``cf_region_points``

The list of pointings for which to run simulations of perturbations due to blended sources, if applicable. If ``cf_region_type`` is ``rectangle``, then ``cf_region_points`` accepts six numbers: ``start longitude, end longitude, number of longitude points, start latitude, end latitude, number of latitude points``; if ``points`` then tuples must be of the syntax ``(a, b), (c, d)`` where ``a`` and ``c`` are RA or Galactic Longitude, and ``b`` and ``d`` are Declination or Galactic Latitude.

.. note::
    For consistency with ``cross_match_extent``, ``cf_region_points`` longitudes may be given with negative coordinates for cases where the region ``cross_match_extent`` defines is both above and below zero degrees, but they can also be given within the [0, 360] phase space, as 350 degrees and -10 degrees are handled the same where needed by ``cf_region_points``.

``real_hankel_points``

The integer number of points, for Hankel (two-dimensional Fourier) transformations, in which to approximate the fourier transformation integral of the AUFs.

``four_hankel_points``

The integer number of points for approximating the inverse Hankel transformation, representing the convolution of two real-space AUFs.

``four_max_rho``

The largest fourier-space value, up to which inverse Hankel transformation integrals are considered. Should typically be larger than the inverse of the smallest typical centroiding Gaussian one-dimensional uncertainty.

``n_pool``

Determines how many CPUs are used when parallelising within ``Python`` using ``multiprocessing``.

``int_fracs``

The integral fractions of the various so-called "error circles" used in the cross-match process. Should be space-separated floats, in the order of: bright error circle fraction, "field" error circle fraction, and potential counterpart cutoff limit. Note that bright and "field" fractions should be reasonably separated in value (more than maybe 0.1) to avoid biasing results that use both to measure photometry-based priors, when applicable.

``output_csv_folder``

The folder path into which to save the stored ``.csv`` files that are created if ``make_output_csv`` is ``True``.

``match_out_csv_name``

Name of the band-merged, cross-matched dataset of counterpart associations and accompanying metadata, including the appropriate file extension (currently ``.csv``).

``nonmatch_out_csv_name``

Filename to save out the respective non-match catalogue objects and metadata to. Will have appended to the front ``cat_name`` to distinguish the two non-match files. ``nonmatch_out_csv_name`` should contain the appropriate file extension.

``num_trials``

The number of PSF realisations to draw when simulating the perturbation component of the AUF. Should be an integer. Only required if ``include_perturb_auf`` is ``True``.

``d_mag``

Bin sizes for magnitudes used to represent the source number density used in the random drawing of perturbation AUF component PSFs. Should be a single float. Only required if ``include_perturb_auf`` is ``True``.


Catalogue-specific Parameters
=============================

These parameters are required in two separate files, one per catalogue to be cross-matched, the files ``cat_a_params.txt`` and ``cat_b_params.txt`` read from sub-folders within ``chunks_folder_path`` as passed to `~macauff.CrossMatch`.

These can be divided into those inputs that are always required:

``cat_folder_path``, ``cat_name``, ``filt_names``, ``auf_folder_path``, ``auf_region_type``, ``auf_region_frame``, ``auf_region_points``, ``correct_astrometry``, and ``compute_snr_mag_relation``;

those that are only required if the `Joint Parameters`_ option ``include_perturb_auf`` is ``True``:

``fit_gal_flag``, ``run_fw_auf``, ``run_psf_auf``, ``psf_fwhms``, ``snr_mag_params_path``, ``download_tri``, ``tri_set_name``, ``tri_filt_names``, ``tri_filt_num``, ``tri_maglim_faint``, ``tri_num_faint``, ``gal_al_avs``, ``dens_dist``, ``dens_hist_tri_location``, ``tri_model_mags_location``, ``tri_model_mag_mids_location``, ``tri_model_mags_interval_location``, and ``tri_n_bright_sources_star_location``;

parameters required if ``run_psf_auf`` is ``True``:

``dd_params_path`` and ``l_cut_path``;

the inputs required in each catalogue parameters file if ``fit_gal_flag`` is ``True`` (and hence ``include_perturb_auf`` is ``True``):

``gal_wavs``, ``gal_zmax``, ``gal_nzs``, ``gal_aboffsets``, and ``gal_filternames``;

inputs required if ``make_output_csv`` is ``True``:

``input_csv_folder``, ``cat_csv_name``, ``cat_col_names``, ``cat_col_nums``, ``input_npy_folder``, ``csv_has_header``, ``extra_col_names``, and ``extra_col_nums``;

the inputs required if either ``correct_astrometry`` or ``compute_snr_mag_relation`` are ``True``:

``correct_astro_save_folder``, ``csv_cat_file_string``, ``mag_indices``, ``mag_unc_indices``, and ``pos_and_err_indices``;

and the inputs required if ``correct_astrometry`` is ``True``:

``best_mag_index``, ``nn_radius``, ``ref_csv_cat_file_string``, ``correct_mag_array``, ``correct_mag_slice``, ``correct_sig_slice``, ``chunk_overlap_col``, ``best_mag_index_col``, and ``saturation_magnitudes``.

.. note::
    ``run_fw_auf``, ``run_psf_auf``, ``psf_fwhms``, ``snr_mag_params_path``, ``download_tri``, ``tri_set_name``, ``tri_filt_names``, ``tri_filt_num``, ``tri_maglim_faint``, ``tri_num_faint``, ``dens_dist``, ``dd_params_path``, ``l_cut_path``, ``gal_wavs``, ``gal_zmax``, ``gal_nzs``, ``gal_aboffsets``, ``gal_filternames``, and ``gal_al_avs`` are all currently required if ``correct_astrometry`` is ``True``, bypassing the nested flags above. For example, ``dens_dist`` is required as an input if ``include_perturb_auf`` is ``True``, or if ``correct_astrometry`` is set. This means that ``AstrometricCorrections`` implicitly always runs and fits for a full Astrometric Uncertainty Function.

.. note::
    ``snr_mag_params_path`` is currently also required if ``compute_snr_mag_relation`` is ``True``, bypassing the above flags. It is therefore currently a required input if any one of ``include_perturb_auf``, ``correct_astrometry``, or ``compute_snr_mag_relation`` are set to ``True``.


Catalogue Parameter Description
-------------------------------

``cat_folder_path``

The folder containing the three files (see :doc:`quickstart` for more details) describing the given input catalogue. Can either be an absolute path, or relative to the folder from which the script was called.

``cat_name``

The name of the catalogue. This is used to generate intermediate folder structure within the cross-matching process, and during any output file creation process.

``filt_names``

The filter names of the photometric bandpasses used in this catalogue, in the order in which they are saved in ``con_cat_photo``. These will be used to describe any output data files generated after the matching process. Should be a space-separated list.

``auf_folder_path``

The folder into which the Astrometric Uncertainty Function (AUF) related files will be, or have been, saved. Can also either be an absolute or relative path, like ``cat_folder_path``. Alternatively, this can (and must) be ``None`` if all parameters related to loading pre-computed TRILEGAL histograms (``dens_hist_tri_location`` et al.) are provided.

``auf_region_type``

Similar to ``cf_region_type``, flag indicating which definition to use for determining the pointings of the AUF simulations; accepts either ``rectangle`` or ``points``. If ``rectangle``, then ``auf_region_points`` will map out a rectangle of evenly spaced points, otherwise it accepts pairs of coordinates at otherwise random coordinates.

``auf_region_frame``

As with ``auf_region_frame``, this flag indicates which frame the data, and thus AUF simulations, are in. Can either be ``equatorial`` or ``galactic``, allowing for data to be input either in Right Ascension and Declination, or Galactic Longitude and Galactic Latitude.

``auf_region_points``

Based on ``auf_region_type``, this must either by six space-separated floats, controlling the start and end, and number of, longitude and latitude points in ``start lon end lon # steps start lat end lat #steps`` order (see ``cf_region_points``), or a series of comma-separated tuples cf. ``(a, b), (c, d)``.

.. note::
    For consistency with ``cross_match_extent``, ``auf_region_points`` longitudes may be given with negative coordinates for cases where the region ``cross_match_extent`` defines is both above and below zero degrees, but they can also be given within the [0, 360] phase space, as 350 degrees and -10 degrees are handled the same where needed by ``auf_region_points``.

``correct_astrometry``

In cases where catalogues have unreliable *centroid* uncertainties, before catalogue matching occurs the dataset can be fit for systematic corrections to its quoted astrometric precisions through ensemble match separation distance distributions to a higher-precision dataset (see the :doc:`Processing<pre_post_process>` section). This flag controls whether this is performed on a chunk-by-chunk basis during the initialisation step of ``CrossMatch``.

.. note::
    If ``correct_astrometry`` is ``True`` then ``dustmaps`` will be called to obtain line-of-sight extinction values. You will need the SFD dustmaps to be pre-downloaded; to do so before you call the cross-match procedure you must run ``dustmaps.sfd.fetch()``; see the ``dustmaps`` documentation for more details on how to specific a particular download location.

``compute_snr_mag_relation``

This flag can be ``False`` if the relationship between signal-to-noise ratio and magnitude is pre-computed; otherwise it indicates that the functional form of SNR vs brightness should be derived for the particular catalogue in question.

``fit_gal_flag``

Optional flag for whether to include simulated external galaxy counts, or just include Galactic sources when deriving the perturbation component of the AUF. Only needed if ``include_perturb_auf`` is ``True``.

``run_fw_auf``

Boolean flag controlling the option to include the flux-weighted algorithm for determining the centre-of-light perturbation with AUF component simulations. Only required if  ``include_perturb_auf`` is ``True``.

``run_psf_auf``

Complementary flag to ``run_fw_auf``, indicates whether to run background-dominated, PSF photometry algorithm for the determination of perturbation due to hidden contaminant objects. If both this and ``run_fw_auf`` are ``True`` a signal-to-noise-based weighting between the two algorithms is implemented. Must be provided if  ``include_perturb_auf`` is ``True``.

``psf_fwhms``

The Full-Width-At-Half-Maximum of each filter's Point Spread Function (PSF), in the same order as in ``filt_names``. These are used to simulate the PSF if ``include_perturb_auf`` is set to ``True``, and are unnecessary otherwise. Should be a space-separated list of floats.

``snr_mag_params_path``

File path, either absolute or relative to the location of the script the cross-matches are run from, of a binary ``.npy`` file containing the parameterisation of the signal-to-noise ratio of sources as a function of magnitude, in a series of given sightlines. Must be of shape ``(N, M, 5)`` where ``N`` is the number of filters in ``filt_names`` order, ``M`` is the number of sightlines for which SNR vs mag has been derived, and the 5 entries for each filter-sightline combination must be in order ``a``, ``b``, ``c``, ``coord1`` (e.g. RA), and ``coord2`` (e.g. Dec). See pre-processing for more information on the meaning of those terms and how ``snr_mag_params`` is used.

``download_tri``

Boolean flag, indicating whether to re-download a TRILEGAL simulation in a given ``auf_region_points`` sky coordinate, once it has successfully been run, and to overwrite the original simulation data or not. Optional if ``include_perturb_aufs`` is ``False``. Alternatively, this can (and must) be ``None`` if all parameters related to loading pre-computed TRILEGAL histograms (``dens_hist_tri_location`` et al.) are provided.

``tri_set_name``

The name of the filter set used to simulate the catalogue's sources in TRILEGAL [#]_. Used to interact with the TRILEGAL API; optional if ``include_perturb_aufs`` is ``False``. Alternatively, this can (and must) be ``None`` if all parameters related to loading pre-computed TRILEGAL histograms (``dens_hist_tri_location`` et al.) are provided.

``tri_filt_names``

The names of the filters, in the same order as ``filt_names``, as given in the data ``tri_set_name`` calls. Optional if ``include_perturb_aufs`` is ``False``. Alternatively, this can (and must) be ``None`` if all parameters related to loading pre-computed TRILEGAL histograms (``dens_hist_tri_location`` et al.) are provided.

``tri_filt_num``

The one-indexed column number of the magnitude, as determined by the column order of the saved data returned by the TRILEGAL API, to which to set the maximum magnitude limit for the simulation. Optional if ``include_perturb_aufs`` is ``False``. Alternatively, this can (and must) be ``None`` if all parameters related to loading pre-computed TRILEGAL histograms (``dens_hist_tri_location`` et al.) are provided.

``tri_maglim_faint``

This is the float that represents the magnitude down to which to simulate TRILEGAL sources in the full-scale simulation, bearing in mind the limiting magnitude cut of the public API but also making sure this value is sufficiently faint that it contains all potentially perturbing objects for the dynamic range of this catalogue (approximately 10 magnitudes fainter than the limiting magnitude of the survey). Alternatively, this can (and must) be ``None`` if all parameters related to loading pre-computed TRILEGAL histograms (``dens_hist_tri_location`` et al.) are provided.

``tri_num_faint``

Integer number of objects to draw from the TRILEGAL simulation -- affecting the area of simulation, up to the limit imposed by TRILEGAL -- down to the full ``tri_maglim_faint`` magnitude. Alternatively, this can (and must) be ``None`` if all parameters related to loading pre-computed TRILEGAL histograms (``dens_hist_tri_location`` et al.) are provided.

``dens_hist_tri_location``

The location on disk of a numpy array, shape ``(len(filt_names), M)`` where ``M`` is a consistent number of magnitude bins, of differential source counts for a given TRILEGAL simulation, in each filter for a specific catalogue. Alternatively, should be ``None`` if ``auf_folder_path`` and associated parameters for the running of TRILEGAL histogram generation within the cross-match run are given.

``tri_model_mags_location``

The location on disk of a numpy array, shape ``(len(filt_names), M)`` where ``M`` is a consistent number of magnitude bins, of the left-hand magnitude bin edges of differential source counts for a given TRILEGAL simulation, in each filter for a specific catalogue. Alternatively, should be ``None`` if ``auf_folder_path`` and associated parameters for the running of TRILEGAL histogram generation within the cross-match run are given.

``tri_model_mag_mids_location``

The location on disk of a numpy array, shape ``(len(filt_names), M)`` where ``M`` is a consistent number of magnitude bins, of magnitude bin-middles of differential source counts for a given TRILEGAL simulation, in each filter for a specific catalogue. Alternatively, should be ``None`` if ``auf_folder_path`` and associated parameters for the running of TRILEGAL histogram generation within the cross-match run are given.

``tri_model_mags_interval_location``

The location on disk of a numpy array, shape ``(len(filt_names), M)`` where ``M`` is a consistent number of magnitude bins, of magnitude bin widths of differential source counts for a given TRILEGAL simulation, in each filter for a specific catalogue. Alternatively, should be ``None`` if ``auf_folder_path`` and associated parameters for the running of TRILEGAL histogram generation within the cross-match run are given.

``tri_n_bright_sources_star_location``

The location on disk of a ``.npy`` file containing the number of simulated bright TRILEGAL objects in the input simulation, one per filter. Should be a 1-D numpy array of shape ``(len(filt_names),)``. Alternatively, should be ``None`` if ``auf_folder_path`` and associated parameters for the running of TRILEGAL histogram generation within the cross-match run are given.

``dd_params_path``

File path containin the ``.npy`` file describing the parameterisations of perturbation offsets due to single hidden contaminating, perturbing objects in the ``run_psf_auf`` background-dominated, PSF photometry algorithm case. See pre-processing documentation for more details on this, and how to generate this file if necessary.

``l_cut_path``

Alongside ``dd_params_path``, path to the ``.npy`` file containing the limiting flux cuts at which various PSF photometry perturbation algorithms apply. See pre-processing documentation for the specifics and how to generate this file if necesssary.

``dens_dist``

The radius, in arcseconds, within which to count internal catalogue sources for each object, to calculate the local source density. Used to scale TRILEGAL simulated source counts to match smaller scale density fluctuations. Only required if ``include_perturb_auf`` is ``True``.

``gal_wavs``

List of floating point central wavelengths, in the order filters are given in ``filt_names``, for each filter, in microns. Used to approximate Schechter function parameters for deriving galaxy counts. Must be given if ``fit_gal_flag`` is ``True``, and hence only required if ``include_perturb_auf`` is ``True``.

``gal_zmax``

Maximum redshift ``z`` to calculate galactic densities out to for Schechter function derivations, one per ``gal_wavs`` point. Only needed if ``fit_gal_flag`` is ``True``.

``gal_nzs``

Integer number of redshift points, from zero to ``gal_zmax``, to evaluate Schechter functions on, for each filter. Must be given if ``fit_gal_flag`` is ``True``.

``gal_aboffsets``

For each filter, floating point offset between the given filter's zeropoint system and that of the AB magnitude system -- in the same that m = m_AB - offset_AB -- for each filter. If ``fit_gal_flag`` is ``True``, must be provided.

``gal_filternames``

Name of each filter as appropriate for providing to ``speclite`` for each filter. See `~macauff.generate_speclite_filters` for how to create appropriate filters if not provided by the module by default. Required if ``fit_gal_flag`` is ``True``.

``gal_al_avs``

Differential extinction relative to the V-band for each filter, a set of space-separated floats. Must be provided if ``include_perturb_auf`` is ``True``.

``input_csv_folder``

Location of the catalogue's original input ``.csv`` file, generally converted to a binary file for use within the main code via ``csv_to_npy``.

``cat_csv_name``

Name, including extension, of the ``.csv`` file located in ``input_csv_folder``.

``cat_col_names``

The names of the mandatory columns from each respctive catalogue. Should contain at least the column name for the name or ID of the object, and the names of the columns containing the two orthogonal sky coordinates, as well as the names of each column containing magnitude information to be transferred to the output match and non-match files.

``cat_col_nums``

For each column name in ``cat_col_names``, ``cat_col_nums`` is the zero-indexed position of the column. For example, if we had ``['ID', 'RA', 'Dec', 'V']`` as our ``cat_col_names``, we might have ``[0, 1, 2, 5]`` as our ``cat_col_nums``, in which our designation and coordinates are the first three columns, but our V-band magnitude is a few columns down.

``input_npy_folder``

The location on disk of the folder that contains the converted binary ``.npy`` files used as inputs to the software. Likely the same as ``cat_folder_path``, or ``None`` can be given if we do not need to load a converted astrometric uncertainty from the binary files and instead can rely solely on the original quoted astrometric uncertainty from the ``.csv`` files.

``csv_has_header``

A boolean, yes/no, for whether there is a header in the first line of the ``.csv`` input catalogue files (``yes``), or if the first line is a line of data (``no``).

``extra_col_names``

Analogous to ``cat_col_names``, a list of the additional columns from the original csv catalogue file that we wish to add to the match and non-match output files.

``extra_col_nums``

The zero-indexed positions of each corresponding column in ``extra_col_names``, much the same as in ``cat_col_nums``, but for additional, optional columns we may wish to transfer from input to output dataset.

``best_mag_index``

For the purposes of correcting systematic biases in a given catalogue, a single photometric band is used. ``best_mag_index`` indicates which filter to use -- e.g., ``best_mag_index = 0`` says to use the first filter as given in ``filt_names`` or ``mag_indices``. Must be a single integer value no larger than ``len(filt_names)-1``.

``nn_radius``

Nearest neighbour radius out to which to search for potential counterparts for the purposes of ensemble match separation distributions; should be a single float.

``correct_astro_save_folder``

File path, relative or absolute, into which to save files as generated by the astrometric correction process.

``csv_cat_file_string``

Path and filename, including extension, all in a single string, containing the location of each correction sightline's dataset to test. Must contain the appropriate number of string format ``{}`` identifiers depending on ``coord_or_chunk`` -- in this case, a single "chunk" identifier for corrections done through ``CrossMatch``. For example, ``/your/path/to/file/data_{}.csv`` where each "chunk" is saved into a csv file called ``data_1``, ``data_2``, ``data_104`` etc.

``ref_csv_cat_file_string``

Similar to ``csv_cat_file_string``, but the path and filename, including extension, of the *reference* dataset used in the matching process. These chunks should correspond one-to-one with those used in ``csv_cat_file_string`` -- i.e., ``data_1.csv`` in ``/your/path/to/file`` should be the same region of the sky as the reference catalogue in ``/another/path/to/elsewhere/reference_data_1.csv``, potentially with some buffer overlap to avoid false matches at the edges.

``correct_mag_array``

List of magnitudes at which to evaluate the distribution of matches to the higher-astrometric-precision dataset in the chosen ``best_mag_index`` filter. Accepts a list of floats.

``correct_mag_slice``

Corresponding to each magnitude in ``correct_mag_array``, each element of this list of floats should be a width around each ``correct_mag_array`` element to select sources, ensuring a small sub-set of similar brightness objects are used to determine the Astrometric Uncertainty Function of.

``correct_sig_slice``

Elementwise with ``correct_mag_array`` and ``correct_mag_slice``, a list of floats of widths of astrometric precision to select a robust sub-sample of objects in each magnitude bin for, ensuring a self-similar AUF.

``pos_and_err_indices``

A list of either three or six whitespace-separated integers. If ``correct_astrometry`` is ``True``, a list of six integers, the first three elements of which are the zero-indexed indices into the *reference* catalogue .csv file (``ref_csv_cat_file_string``) for the longitudinal coordinate, latitudinal coordinate, and circular astrometric precision respectively, followed by the lon/lat/uncert of the *input* catalogue. For example, ``0 1 2 10 9 8`` suggests that the reference catalogue begins with the position and uncertainty of its objects while the catalogue "a" or "b" sources have, in their original .csv file, a backwards list of coordinates and precisions towards the final columns of the filing system. If ``compute_snr_mag_relation`` is ``True``, then only three integers should be passed, the respective coordinates for its own catalogue (dropping the indices of the reference catalogue); in the above example we would therefore only pass ``10 9 8``.

``mag_indices``

Just for the input catalogue, a list of ``len(filt_names)`` space-separated integers detailing the zero-indexed column number of the magnitudes in the dataset.

``mag_unc_indices``

Similar to ``mag_indices``, a list of ``len(mag_indices)`` space-separated integers, one for each column in ``mag_indices`` for where the corresponding uncertainty column is held for each magnitude in the input .csv file.

``chunk_overlap_col``

Column number in the original csv file for the column containing the boolean flag indicating whether sources are in the "halo" or "core" of the chunk. Used within ``CrossMatch`` after calling ``AstrometricCorrections`` to create final npy files via ``csv_to_npy``. Should be a single integer number.

``best_mag_index_col``

The zero-indexed integer column number in the original input csv file used in ``AstrometricCorrections`` that corresponds to the column containing the highest quality detection for each source in the catalogue, used when calling ``csv_to_npy``.

``use_photometric_uncertainties``

Boolean flag indicating whether the astrometric or photometric uncertainties of the input catalogue should be used to derive the astrometric uncertainties from ensemble statistics in ``AstrometricCorrections``.

``saturation_magnitudes``

A list, one float per filter, of the magnitudes in the given filter at which the telescope or survey saturates, used in the filtering of source counts for model-fitting purposes in ``AstrometricCorrections``.


Parameter Dependency Graph
==========================

The inter-dependency of input parameters on one another, and the output ``CrossMatch`` attribute if different, are given below::

    ├─> include_perturb_auf
    │                     ├─> num_trials
    │                     ├─> d_mag
    │                     ├─* dens_dist
    │                     ├─* fit_gal_flag
    │                     │             ├─* gal_wavs
    │                     │             ├─* gal_zmax
    │                     │             ├─* gal_nzs
    │                     │             ├─* gal_aboffsets
    │                     │             └─* gal_filternames
    │                     ├─* snr_mag_params_path -> snr_mag_params
    │                     ├─* tri_set_name[3a]
    │                     ├─* tri_filt_names[3a]
    │                     ├─* tri_filt_num[3a]
    │                     ├─* download_tri[3a]
    │                     ├─* psf_fwhms
    │                     ├─* run_fw_auf
    │                     ├─* run_psf_auf
    │                     │             ├─* dd_params_path -> dd_params
    │                     │             └─* l_cut_path -> l_cut
    │                     ├─* tri_maglim_faint[3a]
    │                     ├─* tri_num_faint[3a]
    │                     ├─* gal_al_avs
    │                     ├─* dens_hist_tri_location[3b] -> dens_hist_tri_list
    │                     ├─* tri_model_mags_location[3b] -> tri_model_mags_list
    │                     ├─* tri_model_mag_mids_location[3b] -> tri_model_mag_mids_list
    │                     ├─* tri_model_mags_interval_location[3b] -> tri_model_mags_interval_list
    │                     ├─* tri_model_mags_uncert_location[3b] -> tri_model_mags_uncert_list
    │                     └─* tri_n_bright_sources_star_location[3b] -> tri_n_bright_sources_star_list
    ├─> include_phot_like
    ├─> use_phot_priors
    ├─> cf_region_type
    ├─> cf_region_frame[2]
    ├─> cf_region_points
    ├─> joint_folder_path
    ├─> pos_corr_dist
    ├─> real_hankel_points
    ├─> four_hankel_points
    ├─> four_max_rho
    ├─> cross_match_extent
    ├─> int_fracs
    ├─> make_output_csv
    │                 ├─> output_csv_folder
    │                 ├─> match_out_csv_name
    │                 ├─> nonmatch_out_csv_name
    │                 ├─* input_csv_folder
    │                 ├─* cat_csv_name
    │                 ├─* cat_col_names
    │                 ├─* cat_col_nums
    │                 ├─* input_npy_folder
    │                 ├─* csv_has_header
    │                 ├─* extra_col_names
    │                 └─* extra_col_nums
    ├─> n_pool
    ├─* auf_region_type
    ├─* auf_region_frame[2]
    ├─* auf_region_points
    ├─* filt_names
    ├─* cat_name
    ├─* auf_folder_path[3a]
    ├─* cat_folder_path
    ├─* correct_astrometry[1]
    │                    ├─* correct_astro_save_folder
    │                    ├─* csv_cat_file_string
    │                    ├─* pos_and_err_indices
    │                    ├─* mag_indices
    │                    ├─* mag_unc_indices
    │                    ├─* best_mag_index
    │                    ├─* nn_radius
    │                    ├─* ref_csv_cat_file_string
    │                    ├─* correct_mag_array
    │                    ├─* correct_mag_slice
    │                    ├─* correct_sig_slice
    │                    ├─* chunk_overlap_col
    │                    ├─* best_mag_index_col
    │                    └─* use_photometric_uncertainties
    ├─* compute_snr_mag_relation[1]
    │                          ├─* correct_astro_save_folder
    │                          ├─* csv_cat_file_string
    │                          ├─* pos_and_err_indices
    │                          ├─* mag_indices
                               └─* mag_unc_indices

List directories end in ``->`` for ``joint`` parameters, ``-*`` for ``catalogue`` parameters. ``catalogue`` level items will have ``a_`` or ``b_`` prepended, depending on which "side" of the cross-match they are from. Items with a second keyword after an arrow ``->`` are the names of the attributes that are saved to ``CrossMatch``, usually when the input parameter is a location on disk.

| [1] - cannot both be ``True``
| [2] - must be the same
| [3] - only one set of [3a] and [3b] should be given, the others should be passed as ``None``

.. rubric:: Footnotes

.. [#] Please see `here <http://stev.oapd.inaf.it/~webmaster/trilegal_1.6/papers.html>`_ to view the TRILEGAL papers to cite, if you use this software in your publication.

