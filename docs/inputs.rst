****************
Input Parameters
****************

This page details the various inputs expected by `~macauff.CrossMatch` via its input parameter files. These are split into two main sections: inputs that both catalogues need separately (contained in their respective catalogue parameters file) and inputs that are defined with respect to the catalogue-catalogue cross-match.

These input parameters are required to be in a file for the `Joint Parameters`_ options, and two further, separate files for the `Catalogue-specific Parameters`_ options.

Depending on the size of the match being performed, it is likely that you may want to use the MPI parallelisation option to save runtime or memory overhead, splitting your matches into "chunks." Even if your cross-match area is small enough that you only have a single file per catalogue, this is treated as a single chunk. Most cross-match parameters are common across all chunks -- such as the input photometric magnitude columns, or whether certain match algorithm components are turned on or not -- but some are unique to each chunk. Certain parameters will be constant but vary with the chunk ID, and will require a ``_{}`` within their input parameter, such that Python's ``.format`` functionality will insert the designation. Others will be totally static, while any input keywords ending in ``_per_chunk``, as well as specifically the ``chunk_id_list`` will be a multi-line YAML entry, one line per chunk (such that the first line of ``chunk_id_list`` matches the first line of each per-chunk entry, and so on).

For a simple example of how this works for a one-chunk cross-match, see :doc:`quickstart`, and for more details on the inputs and outputs from individual functions within the cross-match process check the :doc:`documentation<macauff>`.

CrossMatch Inputs
=================

As well as the parameters required that are ingested through the input parameters files, `~macauff.CrossMatch` itself requires some input arguments.

- ``crossmatch_params_file_path``, ``cat_a_params_file_path``, and ``cat_b_params_file_path``: the file paths to the three input metadata files containing parameters files

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

``output_save_folder``, ``include_perturb_auf``, ``include_phot_like``, ``use_phot_priors``, ``pos_corr_dist``, ``cf_region_type``, ``cf_region_frame``, ``cf_region_points_per_chunk``, ``chunk_id_list``, ``real_hankel_points``, ``four_hankel_points``, ``four_max_rho``, ``int_fracs``, ``make_output_csv``, and ``n_pool``;

options which need to be supplied if ``make_output_csv`` is ``True``:

``match_out_csv_name``, and ``nonmatch_out_csv_name``;

options that must be supplied if ``include_phot_like`` is ``True``:

``with_and_without_photometry``;

those options which only need to be supplied if ``include_perturb_auf`` is ``True``:

``num_trials``, and ``d_mag``;

.. note::
    ``num_trials`` and ``d_mag`` currently need to be supplied if either ``correct_astrometry`` option in the two `Catalogue-specific Parameters`_ config files is ``True`` as well.

and those options which need to be supplied if either of ``apply_proper_motion`` or ``ref_apply_proper_motion`` is ``True``, in either catalogue:

``move_to_epoch``.

Common Parameter Description
----------------------------

``output_save_folder``

The folder path into which to save the stored ``.csv`` files that are created if ``make_output_csv`` is ``True``, or ``.npy`` files are saved otherwise.

``include_perturb_auf``

Flag for whether to include the simulated effects of blended sources on the measured astrometry in the two catalogues or not.

.. note::
    If ``include_perturb_auf`` is ``True`` then ``dustmaps`` will be called to obtain line-of-sight extinction values. You will need the SFD dustmaps to be pre-downloaded; to do so before you call the cross-match procedure you must run ``dustmaps.sfd.fetch()``; see the ``dustmaps`` documentation for more details on how to specific a particular download location.

``include_phot_like``

Flag for the inclusion of the likelihood of match or non-match based on the photometric information in the two catalogues.

``use_phot_priors``

Flag to determine whether to calculate the priors on match or non-match using the photometry (if set to ``True``) or calculate them based on a naive asymmetric density argument (``False``).

``pos_corr_dist``

The floating point precision number determining the maximum possible separation between two sources in opposing catalogues.

``cf_region_type``

This flag controls whether the areas in which photometry-related variables (likelihoods, priors, etc.) are calculated is determined by ``rectangle`` -- evenly spaced longitude/latitude pairings -- or ``points`` -- tuples of randomly placed coordinates.

``cf_region_frame``

This allows either ``equatorial`` or ``galactic`` frame coordinates to be used in the match process.

``cf_region_points_per_chunk``

The list of pointings for which to run simulations of perturbations due to blended sources, if applicable. If ``cf_region_type`` is ``rectangle``, then ``cf_region_points`` accepts six numbers: ``start longitude, end longitude, number of longitude points, start latitude, end latitude, number of latitude points``; if ``points`` then tuples must be of the syntax ``[[a, b], [c, d]]`` where ``a`` and ``c`` are RA or Galactic Longitude, and ``b`` and ``d`` are Declination or Galactic Latitude. Each chunk must have an element of these pointing lists; these can be produced on a single line, or make use of YAML multi-line formatting to more easily visualise the list-of-point-values, which will produce a list of length ``N``, of which each element will either be a length-six list or an ``Mx2`` nested list.

.. note::
    ``cf_region_points`` longitudes may be given with negative coordinates for cases where the match area is both above and below zero degrees, but they can also be given within the [0, 360] phase space, as 350 degrees and -10 degrees are handled the same where needed by ``cf_region_points``.

``chunk_id_list``

A single entry per chunk, to have the same length as ``cf_region_points_per_chunk``, of unique IDs for each chunk. This is the list of chunks to run cross-matches for, and must be contained within the lists of chunks given in the ``chunk_id_list`` entries of each input catalogue. However -- for example, when an all-sky catalogue matches with a non-all-sky dataset -- the joint-parameter list of chunk IDs can be smaller in number, and a subset of those given in the catalogue-specific parameter files.

``real_hankel_points``

The integer number of points, for Hankel (two-dimensional Fourier) transformations, in which to approximate the fourier transformation integral of the AUFs.

``four_hankel_points``

The integer number of points for approximating the inverse Hankel transformation, representing the convolution of two real-space AUFs.

``four_max_rho``

The largest fourier-space value, up to which inverse Hankel transformation integrals are considered. Should typically be larger than the inverse of the smallest typical centroiding Gaussian one-dimensional uncertainty.

``n_pool``

Determines how many CPUs are used when parallelising within ``Python`` using ``multiprocessing``.

``int_fracs``

The integral fractions of the various so-called "error circles" used in the cross-match process. Should be list of floats, in the order of: bright error circle fraction, "field" error circle fraction, and potential counterpart cutoff limit. Note that bright and "field" fractions should be reasonably separated in value (more than maybe 0.1) to avoid biasing results that use both to measure photometry-based priors, when applicable.

``match_out_csv_name``

Name of the band-merged, cross-matched dataset of counterpart associations and accompanying metadata, including the appropriate file extension. Must be a single string containing ``_{}``, into which the chunk ID is inserted.

``nonmatch_out_csv_name``

Filename to save out the respective non-match catalogue objects and metadata to. Will have appended to the front ``cat_name`` to distinguish the two non-match files. ``nonmatch_out_csv_name`` should contain the appropriate file extension. Must be a single string containing ``_{}``, into which the chunk ID is inserted.

``with_and_without_photometry``

Boolean flag that should be given if ``include_phot_like`` is ``True``, to indicate whether to run an astrometry-only cross-match in addition to a full astrometry-plus-photometry match. In this case, a second counterpart determination is called with photometric likelihoods ignored, and a second set of counterparts, match probabilities, etc. is recorded.

``num_trials``

The number of PSF realisations to draw when simulating the perturbation component of the AUF. Should be an integer. Only required if ``include_perturb_auf`` is ``True``.

``d_mag``

Bin sizes for magnitudes used to represent the source number density used in the random drawing of perturbation AUF component PSFs. Should be a single float. Only required if ``include_perturb_auf`` is ``True``.

``move_to_epoch``

The epoch to which proper motions should be fast-forwarded or re-wound, in either catalogue (and astrometric correction reference catalogues, as appropriate). Should be a single string formatted in such a way that it can be read by ``astropy``'s ``Time`` function, such as ``J2000`` or ``2000-01-01``.


Catalogue-specific Parameters
=============================

These parameters are required in two separate files, one per catalogue to be cross-matched, the files ``cat_a_params.txt`` and ``cat_b_params.txt`` read from sub-folders within ``chunks_folder_path`` as passed to `~macauff.CrossMatch`.

These can be divided into those inputs that are always required:

``cat_csv_file_path``, ``cat_name``, ``pos_and_err_indices``, ``mag_indices``, ``chunk_overlap_col``, ``best_mag_index_col``, ``csv_has_header``, ``filt_names``, ``auf_file_path``, ``auf_region_type``, ``auf_region_frame``, ``auf_region_points_per_chunk``, ``chunk_id_list``, ``apply_proper_motion``, and ``correct_astrometry``;

those that are only required if the `Joint Parameters`_ option ``include_perturb_auf`` is ``True``:

``fit_gal_flag``, ``run_fw_auf``, ``run_psf_auf``, ``psf_fwhms``, ``download_tri``, ``tri_set_name``, ``tri_filt_names``, ``tri_filt_num``, ``tri_maglim_faint``, ``tri_num_faint``, ``gal_al_avs``, ``dens_dist``, ``snr_indices``, ``tri_dens_cube_location``, and ``tri_dens_array_location``;

the inputs required in each catalogue parameters file if ``fit_gal_flag`` is ``True`` (and hence ``include_perturb_auf`` is ``True``):

``gal_wavs``, ``gal_zmax``, ``gal_nzs``, ``gal_aboffsets``, and ``gal_filternames``;

inputs required if ``make_output_csv`` is ``True``:

``input_csv_file_path``, ``cat_col_names``, ``cat_col_nums``, ``extra_col_names``, and ``extra_col_nums``;

the inputs required if ``correct_astrometry`` is ``True``:

``correct_astro_save_folder``, ``correct_astro_mag_indices_index``, ``nn_radius``, ``ref_cat_csv_file_path``, ``correct_mag_array``, ``correct_mag_slice``, ``correct_sig_slice``, ``use_photometric_uncertainties``, ``saturation_magnitudes``, and ``ref_apply_proper_motion``;

.. note::
    ``run_fw_auf``, ``run_psf_auf``, ``psf_fwhms``, ``download_tri``, ``tri_set_name``, ``tri_filt_names``, ``tri_filt_num``, ``tri_maglim_faint``, ``tri_num_faint``, ``dens_dist``, ``gal_wavs``, ``gal_zmax``, ``gal_nzs``, ``gal_aboffsets``, ``gal_filternames``, ``gal_al_avs``, and ``snr_indices`` are all currently required if ``correct_astrometry`` is ``True``, bypassing the nested flags above. For example, ``dens_dist`` is required as an input if ``include_perturb_auf`` is ``True``, or if ``correct_astrometry`` is set. This means that ``AstrometricCorrections`` implicitly always runs and fits for a full Astrometric Uncertainty Function.

the inputs required if ``apply_proper_motion`` is ``True``:

``pm_indices``, and ``ref_epoch_or_index``;

and the inputs required if ``ref_apply_proper_motion`` is ``True`` (and hence ``correct_astrometry`` is also ``True``):

``ref_pm_indices``, and ``ref_ref_epoch_or_index``.


Catalogue Parameter Description
-------------------------------

``cat_csv_file_path``

The filepath to the ``.csv`` file of the input catalogue (see :doc:`quickstart` for more details). Can either be an absolute path, or relative to the folder from which the script was called, including the ``_{}`` chunk ID flag requirement.

``cat_name``

The name of the catalogue. This is used to generate intermediate folder structure within the cross-matching process, and during any output file creation process.

``pos_and_err_indices``

A list of either three, six, or N integers. If ``correct_astrometry`` is ``True``, a list of either six or N integers. If ``use_photometric_uncertainties`` is ``False`` then the first three elements are the zero-indexed indices into the *input* catalogue .csv file (``cat_csv_file_path``) for the longitudinal coordinate, latitudinal coordinate, and circular astrometric precision respectively, followed by the lon/lat/uncert of the *reference* catalogue (``ref_cat_csv_file_path``). For example, ``[10, 9, 8, 0, 1, 2]`` suggests that the reference catalogue begins with the position and uncertainty of its objects while the catalogue "a" or "b" sources have, in their original .csv file, a backwards list of coordinates and precisions towards the final columns of the filing system. If photometric uncertainties are to be used to correct astrometric uncertainties, then the first two elements should be input longitude and latitude, followed by all indices for the input catalogue's photometric uncertainty columns, with the three reference catalogue longitude, latitude, and circular astrometric precision columns. Otherwise for ``correct_astrometry`` being ``False`` then only three integers should be passed, the respective coordinates for its own catalogue (dropping the indices of the reference catalogue and requiring that astrometric uncertainty be the third index); in the above example we would therefore only pass ``[10, 9, 8]``.

``mag_indices``

Just for the input catalogue, a list of ``len(filt_names)`` integers detailing the zero-indexed column number of the magnitudes in the dataset.

``chunk_overlap_col``

Column number in the original csv file for the column containing the boolean flag indicating whether sources are in the "halo" or "core" of the chunk. Used within ``CrossMatch`` after calling ``AstrometricCorrections`` to create final npy arrays via ``csv_to_npy``. Should be a single integer number.

``best_mag_index_col``

The zero-indexed integer column number in the original input csv file used in ``AstrometricCorrections`` that corresponds to the column containing the highest quality detection for each source in the catalogue, used in ``csv_to_npy``.

``csv_has_header``

A boolean, yes/no, for whether there is a header in the first line of the ``.csv`` input catalogue files (``True``), or if the first line is a line of data (``False``).

``filt_names``

The filter names of the photometric bandpasses used in this catalogue, in the order in which they are saved in ``con_cat_photo``. These will be used to describe any output data files generated after the matching process. Should be a list.

``auf_file_path``

The folder and file name into which the Astrometric Uncertainty Function (AUF) related files will be, or have been, saved. Can also either be an absolute or relative path, like ``cat_csv_file_path``. Alternatively, this can (and must) be ``None`` if all parameters related to loading pre-computed TRILEGAL histograms (``tri_dens_cube_location``) are provided. Requires ``_{}`` in the string for chunking purposes.

``auf_region_type``

Similar to ``cf_region_type``, flag indicating which definition to use for determining the pointings of the AUF simulations; accepts either ``rectangle`` or ``points``. If ``rectangle``, then ``auf_region_points`` will map out a rectangle of evenly spaced points, otherwise it accepts pairs of coordinates at otherwise random coordinates.

``auf_region_frame``

As with ``auf_region_frame``, this flag indicates which frame the data, and thus AUF simulations, are in. Can either be ``equatorial`` or ``galactic``, allowing for data to be input either in Right Ascension and Declination, or Galactic Longitude and Galactic Latitude.

``auf_region_points_per_chunk``

Based on ``auf_region_type``, this must either by list of six floats, controlling the start and end, and number of, longitude and latitude points in ``start lon end lon # steps start lat end lat #steps`` order (see ``cf_region_points``), or a nested list of lists cf. ``[[a, b], [c, d]]``. Similar to ``cf_region_points_per_chunk``, must be a value per chunk ID of ``chunk_id_list`` in the specific catalogue parameter file, to be bundled into a list wrapper of length the number of chunks.

.. note::
    ``auf_region_points`` longitudes may be given with negative coordinates for cases where the match area is both above and below zero degrees, but they can also be given within the [0, 360] phase space, as 350 degrees and -10 degrees are handled the same where needed by ``auf_region_points``.

``chunk_id_list``

A single entry per chunk, to have the same length as ``auf_region_points_per_chunk``, of unique IDs for each chunk. Must agree with the list in ``chunk_id_list`` in the joint-catalogue parameter file, and be a super-set of those chunks to be matched (i.e., no chunks can be in the joint catalogue match file without being in both catalogue-only input files).

``apply_proper_motion``

Boolean flag indicating whether, for this catalogue, we should apply known proper motion information to project the particular epoch (see ``ref_epoch_or_index``) to another time (see ``move_to_epoch``).

``correct_astrometry``

In cases where catalogues have unreliable *centroid* uncertainties, before catalogue matching occurs the dataset can be fit for systematic corrections to its quoted astrometric precisions through ensemble match separation distance distributions to a higher-precision dataset (see the :doc:`Processing<pre_post_process>` section). This flag controls whether this is performed on a chunk-by-chunk basis during the initialisation step of ``CrossMatch``.

.. note::
    If ``correct_astrometry`` is ``True`` then ``dustmaps`` will be called to obtain line-of-sight extinction values. You will need the SFD dustmaps to be pre-downloaded; to do so before you call the cross-match procedure you must run ``dustmaps.sfd.fetch()``; see the ``dustmaps`` documentation for more details on how to specific a particular download location.

``fit_gal_flag``

Optional flag for whether to include simulated external galaxy counts, or just include Galactic sources when deriving the perturbation component of the AUF. Only needed if ``include_perturb_auf`` is ``True``.

``run_fw_auf``

Boolean flag controlling the option to include the flux-weighted algorithm for determining the centre-of-light perturbation with AUF component simulations. Only required if  ``include_perturb_auf`` is ``True``.

``run_psf_auf``

Complementary flag to ``run_fw_auf``, indicates whether to run background-dominated, PSF photometry algorithm for the determination of perturbation due to hidden contaminant objects. If both this and ``run_fw_auf`` are ``True`` a signal-to-noise-based weighting between the two algorithms is implemented. Must be provided if  ``include_perturb_auf`` is ``True``.

``psf_fwhms``

The Full-Width-At-Half-Maximum of each filter's Point Spread Function (PSF), in the same order as in ``filt_names``. These are used to simulate the PSF if ``include_perturb_auf`` is set to ``True``, and are unnecessary otherwise. Should be a list of floats.

``download_tri``

Boolean flag, indicating whether to re-download a TRILEGAL simulation in a given ``auf_region_points`` sky coordinate, once it has successfully been run, and to overwrite the original simulation data or not. Optional if ``include_perturb_aufs`` is ``False``. Alternatively, this can (and must) be ``None`` if all parameters related to loading pre-computed TRILEGAL histograms (``tri_dens_cube_location``) are provided.

``tri_set_name``

The name of the filter set used to simulate the catalogue's sources in TRILEGAL [#]_. Used to interact with the TRILEGAL API; optional if ``include_perturb_aufs`` is ``False``. Alternatively, this can (and must) be ``None`` if all parameters related to loading pre-computed TRILEGAL histograms (``tri_dens_cube_location``) are provided.

``tri_filt_names``

The names of the filters, in the same order as ``filt_names``, as given in the data ``tri_set_name`` calls. Optional if ``include_perturb_aufs`` is ``False``. Alternatively, this can (and must) be ``None`` if all parameters related to loading pre-computed TRILEGAL histograms (``tri_dens_cube_location``) are provided.

``tri_filt_num``

The one-indexed column number of the magnitude, as determined by the column order of the saved data returned by the TRILEGAL API, to which to set the maximum magnitude limit for the simulation. Optional if ``include_perturb_aufs`` is ``False``. Alternatively, this can (and must) be ``None`` if all parameters related to loading pre-computed TRILEGAL histograms (``tri_dens_cube_location``) are provided.

``tri_maglim_faint``

This is the float that represents the magnitude down to which to simulate TRILEGAL sources in the full-scale simulation, bearing in mind the limiting magnitude cut of the public API but also making sure this value is sufficiently faint that it contains all potentially perturbing objects for the dynamic range of this catalogue (approximately 10 magnitudes fainter than the limiting magnitude of the survey). Alternatively, this can (and must) be ``None`` if all parameters related to loading pre-computed TRILEGAL histograms (``tri_dens_cube_location``) are provided.

``tri_num_faint``

Integer number of objects to draw from the TRILEGAL simulation -- affecting the area of simulation, up to the limit imposed by TRILEGAL -- down to the full ``tri_maglim_faint`` magnitude. Alternatively, this can (and must) be ``None`` if all parameters related to loading pre-computed TRILEGAL histograms (``tri_dens_cube_location``) are provided.

``tri_dens_cube_location``

The location on disk of a numpy array, shape ``(A, len(filt_names), B, M)``, of differential source counts, magnitudes bins, and magnitude bin widths for a series TRILEGAL simulations at ``A`` number of sky positions, in each filter, for a specific catalogue. Alternatively, should be ``None`` if ``auf_file_path`` and associated parameters for the running of TRILEGAL histogram generation within the cross-match run are given.

``tri_dens_array_location``

Location on disk of a numpy array to accompany ``tri_dens_cube_location``, of shape ``(A, 2)``, with the corresponding sky coordinates for each simulation.

``dens_dist``

The radius, in arcseconds, within which to count internal catalogue sources for each object, to calculate the local source density. Used to scale TRILEGAL simulated source counts to match smaller scale density fluctuations. Only required if ``include_perturb_auf`` is ``True``.

``snr_indices``

Similar to ``mag_indices``, a list of ``len(mag_indices)`` integers, one for each column in ``mag_indices`` for where the corresponding signal-to-noise ratio column is held for each magnitude in the input .csv file.

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

Differential extinction relative to the V-band for each filter, a list of floats. Must be provided if ``include_perturb_auf`` is ``True``.

``input_csv_file_path``

Full filepath of the catalogue's original input ``.csv`` file, generally converted to a binary file for use within the main code via ``csv_to_npy``, including filename and extension. Should have ``_{}`` formatting within the string for chunk-identification purposes.

``cat_col_names``

The names of the mandatory columns from each respctive catalogue. Should contain at least the column name for the name or ID of the object, and the names of the columns containing the two orthogonal sky coordinates, as well as the names of each column containing magnitude information to be transferred to the output match and non-match files.

``cat_col_nums``

For each column name in ``cat_col_names``, ``cat_col_nums`` is the zero-indexed position of the column. For example, if we had ``['ID', 'RA', 'Dec', 'V']`` as our ``cat_col_names``, we might have ``[0, 1, 2, 5]`` as our ``cat_col_nums``, in which our designation and coordinates are the first three columns, but our V-band magnitude is a few columns down.

``extra_col_names``

Analogous to ``cat_col_names``, a list of the additional columns from the original csv catalogue file that we wish to add to the match and non-match output files.

``extra_col_nums``

The zero-indexed positions of each corresponding column in ``extra_col_names``, much the same as in ``cat_col_nums``, but for additional, optional columns we may wish to transfer from input to output dataset.

``correct_astro_mag_indices_index``

For the purposes of correcting systematic biases in a given catalogue, a single photometric band is used. ``correct_astro_mag_indices_index`` indicates which filter to use -- e.g., ``correct_astro_mag_indices_index = 0`` says to use the first filter as given in ``filt_names`` or ``mag_indices``. Must be a single integer value no larger than ``len(filt_names)-1``.

``nn_radius``

Nearest neighbour radius out to which to search for potential counterparts for the purposes of ensemble match separation distributions; should be a single float.

``correct_astro_save_folder``

File path, relative or absolute, into which to save files as generated by the astrometric correction process. Must include ``_{}`` to allow for formatting for each chunk separately.

``ref_cat_csv_file_path``

Similar to ``cat_csv_file_path``, but the path and filename, including extension, of the *reference* dataset used in the matching process. These chunks should correspond one-to-one with those used in ``cat_csv_file_path`` -- i.e., ``data_1.csv`` in ``/your/path/to/file`` should be the same region of the sky as the reference catalogue in ``/another/path/to/elsewhere/reference_data_1.csv``, potentially with some buffer overlap to avoid false matches at the edges. Must include ``_{}`` to allow for formatting for each chunk separately.

``correct_mag_array``

List of magnitude arrays at which to evaluate the distribution of matches to the higher-astrometric-precision dataset in each input-catalogue photometric filter. Accepts a list of lists of floats, or two-dimensional array, with the first axis the same length as ``mag_indices``.

``correct_mag_slice``

Corresponding to each magnitude in ``correct_mag_array``, each element of this list of lists of floats should be a width around each ``correct_mag_array`` element to select sources, ensuring a small sub-set of similar brightness objects are used to determine the Astrometric Uncertainty Function of.

``correct_sig_slice``

Elementwise with ``correct_mag_array`` and ``correct_mag_slice``, a list of lists of floats of widths of astrometric precision to select a robust sub-sample of objects in each magnitude bin for, ensuring a self-similar AUF.

``use_photometric_uncertainties``

Boolean flag indicating whether the astrometric or photometric uncertainties of the input catalogue should be used to derive the astrometric uncertainties from ensemble statistics in ``AstrometricCorrections``.

``saturation_magnitudes``

A list, one float per filter, of the magnitudes in the given filter at which the telescope or survey saturates, used in the filtering of source counts for model-fitting purposes in ``AstrometricCorrections``.

``pm_indices``

List of integers for the two orthogonal sky indices that contain the proper motion information in the catalogue being cross-matched, used to propagate its given epoch to that declared in the joint-match configuration, ``move_to_epoch``.

``ref_epoch_or_index``

Either a single epoch that all observations are provided at, in which case -- similar to ``move_to_epoch`` -- the string should be formatted such that ``astropy``'s ``Time`` function can parse it (e.g. ``J2000`` or ``2000-01-01``); or an index into the file that contains the epochs of each observation, one per row, to be moved to ``move_to_epoch`` on a case-by-case basis.

``ref_apply_proper_motion``

Boolean flag for whether to apply proper motions to this catalogue's *reference* catalogue during astrometric precision parameterisation and characterisation.

``ref_pm_indices``

A list of integers, the index for the two ortogonal sky proper motion columns in the *reference* catalogue being loaded during astrometric corrections, similar to ``pm_indices``, if supplied, for the catalogue being parameterised and/or cross-matched.

``ref_ref_epoch_or_index``

The *reference* catalogue's "reference" epoch, used to propagate the reference cataloue to ``move_to_epoch`` before this catalogue has its uncertainties parameterised, as necessary. Formats expected are the same as ``ref_epoch_or_index``: single, ``Time``-compatible string, or index into reference catalogue from which an epoch per row can be extracted.

Parameter Dependency Graph
==========================

The inter-dependency of input parameters on one another, and the output ``CrossMatch`` attribute if different, are given below::

    ├─> include_perturb_auf
    │                     ├─> num_trials
    │                     ├─> d_mag
    │                     ├─* dens_dist
    │                     ├─* fit_gal_flag
    │                     │              ├─* gal_wavs
    │                     │              ├─* gal_zmax
    │                     │              ├─* gal_nzs
    │                     │              ├─* gal_aboffsets
    │                     │              ├─* gal_filternames
    │                     │              └─* saturation_magnitudes
    │                     ├─* snr_indices
    │                     ├─* tri_set_name[2a]
    │                     ├─* tri_filt_names[2a]
    │                     ├─* tri_filt_num[2a]
    │                     ├─* tri_maglim_faint[2a]
    │                     ├─* tri_num_faint[2a]
    │                     ├─* download_tri[2a]
    │                     ├─* psf_fwhms
    │                     ├─* run_fw_auf
    │                     ├─* run_psf_auf
    │                     ├─* gal_al_avs
    │                     ├─* tri_dens_cube_location[2b] => tri_dens_cube
    │                     └─* tri_dens_array_location[2b] => tri_dens_array
    ├─> include_phot_like
    │                   └─> with_and_without_photometry
    ├─> use_phot_priors
    ├─> cf_region_type
    ├─> cf_region_frame[1]
    ├─> cf_region_points_per_chunk[4]
    ├─> chunk_id_list[4]
    ├─> output_save_folder
    ├─> pos_corr_dist
    ├─> real_hankel_points
    ├─> four_hankel_points
    ├─> four_max_rho
    ├─> int_fracs
    ├─> make_output_csv
    │                 ├─> match_out_csv_name[3]
    │                 ├─> nonmatch_out_csv_name[3]
    │                 ├─* cat_col_names
    │                 ├─* cat_col_nums
    │                 ├─* extra_col_names
    │                 └─* extra_col_nums
    ├─> n_pool
    ├─* auf_region_type
    ├─* auf_region_frame[1]
    ├─* auf_region_points_per_chunk[4]
    ├─* chunk_id_list[4]
    ├─* filt_names
    ├─* cat_name
    ├─* auf_file_path[2a, 3]
    ├─* cat_csv_file_path[3]
    ├─* pos_and_err_indices
    ├─* mag_indices
    ├─* chunk_overlap_col
    ├─* best_mag_index_col
    ├─* csv_has_header
    ├─* apply_proper_motion
    │                     ├─> move_to_epoch[5]
    │                     ├─* pm_indices
    │                     └─* ref_epoch_or_index
    └─* correct_astrometry
                         ├─* correct_astro_save_folder[3]
                         ├─* snr_indices
                         ├─* ref_apply_proper_motion
                         │                         ├─> move_to_epoch[5]
                         │                         ├─* ref_pm_indices
                         │                         └─* ref_ref_epoch_or_index
                         ├─* correct_astro_mag_indices_index
                         ├─* nn_radius
                         ├─* ref_cat_csv_file_path[3]
                         ├─* correct_mag_array
                         ├─* correct_mag_slice
                         ├─* correct_sig_slice
                         ├─* use_photometric_uncertainties
                         └─* saturation_magnitudes

List directories end in ``->`` for ``joint`` parameters, ``-*`` for ``catalogue`` parameters. ``catalogue`` level items will have ``a_`` or ``b_`` prepended, depending on which "side" of the cross-match they are from. Items with a second keyword after an arrow ``=>`` are the names of the attributes that are saved to ``CrossMatch``, usually when the input parameter is a location on disk.

| [1] - must be the same
| [2] - only one set of [2a] and [2b] should be given, the others should be passed as ``None``
| [3] - must have ``_{}`` in its string, into which the chunk ID will be inserted
| [4] - must have relevant input entry per chunk, e.g. in a YAML multi-line format, aligned with the chunk ID of ``chunk_id_list`` of the relevant input parameter file
| [5] - must be provided in either ``apply_proper_motion`` or ``ref_apply_proper_motion`` are ``True`` in either input catalogue

.. rubric:: Footnotes

.. [#] Please see `here <http://stev.oapd.inaf.it/~webmaster/trilegal_1.6/papers.html>`_ to view the TRILEGAL papers to cite, if you use this software in your publication.

