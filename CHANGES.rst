0.1.3 (unreleased)
------------------

General
^^^^^^^

- Refactored folder structure and input parameter file workflow, removing the
  parameter file, AUF files, and saved output sub-folders. [#90]

- Removed secondary file conversion requirements, loading directly from plain-text
  input and saving directly to plain-text output as necessary. [#90]

- Relaxed the assumption of rectlinearity in shape of match regions, no longer
  assuming rectangles aligned with a particular coordinate system. [#88]

New Features
^^^^^^^^^^^^

- New function ``apply_proper_motion``, allowing for catalogues with motion data
  to be moved forwards or backwards in time, to align with a dataset with
  differing observation dates. [#91]

- ``AstrometricCorrections`` runs per-band parameterisation when using photometry
  to calibrate astrometric precisions, with an "astrometric scatter"-based relation
  for each filter. [#91]

- Added ``with_and_without_photometry`` input keyword, to allow for cases where
  ``include_phot_like`` is ``True`` but an astrometry-only cross-match is also
  desired, saving two sets of counterparts and corresponding data in one
  run. [#89]

- ``_calculate_cf_areas`` and ``create_c_and_f`` use convex hull overlap for
  both catalogues, as well as nearest-``cf_points`` calculations, to determine
  region areas and photometric priors and likelihoods. [#88]

- Split new class ``Macauff`` out from ``CrossMatch``, with the latter now handling
  the I/O and the former a smaller class to run the actual cross-match. [#76]

Bug Fixes
^^^^^^^^^

- Issue resolved with ``convex_hull_area`` negative longitude resulting in
  coordinates higher than 360 degrees. [#90]

- Fixed issue with rounding of coordinates when calling ``AstrometricCorrections``
  from within ``macauff``, resulting in two separately saved sets of Galaxy
  source counts in two different files.  [#88]

- Fixed ``create_c_and_f`` magnitude indexing issue. [#88]

- Fixed an issue where, in very crowded but asymmetrically dense fields, photometric
  priors could become negative. [#86]

API Changes
^^^^^^^^^^^

- Added catalogue-specific parameters ``apply_proper_motion``, ``pm_indices``,
  and ``ref_epoch_or_index``, along with joint-parameter
  ``move_to_epoch``. Parameters are propagated through ``CrossMatch`` to
  ``csv_to_npy`` and ``AstrometricCorrections`` were necessary. [#91]

- ``CrossMatch`` replaced ``mag_unc_indices`` with ``snr_indices``, requiring
  pre-computed signal-to-noise ratios rather than photometric uncertainties where
  required. [#91]

- Removed ``compute_snr_mag_relation`` and ``snr_mag_params_file_path`` now that
  SNRs are required as inputs and hence it is no longer necessary to compute the
  the scaling with magnitude. [#91]

- ``correct_mag_array``, ``correct_mag_slice``, and ``correct_sig_slice`` require,
  both as inputs to ``CrossMatch`` but generally through to
  ``AstrometricCorrections``, per-band lists of such parameters if photometric
  calibration is required. [#91]

- ``pos_and_err_indices`` now has a third permutation of accepted input, where
  photometry-based calibration through ``AstrometricCorrections`` will require
  three reference-catalogue indices, two position indices in the catatlogue to
  be calibrated, and a number of photometric-uncertainty column indices as
  appropriate. [#91]

- ``AstrometricCorrections`` takes full lists of photometric band-related inputs
  than a single, best parameter, mirroring the format in ``CrossMatch``. [#91]

- ``csv_to_npy`` requires SNR column indices where relevant, and handles the
  per-band astrometric parameterisation accordingly. [#91]

- ``joint_folder_path`` and ``output_csv_folder`` merged into
  ``output_save_folder``. [#90]

- ``cf_regions_points`` and ``auf_region_points`` changed to be a list of
  inputs, one per chunk, with ``chunk_id_list`` added to map the inputs to
  each parallelisation. [#90]

- Parameter file extension changed to YAML syntax, with ``chunks_folder_path``
  changed to separate input file paths in ``crossmatch_params_file_path``,
  ``cat_a_params_file_path``, and ``cat_b_params_file_path``. [#90]

- ``match_out_csv_name``, ``nonmatch_out_csv_name``, ``cat_csv_file_path``
  (previously ``cat_folder_path``), ``auf_file_path`` (previously
  ``auf_folder_path``), ``snr_mag_params_file_path`` (previously
  ``snr_mag_params_path``), ``input_csv_file_path`` (previously
  ``input_csv_folder``),  ``correct_astro_save_folder``, ``ref_cat_csv_file_path``
  (previously ``ref_csv_cat_file_string``), and  ``dens_hist_tri_location`` et
  al., when not None, must contain string formatting to insert chunk IDs
  into. [#90]

- Removed ``csv_cat_file_string``, ``cat_csv_name``, and ``input_csv_folder`` as
  input parameters. [#90]

- ``best_mag_index`` was renamed ``correct_astro_mag_indices_index``. [#90]

- ``pos_and_err_indices`` had its list of integers reversed, always passing
  the current-catalogue triplet first in both the case of ``correct_astrometry``
  and ``compute_snr_mag_relation``. [#90]

- CSV-related input parameters made required entry instead of only being asked for
  when ``correct_astrometry`` was ``True``. [#90]

- ``csv_to_npy`` now returns the arrays created rather than saving to disk. [#90]

- ``npy_to_csv`` requires the ``CrossMatch`` class be passed to it, and no longer
  reads necessary arrays from disk. It also requires the ``correct_astrometry``
  flags rather than the now-defunct ``input_npy_folder`` keyword. [#90]

- ``generate_random_data`` generalised to allow for circular test regions to be
  generated, as well as rectangular ones. [#88]

- Moved ``get_random_seed_size`` from ``perturbation_auf_fortran`` to
  ``misc_functions_fortran``. [#88]

- Removed ``load_small_ref_auf_grid``, ``_load_fourier_grid_cutouts``,
  ``hav_dist_constant_lat``, and ``_clean_overlaps``. [#88]

- ``group_sources_fortran.get_overlap_indices`` uses pre-generated potential overlaps
  in its call, instead of determining potential counterparts from scratch.  [#88]

- Removed ``group_sources_fortran.get_max_overlap``. [#88]

- Removed ``_distance_check``, using ``get_circle_area_overlap`` instead. [#88]

- Removed ``calculate_local_density``, using ``create_densities`` where called. [#88]

- Moved ``create_densities``, generalising to use ``calculate_overlap_counts`` and
  a call to the also-moved ``misc_functions_fortran.get_circle_area_overlap``,
  which also saw algorithmic improvements. [#88]

- Added ``convex_hull_area``, ``coord_inside_convex_hull``, and
  ``generate_avs_inside_hull`` to handle generalised convex hull maths. [#88]

- Removed ``cross_match_extent`` as necessary input parameter. [#88]

- ``AstrometricCorrections`` has keyword inputs ``mn_fit_type``, ``seeing_ranges``,
  ``single_or_repeat``, and ``repeat_unique_visits_list``, which are accordingly
  optional inputs into ``CrossMatch``. [#85]

- ``CrossMatch`` now expects ``saturation_magnitudes`` as an input parameter in
  its input files, if ``fit_gal_flag`` or ``correct_astrometry`` are
  ``True``. [#81]

- ``AstrometricCorrections`` accepts pre-computed TRILEGAL histograms, following
  the expansion of ``CrossMatch`` accepting them. [#79]

- ``AstrometricCorrections`` and ``SNRMagnitudeRelationship`` are able to now
  accept pre-loaded catalogues via ``a_cat`` and ``b_cat``, instead of passing
  ``a_cat_name`` and ``b_cat_name``. [#79]

- Outputs in ``joint_folder_path`` are no longer saved to sub-folders; instead,
  all final saving of file to disk is done within the post-process step of the
  I/O wrapper and into the base ``joint_folder_path`` folder. [#79]

- Added ``dens_hist_tri_location``, ``tri_model_mags_location``,
  ``ntri_model_mag_mids_location``, ``tri_model_mags_interval_location``,
  ``tri_dens_uncert_location``, and ``tri_n_bright_sources_star_location`` as
  input parameters into catalogue configuration files. These must be provided,
  and should be ``None`` if previous TRILEGAL histogram-generation parameters
  (``auf_folder_path``, ``tri_set_name``,  ``tri_filt_names``, ``tri_filt_num``,
  ``download_tri``, ``tri_maglim_faint``, ``tri_num_faint``) are provided,
  while those parameters must be ``None`` if the new set are given. [#79]

- ``AstrometricCorrections`` and ``SNRMagnitudeRelationship`` accept ``return_nm``
  as an optional keyword, allowing for the non-saving of arrays to disk, instead
  returning the arrays after calling. [#79]

- ``Macauff`` expects input IO wrapper ``CrossMatch`` class to have pre-loaded
  datasets in the form of astrometry, photometry, and reference magnitude
  respectively. [#79]

- Removed ``StageData``. [#76]

- Removed ``use_memmap_files`` as an input into ``CrossMatch``, along with
  ``run_auf``, ``run_group``, ``run_cf``, and ``run_source`` from parameters
  input into the cross-match process. This means that there is no option to
  run larger matches by slicing one large input catalogue file, and runs should
  be broken into smaller runs to be parallelised via chunking instead. [#71]

- Removed ``mem_chunk_num`` as input configuration parameter, dealing with the
  entire catalogue match in memory in one go. [#71]

- Removed hard-coded SFD dustmaps, using the ``dustmaps`` package to manage the
  dataset instead. [#69]

Other Changes
^^^^^^^^^^^^^

- Pinned ``numpy`` to minimum v2.0 for compatibility with new features. [#85]

- Pinned ``speclite`` to minimum v0.18 for additional filters. [#82]

- Added ``dustmaps`` as a dependency. [#69]


0.1.2 (2023-10-27)
------------------

General
^^^^^^^

New Features
^^^^^^^^^^^^

- ``AstrometricCorrections`` implemented a simultaneous fit for ``m`` and ``n``
  within the uncertainty correction routine, instead of fitting for each
  uncertaintiy separately and fitting for m and n after the fact. [#67]

- Added ``SNRMagnitudeRelationship`` as a subclass of ``AstrometricCorrections``
  to run solely the SNR-magnitude derivation part of the larger astrometric
  solutions pipeline, in cases where the astrometry of a catalogue is trustworthy
  but we still require the signal-to-noise ratio of sources at a given
  brightness. [#67]

- B-V reddening calculator added directly through ``SFDEBV``, using Schlegel,
  Finkbeiner & Davis (1998), replacing the original NED website lookup call. [#67]

- Added chunk post-processing, removing duplicate sources where in the "halo" of
  a particular region, if desired. [#58]

- ``CrossMatch`` can now generate output csv files during the matching process if
  ``make_output_csv`` is set to ``True``. [#58]

- Added new algorithm, based on the assumption that objects within a photometric
  catalogue were fit with PSF photometry in the sky background dominated regime,
  where noise is constant, extending Plewa & Sari (2018, MNRAS, 476, 4372). This
  can then be combined with the original aperture photometry/photon-noise
  dominated case from Wilson & Naylor (2018, MNRAS, 481, 2148) using
  signal-to-noise ratio as a measure of the weight to apply to each
  algorithm. [#50]

- Added ``fit_astrometry`` and ``AstrometricCorrections`` to allow for fitting
  well-understood datasets against one another to account for systematic
  astrometric uncertainties not present in the photometric catalogues as
  given. [#50]

- Added ``derive_psf_auf_params`` and ``FitPSFPerturbations`` to calculate the
  parameters necessary to fit for the PSF photometry, sky background-dominated
  algorithm for perturbation due to unresolved contaminant objects. [#50]

- ``csv_to_npy`` now has the option to pre-process astrometric uncertainties
  based on ``AstrometricCorrections`` outputs. [#50]

- ``npy_to_csv`` now has the option to include within the final output .csv
  tables made from cross-match results the pre-processed, updated astrometric
  uncertainties that result from ``AstrometricCorrections``. [#50]

- Added MPI parallelisation and checkpointing. [#49]

- Added option to disable use of memory-mapped files for internal arrays.
  Reduces I/O operations at the cost of increased memory consuption. [#49]

- Inclusion of galaxy count model, used in the generation of perturbation
  AUF components. [#41, #44]

- Creation of initial Galactic proper motion model, for inclusion within the
  cross-match framework in a future release. [#39]

- Added additional data outputs to ``counterpart_pairing``: match separations, as
  well as the nearest neighbour non-match for each source within a given island,
  and its eta/xi and average contamination derived values. [#37]

- Added functionality to convert .csv files to internal files used in the
  matching process, and create output .csv files from the resulting merged
  datasets created as a result of the matching. [#34]

- Added option to include the full perturbation AUF component, based on
  simulated Galactic source modelling. [#27]

- Added options to photometric routines to create full photometry-based
  likelihood and prior, or just use the photometric prior and use the naive
  equal-probability likelihood. [#25]

Bug Fixes
^^^^^^^^^

- In rare cases ``G`` can be incorrectly negative calculated from
  ``find_single_island_prob``, and now gets a threshold low-but-positive value
  set in these instances. [#67]

- Fixed issue reading ``local_N`` when ``compute_local_density`` is used in
  combination with no memmapping. [#67]

- Fixed "fire extinguisher" priors and likelihoods, used in cases where both c
  and f are zero, not cancelling to one in the likelihood ratio. [#67]

- Fixed cases where wavelength range of filter response can cause a non-shifted
  spectrum to fail due to non-padding in ``create_galaxy_counts``. [#67]

- ``create_single_perturb_auf`` raises an error if the number of simulated
  sources in a given sightline is insufficient to draw reliable number density
  measurements from. [#67]

- ``make_perturb_aufs`` checks for ``compute_local_density`` and
  ``use_memmap_files`` before loading local normalising density binary
  files, and otherwise uses pre-computed in-memory array values. [#67]

- ``input_npy_folder`` correctly set as ``None`` if passed as such through
  the input parameter file. [#67]

- If ``use_memmap_files`` is ``False`` but any of the flags for running steps
  of the cross-match process are also ``False`` a warning will be raised and
  the run flags set to ``True``, since there are no fallback files to load. [#67]

- Calls to ``make_tri_counts`` and ``create_galaxy_counts`` changed to use a
  grid of extinction vectors within the chosen field of regard to better
  handle differential reddening instead of relying on a single Av at a
  particular precise set of coordinates. [#67]

- ``make_tri_counts`` gains ``brightest_source_mag`` and ``density_mag``
  keywords, returning ``num_bright_obj``. [#67]

- Convenience function ``min_max_lon`` added, to account for issues where
  the minimum and maximum longitude in a given region of space could sit either
  side of the 0-360 boundary, and hence the usual x < l < y conditions would
  fail. [#67]

- ``counterpart_pairing_fortran``'s ``factorial`` function changed from
  calculating N! to directly calculating N! / (N-M)! as the previous function
  had the potential to overflow unnecessarily. [#67]

- Added ``outfolder`` to ``trilegal_webcall`` to avoid a parallelisation race
  condition with saving outputs. [#67]

- ``mag_h_params`` renamed to ``snr_mag_params`` to ensure commonality of the
  reference and parameter without the codebase. [#62]

- ``AstrometricCorrections`` makes a correctly multi-magnitude SNR model
  array. [#59]

- ``npy_to_csv`` expected contamination probability arrays to be transposed from
  their ``CrossMatch`` output shape, but they now correctly assume
  fortran-ordering. [#58]

- Pass ``tri_maglim_bright``, ``tri_maglim_faint``, ``tri_num_bright``, and
  ``tri_num_faint`` through to ``make_perturb_aufs`` in ``CrossMatch`` call. [#56]

- Replaced ``datetime.strptime`` in the ``CrossMatch`` constructor with a
  string ``split`` to fix a crash when given walltime is greater than
  ``24:00:00``. [#52]

- Updated ``fit_gal_flag`` keyword as passed through to ``make_perturb_aufs``
  incorrectly using ``self.a_fit_gal_flag`` when running catalogue "b" AUF
  component generation. [#50]

- Corrected issue where ``local_N`` wasn't having entries saved to memmapped
  array in ``make_perturb_auf``. [#38]

- Updated ``local_N`` to keep the local densities of catalogue in each filter,
  instead of overwriting each time. [#38]

- Set minimum density of local sources to one source in the region in question,
  instead of allowing for a floor of zero density, to avoid issues with AUF
  simulations. [#38]

- Avoided re-using the same random seed in each density-magnitude combination
  during AUF simulations. [#38]

- Changed limits on photometric likelihoods and priors to avoid cases where
  both field and counterpart posteriors are zero, and hence no matches can be
  made in a given island. [#38]

- Fixed issue in ``source_pairing`` where incorrect island lengths could be used
  for field and counterpart arrays. [#38]

- Fixed ordering issue with ``acontamprob`` and ``bcontamprob`` in
  ``source_pairing``. [#38]

- Fix to issue with np.where test in ``test_counterpart_pairing`` causing incorrect
  failure to match probabilities. [#36]

- Fixes to various minor typos in variables in the cross-match workflow. [#32]

- Allow for the non-existence of a TRILEGAL simulation in any folder, and download
  new files even if ``tri_download_flag`` was set to ``False``. [#32]

- Save local normalising densities to file if ``compute_local_density`` was set
  to ``True``, to allow for its non-calculation in the future. [#32]

- Overload ``compute_local_density`` if it is set to ``False`` and the file
  storing local densities does not exist. [#32]

- ``create_single_perturb_auf`` corrected to run on a single filter, as its input
  intended, instead of looping through all filters. [#32]

- Removed final right-hand bin from consideration when identifying which magnitude
  bin each source should be assigned to in ``create_c_and_f``, to avoid an issue
  where sources of exactly the last bin are assigned outside the allowed range
  of indices. [#32]

- Fixed inefficiencies in both group sources and perturbation AUF creation runtime,
  significantly improving the speed of those parts of a cross-match. [#31]

- Corrected an error in ``tests.generate_random_data``, where only one catalogue
  had its source uncertainties simulated. [#23]

API Changes
^^^^^^^^^^^

- ``use_photometric_uncertainties`` added as an optional keyword to
  ``AstrometricCorrections``, allowing for the use of photometric instead of
  astrometric uncertainties as a slicing to determine best-fit astrometric
  uncertainties. [#67]

- ``csv_to_npy``, ``npy_to_csv``, and ``rect_slice_csv`` now expect filenames to
  include their extensions. [#67]

- ``mn_to_radec`` added to ``csv_to_npy``, to convert any astrometric correction
  array coordinates to match catalogue coordinates, with analogous variable
  ``cat_in_radec``, which now controls the coordinate system of the data. [#67]

- Explicitly load save-state data into ``CrossMatch`` and/or ``StageData`` as
  appropriate to match ``use_memmap_files`` boolean in both configurations. [#67]

- Added ``compute_snr_mag_relation`` as expected keyword into ``CrossMatch``
  for each catalogue, calling ``SNRMagnitudeRelationship`` if ``True``. [#67]

- Added checks for ``correct_astro_save_folder``, ``csv_cat_file_string``,
  ``pos_and_err_indices``, ``mag_indices``, and ``mag_unc_indices`` in the case
  of ``compute_snr_mag_relation`` as well as ``correct_astrometry``. [#67]

- Changed dependencies of ``snr_mag_params_path`` to include the requirement
  for just calculating SNR-mag relationships. [#67]

- ``csv_cat_file_string``, ``match_out_csv_name``, and ``nonmatch_out_csv_name``
  now all explicitly require file extensions, generally ``.csv``. [#67]

- Removed ``dens_mag`` as input into ``CrossMatch``, and ``density_mags`` from
  ``make_perturb_aufs``. [#67]

- Changed the requirements of ``al_avs`` in ``make_perturb_aufs`` to not require
  ``fit_gal_flags``. [#67]

- ``gal_al_avs`` is now required if ``include_perturb_auf`` or
  ``correct_astrometry`` is ``True``, instead of being tied to
  ``fit_gal_flag``, as all other galaxy-related inputs are. [#67]

- ``create_galaxy_counts`` takes ``al_grid`` rather than ``al_inf``, which is
  now a list of floats rather than a singular float value, using an average
  galaxy count distribution across all extinctions in ``al_grid``. [#67]

- ``AstrometricCorrections``'s ``create_densities`` and ``create_distances``
  always save binary files instead of checking for their non-existence, due to
  the re-structuring of the looping of sightlines and pipeline steps. [#67]

- ``check_b_only`` flag added to ``make_ax_coords`` function within
  ``AstrometricCorrections`` for cases where we only need to run a sub-set
  of functions on one catalogue, instead of the two-sided approach for the full
  suite of astrometric correction tools. [#67]

- ``dens_search_radius`` changed to degrees, instead of arcseconds, in
  ``AstrometricCorrections``, to match ``CrossMatch`` requirements. [#67]

- Removed ``bright_mag`` from input to ``AstrometricCorrections``. [#67]

- Added ``AV`` and ``sigma_AV`` as input keywords to
  ``download_trilegal_simulation`` and ``get_trilegal`` to allow for the manual
  passing of specific V-band extinctions to API call. If not passed to it, a
  value is still calculated in ``get_trilegal``, and ``AV`` is returned by the
  function. [#67]

- Added expected area of TRILEGAL simulation as keyword to
  ``download_trilegal_simulation``. [#67]

- ``download_trilegal_simulation`` and ``get_trilegal`` have been re-arranged to
  move the try-except loop out of the API call function and into the larger
  function. ``get_trilegal`` will therefore either return an API call or fail,
  without trying to fetch. [#67]

- ``trilegall_webcall`` returns either ``timeout`` or ``good``, allowing for the
  re-starting of failed API calls due to e.g. the remote server being busy. [#67]

- In ``AstrometricCorrections``, ``triname`` now requires either one or two
  ``{}`` Python string formats, depending on ``coord_or_chunk``. [#62]

- All ``recreate`` flags all removed from ``AstrometricCorrections``, which now
  loops on a per-sightline basis instead of using per-step loops. [#62]

- Added ``n_pool`` as input to ``CrossMatch`` to control the number of threads used
  in ``multiprocessing`` calls. [#62]

- Added parameters ``correct_astrometry``, ``best_mag_index``, ``nn_radius``,
  ``correct_astro_save_folder``, ``csv_cat_file_string``,
  ``ref_csv_cat_file_string``, ``correct_mag_array``, ``correct_mag_slice``,
  ``correct_sig_slice``, ``pos_and_err_indices``, ``mag_indices``,
  ``mag_unc_indices``, ``chunk_overlap_col``, and ``best_mag_index_col`` as
  catalogue-level inputs to ``CrossMatch`` to allow for astrometric corrections
  through ``AstrometricCorrections`` directly before a cross-match. [#62]

- Requirements for ``num_trials``, ``d_mag``, ``run_fw_auf``, ``run_psf_auf``,
  ``psf_fwhms``, ``dens_mags``, ``snr_mag_params_path``, ``download_tri``,
  ``tri_set_name``, ``tri_filt_names``, ``tri_filt_num``, ``tri_maglim_faint``,
  ``tri_num_faint``, ``dens_dist``, ``dd_params_path``, ``l_cut_path``,
  ``gal_wavs``, ``gal_zmax``, ``gal_nzs``, ``gal_aboffsets``,
  ``gal_filternames``, and ``gal_al_avs`` inputs to ``CrossMatch`` changed to
  either require ``include_perturb_auf`` (and lower-level input criteria) or
  ``correct_astrometry``. [#62]

- Removed expectation of parameters ``tri_num_bright`` and ``tri_maglim_bright`` from
  ``CrossMatch`` input parameter files. Currently only expect the "faint" versions
  due to limits with requesting significant numbers of bright TRILEGAL objects. [#61]

- Added ``tri_num_faint`` to ``AstrometricCorrections`` to control the resolution of
  TRILEGAL simulations used in fitting for astrometry systematics, and removed
  ``maglim_b`` from expected keywords, limiting the number of TRILEGAL simulations
  to just one across the entire dynamic range, as with ``CrossMatch``. [#61]

- Added new keyword ``pregenerate_cutouts`` to ``AstometricCorrections``, indicating
  whether sightlines can be assumed to be pre-made or if they should be able to be
  made on-the-fly as part of the correction-fitting process. [#59]

- ``AstrometricCorrection`` had ``cutout_area`` and ``cutout_height``, as well as
  ``a_cat_func`` and ``b_cat_func``, made optional keywords. [#59]

- ``AstrometricCorrections`` now takes keyword input ``coord_system`` to determine
  whether coordinates fed into the class are in equatorial or galactic coordinates,
  handling conversions and consistency where necessary. Additionally, keywords were
  given more general names reflecting this change and now the class requires
  ``ax1_mids``, ``ax2_mids``, and ``ax_dimension`` instead of ``lmids``, ``bmids``,
  or ``lb_dimension``. [#59]

- ``CrossMatch`` now expects ``snr_mag_params_path`` rather than
  ``mag_h_params_path``, and ``CrossMatch`` loads and ``AstrometricCorrections``
  saves ``snr_mag_params.npy`` as the file containing the magnitude-SNR
  correlation parameterisation. [#59]

- Added new input keywords to ``AstrometricCorrections`` for the indexes of position
  and magnitudes and their uncertainties, along with the most complete magnitude to
  use in construction of any updates to astrometry of a given catalogue. [#59]

- ``AstrometricCorrections`` accepts three new keywords: ``npy_or_csv``,
  ``coords_or_chunk``, and ``chunks`` which allow for the specification of file
  type and structure of small sightlines used to check astrometry of a
  catalogue. [#59]

- ``npy_to_csv`` always requires two nested lists when using ``extra_col_*_lists``,
  rather than allowing a singular ``None``. The default is now ``[None, None]`` for
  the passing of no extra columns to be propagated to the output csv file. [#58]

- ``tri_maglim_bright``, ``tri_maglim_faint``, ``tri_num_bright``, and
  ``tri_num_faint`` are only required if ``tri_download_flag`` is ``True``. [#56]

- ``tri_filt_num``, ``tri_set_name``, and ``auf_region_frame`` updated to be
  necessary inputs into ``make_perturb_aufs`` even if ``tri_download_flag``
  is not set. [#56]

- Added ``run_fw_auf``, ``run_psf_auf``, ``mag_h_params_path``,
  ``tri_maglim_bright``, ``tri_maglim_faint``, ``tri_num_bright``, and
  ``tri_num_faint`` as required input parameters to ``CrossMatch`` if
  ``include_perturb_auf`` is ``True``. [#50]

- Added ``tri_maglim_bright``, ``tri_maglim_faint``, ``tri_num_bright``,
  ``tri_num_faint``, ``run_fw``, ``run_psf``, ``dd_params``, ``l_cut``, and
  ``mag_h_params`` as optional inputs to ``make_perturb_aufs``. [#50]

- Added ``dd_params_path`` and ``l_cut_path`` as required input parameters if
  ``include_perturb_auf`` and ``run_psf_auf`` are both ``True``. [#50]

- Removed ``dm_max`` as an input to ``CrossMatch``, now being calculated based
  on secondary perturber flux vs primary noise and chance of zero perturbers
  in ``_calculate_magnitude_offsets``. Also removed as input to
  ``make_perturb_aufs``. [#50]

- ``csv_to_npy`` has ``process_uncerts``, ``astro_sig_fits_filepath``, and
  ``cat_in_radec`` as optional input parameters. [#50]

- ``npy_to_csv`` added ``input_npy_folders`` as an input parameter. [#50]

- Removed ``joint_file_path``, ``cat_a_file_path`` and ``cat_b_file_path``
  from ``CrossMatch`` constructor and added ``chunks_folder_path``,
  ``use_memmap_files``, ``resume_file_path``, ``walltime``, ``end_within``,
  and ``polling_rate``. [#49]

- Added ``use_memmap_files`` as input parameter to relevant functions. [#49]

- Added ``StageData`` class to ``misc_functions``. [#49]

- Added ``npool`` as input parameter to ``make_island_groupings``. [#38]

- Removed ``npool`` as input parameter to ``source_pairing``. [#38]

- Added extra columns derived in ``counterpart_pairing`` to output datafiles in
  ``npy_to_csv``. [#37]

- ``npy_to_csv`` now has ``extra_col_name_lists``, allowing for the inclusion of
  extra columns from the original catalogue .csv file to be passed through to the
  output merged datafiles. [#37]

- Moved several functions (``_load_single_sky_slice``, ``_load_rectangular_slice``,
  ``_lon_cut``, ``_lat_cut``) out of individual Python scripts into
  ``misc_functions`` to generalise their use in the codebase. [#27]

- Removed ``norm_scale_laws`` as an input to catalogue configuration files. [#27]

- Added ``dens_mags``, ``num_trials``, ``dm_max``, ``d_mag``, and
  ``compute_local_density`` as inputs to the joint and catalogue-specific
  configuration files [#27]

- Added ``int_fracs`` as an input to the joint configuration file for a
  cross-match. [#25]

Other Changes
^^^^^^^^^^^^^

- Updated documentation to reflect previous improvements to codebase, and add further
  introductory and explanatory material. [#54]

- Changed ``_make_chunk_queue`` to return a queue ordered by file size in bytes
  and improve load balancing in MPI parallelised jobs. [#52]

- Added ``matplotlib`` as a dependency, and explictly defined ``pytest-cov`` as a
  test dependency. [#50]

- Added ``mpi4py`` as a dependency [#49]

- Added ``skypy`` and ``speclite`` as dependencies. [#41]

- Improved github actions matrix testing coverage. [#40]

- Added ``pandas`` as a dependency. [#34]

- Updates to documentation to reflect the relaxing of photometric likelihood and
  perturbation AUF component options. Other minor changes to documentation
  layout. [#30]

- GitHub Actions will only run remote data dependent tests (those marked with
  ``pytest.mark.remote_data``) on a pull request merge. [#27]

- Added ``astropy`` as a dependency. [#27]



0.1.1 (2021-01-06)
------------------

General
^^^^^^^

- Preliminary creation of user documentation. [#22]

- Established changelog [#8]

New Features
^^^^^^^^^^^^

- Created ``generate_random_data``, to create simulated catalogues for testing
  full end-to-end matches. [#20]

- Implemented computation of match probabilities for islands of sources,
  and secondary parameters such as flux contamination likelihood. [#19]

- Added naive Bayes priors based on the relative local densities of the two
  catalogues. [#18]

- Functionality added to create "island" groupings of sources across the two
  catalogues. [#16]

- Creation of the perturbation aspect of the AUF, in the limit that it is
  unused (i.e., the AUF is assumed to be Gaussian). [#12]

Bug Fixes
^^^^^^^^^

- Correct typing of ``point_ind`` in ``misc_function_fortran``'s
  ``find_nearest_point``. [#18]

- Fix mistake in ``haversine`` formula in ``perturbation_auf_fortran``. [#15]

API Changes
^^^^^^^^^^^

- Moved ``delta_mag_cut`` from ``make_perturb_aufs`` to an input variable, defined
  in ``create_perturb_auf``. [#19]

- Moved ``find_nearest_auf_point`` from being specific to ``perturbation_auf``,
  now located in ``misc_functions_fortran`` as ``find_nearest_point``. [#18]

- Update ``run_star`` to ``run_source``, avoiding any specific match
  implication. [#16]

- Require ``psf_fwhms`` regardless of whether ``include_perturb_auf`` is yes or
  not. [#9, #10]

- Preliminary API established, with parameters ingested from several
  input files. [#7]

Other Changes
^^^^^^^^^^^^^

- Added ``sphinx-fortran`` as a dependency. [#22]

- Added ``pytest-astropy`` as a dependency. [#17]

- Added ``scipy`` as a dependency. [#16]
