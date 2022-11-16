0.1.2 (unreleased)
------------------

General
^^^^^^^

New Features
^^^^^^^^^^^^

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
