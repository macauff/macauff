****************
Input Parameters
****************

This page details the various inputs expected by `~macauff.CrossMatch` via its input parameter files. These are split into two main sections: inputs that both catalogues need separately (contained in their respective catalogue parameters file) and inputs that are defined with respect to the catalogue-catalogue cross-match.

These input parameters are required to be in ``crossmatch_params*.txt`` for the `Joint Parameters`_ options, and ``cat_a_params*.txt``, ``cat_b_params*.txt`` for the `Catalogue-specific Parameters`_ options, where the asterisk indicates a wildcard -- i.e., the text files must end ``.txt`` and begin e.g. ``crossmatch_params``,  but can contain text between those two parts of the file.

Depending on the size of the match being performed, it is likely that you may want to use the MPI parallelisation option to save runtime or memory overhead, splitting your matches into "chunks." Even if your cross-match area is small enough that you only have a single file per catalogue, this is treated as a single chunk. Each individual chunk -- or your only chunk -- is required to have its own set of three input text parameter files, within folders inside ``chunks_folder_path`` as passed to `~macauff.CrossMatch`. Most of the parameters within these files will be the same across all chunks -- you'll likely always want to include the perturbation component of the AUF or include photometric likelihoods, or use the same filters across all sub-catalogue cross-matches -- but some will vary on a per-chunk basis, most notably anything that involves astrometric coordiantes, like ``cross_match_extent``.

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

- ``use_memmap_files``: a boolean flag indicating whether or not to save temporary, intermediate arrays to disk and load variables through memmapping. Should be used if the catalogue regions being matched are sufficiently large that they cannot be stored in memory at once. Alternatively, consider smaller chunks. Optional, defaulting to ``False``.

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

``joint_folder_path``, ``run_auf``, ``run_group``, ``run_cf``, ``run_source``, ``include_perturb_auf``, ``include_phot_like``, ``use_phot_priors``, ``cross_match_extent``, ``mem_chunk_num``, ``pos_corr_dist``, ``cf_region_type``, ``cf_region_frame``, ``cf_region_points``, ``real_hankel_points``, ``four_hankel_points``, ``four_max_rho``, ``int_fracs``, and ``n_pool``;

and those options which only need to be supplied if ``include_perturb_auf`` is ``True``:

``num_trials``, ``compute_local_density``, and ``d_mag``.

.. note::
    ``num_trials`` and ``d_mag`` currently need to be supplied if either ``correct_astrometry`` option in the two `Catalogue-specific Parameters`_ config files is ``True`` as well.

Common Parameter Description
----------------------------

``joint_folder_path``

The top-level folder location, into which all intermediate files and folders are placed, when created during the cross-match process. This can either be an absolute file path, or relative to the folder from which your script that called `CrossMatch()` is based.

.. note::
    The four ``run_`` parameters below are called in order. If an earlier stage flag is set to ``True``, an error will be raised in a subsequent flag is set to ``False``.

``run_auf``

Flag to determine if the AUF simulation stage of the cross-match process should be run, or if previously generated files should be used when present.

``run_group``

Flag dictating whether the source grouping -- and island creation -- stage of the process is run, or if previously created islands of sources should be used for this match.

``run_cf``

Flag controlling whether or not to calculate the photometric likelihood information, as determined by ``include_phot_like`` and ``use_phot_priors``, for this cross-match.

``run_source``

Boolean determining whether to run the final stage of the cross-match process, in which posterior probabilities of matches and non-matches for each island of sources are calculated.

``include_perturb_auf``

Flag for whether to include the simulated effects of blended sources on the measured astrometry in the two catalogues or not.

``include_phot_like``

Flag for the inclusion of the likelihood of match or non-match based on the photometric information in the two catalogues.

``use_phot_priors``

Flag to determine whether to calculate the priors on match or non-match using the photometry (if set to ``True``) or calculate them based on a naive asymmetric density argument (``False``).

``cross_match_extent``

The maximum extent of the matching process. When not matching all-sky catalogues, these extents are used to eliminate potential matches within "island" overlap range of the edge of the data, whose potential incompleteness renders the probabilities of match derived uncertain. Must be of the form ``lower-longitude upper-longitude lower-latitude upper-latitude``; accepts four space-separated floats.

``mem_chunk_num``

The number of smaller subsets into which to break various loops throughout the cross-match process. Used to reduce the memory usage of the process at any given time, in case of catalogues too large to fit into memory at once.

``pos_corr_dist``

The floating point precision number determining the maximum possible separation between two sources in opposing catalogues.

``cf_region_type``

This flag controls whether the areas in which photometry-related variables (likelihoods, priors, etc.) are calculated is determined by ``rectangle`` -- evenly spaced longitude/latitude pairings -- or ``points`` -- tuples of randomly placed coordinates.

``cf_region_frame``

This allows either ``equatorial`` or ``galactic`` frame coordinates to be used in the match process.

``cf_region_points``

The list of pointings for which to run simulations of perturbations due to blended sources, if applicable. If ``cf_region_type`` is ``rectangle``, then ``cf_region_points`` accepts six numbers: ``start longitude, end longitude, number of longitude points, start latitude, end latitude, number of latitude points``; if ``points`` then tuples must be of the syntax ``(a, b), (c, d)`` where ``a`` and ``c`` are RA or Galactic Longitude, and ``b`` and ``d`` are Declination or Galactic Latitude.

``real_hankel_points``

The integer number of points, for Hankel (two-dimensional Fourier) transformations, in which to approximate the fourier transformation integral of the AUFs.

``four_hankel_points``

The integer number of points for approximating the inverse Hankel transformation, representing the convolution of two real-space AUFs.

``four_max_rho``

The largest fourier-space value, up to which inverse Hankel transformation integrals are considered. Should typically be larger than the inverse of the smallest typical centroiding Gaussian one-dimensional uncertainty.

``n_pool``

Determines how many CPUs are used when parallelising within ``Python`` using ``multiprocessing``.

``int_fracs``

The integral fractions of the various so-called "error circles" used in the cross-match process. Should be space-separated floats, in the order of: bright error circle fraction, "field" error circle fraction, and potential counterpart cutoff limit.

``num_trials``

The number of PSF realisations to draw when simulating the perturbation component of the AUF. Should be an integer. Only required if ``include_perturb_auf`` is ``True``.

``compute_local_density``

Boolean flag, ``yes`` or ``no``, to indicate whether to on-the-fly compute the local densities of sources in each catalogue for use in its perturbation AUF component, or to use pre-computed values. ``yes`` indicates values will be computed during the cross-match process. Only required if ``include_perturb_auf`` is ``True``.

``d_mag``

Bin sizes for magnitudes used to represent the source number density used in the random drawing of perturbation AUF component PSFs. Should be a single float. Only required if ``include_perturb_auf`` is ``True``.


Catalogue-specific Parameters
=============================

These parameters are required in two separate files, one per catalogue to be cross-matched, the files ``cat_a_params.txt`` and ``cat_b_params.txt`` read from sub-folders within ``chunks_folder_path`` as passed to `~macauff.CrossMatch`.

These can be divided into those inputs that are always required:

``cat_folder_path``, ``cat_name``, ``filt_names``, ``auf_folder_path``, ``auf_region_type``, ``auf_region_frame``, ``auf_region_points``, and ``correct_astrometry``;

those that are only required if the `Joint Parameters`_ option ``include_perturb_auf`` is ``True``:

``fit_gal_flag``, ``run_fw_auf``, ``run_psf_auf``, ``psf_fwhms``, ``dens_mags``, ``snr_mag_params_path``, ``download_tri``, ``tri_set_name``, ``tri_filt_names``, ``tri_filt_num``, ``tri_maglim_faint``, and ``tri_num_faint``;

parameters required if ``run_psf_auf`` is ``True``:

``dd_params_path`` and ``l_cut_path``;

the parameter needed if `Joint Parameters`_ option ``compute_local_density`` is ``True`` (and hence ``include_perturb_auf`` is ``True``):

``dens_dist``;

the inputs required in each catalogue parameters file if ``fit_gal_flag`` is ``True`` (and hence ``include_perturb_auf`` is ``True``):

``gal_wavs``, ``gal_zmax``, ``gal_nzs``, ``gal_aboffsets``, ``gal_filternames``, and ``gal_al_avs``;

and the inputs required if ``correct_astrometry`` is ``True``:

``best_mag_index``, ``nn_radius``, ``correct_astro_save_folder``, ``csv_cat_file_string``, ``ref_csv_cat_file_string``, ``correct_mag_array``, ``correct_mag_slice``, ``correct_sig_slice``, ``pos_and_err_indices``, ``mag_indices``, and ``mag_unc_indices``.

.. note::
    ``run_fw_auf``, ``run_psf_auf``, ``psf_fwhms``, ``dens_mags``, ``snr_mag_params_path``, ``download_tri``, ``tri_set_name``, ``tri_filt_names``, ``tri_filt_num``, ``tri_maglim_faint``, ``tri_num_faint``, ``dens_dist``, ``dd_params_path``, ``l_cut_path``, ``gal_wavs``, ``gal_zmax``, ``gal_nzs``, ``gal_aboffsets``, ``gal_filternames``, and ``gal_al_avs`` are all currently required if ``correct_astrometry`` is ``True``, bypassing the nested flags above. For example, ``dens_dist`` is required as an input if ``compute_local_density`` and ``include_perturb_auf`` are both ``True``, or if ``correct_astrometry`` is set. This means that ``AstrometricCorrections`` implicitly always runs and fits for a full Astrometric Uncertainty Function.


Catalogue Parameter Description
-------------------------------

``cat_folder_path``

The folder containing the three files (see :doc:`quickstart` for more details) describing the given input catalogue. Can either be an absolute path, or relative to the folder from which the script was called.

``cat_name``

The name of the catalogue. This is used to generate intermediate folder structure within the cross-matching process, and during any output file creation process.

``filt_names``

The filter names of the photometric bandpasses used in this catalogue, in the order in which they are saved in ``con_cat_photo``. These will be used to describe any output data files generated after the matching process. Should be a space-separated list.

``auf_folder_path``

The folder into which the Astrometric Uncertainty Function (AUF) related files will be, or have been, saved. Can also either be an absolute or relative path, like ``cat_folder_path``.

``auf_region_type``

Similar to ``cf_region_type``, flag indicating which definition to use for determining the pointings of the AUF simulations; accepts either ``rectangle`` or ``points``. If ``rectangle``, then ``auf_region_points`` will map out a rectangle of evenly spaced points, otherwise it accepts pairs of coordinates at otherwise random coordinates.

``auf_region_frame``

As with ``auf_region_frame``, this flag indicates which frame the data, and thus AUF simulations, are in. Can either be ``equatorial`` or ``galactic``, allowing for data to be input either in Right Ascension and Declination, or Galactic Longitude and Galactic Latitude.

``auf_region_points``

Based on ``auf_region_type``, this must either by six space-separated floats, controlling the start and end, and number of, longitude and latitude points in ``start lon end lon # steps start lat end lat #steps`` order (see ``cf_region_points``), or a series of comma-separated tuples cf. ``(a, b), (c, d)``.

``correct_astrometry``

In cases where catalogues have unreliable *centroid* uncertainties, before catalogue matching occurs the dataset can be fit for systematic corrections to its quoted astrometric precisions through ensemble match separation distance distributions to a higher-precision dataset (see the :doc:`Processing<pre_post_process>` section). This flag controls whether this is performed on a chunk-by-chunk basis during the initialisation step of ``CrossMatch``.

``fit_gal_flag``

Optional flag for whether to include simulated external galaxy counts, or just include Galactic sources when deriving the perturbation component of the AUF. Only needed if ``include_perturb_auf`` is ``True``.

``run_fw_auf``

Boolean flag controlling the option to include the flux-weighted algorithm for determining the centre-of-light perturbation with AUF component simulations. Only required if  ``include_perturb_auf`` is ``True``.

``run_psf_auf``

Complementary flag to ``run_fw_auf``, indicates whether to run background-dominated, PSF photometry algorithm for the determination of perturbation due to hidden contaminant objects. If both this and ``run_fw_auf`` are ``True`` a signal-to-noise-based weighting between the two algorithms is implemented. Must be provided if  ``include_perturb_auf`` is ``True``.

``psf_fwhms``

The Full-Width-At-Half-Maximum of each filter's Point Spread Function (PSF), in the same order as in ``filt_names``. These are used to simulate the PSF if ``include_perturb_auf`` is set to ``True``, and are unnecessary otherwise. Should be a space-separated list of floats.

``dens_mags``

The magnitude, in each bandpass -- the same order as ``filt_names`` -- down to which to count the number of nearby sources when deriving the local normalising density of each object. Should be space-separated floats, of the same number as those given in ``filt_names``.

``snr_mag_params_path``

File path, either absolute or relative to the location of the script the cross-matches are run from, of a binary ``.npy`` file containing the parameterisation of the signal-to-noise ratio of sources as a function of magnitude, in a series of given sightlines. Must be of shape ``(N, M, 5)`` where ``N`` is the number of filters in ``filt_names`` order, ``M`` is the number of sightlines for which SNR vs mag has been derived, and the 5 entries for each filter-sightline combination must be in order ``a``, ``b``, ``c``, ``coord1`` (e.g. RA), and ``coord2`` (e.g. Dec). See pre-processing for more information on the meaning of those terms and how ``snr_mag_params`` is used.

``download_tri``

Boolean flag, indicating whether to re-download a TRILEGAL simulation in a given ``auf_region_points`` sky coordinate, once it has successfully been run, and to overwrite the original simulation data or not. Optional if ``include_perturb_aufs`` is ``False``.


``tri_set_name``
The name of the filter set used to simulate the catalogue's sources in TRILEGAL [#]_. Used to interact with the TRILEGAL API; optional if ``include_perturb_aufs`` is ``False``.

``tri_filt_names``

The names of the filters, in the same order as ``filt_names``, as given in the data ``tri_set_name`` calls. Optional if ``include_perturb_aufs`` is ``False``.

``tri_filt_num``

The one-indexed column number of the magnitude, as determined by the column order of the saved data returned by the TRILEGAL API, to which to set the maximum magnitude limit for the simulation. Optional if ``include_perturb_aufs`` is ``False``.

``tri_maglim_faint``

This is the float that represents the magnitude down to which to simulate TRILEGAL sources in the full-scale simulation, bearing in mind the limiting magnitude cut of the public API but also making sure this value is sufficiently faint that it contains all potentially perturbing objects for the dynamic range of this catalogue (approximately 10 magnitudes fainter than the limiting magnitude of the survey)

``tri_num_faint``

Integer number of objects to draw from the TRILEGAL simulation -- affecting the area of simulation, up to the limit imposed by TRILEGAL -- down to the full ``tri_maglim_faint`` magnitude.

``dd_params_path``

File path containin the ``.npy`` file describing the parameterisations of perturbation offsets due to single hidden contaminating, perturbing objects in the ``run_psf_auf`` background-dominated, PSF photometry algorithm case. See pre-processing documentation for more details on this, and how to generate this file if necessary.

``l_cut_path``

Alongside ``dd_params_path``, path to the ``.npy`` file containing the limiting flux cuts at which various PSF photometry perturbation algorithms apply. See pre-processing documentation for the specifics and how to generate this file if necesssary.

``dens_dist``

The radius, in arcseconds, within which to count internal catalogue sources for each object, to calculate the local source density. Used to scale TRILEGAL simulated source counts to match smaller scale density fluctuations. Only required if ``compute_local_density`` is ``True`` (and hence ``include_perturb_auf`` is also ``True``).

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

Differential extinction relative to the V-band for each filter, a set of space-separated floats. Must be provided if ``fit_gal_flag`` is ``True``.

``best_mag_index``

For the purposes of correcting systematic biases in a given catalogue, a single photometric band is used. ``best_mag_index`` indicates which filter to use -- e.g., ``best_mag_index = 0`` says to use the first filter as given in ``filt_names`` or ``mag_indices``. Must be a single integer value no larger than ``len(filt_names)-1``.

``nn_radius``

Nearest neighbour radius out to which to search for potential counterparts for the purposes of ensemble match separation distributions; should be a single float.

``correct_astro_save_folder``

File path, relative or absolute, into which to save files as generated by the astrometric correction process.

``csv_cat_file_string``

Path and filename, all in a single string, containing the location of each correction sightline's dataset to test. Must contain the appropriate number of string format ``{}`` identifiers depending on ``coord_or_chunk`` -- in this case, a single "chunk" identifier for corrections done through ``CrossMatch``. For example, ``/your/path/to/file/data_{}.csv`` where each "chunk" is saved into a csv file called ``data_1``, ``data_2``, ``data_104`` etc.

``ref_csv_cat_file_string``

Similar to ``csv_cat_file_string``, but the path and filename of the *reference* dataset used in the matching process. These chunks should correspond one-to-one with those used in ``csv_cat_file_string`` -- i.e., ``data_1.csv`` in ``/your/path/to/file`` should be the same region of the sky as the reference catalogue in ``/another/path/to/elsewhere/reference_data_1.csv``, potentially with some buffer overlap to avoid false matches at the edges.

``correct_mag_array``

List of magnitudes at which to evaluate the distribution of matches to the higher-astrometric-precision dataset in the chosen ``best_mag_index`` filter. Accepts a list of floats.

``correct_mag_slice``

Corresponding to each magnitude in ``correct_mag_array``, each element of this list of floats should be a width around each ``correct_mag_array`` element to select sources, ensuring a small sub-set of similar brightness objects are used to determine the Astrometric Uncertainty Function of.

``correct_sig_slice``

Elementwise with ``correct_mag_array`` and ``correct_mag_slice``, a list of floats of widths of astrometric precision to select a robust sub-sample of objects in each magnitude bin for, ensuring a self-similar AUF.

``pos_and_err_indices``

A list of six integers, the first three elements of which are the zero-indexed indices into the *reference* catalogue .csv file (``ref_csv_cat_file_string``) for the longitudinal coordinate, latitudinal coordinate, and circular astrometric precision respectively, followed by the lon/lat/uncert of the *input* catalogue. For example, ``0 1 2 10 9 8`` suggests that the reference catalogue begins with the position and uncertainty of its objects while the catalogue "a" or "b" sources have, in their original .csv file, a backwards list of coordinates and precisions towards the final columns of the filing system.

``mag_indices``

Just for the input catalogue, a list of ``len(filt_names)`` space-separated integers detailing the zero-indexed column number of the magnitudes in the dataset.

``mag_unc_indices``

Similar to ``mag_indices``, a list of ``len(mag_indices)`` space-separated integers, one for each column in ``mag_indices`` for where the corresponding uncertainty column is held for each magnitude in the input .csv file.

.. rubric:: Footnotes

.. [#] Please see `here <http://stev.oapd.inaf.it/~webmaster/trilegal_1.6/papers.html>`_ to view the TRILEGAL papers to cite, if you use this software in your publication.

