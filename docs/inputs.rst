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

Joint Parameters
================

These parameters are only provided in the single, common-parameter input file, and given as ``joint_file_path`` in `~macauff.CrossMatch`.

There are some parameters that must be given in all runs:

``joint_folder_path``, ``run_auf``, ``run_group``, ``run_cf``, ``run_source``, ``include_perturb_auf``, ``include_phot_like``, ``use_phot_priors``, ``cross_match_extent``, ``mem_chunk_num``, ``pos_corr_dist``, ``cf_region_type``, ``cf_region_frame``, ``cf_region_points``, ``real_hankel_points``, ``four_hankel_points``, ``four_max_rho``, and ``int_fracs``;

and those options which only need to be supplied if ``include_perturb_auf`` is ``True``:

``num_trials``, ``compute_local_density``, and ``d_mag``.

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

``cat_folder_path``, ``cat_name``, ``filt_names``, ``auf_folder_path``, ``auf_region_type``, ``auf_region_frame``, and ``auf_region_points``;

those that are only required if the `Joint Parameters`_ option ``include_perturb_auf`` is ``True``:

``fit_gal_flag``, ``run_fw_auf``, ``run_psf_auf``, ``psf_fwhms``, ``dens_mags``, ``mag_h_params_path``, ``download_tri``, ``tri_set_name``, ``tri_filt_names``, ``tri_filt_num``, ``tri_maglim_bright``, ``tri_maglim_faint``, ``tri_num_bright``, and ``tri_num_faint``;

parameters required if ``run_psf_auf`` is ``True``:

``dd_params_path`` and ``l_cut_path``;

the parameter only needed if `Joint Parameters`_ option ``compute_local_density`` is ``True``:

``dens_dist``;

and the inputs required in each catalogue parameters file if ``fit_gal_flag`` is ``True``:

``gal_wavs``, ``gal_zmax``, ``gal_nzs``, ``gal_aboffsets``, ``gal_filternames``, and ``gal_al_avs``.


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

``mag_h_params_path``

File path, either absolute or relative to the location of the script the cross-matches are run from, of a binary ``.npy`` file containing the parameterisation of the signal-to-noise ratio of sources as a function of magnitude, in a series of given sightlines. Must be of shape ``(N, M, 5)`` where ``N`` is the number of filters in ``filt_names`` order, ``M`` is the number of sightlines for which SNR vs mag has been derived, and the 5 entries for each filter-sightline combination must be in order ``a``, ``b``, ``c``, ``coord1`` (e.g. RA), and ``coord2`` (e.g. Dec). See pre-processing for more information on the meaning of those terms and how ``mag_h_params`` is used.

``download_tri``

Boolean flag, indicating whether to re-download a TRILEGAL simulation in a given ``auf_region_points`` sky coordinate, once it has successfully been run, and to overwrite the original simulation data or not. Optional if ``include_perturb_aufs`` is ``False``.


``tri_set_name``
The name of the filter set used to simulate the catalogue's sources in TRILEGAL [#]_. Used to interact with the TRILEGAL API; optional if ``include_perturb_aufs`` is ``False``.

``tri_filt_names``

The names of the filters, in the same order as ``filt_names``, as given in the data ``tri_set_name`` calls. Optional if ``include_perturb_aufs`` is ``False``.

``tri_filt_num``

The one-indexed column number of the magnitude, as determined by the column order of the saved data returned by the TRILEGAL API, to which to set the maximum magnitude limit for the simulation. Optional if ``include_perturb_aufs`` is ``False``.

``tri_maglim_bright``

When using TRILEGAL to simulated Galactic sources for perturbation AUF component purposes, two simulations are run: one, with a brighter limiting magnitude, to capture the shape of the differential source counts at bright magnitudes, and one, with a sufficiently deep limiting magnitude that it contains all potentially perturbing objects for the dynamic range of this catalogue (approximately 10 magnitudes fainter than the limiting magnitude of the survey). ``tri_maglim_bright`` is the faintest magnitude down to which to draw TRILEGAL sources for the bright-end simulation.

``tri_maglim_faint``

Complementary to ``tri_maglim_bright``, this is the float that represents the magnitude down to which to simulate TRILEGAL sources in the full-scale simulation, bearing in mind the limiting magnitude cut of the public API.

``tri_num_bright``

The integer number of sources to simulate -- affecting the area of simulation, up to the limit imposed by TRILEGAL -- for the bright simulation of Galactic sources.

``tri_num_faint``

Number of objects to draw from the TRILEGAL simulation down to the full ``tri_maglim_faint`` magnitude.

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

.. rubric:: Footnotes

.. [#] Please see `here <http://stev.oapd.inaf.it/~webmaster/trilegal_1.6/papers.html>`_ to view the TRILEGAL papers to cite, if you use this software in your publication.

