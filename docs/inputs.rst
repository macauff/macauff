****************
Input Parameters
****************

This page details the various inputs expected by `~macauff.CrossMatch`.


.. note::
	Currently only a naive, pure Bayes match is implemented. Thus ``include_perturb_auf``, ``include_phot_like``, and ``use_phot_priors`` must all be set to ``False``, and any parameters only used when either of these parameters is ``True`` can be ignored.

Catalogue-specific Parameters
=============================

These parameters are required in two separate files, one per catalogue to be cross-matched., the inputs ``cat_a_file_path`` and ``cat_b_file_path`` in `~macauff.CrossMatch`.

Catalogue Parameter Description
-------------------------------

``cat_name``

The name of the catalogue. This is used to generate intermediate folder structure within the cross-matching process, and during any output file creation process.

``cat_folder_path``

The folder containing the three files (see :doc:`quickstart` for more details) describing the given input catalogue. Can either be an absolute path, or relative to the folder from which the script was called.

``auf_folder_path``

The folder into which the Astrometric Uncertainty Function (AUF) related files will be, or have been, saved. Can also either be an absolute or relative path, like ``cat_folder_path``.

``filt_names``

The filter names of the photometric bandpasses used in this catalogue, in the order in which they are saved in ``con_cat_photo``. These will be used to describe any output data files generated after the matching process. Should be a space-separated list.

``psf_fwhms``

The Full-Width-At-Half-Maximum of each filter's Point Spread Function (PSF), in the same order as in ``filt_names``. These are used to simulate the PSF if ``include_perturb_auf`` is set to ``True``, and are unnecessary otherwise. Should be a space-separated list of floats.

``norm_scale_laws``

The scaling relations, number density of sources as a function of magnitude, for the bright end of the catalogue. Should be of the form :math:`N = N_0 z^m`, where :math:`z` is the scaling law parameter. Should be a space-separated list of floats. Optional if ``include_perturb_aufs`` is ``False``.

``tri_set_name``
The name of the filter set used to simulate the catalogue's sources in TRILEGAL [#]_. Used to interact with the TRILEGAL API; optional if ``include_perturb_aufs`` is ``False``.

``tri_filt_names``

The names of the filters, in the same order as ``filt_names``, as given in the data ``tri_set_name`` calls. Optional if ``include_perturb_aufs`` is ``False``.

``tri_filt_num``

The one-indexed column number of the magnitude, as determined by the column order of the saved data returned by the TRILEGAL API, to which to set the maximum magnitude limit for the simulation. Optional if ``include_perturb_aufs`` is ``False``.

``download_tri``

Boolean flag, indicating whether to re-download a TRILEGAL simulation in a given ``auf_region_points`` sky coordinate, once it has successfully been run, and to overwrite the original simulation data or not. Optional if ``include_perturb_aufs`` is ``False``.

``auf_region_type``

Flag indicating which definition to use for determining the pointings of the AUF simulations; accepts either ``rectangle`` or ``points``. If ``rectangle``, then ``auf_region_points`` will map out a rectangle of evenly spaced points, otherwise it accepts pairs of coordinates at otherwise random coordinates.

``auf_region_frame``

Flag to indicate which frame the data, and thus AUF simulations, are in. Can either be ``equatorial`` or ``galactic``, allowing for data to be input either in Right Ascension and Declination, or Galactic Longitude and Galactic Latitude.

``auf_region_points``

The list of pointings for which to run simulations of perturbations due to blended sources, if applicable. If ``auf_region_type`` is ``rectangle``, then ``auf_region_points`` accepts six numbers: ``start longitude, end longitude, number of longitude points, start latitude, end latitude, number of latitude points``; if ``points`` then tuples must be of the syntax ``(a, b), (c, d)`` where ``a`` and ``c`` and RA or Galactic Longitude, and ``b`` and ``d`` are Declination or Galactic Latitude.

``dens_dist``

The radius, in arcseconds, within which to count internal catalogue sources for each object, to calculate the local source density. Used to scale TRILEGAL simulated source counts to match smaller scale density fluctuations.


Joint Parameters
================

These parameters are only provided in the single, common-parameter input file, and given as ``joint_file_path`` in `~macauff.CrossMatch`.

Common Parameter Description
----------------------------

``include_perturb_auf``

Flag for whether to include the simulated effects of blended sources on the measured astrometry in the two catalogues or not. Currently must be ``False``.


``include_phot_like``

Flag for the inclusion of the likelihood of match or non-match based on the photometric information in the two catalogues. Must be ``False``.

``use_phot_priors``

Flag to determine whether to calculate the priors on match or non-match using the photometry (if set to ``True``) or calculate them based on a naive asymmetric density argument (``False``). Currently should be ``False``.

``joint_folder_path``

The top-level folder location, into which all intermediate files and folders are placed, when created during the cross-match process.

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

``cf_region_type``

Similar to ``auf_region_type``, this flag controls whether the areas in which photometric likelihoods are calculated is determined by ``rectangle`` -- evenly spaced longitude/latitude pairings -- or ``points`` -- tuples of randomly placed coordinates.

``cf_region_frame``

As with ``auf_region_frame``, this allows either ``equatorial`` or ``galactic`` frame coordinates to be used in the match process.

``cf_region_points``

Based on ``cf_region_type``, this must either by six space-separated floats, controlling the start and end, and number of, longitude and latitude points in ``start lon end lon # steps start lat end lat #steps`` order (see ``auf_region_points``), or a series of comma-separated tuples cf. ``(a, b), (c, d)``.

``pos_corr_dist``

The floating point precision number determining the maximum possible separation between two sources in opposing catalogues.

``real_hankel_points``

The integer number of points, for Hankel (two-dimensional Fourier) transformations, in which to approximate the fourier transformation integral of the AUFs.

``four_hankel_points``

The integer number of points for approximating the inverse Hankel transformation, representing the convolution of two real-space AUFs.

``four_max_rho``

The largest fourier-space value, up to which inverse Hankel transformation integrals are considered. Should typically be larger than the inverse of the smallest typical centroiding Gaussian one-dimensional uncertainty.

``cross_match_extent``

The maximum extent of the matching process. When not matching all-sky catalogues, these extents are used to eliminate potential matches within "island" overlap range of the edge of the data, whose potential incompleteness renders the probabilities of match derived uncertain. Must be of the form ``lower longitude upper longitude lower latitude upper latitude``; accepts four space-separated floats.

``mem_chunk_num``

The number of smaller subsets into which to break various loops throughout the cross-match process. Used to reduce the memory usage of the process at any given time, in case of catalogues too large to fit into memory at once.



.. rubric:: Footnotes

.. [#] Please see `here <http://stev.oapd.inaf.it/~webmaster/trilegal_1.6/papers.html>`_ to view the TRILEGAL papers to cite, if you use this software in your publication.
