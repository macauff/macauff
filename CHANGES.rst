0.1.1 (unreleased)
----------------

General
^^^^^^^

- Established changelog [#8]

New Features
^^^^^^^^^^^^

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
