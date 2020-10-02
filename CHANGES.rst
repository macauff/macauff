0.1.1 (unreleased)
----------------

General
^^^^^^^

- Established changelog [#8]

New Features
^^^^^^^^^^^^

- Creation of the perturbation aspect of the AUF, in the limit that it is
  unused (i.e., the AUF is assumed to be Gaussian). [#12]

Bug Fixes
^^^^^^^^^

API Changes
^^^^^^^^^^^

- Require `psf_fwhms` regardless of whether `include_perturb_auf` is yes or
  not. [#9, #10]

- Preliminary API established, with parameters ingested from several
  input files. [#7]

Other Changes
^^^^^^^^^^^^^

- Consistency within documentation strings for ``CrossMatch`` [#11]