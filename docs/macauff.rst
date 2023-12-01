*************************
``macauff`` Documentation
*************************

This page details the inputs and outputs for various Python and Fortran functions/subroutines used internally within macauff. While `~macauff.CrossMatch` is the main input for most users, it is possible to overwrite each of the four main steps within the matching process (AUF component creation, island group creation, photometric likelihood derivation, and final match assignment) and hence it may be important to understand the required I/O to a given step for compatibility purposes.

For the details of the inputs that `~macauff.CrossMatch` expects and parses through its `read_metadata` function, see :doc:`inputs` for specifics.

Python
======
.. automodapi:: macauff
    :no-inheritance-diagram:
    :no-heading:

Fortran
=======

:doc:`Perturbation AUF<f90_docs/perturb_auf_docs>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:doc:`Group Sources<f90_docs/group_sources_docs>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:doc:`Photometric Likelihood<f90_docs/phot_like_docs>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:doc:`Counterpart Pairing<f90_docs/pairing_docs>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:doc:`Miscellaneous Functions<f90_docs/misc_func_docs>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:doc:`Shared Library<f90_docs/shared_library_docs>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
