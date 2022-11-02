.. macauff documentation master file, created by
   sphinx-quickstart on Tue Jun 23 14:13:11 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

#######
macauff
#######
**The Python package for Matching Across Catalogues using the Astrometric Uncertainty Function and Flux**

``macauff`` is a package for cross-matching photometric catalogues. Using the positions, uncertainties, and flux measurements of sources, as well as modelling of the level to which objects are affected by hidden, blended contaminants, ``macauff`` provides posterior probabilities of "many-to-many" matches and non-matches between the two catalogues being merged. It also provides numerous secondary parameters, such as the level to which sources are flux contaminated, and the probability of their suffering blended by a source of a given flux ratio.

.. _getting-started:

************
Installation
************

The instructions for installing ``macauff`` can be found :doc:`here<installation>`.

.. _quick-start:

***********
Quick Start
***********

A quick-start guide is available :doc:`here<quickstart>`.

****************
Input Parameters
****************

Input parameters are detailed on :doc:`this<inputs>` page.

*********************
Real-World Match Case
*********************

A more complex match case, using pre-existing photometric catalogues, is described :doc:`here<real_world_matches>`.

******************
User Documentation
******************

.. toctree::
   :maxdepth: 1

   macauff

************************
Pre- and Post-Processing
************************


..
   ********************************
   Interpreting Cross-Match Results
   ********************************



*******************
Algorithmic Details
*******************

For specific implementation details and the mathematics used in macauff, see the :doc:`algorithms<algorithms>` page.

..
   **************************
   Starting a New Cross-Match
   **************************



******
Search
******

* :ref:`search`

.. toctree::
   :hidden:

   installation
   quickstart
   inputs
   real_world_matches
   algorithms
   f90_docs/perturb_auf_docs
   f90_docs/group_sources_docs
   f90_docs/phot_like_docs
   f90_docs/pairing_docs
   f90_docs/misc_func_docs
   f90_docs/shared_library_docs
