***********
Quick Start
***********

To get started quickly with ``macauff``, you will need nine items: three binary, ``.npy`` files for each of the two catalogues to be cross-matched, and three input plain text files.

Input Data
==========

The three input files must be contained within a single folder, with the respective names:

* ``con_cat_astro``, which must be of shape ``(N, 3)``, and contain, for each of the ``N`` objects to be matched in this catalogue, two astrometric position coordinates: the azimuthal angle (typically Right Ascension or Longitude) and polar angle (typically Declination or Latitude) -- both in degrees -- and a single, *circular* astrometric uncertainty, in arcseconds. If your data are described by covariance matrices for their uncertainty, then we require the average of the semi-major and semi-minor axes.

* ``con_cat_photo``, of shape ``(N, F)`` for a catalogue with ``F`` filters available for use. If a detection is not available for any reason -- either the source is below the detection limit of the survey, or it was removed for any reason during the catalogue creation process -- it must be replaced with a ``NaN`` value.

* ``magref``, of shape ``(N,)``. This "magnitude reference" array contains, for each of the ``N`` elements, the best available of the 1 to ``F`` detections of that object. Best is a subjective term, and left to the user, but general considerations to take into account are the closeness of the wavelength coverage to the opposing catalogue (where more similar wavelength ranges are better), quality of the detection, or if the source suffered any artefacts during its observation that may affect the photometry.

These files should be present for each catalogue, in separate folders.

An example function to create simple test catalogues is available :func:`here<macauff.tests.generate_random_data>`, and can be imported within Python as::

	from macauff.tests import generate_random_data

and called with

.. code-block:: python

	num_a_source, num_b_source, num_common = 50, 100, 40
	extent = [0, 0.25, 50, 50.3]
	num_filters_a, num_filters_b = 3, 2
	a_uncert, b_uncert = 0.1, 0.3
	a_cat_path, b_cat_path = 'name_of_a_folder', 'name_of_b_folder'
	generate_random_data(num_a_source, num_b_source, num_common, extent, num_filters_a,
	                     num_filters_b, a_uncert, b_uncert, a_cat_path, b_cat_path)

This will save the appropriate three files -- ``con_cat_astro``, ``con_cat_photo``, and ``magref`` -- in the respective catalogue path folders. Additionally, it will save the (zero-indexed) indices in each catalogue that correspond to the counterpart sources in the two catalogues, for "ground truth" comparison.

Input Parameters
================

Once you have your catalogue files, you will next require the files containing the input parameters. Examples of these files can be found in the ``macauff`` test data folder, for `catalogue "a" <https://raw.githubusercontent.com/Onoddil/macauff/master/macauff/tests/data/cat_a_params.txt>`_, `catalogue "b" <https://raw.githubusercontent.com/Onoddil/macauff/master/macauff/tests/data/cat_b_params.txt>`_, and the `common cross-match <https://raw.githubusercontent.com/Onoddil/macauff/master/macauff/tests/data/crossmatch_params.txt>`_ parameters.

These files are the only inputs to `~macauff.CrossMatch`, the main class for performing a cross-match between the two catalogues.

By default, the parameters in the files above are set up such that a cross-match will be performed in the equatorial coordinate frame region :math:`131 \leq \alpha \leq 138,\ -3 \leq \delta \leq 3`. Thus, depending on the coordinates you used when calling `~macauff.tests.generate_random_data`, you may wish to edit ``cross_match_extent`` and ``cf_region_points`` within ``crossmatch_params.txt``, to match your chosen sky coordinates.

.. note::
	Discussion of the input parameters available in the catalogue-specific and joint match-specific input files is provided in more detail :doc:`here<inputs>`.

Running the Matches
===================

With both your data and input files, you are now ready to perform your first cross-match! This should be as straightforward as running

.. code-block:: python
	
	from macauff import CrossMatch
	joint_file_path = 'some_location_here/common_match_parameters.txt'
	cat_a_file, cat_b_file = 'loc_one/catalogue_a_params.txt', 'loc_two/catalogue_b_params.txt'
	cross_match = CrossMatch(joint_folder_path, cat_a_file, cat_b_file)
	cross_match()

which will save all intermediate match data to the ``joint_folder_path`` parameter in ``joint_file_path``, and eventually produce a list of indices of matches for the two catalogues. Within Python these can be loaded by calling the original binary files::

	import numpy as np
	a = np.load('{}/con_cat_astro.npy'.format(a_cat_folder_path))
	b = np.load('{}/con_cat_astro.npy'.format(b_cat_folder_path))
	cat_a_match_inds = np.load('{}/pairing/ac.npy'.format(joint_folder_path))
	cat_b_match_inds = np.load('{}/pairing/bc.npy'.format(joint_folder_path))

	a_matches, b_matches = a[cat_a_match_inds], b[cat_b_match_inds]

You can then, for example, calculate the on-sky separations between these sources::

	import numpy as np
	from macauff.misc_functions_fortran import misc_functions_fortan as mff
	arcsec_seps = np.array([3600 * mff.haversine_wrapper(a_matches[i, 0], b_matches[i, 0],
	                        a_matches[i, 1], b_matches[i, 1]) for i in range(len(a_matches))])
