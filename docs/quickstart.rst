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
    a_cat_path = 'test_macauff_outputs/name_of_a_folder'
    b_cat_path = 'test_macauff_outputs/name_of_b_folder'
    generate_random_data(num_a_source, num_b_source, num_common, extent, num_filters_a,
                         num_filters_b, a_uncert, b_uncert, a_cat_path, b_cat_path)

This will save the appropriate three files -- ``con_cat_astro``, ``con_cat_photo``, and ``magref`` -- in the respective catalogue path folders. Additionally, it will save the (zero-indexed) indices in each catalogue that correspond to the counterpart sources in the two catalogues, for "ground truth" comparison.

Input Parameters
================

Once you have your catalogue files, you will next require the files containing the input parameters. Examples of these files can be found in the ``macauff`` test data folder, for `catalogue "a" <https://raw.githubusercontent.com/Onoddil/macauff/main/macauff/tests/data/cat_a_params.txt>`_, `catalogue "b" <https://raw.githubusercontent.com/Onoddil/macauff/main/macauff/tests/data/cat_b_params.txt>`_, and the `common cross-match <https://raw.githubusercontent.com/Onoddil/macauff/main/macauff/tests/data/crossmatch_params.txt>`_ parameters.

These files contain all of the inputs to `~macauff.CrossMatch`, the main class for performing a cross-match between the two catalogues.

Quick, self-consistent examples of the three files are:

crossmatch_params.txt::

    # High level match type
    include_perturb_auf = no
    include_phot_like = no
    use_phot_priors = no

    # File paths
    # Top-level folder for all temporary cross-match files to be created in. Should be absolute path, or relative to folder script called in
    joint_folder_path = test_macauff_outputs/test_path

    # Flags for each stage of the match process - must be "yes"/"no", "true"/"false", "t"/"f", or "1"/"0"
    run_auf = yes
    run_group = yes
    run_cf = yes
    run_source = yes

    # c/f region definition for photometric likelihood - either "rectangle" for NxM evenly spaced grid points, or "points" to define a list of two-point tuple coordinates, separated by a comma
    cf_region_type = rectangle
    # Frame of the coordinates must be specified -- either "equatorial" or "galactic"
    cf_region_frame = equatorial
    # For "points" this should be individually specified (ra, dec) or (l, b) coordinates [as "(a, b), (c, d)"]
    # For "rectangle" this should be 6 numbers: start coordinate, end coordinate, integer number of data points from start to end (inclusive of both start and end), first for ra/l, then for dec/b (depending on cf_region_type), all separated by spaces
    cf_region_points = 0.1 0.2 2 50.15 50.15 1

    # Maximum separation, in arcseconds, between two sources for them to be deemed positionally correlated
    pos_corr_dist = 2

    # Convolution (fourier transform) parameters
    # Integer number of real space grid points, for Hankel transformations
    real_hankel_points = 10000
    # Integer number of fourier space grid points
    four_hankel_points = 10000
    # Maximum fourier space "rho" parameter considered (typically larger than the inverse of the smallest Gaussian sigma)
    four_max_rho = 100

    # Maximum extent of cross-match, used in non-all-sky cases to remove sources suffering potential edge effects -- min/max first axis coordinates (ra/l) then min/max second axis coordinates (dec/b)
    cross_match_extent = 0 0.25 50 50.3

    # Number of chunks to break each catalogue into when splitting larger catalogues up for memory reasons
    mem_chunk_num = 2

    # Integral fractions for various error circle cutouts used during the cross-match process. Should be space-separated floats, in the order of <bright error circle fraction>, <field error circle fraction>, <potential counterpart integral limit>
    int_fracs = 0.63 0.9 0.999

cat_a_params.txt::

    # Catalogue name -- used both for folder creation and output file names
    cat_name = catalogue_a
    cat_folder_path = test_macauff_outputs/name_of_a_folder
    # Folder for all AUF-related files to be created in. Should be an absolute path, or relative to folder script called in.
    auf_folder_path = test_macauff_outputs/cat_a_auf_folder

    # Filter names are also used in any output file created
    filt_names = G_BP G G_RP

    # AUF region definition - either "rectangle" for NxM evenly spaced grid points, or "points" to define a list of two-point tuple coordinates, separated by a comma
    auf_region_type = rectangle
    # Frame of the coordinates must be specified -- either "equatorial" or "galactic"
    auf_region_frame = equatorial
    # For "points" this should be individually specified (ra, dec) or (l, b) coordinates [as "(a, b), (c, d)"]
    # For "rectangle" this should be 6 numbers: start coordinate, end coordinate, integer number of data points from start to end (inclusive of both start and end), first for ra/l, then for dec/b (depending on auf_region_type), all separated by spaces
    auf_region_points = 0.1 0.2 2 50.15 50.15 1

    # Local density calculation radius, in degrees
    dens_dist = 0.25

cat_b_params.txt::

    # Catalogue name -- used both for folder creation and output file names
    cat_name = catalogue_b
    cat_folder_path = test_macauff_outputs/name_of_b_folder
    # Folder for all AUF-related files to be created in. Should be an absolute path, or relative to folder script called in.
    auf_folder_path = test_macauff_outputs/cat_b_auf_folder

    # Filter names are also used in any output file created
    filt_names = W1 W2

    # AUF region definition - either "rectangle" for NxM evenly spaced grid points, or "points" to define a list of two-point tuple coordinates, separated by a comma
    auf_region_type = rectangle
    # Frame of the coordinates must be specified -- either "equatorial" or "galactic"
    auf_region_frame = equatorial
    # For "points" this should be individually specified (ra, dec) or (l, b) coordinates [as "(a, b), (c, d)"]
    # For "rectangle" this should be 6 numbers: start coordinate, end coordinate, integer number of data points from start to end (inclusive of both start and end), first for ra/l, then for dec/b (depending on auf_region_type), all separated by spaces
    auf_region_points = 0.1 0.2 2 50.15 50.15 1

    # Local density calculation radius, in degrees
    dens_dist = 0.25

.. note::
    Discussion of the input parameters available in the catalogue-specific and joint match-specific input files is provided in more detail :doc:`here<inputs>`.

Running the Matches
===================

With both your data and input files, you are now ready to perform your first cross-match! This should be as straightforward as saving the three above text files into a folder within ``test_macauff_inputs`` (e.g. ``match_run``) and, from the same folder as ``test_macauff_inputs`` is located in, running

.. code-block:: python

    if __name__ == '__main__':
        from macauff import CrossMatch
        parameter_file_path = 'test_macauff_inputs'
        cross_match = CrossMatch(parameter_file_path, use_mpi=False)
        cross_match()

which will save all intermediate match data to the ``joint_folder_path`` parameter in ``joint_file_path`` (``test_macauff_outputs/test_path`` if you used the files as given above), and eventually produce a list of indices of matches for the two catalogues. Within Python these can be loaded by calling the original binary files

.. code-block:: python

        import numpy as np
        joint_folder_path = 'test_macauff_outputs/test_path'
        a = np.load('{}/con_cat_astro.npy'.format(a_cat_path))
        b = np.load('{}/con_cat_astro.npy'.format(b_cat_path))
        cat_a_match_inds = np.load('{}/pairing/ac.npy'.format(joint_folder_path))
        cat_b_match_inds = np.load('{}/pairing/bc.npy'.format(joint_folder_path))

        a_matches, b_matches = a[cat_a_match_inds], b[cat_b_match_inds]

You can then, for example, calculate the on-sky separations between these sources

.. code-block:: python

        from macauff.misc_functions_fortran import misc_functions_fortan as mff
        arcsec_seps = np.array([3600 * mff.haversine_wrapper(a_matches[i, 0], b_matches[i, 0],
                                a_matches[i, 1], b_matches[i, 1]) for i in range(len(a_matches))])

Running More Complex Matches
============================

For example cross-matches, including some more advanced features available within ``macauff``, check out the :doc:`Real-World Matching<real_world_matches>` examples.

Documentation
=============

For the full documentation, click :doc:`here<macauff>`.