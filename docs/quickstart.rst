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

An example function to create simple test catalogues is available :func:`here<macauff.utils.generate_random_data>`, and can be imported within Python as::

    from macauff.utils import generate_random_data

and called with

.. code-block:: python

    num_a_source, num_b_source, num_common = 50, 100, 40
    extent = [0, 0.25, 50, 50.3]
    num_filters_a, num_filters_b = 3, 2
    a_uncert, b_uncert = 0.1, 0.3
    a_file_path = 'test_macauff_outputs/name_of_a_file.csv'
    b_file_path = 'test_macauff_outputs/name_of_b_file.csv'
    (a_astro, b_astro, a_photo, b_photo, a_mag_ind, b_mag_ind, a_true,
     b_true) = generate_random_data(
        num_a_source, num_b_source, num_common, extent, num_filters_a,
        num_filters_b, a_uncert, b_uncert)
    a_array = np.hstack((a_astro, a_photo, np.zeros((len(a_astro)), bool), a_mag_ind))
    with open(a_file_path, "w", encoding='utf-8') as f:
        np.savetxt(f, a_array, delimiter=',')
    b_array = np.hstack((b_astro, b_photo, np.zeros((len(b_astro)), bool), b_mag_ind))
    with open(b_file_path, "w", encoding='utf-8') as f:
        np.savetxt(f, b_array, delimiter=',')

This will provide three fake-data arrays -- astrometry (right ascension, declination, uncertainty), photometry (``num_filters_*`` per source), and a "best magnitude" array -- per catalogue, which can then be saved as a comma-separated file. Additionally, it will return the (zero-indexed) indices in each catalogue that correspond to the counterpart sources in the two catalogues, for "ground truth" comparison.

Input Parameters
================

Once you have your catalogue files, you will next require the files containing the input parameters. Examples of these files can be found in the ``macauff`` test data folder, for `catalogue "a" <https://raw.githubusercontent.com/macauff/macauff/main/tests/macauff/data/cat_a_params.yaml>`_, `catalogue "b" <https://raw.githubusercontent.com/macauff/macauff/main/tests/macauff/data/cat_b_params.yaml>`_, and the `common cross-match <https://raw.githubusercontent.com/macauff/macauff/main/tests/macauff/data/crossmatch_params.yaml>`_ parameters.

These files contain all of the inputs to `~macauff.CrossMatch`, the main class for performing a cross-match between the two catalogues.

Quick, self-consistent examples of the three files are:

crossmatch_params.yaml::

    # High level match type
    include_perturb_auf: False
    include_phot_like: False
    use_phot_priors: False

    # File paths
    # Top-level folder for all temporary cross-match files to be created in. Should be absolute path, or relative to folder script called in
    output_save_folder: test_macauff_outputs/test_path

    make_output_csv: False

    # c/f region definition for photometric likelihood - either "rectangle" for NxM evenly spaced grid points, or "points" to define a list of two-point tuple coordinates, separated by a comma
    cf_region_type: rectangle
    # Frame of the coordinates must be specified -- either "equatorial" or "galactic"
    cf_region_frame: equatorial
    # For "points" this should be individually specified (ra, dec) or (l, b) coordinates, as [[a, b], [c, d], [e, f]].
    # For "rectangle" this should be 6 numbers: start coordinate, end coordinate, integer number of data points from start to end (inclusive of both start and end), first for ra/l, then for dec/b (depending on cf_region_type), all separated by spaces
    cf_region_points_per_chunk:
      - [131, 134, 4, -1, 1, 3]
    chunk_id_list:
      - 1

    # Maximum separation, in arcseconds, between two sources for them to be deemed positionally correlated
    pos_corr_dist: 2

    # Convolution (fourier transform) parameters
    # Integer number of real space grid points, for Hankel transformations
    real_hankel_points: 10000
    # Integer number of fourier space grid points
    four_hankel_points: 10000
    # Maximum fourier space "rho" parameter considered (typically larger than the inverse of the smallest Gaussian sigma)
    four_max_rho: 100

    # Integral fractions for various error circle cutouts used during the cross-match process. Should be space-separated floats, in the order of <bright error circle fraction>, <field error circle fraction>, <potential counterpart integral limit>
    int_fracs: [0.63, 0.9, 0.999]

    # Multiprocessing CPU count
    n_pool: 2


cat_a_params.yaml::

    # Catalogue name -- used both for folder creation and output file names
    cat_name: Gaia
    cat_csv_file_path: test_macauff_outputs/name_of_a_folder/catalogue_a_{}.csv
    # Folder for all AUF-related files to be created in. Should be an absolute path, or relative to folder script called in.
    auf_folder_path: test_macauff_outputs/cat_a_auf_folder_{}

    pos_and_err_indices: [0, 1, 2]
    mag_indices: [3, 4, 5]
    chunk_overlap_col: 6
    best_mag_index_col: 7
    csv_has_header: False

    # Filter names are also used in any output file created
    filt_names: [G_BP, G, G_RP]

    # Flags for which of the two AUF simulation algorithms to run
    run_fw_auf: True
    run_psf_auf: False

    # Catalogue PSF parameters
    # Full-width at half maximums for each filter, in order, in arcseconds
    psf_fwhms: [0.12, 0.12, 0.12]

    # AUF region definition - either "rectangle" for NxM evenly spaced grid points, or "points" to define a list of two-point tuple coordinates, separated by a comma
    auf_region_type: rectangle
    # Frame of the coordinates must be specified -- either "equatorial" or "galactic"
    auf_region_frame: equatorial
    # For "points" this should be individually specified (ra, dec) or (l, b) coordinates [as "(a, b), (c, d)"]
    # For "rectangle" this should be 6 numbers: start coordinate, end coordinate, integer number of data points from start to end (inclusive of both start and end), first for ra/l, then for dec/b (depending on auf_region_type), all separated by spaces
    auf_region_points_per_chunk:
      - [131, 134, 4, -1, 1, 3]
    chunk_id_list:
      - 9

    # Local density calculation radius, in degrees
    dens_dist: 0.25

    # Test for whether we need to correct astrometry of catalogue for systematic biases before performing matches
    correct_astrometry: False


cat_b_params.yaml::

    # Catalogue name -- used both for folder creation and output file names
    cat_name: Gaia
    cat_csv_file_path: test_macauff_outputs/name_of_b_folder_{}
    # Folder for all AUF-related files to be created in. Should be an absolute path, or relative to folder script called in.
    auf_folder_path: test_macauff_outputs/cat_b_auf_folder/catalogue_b_{}.csv

    pos_and_err_indices: [0, 1, 2]
    mag_indices: [3, 4, 5, 6]
    chunk_overlap_col: 7
    best_mag_index_col: 8
    csv_has_header: False

    # Filter names are also used in any output file created
    filt_names: [W1, W2]

    # Flags for which of the two AUF simulation algorithms to run
    run_fw_auf: True
    run_psf_auf: False

    # Catalogue PSF parameters
    # Full-width at half maximums for each filter, in order, in arcseconds
    psf_fwhms: [0.12, 0.12, 0.12]

    # AUF region definition - either "rectangle" for NxM evenly spaced grid points, or "points" to define a list of two-point tuple coordinates, separated by a comma
    auf_region_type: rectangle
    # Frame of the coordinates must be specified -- either "equatorial" or "galactic"
    auf_region_frame: equatorial
    # For "points" this should be individually specified (ra, dec) or (l, b) coordinates [as "(a, b), (c, d)"]
    # For "rectangle" this should be 6 numbers: start coordinate, end coordinate, integer number of data points from start to end (inclusive of both start and end), first for ra/l, then for dec/b (depending on auf_region_type), all separated by spaces
    auf_region_points_per_chunk:
      - [0.1, 0.2, 2, 50.15, 50.15, 1]
    chunk_id_list:
      - 9

    # Local density calculation radius, in degrees
    dens_dist: 0.25

    # Test for whether we need to correct astrometry of catalogue for systematic biases before performing matches
    correct_astrometry: False

.. note::
    Discussion of the input parameters available in the catalogue-specific and joint match-specific input files is provided in more detail :doc:`here<inputs>`.

Running the Matches
===================

With both your data and input files, you are now ready to perform your first cross-match! This should be as straightforward as saving the three above text files into a folder within ``test_macauff_inputs`` (e.g. ``match_run``) and, from the same folder as ``test_macauff_inputs`` is located in, running

.. code-block:: python

    if __name__ == '__main__':
        from macauff import CrossMatch
        parameter_file_path = 'test_macauff_inputs'
        cross_match = CrossMatch(path_to_crossmatch_params_file,
                                 path_to_a_params_file, path_to_b_params_file,
                                 use_mpi=False)
        cross_match()

which will save all intermediate match data to the ``output_save_folder`` parameter in ``joint_file_path`` (``test_macauff_outputs/test_path`` if you used the files as given above), and eventually produce a list of indices of matches for the two catalogues. Within Python these can be loaded by calling the original binary files

.. code-block:: python

    import numpy as np
    output_save_folder = 'test_macauff_outputs/test_path'
    # Alternatively, load a saved file depending on e.g.
    # make_output_csv being set to True.
    cat_a_match_inds = cross_match.ac
    cat_b_match_inds = cross_match.bc

    a_matches, b_matches = a_astro[cat_a_match_inds], b_astro[cat_b_match_inds]

You can then, for example, calculate the on-sky separations between these sources

.. code-block:: python

    from macauff.misc_functions_fortran import misc_functions_fortan as mff
    arcsec_seps = np.array([3600 * mff.haversine_wrapper(a_matches[i, 0], b_matches[i, 0],
                            a_matches[i, 1], b_matches[i, 1]) for i in range(len(a_matches))])

..
    Running More Complex Matches
    ============================

    For example cross-matches, including some more advanced features available within ``macauff``, check out the :doc:`Real-World Matching<real_world_matches>` examples.

Documentation
=============

For the full documentation, click :doc:`here<macauff>`.