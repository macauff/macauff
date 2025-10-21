# Licensed under a 3-clause BSD style license - see LICENSE
'''
Module for reading the metadata that macauff requires to set various flags and input
parameters for running cross-matches.
'''

import os

import numpy as np
import yaml
from astropy.time import Time


# pylint: disable=too-many-statements,too-many-branches
def read_metadata(self):
    '''
    Helper function to read in metadata and set various class attributes.

    Returns
    -------
    joint_config : dict
        Dictionary with all loaded parameters needed for the cross-match
        from the ``crossmatch_params_file_path`` file.
    cat_a_config : dict
        Dictionary with all cross-match parameters solely related to the
        first of the two photometric catalogues.
    cat_b_config : dict
        Dictionary with all of catalogue b's metadata parameters.
    '''
    with open(self.crossmatch_params_file_path, encoding='utf-8') as f:
        joint_config = yaml.safe_load(f)
    with open(self.cat_a_params_file_path, encoding='utf-8') as f:
        cat_a_config = yaml.safe_load(f)
    with open(self.cat_b_params_file_path, encoding='utf-8') as f:
        cat_b_config = yaml.safe_load(f)

    joint_config, cat_a_config, cat_b_config = _read_metadata_common(joint_config, cat_a_config, cat_b_config)

    joint_config, cat_a_config, cat_b_config = _read_metadata_perturb_auf(
        joint_config, cat_a_config, cat_b_config)

    if cat_a_config['correct_astrometry'] or cat_b_config['correct_astrometry']:
        joint_config, cat_a_config, cat_b_config = _read_metadata_correct_astro(
            joint_config, cat_a_config, cat_b_config)

    if joint_config["make_output_csv"]:
        joint_config, cat_a_config, cat_b_config = _read_metadata_csv(
            joint_config, cat_a_config, cat_b_config)

    # pylint: disable-next=too-many-boolean-expressions
    if (cat_a_config['apply_proper_motion'] or cat_b_config['apply_proper_motion'] or
            (cat_a_config['correct_astrometry'] and cat_a_config['ref_apply_proper_motion']) or
            (cat_b_config['correct_astrometry'] and cat_b_config['ref_apply_proper_motion'])):
        joint_config, cat_a_config, cat_b_config = _read_metadata_pm(
            joint_config, cat_a_config, cat_b_config)

    return joint_config, cat_a_config, cat_b_config


def _read_metadata_common(joint_config, cat_a_config, cat_b_config):
    """
    Read metadata from input config files that are common to all match runs.

        Parameters
    ----------
    joint_config : dict
        The pre-loaded set of input configuration parameters as related
        to the joint match between catalogues a and b.
    cat_a_config : dict
        Configuration parameters that are solely related to catalogue "a".
    cat_b_config : dict
        Configuration parameters that are just relate to catalogue "b".

    Returns
    -------
    joint_config : dict
        Dictionary with all loaded parameters needed for the cross-match
        from the ``crossmatch_params_file_path`` file.
    cat_a_config : dict
        Dictionary with all cross-match parameters solely related to the
        first of the two photometric catalogues.
    cat_b_config : dict
        Dictionary with all of catalogue b's metadata parameters.
    """
    for check_flag in ['include_perturb_auf', 'include_phot_like', 'use_phot_priors',
                       'cf_region_type', 'cf_region_frame', 'cf_region_points_per_chunk',
                       'output_save_folder', 'pos_corr_dist', 'real_hankel_points', 'chunk_id_list',
                       'four_hankel_points', 'four_max_rho', 'int_fracs', 'make_output_csv', 'n_pool']:
        if check_flag not in joint_config:
            raise ValueError(f"Missing key {check_flag} from joint metadata file.")

    for config, catname in zip([cat_a_config, cat_b_config], ['"a"', '"b"']):
        for check_flag in ['auf_region_type', 'auf_region_frame', 'auf_region_points_per_chunk',
                           'filt_names', 'cat_name', 'auf_file_path', 'cat_csv_file_path',
                           'correct_astrometry', 'chunk_id_list', 'pos_and_err_indices', 'mag_indices',
                           'chunk_overlap_col', 'best_mag_index_col', 'csv_has_header',
                           'apply_proper_motion']:
            if check_flag not in config:
                raise ValueError(f"Missing key {check_flag} from catalogue {catname} metadata file.")

    for config, flag, correct_astro in zip(
            [cat_a_config, cat_b_config], ['a_', 'b_'], [cat_a_config['correct_astrometry'],
                                                         cat_b_config['correct_astrometry']]):
        a = config['mag_indices']
        try:
            b = np.array([float(f) for f in a])
        except ValueError as exc:
            raise ValueError('mag_indices should be a list of integers '
                             f'in the catalogue "{flag[0]}" metadata file') from exc
        if len(b) != len(config['filt_names']):
            raise ValueError(f'{flag}filt_names and {flag}mag_indices should contain the '
                             'same number of entries.')
        if not np.all([c.is_integer() for c in b]):
            raise ValueError(f'All elements of {flag}mag_indices should be '
                             'integers.')

        # pos_and_err_indices should be a three-, six- or N-integer list
        # that we then transform into [this_cat_inds, reference_cat_inds]
        # where reference_cat_inds is a three-element list [x, y, z],
        # necessary when correct_astrometry is set, and this_cat_inds,
        # depending on whether use_photometric_uncertainties is set, is
        # either a three- or N-element (N>=3) list.
        a = config['pos_and_err_indices']
        try:
            b = np.array([float(f) for f in a])
        except ValueError as exc:
            raise ValueError('pos_and_err_indices should be a list of integers '
                             f'in the catalogue "{flag[0]}"" metadata file') from exc
        if len(b) != 3 and not correct_astro:
            raise ValueError(f'{flag}pos_and_err_indices should contain three elements '
                             'when correct_astrometry is False.')
        if not np.all([c.is_integer() for c in b]):
            raise ValueError(f'All elements of {flag}pos_and_err_indices should be integers.')

        a = config['best_mag_index_col']
        try:
            a = float(a)
        except (ValueError, TypeError) as exc:
            raise ValueError(f'best_mag_index_col should be an integer in the catalogue "{flag[0]}" '
                             'metadata file.') from exc
        if not a.is_integer():
            raise ValueError(f'best_mag_index_col should be an integer in the catalogue "{flag[0]}" '
                             'metadata file.')

        a = config['chunk_overlap_col']
        if a == "None":
            config['chunk_overlap_col'] = None
        else:
            try:
                a = float(a)
            except (ValueError, TypeError) as exc:
                raise ValueError('chunk_overlap_col should be an integer in the '
                                 f'catalogue "{flag[0]}" metadata file.') from exc
            if not a.is_integer():
                raise ValueError('chunk_overlap_col should be an integer in the '
                                 f'catalogue "{flag[0]}" metadata file.')

        if config['csv_has_header'] not in (True, False):
            raise ValueError('Boolean flag key csv_has_header not set to allowed value in catalogue '
                             f'{flag[0]} metadata file.')

    for run_flag in ['include_perturb_auf', 'include_phot_like', 'use_phot_priors']:
        if joint_config[run_flag] not in (True, False):
            raise ValueError(f'Boolean flag key {run_flag} not set to allowed value in joint metadata '
                             'file.')

    if joint_config['include_phot_like']:
        if "with_and_without_photometry" not in joint_config:
            raise ValueError("Missing key with_and_without_photometry from joint metadata file.")
        if joint_config["with_and_without_photometry"] not in (True, False):
            raise ValueError('Boolean flag key with_and_without_photometry not set to allowed value '
                             'in joint metadata file.')

    for region_frame, x, y in zip([joint_config['cf_region_frame'], cat_a_config['auf_region_frame'],
                                   cat_b_config['auf_region_frame']],
                                  ['cf_region_frame', 'auf_region_frame', 'auf_region_frame'],
                                  ['joint', 'catalogue a', 'catalogue b']):
        if isinstance(region_frame, str):
            rf = region_frame.lower()
            if rf not in ('equatorial', 'galactic'):
                raise ValueError(f"{x} should either be 'equatorial' or 'galactic' in the {y} "
                                 "metadata file.")
        else:
            raise ValueError(f"{x} should either be 'equatorial' or 'galactic' in the {y} "
                             "metadata file.")

    for region_type, x, y in zip([joint_config['cf_region_type'], cat_a_config['auf_region_type'],
                                  cat_b_config['auf_region_type']],
                                 ['cf_region_type', 'auf_region_type', 'auf_region_type'],
                                 ['joint', 'catalogue a', 'catalogue b']):
        if isinstance(region_type, str):
            rf = region_type.lower()
            if rf not in ('rectangle', 'points'):
                raise ValueError(f"{x} should either be 'rectangle' or 'points' in the {y} "
                                 "metadata file.")
        else:
            raise ValueError(f"{x} should either be 'rectangle' or 'points' in the {y} "
                             "metadata file.")

    # If the frame of the two AUF parameter files and the 'cf' frame are
    # not all the same then we have to raise an error.
    if (cat_a_config['auf_region_frame'] != cat_b_config['auf_region_frame'] or
            cat_a_config['auf_region_frame'] != joint_config['cf_region_frame']):
        raise ValueError("Region frames for c/f and AUF creation must all be the same.")

    joint_config['output_save_folder'] = os.path.abspath(joint_config['output_save_folder'])

    if cat_a_config['auf_file_path'] == "None":
        cat_a_config['auf_file_path'] = None
    else:
        cat_a_config['auf_file_path'] = os.path.abspath(cat_a_config['auf_file_path'])
    if cat_b_config['auf_file_path'] == "None":
        cat_b_config['auf_file_path'] = None
    else:
        cat_b_config['auf_file_path'] = os.path.abspath(cat_b_config['auf_file_path'])

    cat_a_config['cat_csv_file_path'] = os.path.abspath(cat_a_config['cat_csv_file_path'])
    cat_b_config['cat_csv_file_path'] = os.path.abspath(cat_b_config['cat_csv_file_path'])

    try:
        a = float(joint_config['pos_corr_dist'])
    except (ValueError, TypeError) as exc:
        raise ValueError("pos_corr_dist must be a float.") from exc

    for flag in ['real_hankel_points', 'four_hankel_points', 'four_max_rho']:
        a = joint_config[flag]
        try:
            a = float(a)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{flag} should be an integer.") from exc
        if not a.is_integer():
            raise ValueError(f"{flag} should be an integer.")

    a = joint_config['int_fracs']
    try:
        b = np.array([float(f) for f in a])
    except ValueError as exc:
        raise ValueError("All elements of int_fracs should be floats.") from exc
    if len(b) != 3:
        raise ValueError("int_fracs should contain three elements.")
    joint_config['int_fracs'] = np.array(joint_config['int_fracs'])

    if joint_config["make_output_csv"] not in (True, False):
        raise ValueError('Boolean flag key make_output_csv not set to allowed value '
                         'in joint metadata file.')

    # Load the multiprocessing Pool count.
    try:
        a = joint_config['n_pool']
        if float(a).is_integer():
            a = int(a)
        else:
            raise ValueError("n_pool should be a single integer number.")
    except (ValueError, TypeError) as exc:
        raise ValueError("n_pool should be a single integer number.") from exc

    for n, config in zip(['a', 'b'], [cat_a_config, cat_b_config]):
        if config['correct_astrometry'] not in (True, False):
            raise ValueError(f"Boolean key correct_astrometry not set to allowed value in catalogue {n} "
                             "metadata file.")

    return joint_config, cat_a_config, cat_b_config


def _read_metadata_perturb_auf(joint_config, cat_a_config, cat_b_config):
    """
    Read metadata from input config files relating to the simulation of position
    shifts due to unresolved, contaminant sources inside detections' PSFs.

    Parameters
    ----------
    joint_config : dict
        The pre-loaded set of input configuration parameters as related
        to the joint match between catalogues a and b.
    cat_a_config : dict
        Configuration parameters that are solely related to catalogue "a".
    cat_b_config : dict
        Configuration parameters that are just relate to catalogue "b".

    Returns
    -------
    joint_config : dict
        Dictionary with all loaded parameters needed for the cross-match
        from the ``crossmatch_params_file_path`` file.
    cat_a_config : dict
        Dictionary with all cross-match parameters solely related to the
        first of the two photometric catalogues.
    cat_b_config : dict
        Dictionary with all of catalogue b's metadata parameters.
    """
    # Only have to check for the existence of Pertubation AUF-related
    # parameters if we are using the perturbation AUF component.
    # However, calling AstrometricCorrections in its current form confuses
    # this, since it always uses the perturbation AUF component. We therefore
    # split out the items that are NOT required for AstrometricCorrections
    # first, if there are any.
    if (joint_config['include_perturb_auf'] or cat_a_config['correct_astrometry'] or
            cat_b_config['correct_astrometry']):
        for check_flag in ['num_trials', 'd_mag']:
            if check_flag not in joint_config:
                raise ValueError(f"Missing key {check_flag} from joint metadata file.")

        a = joint_config['num_trials']
        try:
            a = float(a)
        except (ValueError, TypeError) as exc:
            raise ValueError("num_trials should be an integer.") from exc
        if not a.is_integer():
            raise ValueError("num_trials should be an integer.")

        for flag in ['d_mag']:
            try:
                a = float(joint_config[flag])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{flag} must be a float.") from exc

    # Nominally these are all of the parameters required if include_perturb_auf
    # is True, but for the minute they're also forced if correct_astrometry
    # is also True instead of those aligning, so we bypass the if statement
    # if self.a_correct_astrometry or self.b_correct_astrometry respectively
    # have been set.
    # pylint: disable-next=too-many-nested-blocks
    for correct_astro, config, catname, flag in zip(
            [cat_a_config['correct_astrometry'], cat_b_config['correct_astrometry']],
            [cat_a_config, cat_b_config], ['"a"', '"b"'], ['a_', 'b_']):
        if joint_config['include_perturb_auf'] or correct_astro:
            for check_flag in ['dens_dist']:
                if check_flag not in config:
                    raise ValueError(f"Missing key {check_flag} from catalogue {catname} metadata file.")
            try:
                a = float(config['dens_dist'])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"dens_dist in catalogue {catname} must be a float.") from exc

        if joint_config['include_perturb_auf']:
            if 'fit_gal_flag' not in config:
                raise ValueError(f"Missing key fit_gal_flag from catalogue {catname} metadata file.")
            if config['fit_gal_flag'] not in (True, False):
                raise ValueError("Boolean key fit_gal_flag not set to allowed value in catalogue "
                                 f"{catname} metadata file.")

        if joint_config['include_perturb_auf'] or correct_astro:
            for check_flag in ['snr_indices', 'tri_set_name', 'tri_filt_names', 'tri_filt_num',
                               'download_tri', 'psf_fwhms', 'run_fw_auf', 'run_psf_auf',
                               'tri_maglim_faint', 'tri_num_faint', 'gal_al_avs',
                               'tri_dens_cube_location', 'tri_dens_array_location']:
                if check_flag not in config:
                    raise ValueError(f"Missing key {check_flag} from catalogue {catname} metadata file.")

            # snr_indices is also needed in correct_astrometry's fork, but for now
            # those are one and the same with the OR statement above.
            a = config['snr_indices']
            try:
                b = np.array([float(f) for f in a])
            except ValueError as exc:
                raise ValueError('snr_indices should be a list of integers '
                                 f'in catalogue {catname} metadata file') from exc
            if len(b) != len(config['mag_indices']):
                raise ValueError(f'{flag}snr_indices and {flag}mag_indices should contain the same '
                                 'number of entries.')
            if not np.all([c.is_integer() for c in b]):
                raise ValueError(f'All elements of {flag}snr_indices should be integers.')

            # Set as a list of floats
            for var in ['gal_al_avs']:
                try:
                    b = np.array([float(f) for f in config[var]])
                except ValueError as exc:
                    raise ValueError(f'{var} should be a list of floats in catalogue '
                                     f'{catname} metadata file') from exc
                if len(b) != len(config['filt_names']):
                    raise ValueError(f'{flag}{var} and {flag}filt_names should contain the same '
                                     'number of entries.')

            if config['download_tri'] == "None":
                config['download_tri'] = None
            else:
                if config['download_tri'] not in (True, False):
                    raise ValueError("Boolean key download_tri not set to allowed value in catalogue "
                                     f"{catname} metadata file.")

            if config['tri_set_name'] == "None":
                config['tri_set_name'] = None
            if config['tri_filt_names'] == "None":
                config['tri_filt_names'] = [None] * len(config['filt_names'])
            else:
                if len(config['tri_filt_names']) != len(config['filt_names']):
                    raise ValueError(f'{flag}tri_filt_names and {flag}filt_names should contain the '
                                     'same number of entries.')

            a = config['psf_fwhms']
            try:
                b = np.array([float(f) for f in a])
            except ValueError as exc:
                raise ValueError(f'psf_fwhms should be a list of floats in catalogue {catname} '
                                 'metadata file.') from exc
            if len(b) != len(config['filt_names']):
                raise ValueError(f'{flag}psf_fwhms and {flag}filt_names should contain the '
                                 'same number of entries.')

            for auf_run_flag in ['run_fw_auf', 'run_psf_auf']:
                if config[auf_run_flag] not in (True, False):
                    raise ValueError(f"Boolean key {auf_run_flag} not set to allowed value in catalogue "
                                     f"{catname} metadata file.")

            try:
                a = config['tri_filt_num']
                if a == "None":
                    config['tri_filt_num'] = None
                else:
                    if not float(a).is_integer():
                        raise ValueError("tri_filt_num should be a single integer number in "
                                         f"catalogue {catname} metadata file, or None.")
            except (ValueError, TypeError) as exc:
                raise ValueError("tri_filt_num should be a single integer number in "
                                 f"catalogue {catname} metadata file, or None.") from exc

            for suffix in ['_faint']:
                try:
                    a = config[f'tri_num{suffix}']
                    if a == "None":
                        config[f'tri_num{suffix}'] = None
                    else:
                        if not float(a).is_integer():
                            raise ValueError(f"tri_num{suffix} should be a single integer number in "
                                             f"catalogue {catname} metadata file, or None.")
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"tri_num{suffix} should be a single integer number in "
                                     f"catalogue {catname} metadata file, or None.") from exc

                try:
                    if config[f'tri_maglim{suffix}'] == "None":
                        config[f'tri_maglim{suffix}'] = None
                    else:
                        a = float(config[f'tri_maglim{suffix}'])
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"tri_maglim{suffix} in catalogue {catname} must be a "
                                     "float, or None.") from exc

            # Assume that we input filenames, including full location, for each
            # pre-computed TRILEGAL histogram file, and that they are all shape
            # (len(filters), ...).
            for name in ['tri_dens_cube', 'tri_dens_array']:
                f = config[f'{name}_location']
                if f == "None":
                    config[name] = None
                else:
                    if not os.path.isfile(f):
                        raise FileNotFoundError(f"File not found for {name}. Please verify "
                                                "the input location on disk.")
                    try:
                        g = np.load(f)
                    except Exception as exc:
                        raise ValueError(f"File could not be loaded from {name}.") from exc
                    if name == "tri_dens_cube":
                        shape_dht = g.shape
                        if g.shape[1] != len(config['filt_names']):
                            raise ValueError(f"The number of filters in {flag}filt_names and "
                                             f"{flag}tri_dens_cube do not match.")
                    else:
                        if g.shape[0] != shape_dht[0]:
                            raise ValueError("The number of sky-elements in tri_dens_cube "
                                             f"and {name} do not match.")
            # Check for inter- and intra-TRILEGAL parameter compatibility.
            # If any one parameter from option A or B is None, they all
            # should be; and if any (all) options from A are None, zero
            # options from B should be None, and vice versa.
            run_internal_none_flag = [config[name] is None for name in
                                      ['auf_file_path', 'tri_set_name', 'tri_maglim_faint',
                                      'tri_num_faint', 'download_tri', 'tri_filt_num']]
            run_internal_none_flag.append(np.all([b is None for b in config['tri_filt_names']]))
            if not (np.sum(run_internal_none_flag) == 0 or
                    np.sum(run_internal_none_flag) == len(run_internal_none_flag)):
                raise ValueError("Either all flags related to running TRILEGAL histogram generation "
                                 f"within the catalogue {catname} cross-match call -- tri_filt_names, "
                                 "tri_set_name, etc. -- should be None or zero of them should be None.")
            run_external_none_flag = [config[name] == "None" for name in
                                      ['tri_dens_cube_location', 'tri_dens_array_location']]
            if not (np.sum(run_external_none_flag) == 0 or
                    np.sum(run_external_none_flag) == len(run_external_none_flag)):
                raise ValueError("Either all flags related to running TRILEGAL histogram generation "
                                 f"externally to the catalogue {catname} cross-match call -- "
                                 "tri_dens_cube and tri_dens_array -- should be None or zero of "
                                 "them should be None.")
            if ((np.sum(run_internal_none_flag) == 0 and np.sum(run_external_none_flag) == 0) or
                (np.sum(run_internal_none_flag) == len(run_internal_none_flag) and
                 np.sum(run_external_none_flag) == len(run_external_none_flag))):
                raise ValueError("Ambiguity in whether TRILEGAL histogram generation is being run "
                                 f"within or prior to cross-match run in catalogue {catname}. Please "
                                 "flag one set of parameters as None and only pass the other set "
                                 "into CrossMatch.")

            if correct_astro or config['fit_gal_flag']:
                for check_flag in ['gal_wavs', 'gal_zmax', 'gal_nzs',
                                   'gal_aboffsets', 'gal_filternames', 'saturation_magnitudes']:
                    if check_flag not in config:
                        raise ValueError(f"Missing key {check_flag} from catalogue {catname} "
                                         "metadata file.")
                # Set all lists of floats
                for var in ['gal_wavs', 'gal_zmax', 'gal_aboffsets', 'saturation_magnitudes']:
                    try:
                        b = np.array([float(f) for f in config[var]])
                    except ValueError as exc:
                        raise ValueError(f'{var} should be a list of floats in catalogue '
                                         f'{catname} metadata file') from exc
                    if len(b) != len(config['filt_names']):
                        raise ValueError(f'{flag}{var} and {flag}filt_names should contain the same '
                                         'number of entries.')
                # galaxy_nzs should be a list of integers.
                a = config['gal_nzs']
                try:
                    b = np.array([float(f) for f in a])
                except ValueError as exc:
                    raise ValueError('gal_nzs should be a list of integers '
                                     f'in catalogue {catname} metadata file') from exc
                if len(b) != len(config['filt_names']):
                    raise ValueError(f'{flag}gal_nzs and {flag}filt_names should contain the same '
                                     'number of entries.')
                if not np.all([c.is_integer() for c in b]):
                    raise ValueError(f'All elements of {flag}gal_nzs should be integers.')
                # Filter names are simple lists of strings
                b = config['gal_filternames']
                if len(b) != len(config['filt_names']):
                    raise ValueError(f'{flag}gal_filternames and {flag}filt_names should contain the '
                                     'same number of entries.')

    return joint_config, cat_a_config, cat_b_config


def _read_metadata_correct_astro(joint_config, cat_a_config, cat_b_config):
    """
    Read metadata from input config files relating to the correction of astrometric
    precisions by the use of standard deviations of truth-measured residuals.

    Parameters
    ----------
    joint_config : dict
        The pre-loaded set of input configuration parameters as related
        to the joint match between catalogues a and b.
    cat_a_config : dict
        Configuration parameters that are solely related to catalogue "a".
    cat_b_config : dict
        Configuration parameters that are just relate to catalogue "b".

    Returns
    -------
    joint_config : dict
        Dictionary with all loaded parameters needed for the cross-match
        from the ``crossmatch_params_file_path`` file.
    cat_a_config : dict
        Dictionary with all cross-match parameters solely related to the
        first of the two photometric catalogues.
    cat_b_config : dict
        Dictionary with all of catalogue b's metadata parameters.
    """
    for correct_astro, config, catname, flag in zip(
            [cat_a_config['correct_astrometry'], cat_b_config['correct_astrometry']],
            [cat_a_config, cat_b_config], ['"a"', '"b"'], ['a_', 'b_']):
        if correct_astro:
            if 'ref_apply_proper_motion' not in config:
                raise ValueError(f'Missing key ref_apply_proper_motion from catalogue "{flag[0]}"" '
                                 'metadata file.')
        if correct_astro and config['ref_apply_proper_motion']:
            for check_flag in ['ref_pm_indices', 'ref_ref_epoch_or_index']:
                if check_flag not in config:
                    raise ValueError(f'Missing key {check_flag} from catalogue "{flag[0]}"" '
                                     'metadata file.')
        if correct_astro:
            # If this particular catalogue requires a systematic correction
            # for astrometric biases from ensemble match distributions before
            # we can do a probability-based cross-match, then load some extra
            # pieces of information.
            for check_flag in ['correct_astro_save_folder', 'correct_astro_mag_indices_index',
                               'nn_radius', 'ref_cat_csv_file_path', 'correct_mag_array',
                               'correct_mag_slice', 'correct_sig_slice', 'use_photometric_uncertainties',
                               'mn_fit_type', 'seeing_ranges']:
                if check_flag not in config:
                    raise ValueError(f"Missing key {check_flag} from catalogue {catname} metadata file.")

            if config['use_photometric_uncertainties'] not in (True, False):
                raise ValueError('Boolean flag key use_photometric_uncertainties not set to allowed '
                                 f'value in catalogue {catname} metadata file.')

            config['correct_astro_save_folder'] = os.path.abspath(config['correct_astro_save_folder'])

            mn_fit_type = config['mn_fit_type']
            if mn_fit_type not in ['quadratic', 'linear']:
                raise ValueError(f"mn_fit_type must be 'quadratic' or 'linear' in catalogue {catname} "
                                 "metadata file.")

            # Since make_plots is always True, we always need seeing_ranges.
            a = config['seeing_ranges']
            try:
                b = np.array([float(f) for f in a])
                if len(b.shape) != 1 or len(b) not in [1, 2, 3]:
                    raise ValueError("seeing_ranges must be a 1-D list or array of ints, length 1, 2, or "
                                     f"3 {catname} metadata file.")
            except ValueError as exc:
                raise ValueError("seeing_ranges must be a 1-D list or array of ints, length 1, 2, or "
                                 f"3 in catalogue {catname} metadata file.") from exc

            # AstrometricCorrections takes both single_or_repeat and
            # repeat_unique_visits_list, but since you can't do a
            # cross-match on multiple observations of the same objects
            # at once, we assume that time-series is outside of the loop
            # and therefore need to pass neither parameter through to
            # the fitting routine.

            a = config['correct_astro_mag_indices_index']
            try:
                a = float(a)
            except (ValueError, TypeError) as exc:
                raise ValueError("correct_astro_mag_indices_index should be an integer in the catalogue "
                                 f"{catname} metadata file.") from exc
            if not a.is_integer():
                raise ValueError("correct_astro_mag_indices_index should be an integer in the catalogue "
                                 f"{catname} metadata file.")
            if int(a) >= len(config['filt_names']):
                raise ValueError("correct_astro_mag_indices_index cannot be a larger index than the list "
                                 f"of filters in the catalogue {catname} metadata file.")

            try:
                a = float(config['nn_radius'])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"nn_radius must be a float in the catalogue {catname} metadata "
                                 "file.") from exc

            config['ref_cat_csv_file_path'] = os.path.abspath(config['ref_cat_csv_file_path'])

            a = config['correct_mag_array']
            try:
                b = np.array([float(f) for mid_list in a for f in mid_list])
            except (ValueError, TypeError) as exc:
                raise ValueError('correct_mag_array should be a list of list of floats in the '
                                 f'catalogue {catname} metadata file.') from exc

            a = config['correct_mag_slice']
            try:
                b = np.array([float(f) for mid_list in a for f in mid_list])
            except ValueError as exc:
                raise ValueError('correct_mag_slice should be a list of list of floats in the '
                                 f'catalogue {catname} metadata file.') from exc
            if (len(a) != len(config['correct_mag_array']) or
                    np.any([len(a[i]) != len(config['correct_mag_array'][i]) for i in range(len(a))])):
                raise ValueError(f'{flag}correct_mag_array and {flag}correct_mag_slice should contain '
                                 'the same number of entries.')

            a = config['correct_sig_slice']
            try:
                b = np.array([float(f) for mid_list in a for f in mid_list])
            except ValueError as exc:
                raise ValueError('correct_sig_slice should be a list of list of floats in the '
                                 f'catalogue {catname} metadata file.') from exc
            if (len(a) != len(config['correct_mag_array']) or
                    np.any([len(a[i]) != len(config['correct_mag_array'][i]) for i in range(len(a))])):
                raise ValueError(f'{flag}correct_mag_array and {flag}correct_sig_slice should contain '
                                 'the same number of entries.')

            # Check for the remaining permutations of pos_and_err_indices
            # within the inner correct_astrometry check.
            a = config['pos_and_err_indices']
            b = np.array([float(f) for f in a])
            if len(b) != 6 and not config['use_photometric_uncertainties']:
                raise ValueError(f'{flag}pos_and_err_indices should contain six elements '
                                 'when correct_astrometry is True and use_photometric_uncertainties is '
                                 'False.')
            if len(b) < 6 and config['use_photometric_uncertainties']:
                raise ValueError(f'{flag}pos_and_err_indices should contain at least six elements '
                                 'when correct_astrometry is True and use_photometric_uncertainties is '
                                 'True.')
            # Three elements of pos_and_err_indices are taken up by the reference
            # catalogue's indices, and two are for the to-be-fit catalogue's
            # positions, but the remaining indices must match the lists of
            # magnitude-related elements.
            if len(b) - 5 != len(config['filt_names']) and config['use_photometric_uncertainties']:
                raise ValueError(f'{flag}pos_and_err_indices should contain the same number of '
                                 'non-reference, non-position, magnitude-uncertainty columns as there '
                                 f'are elements in {flag}filt_names when correct_astrometry is True and '
                                 'use_photometric_uncertainties is True.')

    return joint_config, cat_a_config, cat_b_config


def _read_metadata_csv(joint_config, cat_a_config, cat_b_config):
    """
    Read metadata from input config files relating to outputting combined
    .csv files during the post-process step, if requested.

    Parameters
    ----------
    joint_config : dict
        The pre-loaded set of input configuration parameters as related
        to the joint match between catalogues a and b.
    cat_a_config : dict
        Configuration parameters that are solely related to catalogue "a".
    cat_b_config : dict
        Configuration parameters that are just relate to catalogue "b".

    Returns
    -------
    joint_config : dict
        Dictionary with all loaded parameters needed for the cross-match
        from the ``crossmatch_params_file_path`` file.
    cat_a_config : dict
        Dictionary with all cross-match parameters solely related to the
        first of the two photometric catalogues.
    cat_b_config : dict
        Dictionary with all of catalogue b's metadata parameters.
    """

    for check_flag in ['match_out_csv_name', 'nonmatch_out_csv_name']:
        if check_flag not in joint_config:
            raise ValueError(f"Missing key {check_flag} from joint metadata file.")

    for config, catname in zip([cat_a_config, cat_b_config], ['"a"', '"b"']):
        for check_flag in ['cat_col_names', 'cat_col_nums', 'extra_col_names', 'extra_col_nums']:
            if check_flag not in config:
                raise ValueError(f"Missing key {check_flag} from catalogue {catname} metadata file.")

    for config, catname in zip([cat_a_config, cat_b_config], ['a_', 'b_']):
        # Non-match csv name should be of the format
        # [cat name]_[some indication this is a non-match], but note that
        # this is defined in joint_config, not each individual
        # catalogue config!
        nonmatch_out_name = joint_config['nonmatch_out_csv_name']
        config['nonmatch_out_csv_name'] = f'{config["cat_name"]}_{nonmatch_out_name}'

        # cat_col_names is simply a list/array of strings. However, to
        # avoid any issues with generic names like "RA" being added to the
        # output .csv file twice, we prepend the catalogue name to the front
        # of them all.
        config['cat_col_names'] = np.array([f'{config["cat_name"]}_{q}' for q in config['cat_col_names']])

        # But cat_col_nums is a list/array of integers, and should be of the
        # same length as cat_col_names.
        try:
            b = np.array([float(f) for f in config['cat_col_nums']])
        except ValueError as exc:
            raise ValueError('cat_col_nums should be a list of integers '
                             f'in catalogue "{catname[0]}" metadata file') from exc
        if len(b) != len(config['cat_col_names']):
            raise ValueError(f'{catname}cat_col_names and {catname}cat_col_nums should contain the same '
                             'number of entries.')
        if not np.all([c.is_integer() for c in b]):
            raise ValueError(f'All elements of {catname}cat_col_nums should be '
                             'integers.')

        # As above, extra_col_names is just strings but extra_col_names
        # is a list of integers.
        # However, both can be None (although if either is None both have to
        # be None), so check for that first
        a = config['extra_col_names']
        b = config['extra_col_nums']
        if a == 'None' and b == 'None':
            config['extra_col_names'] = None
            config['extra_col_nums'] = None
        else:
            if a == 'None' and b != 'None' or a != 'None' and b == 'None':
                raise ValueError('Both extra_col_names and extra_col_nums must be None if '
                                 f'either is None in catalogue "{catname[0]}".')
            config['extra_col_names'] = np.array([f'{config["cat_name"]}_{q}' for q in
                                                  config['extra_col_names']])
            a = config['extra_col_nums']
            try:
                b = np.array([float(f) for f in a])
            except ValueError as exc:
                raise ValueError('extra_col_nums should be a list of integers '
                                 f'in catalogue "{catname[0]}" metadata file') from exc
            if len(b) != len(config['extra_col_names']):
                raise ValueError(f'{catname}extra_col_names and {catname}extra_col_nums should '
                                 'contain the same number of entries.')
            if not np.all([c.is_integer() for c in b]):
                raise ValueError(f'All elements of {catname}extra_col_nums should be integers.')

    return joint_config, cat_a_config, cat_b_config


def _read_metadata_pm(joint_config, cat_a_config, cat_b_config):
    """
    Read metadata from input config files relating to the handling of
    proper motions, if applied to any catalogues involved in the cross-match.

    Parameters
    ----------
    joint_config : dict
        The pre-loaded set of input configuration parameters as related
        to the joint match between catalogues a and b.
    cat_a_config : dict
        Configuration parameters that are solely related to catalogue "a".
    cat_b_config : dict
        Configuration parameters that are just relate to catalogue "b".

    Returns
    -------
    joint_config : dict
        Dictionary with all loaded parameters needed for the cross-match
        from the ``crossmatch_params_file_path`` file.
    cat_a_config : dict
        Dictionary with all cross-match parameters solely related to the
        first of the two photometric catalogues.
    cat_b_config : dict
        Dictionary with all of catalogue b's metadata parameters.
    """
    for config, flag, apply_pm in zip(
            [cat_a_config, cat_b_config], ['a_', 'b_'],
            [cat_a_config['apply_proper_motion'], cat_b_config['apply_proper_motion']]):
        if apply_pm:
            for check_flag in ['pm_indices', 'ref_epoch_or_index']:
                if check_flag not in config:
                    raise ValueError(f'Missing key {check_flag} from catalogue "{flag[0]}"" '
                                     'metadata file.')

            a = config['pm_indices']
            try:
                b = np.array([float(f) for f in a])
            except (ValueError, TypeError) as exc:
                raise ValueError('pm_indices should be a list of integers '
                                 f'in the catalogue "{flag[0]}" metadata file') from exc
            if len(b) != 2:
                raise ValueError(f'{flag}pm_indices should contain two entries.')
            if not np.all([c.is_integer() for c in b]):
                raise ValueError(f'All elements of {flag}pm_indices should be integers.')

            a = config['ref_epoch_or_index']
            if isinstance(config['ref_epoch_or_index'], str):
                try:
                    Time(a)
                except ValueError as exc:
                    raise ValueError(f"{flag}ref_epoch_or_index, if given as a constant string input, "
                                     "must be a string that astropy's Time function accepts, such as "
                                     "JYYYY or YYYY-MM-DD.") from exc
            else:
                try:
                    a = float(a)
                except (ValueError, TypeError) as exc:
                    raise ValueError('ref_epoch_or_index, if indicating a column index, should be an '
                                     f'integer in the catalogue "{flag[0]}" metadata file.') from exc
                if not a.is_integer():
                    raise ValueError('ref_epoch_or_index, if indicating a column index, should be an '
                                     f'integer in the catalogue "{flag[0]}" metadata file.')

    for config, flag, correct_astro in zip(
            [cat_a_config, cat_b_config], ['a_', 'b_'],
            [cat_a_config['correct_astrometry'], cat_b_config['correct_astrometry']]):
        if correct_astro and config['ref_apply_proper_motion']:
            a = config['ref_pm_indices']
            try:
                b = np.array([float(f) for f in a])
            except (ValueError, TypeError) as exc:
                raise ValueError('ref_pm_indices should be a list of integers '
                                 f'in the catalogue "{flag[0]}" metadata file') from exc
            if len(b) != 2:
                raise ValueError(f'{flag}ref_pm_indices should contain two entries.')
            if not np.all([c.is_integer() for c in b]):
                raise ValueError(f'All elements of {flag}ref_pm_indices should be integers.')

            a = config['ref_ref_epoch_or_index']
            if isinstance(config['ref_ref_epoch_or_index'], str):
                try:
                    Time(a)
                except ValueError as exc:
                    raise ValueError(f"{flag}ref_ref_epoch_or_index, if given as a constant string "
                                     "input, must be a string that astropy's Time function accepts, "
                                     "such as JYYYY or YYYY-MM-DD.") from exc
            else:
                try:
                    a = float(a)
                except (ValueError, TypeError) as exc:
                    raise ValueError('ref_ref_epoch_or_index, if indicating a column index, should be an '
                                     f'integer in the catalogue "{flag[0]}" metadata file.') from exc
                if not a.is_integer():
                    raise ValueError('ref_ref_epoch_or_index, if indicating a column index, should be an '
                                     f'integer in the catalogue "{flag[0]}" metadata file.')

    # Always need to check for a/b catalogue PM application, but only check
    # for their respective reference catalogues' PM application if they
    # get a correct_astrometry call at all.
    # pylint: disable-next=too-many-boolean-expressions
    if (cat_a_config['apply_proper_motion'] or cat_b_config['apply_proper_motion'] or
            (cat_a_config['correct_astrometry'] and cat_a_config['ref_apply_proper_motion']) or
            (cat_b_config['correct_astrometry'] and cat_b_config['ref_apply_proper_motion'])):
        if 'move_to_epoch' not in joint_config:
            raise ValueError("Missing key move_to_epoch from joint metadata file.")

        a = joint_config['move_to_epoch']
        try:
            Time(a)
        except ValueError as exc:
            raise ValueError("move_to_epoch must be a string that astropy's Time "
                             "function accepts, such as JYYYY or YYYY-MM-DD.") from exc

    return joint_config, cat_a_config, cat_b_config
