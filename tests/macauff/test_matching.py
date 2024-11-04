# Licensed under a 3-clause BSD style license - see LICENSE
'''
Tests for the "matching" module.
'''

# pylint: disable=too-many-lines,duplicate-code

import os
from configparser import ConfigParser

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from test_fit_astrometry import TestAstroCorrection as TAC

# pylint: disable=import-error,no-name-in-module
from macauff.macauff import Macauff
from macauff.matching import CrossMatch
from macauff.perturbation_auf import make_tri_counts

# pylint: enable=import-error,no-name-in-module


def _replace_line(file_name, line_num, text, out_file=None):
    '''
    Helper function to update the metadata file on-the-fly, allowing for
    "run" flags to be set from run to no run once they have finished.

    Parameters
    ----------
    file_name : string
        Name of the file to read in and change lines of.
    line_num : integer
        Line number of line to edit in ``file_name``.
    text : string
        New line to replace original line in ``file_name`` with.
    out_file : string, optional
        Name of the file to save new, edited version of ``file_name`` to.
        If ``None`` then ``file_name`` is overwritten.
    '''
    if out_file is None:
        out_file = file_name
    with open(file_name, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    lines[line_num] = text
    with open(out_file, 'w', encoding='utf-8') as out:
        out.writelines(lines)


class TestInputs:
    def setup_class(self):
        os.makedirs('a_cat', exist_ok=True)
        os.makedirs('b_cat', exist_ok=True)
        joint_config = ConfigParser()
        with open(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                  encoding='utf-8') as f:
            joint_config.read_string('[config]\n' + f.read())
        joint_config = joint_config['config']
        cat_a_config = ConfigParser()
        with open(os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'), encoding='utf-8') as f:
            cat_a_config.read_string('[config]\n' + f.read())
        cat_a_config = cat_a_config['config']
        cat_b_config = ConfigParser()
        with open(os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'), encoding='utf-8') as f:
            cat_b_config.read_string('[config]\n' + f.read())
        cat_b_config = cat_b_config['config']
        self.a_cat_folder_path = os.path.abspath(cat_a_config['cat_folder_path'])
        self.b_cat_folder_path = os.path.abspath(cat_b_config['cat_folder_path'])

        os.makedirs(joint_config['joint_folder_path'], exist_ok=True)
        os.makedirs(self.a_cat_folder_path, exist_ok=True)
        os.makedirs(self.b_cat_folder_path, exist_ok=True)

        np.save(f'{self.a_cat_folder_path}/con_cat_astro.npy', np.zeros((2, 3), float))
        np.save(f'{self.a_cat_folder_path}/con_cat_photo.npy', np.zeros((2, 3), float))
        np.save(f'{self.a_cat_folder_path}/magref.npy', np.zeros(2, float))

        np.save(f'{self.b_cat_folder_path}/con_cat_astro.npy', np.zeros((2, 3), float))
        np.save(f'{self.b_cat_folder_path}/con_cat_photo.npy', np.zeros((2, 4), float))
        np.save(f'{self.b_cat_folder_path}/magref.npy', np.zeros(2, float))

        os.makedirs('a_snr_mag', exist_ok=True)
        os.makedirs('b_snr_mag', exist_ok=True)
        np.save('a_snr_mag/snr_mag_params.npy', np.ones((3, 3, 5), float))
        np.save('b_snr_mag/snr_mag_params.npy', np.ones((4, 3, 5), float))

    def test_crossmatch_run_input(self):
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'))
        with pytest.raises(FileNotFoundError):
            cm._initialise_chunk('./file.txt', './file2.txt', './file3.txt')
        with pytest.raises(FileNotFoundError):
            cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                 'data/crossmatch_params.txt'), './file2.txt', './file3.txt')
        with pytest.raises(FileNotFoundError):
            cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                 'data/crossmatch_params.txt'),
                                 os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                                 './file3.txt')

    # pylint: disable-next=too-many-statements
    def test_crossmatch_auf_cf_input(self):
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'))
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        assert cm.cf_region_frame == 'equatorial'  # pylint: disable=no-member
        assert_allclose(cm.cf_region_points,
                        np.array([[131, -1], [132, -1], [133, -1], [134, -1],
                                  [131, 0], [132, 0], [133, 0], [134, 0],
                                  [131, 1], [132, 1], [133, 1], [134, 1]]))

        with open(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        old_line = 'include_perturb_auf = no'
        new_line = 'include_perturb_auf = yes\n'
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                      idx, new_line, out_file=os.path.join(
                      os.path.dirname(__file__), 'data/crossmatch_params_.txt'))

        with open(os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        old_line = 'auf_folder_path = gaia_auf_folder'
        new_line = ('auf_folder_path = gaia_auf_folder\ndens_hist_tri_location = None\n'
                    'tri_model_mags_location = None\ntri_model_mag_mids_location = None\n'
                    'tri_model_mags_interval_location = None\ntri_dens_uncert_location = None\n'
                    'tri_n_bright_sources_star_location = None\n')
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                      idx, new_line, out_file=os.path.join(
                      os.path.dirname(__file__), 'data/cat_a_params_new.txt'))
        with open(os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        old_line = 'auf_folder_path = wise_auf_folder'
        new_line = ('auf_folder_path = wise_auf_folder\ndens_hist_tri_location = None\n'
                    'tri_model_mags_location = None\ntri_model_mag_mids_location = None\n'
                    'tri_model_mags_interval_location = None\ntri_dens_uncert_location = None\n'
                    'tri_n_bright_sources_star_location = None\n')
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'),
                      idx, new_line, out_file=os.path.join(
                      os.path.dirname(__file__), 'data/cat_b_params_new.txt'))
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params_.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params_new.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params_new.txt'))
        assert cm.a_auf_region_frame == 'equatorial'  # pylint: disable=no-member
        assert_allclose(cm.a_auf_region_points,
                        np.array([[131, -1], [132, -1], [133, -1], [134, -1],
                                  [131, 0], [132, 0], [133, 0], [134, 0],
                                  [131, 1], [132, 1], [133, 1], [134, 1]]))
        assert_allclose(cm.b_auf_region_points,
                        np.array([[131, -1], [132, -1], [133, -1], [134, -1],
                                  [131, -1/3], [132, -1/3], [133, -1/3], [134, -1/3],
                                  [131, 1/3], [132, 1/3], [133, 1/3], [134, 1/3],
                                  [131, 1], [132, 1], [133, 1], [134, 1]]))

        for kind in ['auf_region_', 'cf_region_']:
            in_file = 'crossmatch_params' if 'cf' in kind else 'cat_a_params_new'
            with open(os.path.join(os.path.dirname(__file__), f'data/{in_file}.txt'),
                      encoding='utf-8') as file:
                f = file.readlines()
            # List of simple one line config file replacements for error message checking
            for old_line, new_line, match_text in zip(
                [f'{kind}type = rectangle', f'{kind}type = rectangle', f'{kind}points = 131 134 4 -1 1 3',
                 f'{kind}points = 131 134 4 -1 1 3', f'{kind}frame = equatorial',
                 f'{kind}points = 131 134 4 -1 1 3'],
                ['', f'{kind}type = triangle\n', f'{kind}points = 131 134 4 -1 1 a\n',
                 f'{kind}points = 131 134 4 -1 1\n', f'{kind}frame = ecliptic\n',
                 f'{kind}points = 131 134 4 -1 1 3.4\n'],
                [f'Missing key {kind}type',
                 f"{'' if 'cf' in kind else 'a_'}{kind}type should either be 'rectangle' or",
                 f"{'' if 'cf' in kind else 'a_'}{kind}points should be 6 numbers",
                 f"{'' if 'cf' in kind else 'a_'}{kind}points should be 6 numbers",
                 f"{'' if 'cf' in kind else 'a_'}{kind}frame should either be 'equatorial' or",
                 f"start and stop values for {'' if 'cf' in kind else 'a_'}{kind}points"]):
                idx = np.where([old_line in line for line in f])[0][0]
                _replace_line(os.path.join(os.path.dirname(__file__), f'data/{in_file}.txt'), idx, new_line,
                              out_file=os.path.join(os.path.dirname(__file__), f'data/{in_file}_.txt'))

                with pytest.raises(ValueError, match=match_text):
                    cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                         f"data/crossmatch_params{'_' if 'cf' in kind else ''}.txt"),
                                         os.path.join(os.path.dirname(__file__),
                                         f"data/cat_a_params{'_new_' if 'cf' not in kind else '_new'}.txt"),
                                         os.path.join(os.path.dirname(__file__), 'data/cat_b_params_new.txt'))

            # Check correct and incorrect *_region_points when *_region_type is 'points'
            idx = np.where([f'{kind}type = rectangle' in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__),
                          f'data/{in_file}.txt'), idx, f'{kind}type = points\n',
                          out_file=os.path.join(os.path.dirname(__file__), f'data/{in_file}_.txt'))

            idx = np.where([f'{kind}points = 131 134 4 -1 1 3' in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__), f'data/{in_file}_.txt'), idx,
                          f'{kind}points = (131, 0), (133, 0), (132, -1)\n',
                          out_file=os.path.join(os.path.dirname(__file__), f'data/{in_file}_2.txt'))

            cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                 f"data/crossmatch_params{'_2' if 'cf' in kind else ''}.txt"),
                                 os.path.join(os.path.dirname(__file__),
                                 f"data/cat_a_params{'_new_2' if 'cf' not in kind else '_new'}.txt"),
                                 os.path.join(os.path.dirname(__file__), 'data/cat_b_params_new.txt'))
            assert_allclose(getattr(cm, f"{'' if 'cf' in kind else 'a_'}{kind}points"),
                            np.array([[131, 0], [133, 0], [132, -1]]))

            old_line = f'{kind}points = 131 134 4 -1 1 3'
            for new_line in [f'{kind}points = (131, 0), (131, )\n', f'{kind}points = (131, 0), (131, 1, 2)\n',
                             f'{kind}points = (131, 0), (131, a)\n']:
                idx = np.where([old_line in line for line in f])[0][0]
                _replace_line(os.path.join(os.path.dirname(__file__), f'data/{in_file}_.txt'), idx, new_line,
                              out_file=os.path.join(os.path.dirname(__file__), f'data/{in_file}_2.txt'))

                with pytest.raises(ValueError):
                    cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                         f"data/crossmatch_params{'_2' if 'cf' in kind else ''}.txt"),
                                         os.path.join(os.path.dirname(__file__),
                                         f"data/cat_a_params{'_new_2' if 'cf' not in kind else '_new'}.txt"),
                                         os.path.join(os.path.dirname(__file__), 'data/cat_b_params_new.txt'))

            # Check single-length point grids are fine
            idx = np.where([f'{kind}points = 131 134 4 -1 1 3' in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__),
                          f'data/{in_file}.txt'), idx, f'{kind}points = 131 131 1 0 0 1\n',
                          out_file=os.path.join(os.path.dirname(__file__), f'data/{in_file}_.txt'))

            cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                 f"data/crossmatch_params{'_' if 'cf' in kind else ''}.txt"),
                                 os.path.join(os.path.dirname(__file__),
                                 f"data/cat_a_params{'_new_' if 'cf' not in kind else '_new'}.txt"),
                                 os.path.join(os.path.dirname(__file__), 'data/cat_b_params_new.txt'))
            assert_allclose(getattr(cm, f"{'' if 'cf' in kind else 'a_'}{kind}points"), np.array([[131, 0]]))

            idx = np.where([f'{kind}type = rectangle' in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__),
                          f'data/{in_file}.txt'), idx, f'{kind}type = points\n',
                          out_file=os.path.join(os.path.dirname(__file__), f'data/{in_file}_.txt'))

            idx = np.where([f'{kind}points = 131 134 4 -1 1 3' in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__),
                          f'data/{in_file}_.txt'), idx, f'{kind}points = (131, 0)\n',
                          out_file=os.path.join(os.path.dirname(__file__), f'data/{in_file}_2.txt'))

            cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                 f"data/crossmatch_params{'_2' if 'cf' in kind else ''}.txt"),
                                 os.path.join(os.path.dirname(__file__),
                                 f"data/cat_a_params{'_new_2' if 'cf' not in kind else '_new'}.txt"),
                                 os.path.join(os.path.dirname(__file__), 'data/cat_b_params_new.txt'))
            assert_allclose(getattr(cm, f"{'' if 'cf' in kind else 'a_'}{kind}points"), np.array([[131, 0]]))

        # Check galactic run is also fine -- here we have to replace all 3 parameter
        # options with "galactic", however.
        for in_file in ['crossmatch_params', 'cat_a_params_new', 'cat_b_params_new']:
            kind = 'cf_region_' if 'h_p' in in_file else 'auf_region_'
            with open(os.path.join(os.path.dirname(__file__), f'data/{in_file}.txt'),
                      encoding='utf-8') as file:
                f = file.readlines()
            idx = np.where([f'{kind}frame = equatorial' in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__),
                          f'data/{in_file}.txt'), idx, f'{kind}frame = galactic\n',
                          out_file=os.path.join(os.path.dirname(__file__), f'data/{in_file}_.txt'))

        cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params_.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params_new_.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params_new_.txt'))
        for kind in ['auf_region_', 'cf_region_']:
            assert getattr(cm, f"{'' if 'cf' in kind else 'a_'}{kind}frame") == 'galactic'
            assert_allclose(getattr(cm, f"{'' if 'cf' in kind else 'a_'}{kind}points"),
                            np.array([[131, -1], [132, -1], [133, -1], [134, -1],
                                      [131, 0], [132, 0], [133, 0], [134, 0],
                                      [131, 1], [132, 1], [133, 1], [134, 1]]))

    def test_crossmatch_load_tri_hists(self):  # pylint: disable=too-many-statements
        # pylint: disable=no-member
        with open(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        old_line = 'include_perturb_auf = no'
        new_line = 'include_perturb_auf = yes\n'
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                      idx, new_line, out_file=os.path.join(
                      os.path.dirname(__file__), 'data/crossmatch_params_.txt'))

        with open(os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        old_line = 'auf_folder_path = gaia_auf_folder'
        new_line = ('auf_folder_path = None\ndens_hist_tri_location = None\n'
                    'tri_model_mags_location = None\ntri_model_mag_mids_location = None\n'
                    'tri_model_mags_interval_location = None\ntri_dens_uncert_location = None\n'
                    'tri_n_bright_sources_star_location = None\n')
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                      idx, new_line, out_file=os.path.join(
                      os.path.dirname(__file__), 'data/cat_a_params_new.txt'))
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'))
        with pytest.raises(ValueError,
                           match="Either all flags related to running TRILEGAL histogram generation within"):
            cm._initialise_chunk(os.path.join(os.path.dirname(__file__), "data/crossmatch_params_.txt"),
                                 os.path.join(os.path.dirname(__file__), "data/cat_a_params_new.txt"),
                                 os.path.join(os.path.dirname(__file__), "data/cat_b_params_new.txt"))
        with open(os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        old_line = 'auf_folder_path = wise_auf_folder'
        new_line = ('auf_folder_path = None\ndens_hist_tri_location = None\n'
                    'tri_model_mags_location = None\ntri_model_mag_mids_location = None\n'
                    'tri_model_mags_interval_location = None\ntri_dens_uncert_location = None\n'
                    'tri_n_bright_sources_star_location = None\n')
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'),
                      idx, new_line, out_file=os.path.join(
                      os.path.dirname(__file__), 'data/cat_b_params_new.txt'))
        for flag in ['a', 'b']:
            for name in ['tri_set_name', 'tri_filt_names', 'tri_filt_num', 'download_tri',
                         'tri_maglim_faint', 'tri_num_faint']:
                with open(os.path.join(os.path.dirname(__file__), f'data/cat_{flag}_params_new.txt'),
                          encoding='utf-8') as file:
                    f = file.readlines()
                old_line = f'{name} = '
                new_line = f'{name} = None\n'
                idx = np.where([old_line in line for line in f])[0][0]
                _replace_line(os.path.join(os.path.dirname(__file__), f'data/cat_{flag}_params_new.txt'),
                              idx, new_line)
        # With everything set to None we hit the "can't have anything set" error:
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'))
        with pytest.raises(ValueError, match="Ambiguity in whether TRILEGAL histogram generation is being "):
            cm._initialise_chunk(os.path.join(os.path.dirname(__file__), "data/crossmatch_params_.txt"),
                                 os.path.join(os.path.dirname(__file__), "data/cat_a_params_new.txt"),
                                 os.path.join(os.path.dirname(__file__), "data/cat_b_params_new.txt"))
        for location, error, match_text in zip(
                ['some_fake_folder', os.path.join(os.path.dirname(__file__), 'data/dens_hist_tri.npy'),
                 os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                 os.path.join(os.path.dirname(__file__), 'data/dens_hist_tri.npy')],
                [FileNotFoundError, ValueError, ValueError, ValueError],
                ['File not found for dens_hist_tri. Please verify',
                 'Either all flags related to running TRILEGAL histogram generation externa',
                 'File could not be loaded from dens_hist_tri',
                 'number of filters in a_filt_names and a_dens_hist_tri']):
            if 'npy' in location and 'number of filters in' in match_text:
                np.save(location, np.ones((5, 10), float))
            elif 'npy' in location:
                np.save(location, np.ones((3, 10), float))
            with open(os.path.join(os.path.dirname(__file__), 'data/cat_a_params_new.txt'),
                      encoding='utf-8') as file:
                f = file.readlines()
            old_line = 'dens_hist_tri_location = '
            new_line = f'dens_hist_tri_location = {location}\n'
            idx = np.where([old_line in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_a_params_new.txt'), idx, new_line)
            cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'))
            with pytest.raises(error, match=match_text):
                cm._initialise_chunk(os.path.join(os.path.dirname(__file__), "data/crossmatch_params_.txt"),
                                     os.path.join(os.path.dirname(__file__), "data/cat_a_params_new.txt"),
                                     os.path.join(os.path.dirname(__file__), "data/cat_b_params_new.txt"))
        np.save(os.path.join(os.path.dirname(__file__), 'data/dens_hist_tri.npy'), np.ones((3, 10), float))
        with open(os.path.join(os.path.dirname(__file__), 'data/cat_a_params_new.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        old_line = 'dens_hist_tri_location = '
        location = os.path.join(os.path.dirname(__file__), 'data/dens_hist_tri.npy')
        new_line = f'dens_hist_tri_location = {location}\n'
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_a_params_new.txt'), idx, new_line)
        for file, match_text in zip([np.ones((5, 10), float), np.ones((3, 4), float)],
                                    ['The number of filter-elements in dens_hist_tri and tri_model_mags',
                                     'The number of magnitude-elements in dens_hist_tri and tri_model_mags']):
            np.save(os.path.join(os.path.dirname(__file__), 'data/tri_model_mags.npy'), file)
            with open(os.path.join(os.path.dirname(__file__), 'data/cat_a_params_new.txt'),
                      encoding='utf-8') as file:
                f = file.readlines()
            old_line = 'tri_model_mags_location = '
            location = os.path.join(os.path.dirname(__file__), 'data/tri_model_mags.npy')
            new_line = f'tri_model_mags_location = {location}\n'
            idx = np.where([old_line in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_a_params_new.txt'), idx, new_line)
            with pytest.raises(ValueError, match=match_text):
                cm._initialise_chunk(os.path.join(os.path.dirname(__file__), "data/crossmatch_params_.txt"),
                                     os.path.join(os.path.dirname(__file__), "data/cat_a_params_new.txt"),
                                     os.path.join(os.path.dirname(__file__), "data/cat_b_params_new.txt"))
        for catname, nfilts in zip(['a', 'b'], [3, 4]):
            for file, file_name in zip(
                    [np.ones((nfilts, 10), float), np.ones((nfilts, 10), float), np.ones((nfilts, 10), float),
                     np.ones((nfilts, 10), float), np.ones((nfilts, 10), float), np.ones((nfilts,), float)],
                    ['dens_hist_tri', 'tri_model_mags', 'tri_model_mag_mids', 'tri_model_mags_interval',
                     'tri_dens_uncert', 'tri_n_bright_sources_star']):
                np.save(os.path.join(os.path.dirname(__file__), f'data/{catname}_{file_name}.npy'), file)
                with open(os.path.join(os.path.dirname(__file__), f'data/cat_{catname}_params_new.txt'),
                          encoding='utf-8') as file:
                    f = file.readlines()
                old_line = f'{file_name}_location = '
                location = os.path.join(os.path.dirname(__file__), f'data/{catname}_{file_name}.npy')
                new_line = f'{file_name}_location = {location}\n'
                idx = np.where([old_line in line for line in f])[0][0]
                _replace_line(os.path.join(os.path.dirname(__file__), f'data/cat_{catname}_params_new.txt'),
                              idx, new_line)
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__), "data/crossmatch_params_.txt"),
                             os.path.join(os.path.dirname(__file__), "data/cat_a_params_new.txt"),
                             os.path.join(os.path.dirname(__file__), "data/cat_b_params_new.txt"))
        assert cm.b_auf_folder_path is None
        assert np.all([b is None for b in cm.a_tri_filt_names])
        assert np.all(cm.a_dens_hist_tri_list == np.ones((3, 10), float))
        assert np.all(cm.a_tri_dens_uncert_list == np.ones((3, 10), float))
        assert np.all(cm.b_tri_n_bright_sources_star_list == np.ones((4,), float))

    def test_crossmatch_folder_path_inputs(self):
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'))
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        assert cm.joint_folder_path == os.path.join(os.getcwd(), 'test_path')
        assert os.path.isdir(os.path.join(os.getcwd(), 'test_path'))
        assert cm.a_auf_folder_path == os.path.join(os.getcwd(), 'gaia_auf_folder')
        assert cm.b_auf_folder_path == os.path.join(os.getcwd(), 'wise_auf_folder')

        # List of simple one line config file replacements for error message checking
        for old_line, new_line, match_text, error, in_file in zip(
                ['joint_folder_path = test_path', 'joint_folder_path = test_path',
                 'auf_folder_path = gaia_auf_folder', 'auf_folder_path = wise_auf_folder'],
                ['', 'joint_folder_path = /User/test/some/path/\n', '',
                 'auf_folder_path = /User/test/some/path\n'],
                ['Missing key', 'Error when trying to check temporary',
                 'Missing key auf_folder_path from catalogue "a"',
                 'folder for catalogue "b" AUF outputs. Please ensure that b_auf_folder_path'],
                [ValueError, OSError, ValueError, OSError],
                ['crossmatch_params', 'crossmatch_params', 'cat_a_params', 'cat_b_params']):
            with open(os.path.join(os.path.dirname(__file__), f'data/{in_file}.txt'),
                      encoding='utf-8') as file:
                f = file.readlines()
            idx = np.where([old_line in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__),
                          f'data/{in_file}.txt'), idx, new_line, out_file=os.path.join(
                          os.path.dirname(__file__), f'data/{in_file}_.txt'))

            with pytest.raises(error, match=match_text):
                cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                     f"data/crossmatch_params{'_' if 'h_p' in in_file else ''}.txt"),
                                     os.path.join(os.path.dirname(__file__),
                                     f"data/cat_a_params{'_' if '_a_' in in_file else ''}.txt"),
                                     os.path.join(os.path.dirname(__file__),
                                     f"data/cat_b_params{'_' if '_b_' in in_file else ''}.txt"))

    def test_crossmatch_tri_inputs(self):
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'))
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        assert not hasattr(cm, 'a_tri_set_name')

        with open(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        old_line = 'include_perturb_auf = no'
        new_line = 'include_perturb_auf = yes\n'
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__),
                      'data/crossmatch_params.txt'), idx, new_line, out_file=os.path.join(
                      os.path.dirname(__file__), 'data/crossmatch_params_.txt'))

        with open(os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        old_line = 'auf_folder_path = gaia_auf_folder'
        new_line = ('auf_folder_path = gaia_auf_folder\ndens_hist_tri_location = None\n'
                    'tri_model_mags_location = None\ntri_model_mag_mids_location = None\n'
                    'tri_model_mags_interval_location = None\ntri_dens_uncert_location = None\n'
                    'tri_n_bright_sources_star_location = None\n')
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                      idx, new_line, out_file=os.path.join(
                      os.path.dirname(__file__), 'data/cat_a_params_new.txt'))
        with open(os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        old_line = 'auf_folder_path = wise_auf_folder'
        new_line = ('auf_folder_path = wise_auf_folder\ndens_hist_tri_location = None\n'
                    'tri_model_mags_location = None\ntri_model_mag_mids_location = None\n'
                    'tri_model_mags_interval_location = None\ntri_dens_uncert_location = None\n'
                    'tri_n_bright_sources_star_location = None\n')
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'),
                      idx, new_line, out_file=os.path.join(
                      os.path.dirname(__file__), 'data/cat_b_params_new.txt'))

        cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params_.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params_new.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params_new.txt'))
        assert cm.a_tri_set_name == 'gaiaDR2'  # pylint: disable=no-member
        assert np.all(cm.b_tri_filt_names == np.array(['W1', 'W2', 'W3', 'W4']))  # pylint: disable=no-member
        assert cm.a_tri_filt_num == 1  # pylint: disable=no-member
        assert not cm.b_download_tri  # pylint: disable=no-member

        # List of simple one line config file replacements for error message checking
        for old_line, new_line, match_text, in_file in zip(
                ['tri_set_name = gaiaDR2', 'tri_filt_num = 11', 'tri_filt_num = 11',
                 'download_tri = no', 'tri_maglim_faint = 32',
                 'tri_maglim_faint = 32', 'tri_num_faint = 1500000', 'tri_num_faint = 1500000',
                 'tri_num_faint = 1500000'],
                ['', 'tri_filt_num = a\n', 'tri_filt_num = 3.4\n', 'download_tri = aye\n',
                 'tri_maglim_faint = 32 33.5\n', 'tri_maglim_faint = a\n',
                 'tri_num_faint = 1500000.1\n', 'tri_num_faint = a\n',
                 'tri_num_faint = 1500000 15\n'],
                ['Missing key tri_set_name from catalogue "a"',
                 'tri_filt_num should be a single integer number in catalogue "b"',
                 'tri_filt_num should be a single integer number in catalogue "b"',
                 'Boolean flag key not set', 'tri_maglim_faint in catalogue "a" must be a float.',
                 'tri_maglim_faint in catalogue "b" must be a float.',
                 'tri_num_faint should be a single integer number in catalogue "b"',
                 'tri_num_faint should be a single integer number in catalogue "a" metadata file',
                 'tri_num_faint should be a single integer number in catalogue "a"'],
                ['cat_a_params', 'cat_b_params', 'cat_b_params', 'cat_a_params', 'cat_a_params',
                 'cat_b_params', 'cat_b_params', 'cat_a_params', 'cat_a_params']):
            with open(os.path.join(os.path.dirname(__file__), f'data/{in_file}_new.txt'),
                      encoding='utf-8') as file:
                f = file.readlines()
            idx = np.where([old_line in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__),
                          f'data/{in_file}_new.txt'), idx, new_line, out_file=os.path.join(
                          os.path.dirname(__file__), f'data/{in_file}_new_.txt'))

            with pytest.raises(ValueError, match=match_text):
                cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                     'data/crossmatch_params_.txt'),
                                     os.path.join(os.path.dirname(__file__),
                                     f"data/cat_a_params{'_new_' if '_a_' in in_file else '_new'}.txt"),
                                     os.path.join(os.path.dirname(__file__),
                                     f"data/cat_b_params{'_new_' if '_b_' in in_file else '_new'}.txt"))

    def test_crossmatch_psf_param_inputs(self):
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'))
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        assert np.all(cm.b_filt_names == np.array(['W1', 'W2', 'W3', 'W4']))

        with open(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        old_line = 'include_perturb_auf = no'
        new_line = 'include_perturb_auf = yes\n'
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__),
                      'data/crossmatch_params.txt'), idx, new_line, out_file=os.path.join(
                      os.path.dirname(__file__), 'data/crossmatch_params_.txt'))

        with open(os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        old_line = 'auf_folder_path = gaia_auf_folder'
        new_line = ('auf_folder_path = gaia_auf_folder\ndens_hist_tri_location = None\n'
                    'tri_model_mags_location = None\ntri_model_mag_mids_location = None\n'
                    'tri_model_mags_interval_location = None\ntri_dens_uncert_location = None\n'
                    'tri_n_bright_sources_star_location = None\n')
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                      idx, new_line, out_file=os.path.join(
                      os.path.dirname(__file__), 'data/cat_a_params_new.txt'))
        with open(os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        old_line = 'auf_folder_path = wise_auf_folder'
        new_line = ('auf_folder_path = wise_auf_folder\ndens_hist_tri_location = None\n'
                    'tri_model_mags_location = None\ntri_model_mag_mids_location = None\n'
                    'tri_model_mags_interval_location = None\ntri_dens_uncert_location = None\n'
                    'tri_n_bright_sources_star_location = None\n')
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'),
                      idx, new_line, out_file=os.path.join(
                      os.path.dirname(__file__), 'data/cat_b_params_new.txt'))

        os.makedirs('a_snr_mag', exist_ok=True)
        os.makedirs('b_snr_mag', exist_ok=True)
        np.save('a_snr_mag/snr_mag_params.npy', np.ones((3, 1, 5), float))
        np.save('b_snr_mag/snr_mag_params.npy', np.ones((4, 1, 5), float))

        cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params_.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params_new.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params_new.txt'))
        assert np.all(cm.a_psf_fwhms == np.array([0.12, 0.12, 0.12]))  # pylint: disable=no-member

        # List of simple one line config file replacements for error message checking
        for old_line, new_line, match_text, in_file in zip(
                ['filt_names = G_BP G G_RP', 'filt_names = G_BP G G_RP',
                 'psf_fwhms = 6.08 6.84 7.36 11.99', 'psf_fwhms = 6.08 6.84 7.36 11.99'],
                ['', 'filt_names = G_BP G\n',
                 'psf_fwhms = 6.08 6.84 7.36\n', 'psf_fwhms = 6.08 6.84 7.36 word\n'],
                ['Missing key filt_names from catalogue "a"',
                 'a_gal_al_avs and a_filt_names should contain the same',
                 'b_psf_fwhms and b_filt_names should contain the same',
                 'psf_fwhms should be a list of floats in catalogue "b".'],
                ['cat_a_params', 'cat_a_params', 'cat_b_params', 'cat_b_params']):
            # For the singular filt_names change we need to dummy snr_mag_params
            # as well, remembering to change it back afterwards
            if 'G\n' in new_line:
                np.save('a_snr_mag/snr_mag_params.npy', np.ones((2, 1, 5), float))
            with open(os.path.join(os.path.dirname(__file__), f'data/{in_file}_new.txt'),
                      encoding='utf-8') as file:
                f = file.readlines()
            idx = np.where([old_line in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__),
                          f'data/{in_file}_new.txt'), idx, new_line, out_file=os.path.join(
                          os.path.dirname(__file__), f'data/{in_file}_new_.txt'))

            with pytest.raises(ValueError, match=match_text):
                cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                     'data/crossmatch_params_.txt'),
                                     os.path.join(os.path.dirname(__file__),
                                     f"data/cat_a_params{'_new_' if '_a_' in in_file else '_new'}.txt"),
                                     os.path.join(os.path.dirname(__file__),
                                     f"data/cat_b_params{'_new_' if '_b_' in in_file else '_new'}.txt"))
            if 'G\n' in new_line:
                np.save('a_snr_mag/snr_mag_params.npy', np.ones((3, 1, 5), float))

    def test_crossmatch_cat_name_inputs(self):
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'))
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        assert cm.b_cat_name == 'WISE'

        with open(os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'), encoding='utf-8') as file:
            f = file.readlines()
        old_line = 'cat_name = Gaia'
        new_line = ''
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__),
                      'data/cat_a_params.txt'), idx, new_line, out_file=os.path.join(
                      os.path.dirname(__file__), 'data/cat_a_params_.txt'))

        match_text = 'Missing key cat_name from catalogue "a"'
        with pytest.raises(ValueError, match=match_text):
            cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                              'data/crossmatch_params.txt'),
                                 os.path.join(os.path.dirname(__file__), 'data/cat_a_params_.txt'),
                                 os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))

    def test_crossmatch_search_inputs(self):
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'))
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        assert cm.pos_corr_dist == 11
        assert not hasattr(cm, 'a_dens_dist')
        with open(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        old_line = 'include_perturb_auf = no'
        new_line = 'include_perturb_auf = yes\n'
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__),
                      'data/crossmatch_params.txt'), idx, new_line, out_file=os.path.join(
                      os.path.dirname(__file__), 'data/crossmatch_params_.txt'))

        with open(os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        old_line = 'auf_folder_path = gaia_auf_folder'
        new_line = ('auf_folder_path = gaia_auf_folder\ndens_hist_tri_location = None\n'
                    'tri_model_mags_location = None\ntri_model_mag_mids_location = None\n'
                    'tri_model_mags_interval_location = None\ntri_dens_uncert_location = None\n'
                    'tri_n_bright_sources_star_location = None\n')
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                      idx, new_line, out_file=os.path.join(
                      os.path.dirname(__file__), 'data/cat_a_params_new.txt'))
        with open(os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        old_line = 'auf_folder_path = wise_auf_folder'
        new_line = ('auf_folder_path = wise_auf_folder\ndens_hist_tri_location = None\n'
                    'tri_model_mags_location = None\ntri_model_mag_mids_location = None\n'
                    'tri_model_mags_interval_location = None\ntri_dens_uncert_location = None\n'
                    'tri_n_bright_sources_star_location = None\n')
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'),
                      idx, new_line, out_file=os.path.join(
                      os.path.dirname(__file__), 'data/cat_b_params_new.txt'))

        cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params_.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params_new.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params_new.txt'))
        assert np.all(cm.a_psf_fwhms == np.array([0.12, 0.12, 0.12]))  # pylint: disable=no-member
        assert cm.b_dens_dist == 0.25

        # List of simple one line config file replacements for error message checking
        for old_line, new_line, match_text, in_file in zip(
                ['pos_corr_dist = 11', 'pos_corr_dist = 11', 'dens_dist = 0.25',
                 'dens_dist = 0.25'],
                ['', 'pos_corr_dist = word\n', '', 'dens_dist = word\n'],
                ['Missing key pos_corr_dist', 'pos_corr_dist must be a float',
                 'Missing key dens_dist from catalogue "b"', 'dens_dist in catalogue "a" must'],
                ['crossmatch_params', 'crossmatch_params', 'cat_b_params_new', 'cat_a_params_new']):
            with open(os.path.join(os.path.dirname(__file__), f'data/{in_file}.txt'),
                      encoding='utf-8') as file:
                f = file.readlines()
            idx = np.where([old_line in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__),
                          f"data/{in_file}{'_' if 'h_p' in in_file else ''}.txt"), idx,
                          new_line, out_file=os.path.join(os.path.dirname(__file__),
                          f"data/{in_file}_{'2' if 'h_p' in in_file else ''}.txt"))
            with pytest.raises(ValueError, match=match_text):
                cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                     f"data/crossmatch_params{'_2' if 'h_p' in in_file else '_'}.txt"),
                                     os.path.join(os.path.dirname(__file__),
                                     f"data/cat_a_params{'_new_' if '_a_' in in_file else '_new'}.txt"),
                                     os.path.join(os.path.dirname(__file__),
                                     f"data/cat_b_params{'_new_' if '_b_' in in_file else '_new'}.txt"))

    # pylint: disable-next=too-many-statements
    def test_crossmatch_perturb_auf_inputs(self):
        with open(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        old_line = 'include_perturb_auf = no'
        new_line = 'include_perturb_auf = yes\n'
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__),
                      'data/crossmatch_params.txt'), idx, new_line, out_file=os.path.join(
                      os.path.dirname(__file__), 'data/crossmatch_params_.txt'))

        with open(os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        old_line = 'auf_folder_path = gaia_auf_folder'
        new_line = ('auf_folder_path = gaia_auf_folder\ndens_hist_tri_location = None\n'
                    'tri_model_mags_location = None\ntri_model_mag_mids_location = None\n'
                    'tri_model_mags_interval_location = None\ntri_dens_uncert_location = None\n'
                    'tri_n_bright_sources_star_location = None\n')
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                      idx, new_line, out_file=os.path.join(
                      os.path.dirname(__file__), 'data/cat_a_params_new.txt'))
        with open(os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        old_line = 'auf_folder_path = wise_auf_folder'
        new_line = ('auf_folder_path = wise_auf_folder\ndens_hist_tri_location = None\n'
                    'tri_model_mags_location = None\ntri_model_mag_mids_location = None\n'
                    'tri_model_mags_interval_location = None\ntri_dens_uncert_location = None\n'
                    'tri_n_bright_sources_star_location = None\n')
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'),
                      idx, new_line, out_file=os.path.join(
                      os.path.dirname(__file__), 'data/cat_b_params_new.txt'))

        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'))
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params_.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params_new.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params_new.txt'))
        assert cm.num_trials == 10000
        assert cm.d_mag == 0.1  # pylint: disable=no-member

        for old_line, new_line, match_text in zip(
                ['num_trials = 10000', 'num_trials = 10000', 'num_trials = 10000',
                 'd_mag = 0.1', 'd_mag = 0.1'],
                ['', 'num_trials = word\n', 'num_trials = 10000.1\n', '', 'd_mag = word\n'],
                ['Missing key num_trials from joint', 'num_trials should be an integer',
                 'num_trials should be an integer', 'Missing key d_mag from joint',
                 'd_mag must be a float']):
            # Make sure to keep the first edit of crossmatch_params, adding each
            # second change in turn.
            with open(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params_.txt'),
                      encoding='utf-8') as file:
                f = file.readlines()
            idx = np.where([old_line in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__),
                          'data/crossmatch_params_.txt'), idx, new_line, out_file=os.path.join(
                          os.path.dirname(__file__), 'data/crossmatch_params_2.txt'))
            with open(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params_2.txt'),
                      encoding='utf-8') as file:
                f = file.readlines()

            with pytest.raises(ValueError, match=match_text):
                cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                     'data/crossmatch_params_2.txt'),
                                     os.path.join(os.path.dirname(__file__),
                                     'data/cat_a_params_new.txt'),
                                     os.path.join(os.path.dirname(__file__),
                                     'data/cat_b_params_new.txt'))

        for old_line, var_name in zip(['fit_gal_flag = no', 'run_fw_auf = yes', 'run_psf_auf = no',
                                       'snr_mag_params_path = '],
                                      ['fit_gal_flag', 'run_fw_auf', 'run_psf_auf',
                                       'snr_mag_params_path']):
            for cat_reg, cat_name in zip(['"a"', '"b"'], ['cat_a_params_new', 'cat_b_params_new']):
                with open(os.path.join(os.path.dirname(__file__), f'data/{cat_name}.txt'),
                          encoding='utf-8') as file:
                    f = file.readlines()
                new_line = ''
                idx = np.where([old_line in line for line in f])[0][0]
                _replace_line(os.path.join(os.path.dirname(__file__),
                              f'data/{cat_name}.txt'), idx, new_line, out_file=os.path.join(
                              os.path.dirname(__file__), f'data/{cat_name}_.txt'))
                with pytest.raises(ValueError, match=f'Missing key {var_name} from catalogue {cat_reg}'):
                    cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                         'data/crossmatch_params_.txt'),
                                         os.path.join(os.path.dirname(__file__),
                                         f"data/cat_a_params_new{'_' if 'a' in cat_reg else ''}.txt"),
                                         os.path.join(os.path.dirname(__file__),
                                         f"data/cat_b_params_new{'_' if 'b' in cat_reg else ''}.txt"))

        os.makedirs('a_snr_mag', exist_ok=True)
        os.makedirs('b_snr_mag', exist_ok=True)
        np.save('a_snr_mag/snr_mag_params.npy', np.ones((3, 1, 5), float))
        np.save('b_snr_mag/snr_mag_params.npy', np.ones((4, 1, 5), float))

        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'))
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                             'data/crossmatch_params_.txt'),
                             os.path.join(os.path.dirname(__file__),
                             'data/cat_a_params_new.txt'),
                             os.path.join(os.path.dirname(__file__),
                             'data/cat_b_params_new.txt'))
        assert not hasattr(cm, 'a_dd_params_path')
        assert not hasattr(cm, 'b_l_cut_path')

        for cat_reg, cat_name in zip(['"a"', '"b"'], ['cat_a_params_new', 'cat_b_params_new']):
            with open(os.path.join(os.path.dirname(__file__), f'data/{cat_name}.txt'),
                      encoding='utf-8') as file:
                f = file.readlines()
            old_line = 'run_psf_auf = no'
            new_line = 'run_psf_auf = yes\n'
            idx = np.where([old_line in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__), f'data/{cat_name}.txt'), idx, new_line,
                          out_file=os.path.join(os.path.dirname(__file__), f'data/{cat_name}_.txt'))
            with open(os.path.join(os.path.dirname(__file__), f'data/{cat_name}_.txt'),
                      encoding='utf-8') as file:
                f = file.readlines()
            old_line = 'snr_mag_params_path = '
            new_line = (f'snr_mag_params_path = {cat_reg[1]}_snr_mag\ndd_params_path = .\n'
                        'l_cut_path = .\n')
            idx = np.where([old_line in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__), f'data/{cat_name}_.txt'), idx, new_line)
            for old_line, var_name in zip(['dd_params_path = .', 'l_cut_path = .'],
                                          ['dd_params_path', 'l_cut_path']):
                with open(os.path.join(os.path.dirname(__file__), f'data/{cat_name}_.txt'),
                          encoding='utf-8') as file:
                    f = file.readlines()
                new_line = ''
                idx = np.where([old_line in line for line in f])[0][0]
                _replace_line(os.path.join(os.path.dirname(__file__),
                              f'data/{cat_name}_.txt'), idx, new_line, out_file=os.path.join(
                              os.path.dirname(__file__), f'data/{cat_name}__.txt'))
                with pytest.raises(ValueError, match=f'Missing key {var_name} from catalogue {cat_reg}'):
                    cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                         'data/crossmatch_params_.txt'),
                                         os.path.join(os.path.dirname(__file__),
                                         f"data/cat_a_params_new{'__' if 'a' in cat_reg else ''}.txt"),
                                         os.path.join(os.path.dirname(__file__),
                                         f"data/cat_b_params_new{'__' if 'b' in cat_reg else ''}.txt"))

        ddp = np.ones((5, 15, 2), float)
        np.save('dd_params.npy', ddp)
        lc = np.ones(3, float)
        np.save('l_cut.npy', lc)
        for cat_name in ['cat_a_params_new', 'cat_b_params_new']:
            with open(os.path.join(os.path.dirname(__file__), f'data/{cat_name}.txt'),
                      encoding='utf-8') as file:
                f = file.readlines()
            old_line = 'run_psf_auf = no'
            new_line = 'run_psf_auf = yes\n'
            idx = np.where([old_line in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__), f'data/{cat_name}.txt'), idx, new_line,
                          out_file=os.path.join(os.path.dirname(__file__), f'data/{cat_name}_.txt'))
            with open(os.path.join(os.path.dirname(__file__), f'data/{cat_name}_.txt'),
                      encoding='utf-8') as file:
                f = file.readlines()
            old_line = f'snr_mag_params_path = {cat_name[4]}_snr_mag'
            new_line = (f'snr_mag_params_path = {cat_name[4]}_snr_mag\ndd_params_path = .\n'
                        'l_cut_path = .\n')
            idx = np.where([old_line in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__), f'data/{cat_name}_.txt'), idx, new_line)
        with open(os.path.join(os.path.dirname(__file__), 'data/cat_b_params_new_.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        old_line = 'snr_mag_params_path = b_snr_mag'
        new_line = 'snr_mag_params_path = /some/path/or/other\n'
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__),
                      'data/cat_b_params_new_.txt'), idx, new_line,
                      out_file=os.path.join(os.path.dirname(__file__), 'data/cat_b_params_new__.txt'))
        with pytest.raises(OSError, match='b_snr_mag_params_path does not exist.'):
            cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                 'data/crossmatch_params_.txt'),
                                 os.path.join(os.path.dirname(__file__),
                                 'data/cat_a_params_new_.txt'),
                                 os.path.join(os.path.dirname(__file__),
                                 'data/cat_b_params_new__.txt'))
        os.remove('a_snr_mag/snr_mag_params.npy')
        with pytest.raises(FileNotFoundError,
                           match='snr_mag_params file not found in catalogue "a" path'):
            cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                 'data/crossmatch_params_.txt'),
                                 os.path.join(os.path.dirname(__file__),
                                 'data/cat_a_params_new_.txt'),
                                 os.path.join(os.path.dirname(__file__),
                                 'data/cat_b_params_new_.txt'))
        for fn, array, err_msg in zip([
                'snr_mag_params', 'snr_mag_params', 'snr_mag_params', 'dd_params', 'dd_params',
                'dd_params', 'dd_params', 'l_cut', 'l_cut'],
                [np.ones(4, float), np.ones((5, 3, 2), float), np.ones((4, 4), float),
                 np.ones(5, float), np.ones((5, 3), float), np.ones((4, 4, 2), float),
                 np.ones((5, 3, 1), float), np.ones((4, 2), float), np.ones(4, float)],
                [r'a_snr_mag_params should be of shape \(X, Y, 5\)',
                 r'a_snr_mag_params should be of shape \(X, Y, 5\)',
                 r'a_snr_mag_params should be of shape \(X, Y, 5\)',
                 r'a_dd_params should be of shape \(5, X, 2\)',
                 r'a_dd_params should be of shape \(5, X, 2\)',
                 r'a_dd_params should be of shape \(5, X, 2\)',
                 r'a_dd_params should be of shape \(5, X, 2\)',
                 r'a_l_cut should be of shape \(3,\) only.',
                 r'a_l_cut should be of shape \(3,\) only.']):
            np.save(f"{'a_snr_mag/' if 'snr_mag' in fn else ''}{fn}.npy", array)
            with pytest.raises(ValueError, match=err_msg):
                cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                     'data/crossmatch_params_.txt'),
                                     os.path.join(os.path.dirname(__file__),
                                     'data/cat_a_params_new_.txt'),
                                     os.path.join(os.path.dirname(__file__),
                                     'data/cat_b_params_new_.txt'))
            # Re-make "good" fake arrays
            snr_mag_params = np.ones((3, 3, 5), float)
            np.save('a_snr_mag/snr_mag_params.npy', snr_mag_params)
            ddp = np.ones((5, 15, 2), float)
            np.save('dd_params.npy', ddp)
            lc = np.ones(3, float)
            np.save('l_cut.npy', lc)

        with open(os.path.join(os.path.dirname(__file__), 'data/cat_a_params_new.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        old_line = 'fit_gal_flag = no'
        new_line = ('fit_gal_flag = yes\ngal_wavs = 0.513 0.641 0.778\n'
                    'gal_zmax = 4.5 4.5 5\ngal_nzs = 46 46 51\n'
                    'gal_aboffsets = 0.5 0.5 0.5\n'
                    'gal_filternames = gaiadr2-BP gaiadr2-G gaiadr2-RP\n'
                    'saturation_magnitudes = 5 5 5\n')
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_a_params_new.txt'), idx,
                      new_line, out_file=os.path.join(os.path.dirname(__file__),
                      'data/cat_a_params_new_.txt'))
        with open(os.path.join(os.path.dirname(__file__), 'data/cat_b_params_new.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        old_line = 'fit_gal_flag = no'
        new_line = ('fit_gal_flag = yes\ngal_wavs = 3.37 4.62 12.08 22.19\n'
                    'gal_zmax = 3.2 4.0 1 4\ngal_nzs = 33 41 11 41\n'
                    'gal_aboffsets = 0.5 0.5 0.5 0.5\n'
                    'gal_filternames = wise2010-W1 wise2010-W2 wise2010-W3 wise2010-W4\n'
                    'saturation_magnitudes = 5 5 5 5\n')
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_b_params_new.txt'), idx,
                      new_line, out_file=os.path.join(os.path.dirname(__file__),
                      'data/cat_b_params_new_.txt'))

        cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                             'data/crossmatch_params_.txt'),
                             os.path.join(os.path.dirname(__file__),
                             'data/cat_a_params_new_.txt'),
                             os.path.join(os.path.dirname(__file__),
                             'data/cat_b_params_new_.txt'))
        assert_allclose(cm.a_gal_zmax, np.array([4.5, 4.5, 5.0]))
        assert np.all(cm.b_gal_nzs == np.array([33, 41, 11, 41]))
        assert np.all(cm.a_gal_filternames == ['gaiadr2-BP', 'gaiadr2-G', 'gaiadr2-RP'])

        with open(os.path.join(os.path.dirname(__file__), 'data/cat_b_params_new_.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        for key in ['gal_wavs', 'gal_zmax', 'gal_nzs', 'gal_aboffsets', 'gal_filternames', 'gal_al_avs']:
            idx = np.where([key in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_b_params_new_.txt'),
                          idx, '', out_file=os.path.join(os.path.dirname(__file__),
                          'data/cat_b_params_new__.txt'))
            with pytest.raises(ValueError,
                               match=f'Missing key {key} from catalogue "b"'):
                cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                     'data/crossmatch_params_.txt'),
                                     os.path.join(os.path.dirname(__file__),
                                     'data/cat_a_params_new_.txt'),
                                     os.path.join(os.path.dirname(__file__),
                                     'data/cat_b_params_new__.txt'))

        for old_line, new_line, match_text, in_file in zip(
                ['gal_wavs = 0.513 0.641 0.778', 'gal_aboffsets = 0.5 0.5 0.5 0.5',
                 'gal_nzs = 46 46 51', 'gal_nzs = 33 41 11 41', 'gal_nzs = 33 41 11 41',
                 'gal_filternames = gaiadr2-BP gaiadr2-G gaiadr2-RP',
                 'gal_al_avs = 1.002 0.789 0.589', 'gal_al_avs = 1.002 0.789 0.589',
                 'saturation_magnitudes = ', 'saturation_magnitudes = '],
                ['gal_wavs = 0.513 0.641\n', 'gal_aboffsets = a 0.5 0.5 0.5\n',
                 'gal_nzs = 46 a 51\n', 'gal_nzs = 33.1 41 11 41\n', 'gal_nzs = 33 41 11\n',
                 'gal_filternames = gaiadr2-BP gaiadr2-G gaiadr2-RP wise2010-W1\n',
                 'gal_al_avs = words\n', 'gal_al_avs = 0.789 1.002\n',
                 'saturation_magnitudes = words\n', 'saturation_magnitudes = 4 4\n'],
                ['a_gal_wavs and a_filt_names should contain the same number',
                 'gal_aboffsets should be a list of floats in catalogue "b"',
                 'gal_nzs should be a list of integers in catalogue "a"',
                 'All elements of b_gal_nzs should be integers.',
                 'b_gal_nzs and b_filt_names should contain the same number of entries.',
                 'a_gal_filternames and a_filt_names should contain the same number of entries.',
                 'gal_al_avs should be a list of floats in catalogue "a"',
                 'a_gal_al_avs and a_filt_names should contain the same number of entries.',
                 'saturation_magnitudes should be a list of floats in catalogue "b"',
                 'b_saturation_magnitudes and b_filt_names should contain the same number of entries.'],
                ['cat_a_params_new', 'cat_b_params_new', 'cat_a_params_new', 'cat_b_params_new',
                 'cat_b_params_new', 'cat_a_params_new', 'cat_a_params_new', 'cat_a_params_new',
                 'cat_b_params_new', 'cat_b_params_new']):
            with open(os.path.join(os.path.dirname(__file__), f'data/{in_file}_.txt'),
                      encoding='utf-8') as file:
                f = file.readlines()
            idx = np.where([old_line in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__),
                          f'data/{in_file}_.txt'), idx, new_line, out_file=os.path.join(
                          os.path.dirname(__file__), f'data/{in_file}__.txt'))

            with pytest.raises(ValueError, match=match_text):
                cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                     'data/crossmatch_params_.txt'),
                                     os.path.join(os.path.dirname(__file__),
                                     f"data/cat_a_params_new{'_' if '_b_' in in_file else '__'}.txt"),
                                     os.path.join(os.path.dirname(__file__),
                                     f"data/cat_b_params_new{'_' if '_a_' in in_file else '__'}.txt"))

    def test_crossmatch_fourier_inputs(self):
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'))
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        assert cm.real_hankel_points == 10000  # pylint: disable=no-member
        assert cm.four_hankel_points == 10000  # pylint: disable=no-member
        assert cm.four_max_rho == 100  # pylint: disable=no-member

        # List of simple one line config file replacements for error message checking
        for old_line, new_line, match_text in zip(
                ['real_hankel_points = 10000', 'four_hankel_points = 10000', 'four_max_rho = 100'],
                ['', 'four_hankel_points = 10000.1\n', 'four_max_rho = word\n'],
                ['Missing key real_hankel_points', 'four_hankel_points should be an integer.',
                 'four_max_rho should be an integer.']):
            with open(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                      encoding='utf-8') as file:
                f = file.readlines()
            idx = np.where([old_line in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__),
                          'data/crossmatch_params.txt'), idx, new_line, out_file=os.path.join(
                          os.path.dirname(__file__), 'data/crossmatch_params_.txt'))

            with pytest.raises(ValueError, match=match_text):
                cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                     'data/crossmatch_params_.txt'),
                                     os.path.join(os.path.dirname(__file__),
                                     'data/cat_a_params.txt'),
                                     os.path.join(os.path.dirname(__file__),
                                     'data/cat_b_params.txt'))

    def test_crossmatch_frame_equality(self):
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'))
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        assert cm.a_auf_region_frame == 'equatorial'  # pylint: disable=no-member
        assert cm.b_auf_region_frame == 'equatorial'  # pylint: disable=no-member
        assert cm.cf_region_frame == 'equatorial'  # pylint: disable=no-member

        # List of simple one line config file replacements for error message checking
        match_text = 'Region frames for c/f and AUF creation must all be the same.'
        for old_line, new_line, in_file in zip(
                ['cf_region_frame = equatorial', 'auf_region_frame = equatorial',
                 'auf_region_frame = equatorial'],
                ['cf_region_frame = galactic\n', 'auf_region_frame = galactic\n',
                 'auf_region_frame = galactic\n'],
                ['crossmatch_params', 'cat_a_params', 'cat_b_params']):
            with open(os.path.join(os.path.dirname(__file__), f'data/{in_file}.txt'),
                      encoding='utf-8') as file:
                f = file.readlines()
            idx = np.where([old_line in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__), f'data/{in_file}.txt'), idx, new_line,
                          out_file=os.path.join(os.path.dirname(__file__), f'data/{in_file}_.txt'))

            with pytest.raises(ValueError, match=match_text):
                cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                     f"data/crossmatch_params{'_' if 'h_p' in in_file else ''}.txt"),
                                     os.path.join(os.path.dirname(__file__),
                                     f"data/cat_a_params{'_' if '_a_' in in_file else ''}.txt"),
                                     os.path.join(os.path.dirname(__file__),
                                     f"data/cat_b_params{'_' if '_b_' in in_file else ''}.txt"))

    def test_cross_match_extent(self):
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'))
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        assert np.all(cm.cross_match_extent == np.array([131, 138, -3, 3]))

        # List of simple one line config file replacements for error message checking
        in_file = 'crossmatch_params'
        with open(os.path.join(os.path.dirname(__file__), f'data/{in_file}.txt'), encoding='utf-8') as file:
            f = file.readlines()
        old_line = 'cross_match_extent = 131 138 -3 3'
        for new_line, match_text in zip(
                ['', 'cross_match_extent = 131 138 -3 word\n', 'cross_match_extent = 131 138 -3\n',
                 'cross_match_extent = 131 138 -3 3 1'],
                ['Missing key cross_match_extent', 'All elements of cross_match_extent should be',
                 'cross_match_extent should contain.', 'cross_match_extent should contain']):
            idx = np.where([old_line in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__), f'data/{in_file}.txt'), idx, new_line,
                          out_file=os.path.join(os.path.dirname(__file__), f'data/{in_file}_.txt'))

            with pytest.raises(ValueError, match=match_text):
                cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                     f"data/crossmatch_params{'_' if 'h_p' in in_file else ''}.txt"),
                                     os.path.join(os.path.dirname(__file__),
                                     f"data/cat_a_params{'_' if '_a_' in in_file else ''}.txt"),
                                     os.path.join(os.path.dirname(__file__),
                                     f"data/cat_b_params{'_' if '_b_' in in_file else ''}.txt"))

    def test_int_fracs(self):
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'))
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        assert np.all(cm.int_fracs == np.array([0.63, 0.9, 0.99]))

        # List of simple one line config file replacements for error message checking
        in_file = 'crossmatch_params'
        with open(os.path.join(os.path.dirname(__file__), f'data/{in_file}.txt'), encoding='utf-8') as file:
            f = file.readlines()
        old_line = 'int_fracs = 0.63 0.9 0.99'
        for new_line, match_text in zip(
                ['', 'int_fracs = 0.63 0.9 word\n', 'int_fracs = 0.63 0.9\n'],
                ['Missing key int_fracs', 'All elements of int_fracs should be',
                 'int_fracs should contain.']):
            idx = np.where([old_line in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__), f'data/{in_file}.txt'), idx, new_line,
                          out_file=os.path.join(os.path.dirname(__file__), f'data/{in_file}_.txt'))

            with pytest.raises(ValueError, match=match_text):
                cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                     f"data/crossmatch_params{'_' if 'h_p' in in_file else ''}.txt"),
                                     os.path.join(os.path.dirname(__file__),
                                     f"data/cat_a_params{'_' if '_a_' in in_file else ''}.txt"),
                                     os.path.join(os.path.dirname(__file__),
                                     f"data/cat_b_params{'_' if '_b_' in in_file else ''}.txt"))

    def test_crossmatch_shared_data(self):
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'))
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        assert np.all(cm.r == np.linspace(0, 11, 10000))
        assert_allclose(cm.dr, np.ones(9999, float) * 11/9999)
        assert np.all(cm.rho == np.linspace(0, 100, 10000))
        assert_allclose(cm.drho, np.ones(9999, float) * 100/9999)

    def test_cat_folder_path(self):
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'))
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        assert os.path.exists(self.a_cat_folder_path)
        assert os.path.exists(self.b_cat_folder_path)
        assert cm.a_cat_folder_path == self.a_cat_folder_path
        assert np.all(np.load(f'{self.a_cat_folder_path}/con_cat_astro.npy').shape == (2, 3))
        assert np.all(np.load(f'{self.b_cat_folder_path}/con_cat_photo.npy').shape == (2, 4))
        assert np.all(np.load(f'{self.a_cat_folder_path}/magref.npy').shape == (2,))

        os.system(f'rm -rf {self.a_cat_folder_path}')
        with pytest.raises(OSError, match="a_cat_folder_path does not exist."):
            cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                              'data/crossmatch_params.txt'),
                                 os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                                 os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        self.setup_class()

        os.system(f'rm -rf {self.b_cat_folder_path}')
        with pytest.raises(OSError, match="b_cat_folder_path does not exist."):
            cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                              'data/crossmatch_params.txt'),
                                 os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                                 os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        self.setup_class()

        for catpath, file in zip([self.a_cat_folder_path, self.b_cat_folder_path],
                                 ['con_cat_astro', 'magref']):
            os.system(f'rm {catpath}/{file}.npy')
            with pytest.raises(FileNotFoundError,
                               match=f'{file} file not found in catalogue '):
                cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                     'data/crossmatch_params.txt'),
                                     os.path.join(os.path.dirname(__file__),
                                                  'data/cat_a_params.txt'),
                                     os.path.join(os.path.dirname(__file__),
                                                  'data/cat_b_params.txt'))
            self.setup_class()

        for name, data, match in zip(['con_cat_astro', 'con_cat_photo', 'con_cat_astro',
                                      'con_cat_photo', 'magref', 'con_cat_astro', 'con_cat_photo',
                                      'magref'],
                                     [np.zeros((2, 2), float), np.zeros((2, 5), float),
                                      np.zeros((2, 3, 4), float), np.zeros(2, float),
                                      np.zeros((2, 2), float), np.zeros((1, 3), float),
                                      np.zeros((3, 4), float), np.zeros(3, float)],
                                     ["Second dimension of con_cat_astro",
                                      "Second dimension of con_cat_photo in",
                                      "Incorrect number of dimensions",
                                      "Incorrect number of dimensions",
                                      "Incorrect number of dimensions",
                                      'Consolidated catalogue arrays for catalogue "b"',
                                      'Consolidated catalogue arrays for catalogue "b"',
                                      'Consolidated catalogue arrays for catalogue "b"']):
            np.save(f'{self.b_cat_folder_path}/{name}.npy', data)
            with pytest.raises(ValueError, match=match):
                cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                     'data/crossmatch_params.txt'),
                                     os.path.join(os.path.dirname(__file__),
                                                  'data/cat_a_params.txt'),
                                     os.path.join(os.path.dirname(__file__),
                                                  'data/cat_b_params.txt'))
            self.setup_class()

    def test_calculate_cf_areas(self):
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'))
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        cm.cross_match_extent = np.array([131, 134, -1, 1])
        cm.cf_region_points = np.array([[a, b] for a in [131.5, 132.5, 133.5]
                                        for b in [-0.5, 0.5]])
        cm.chunk_id = 1
        mcff = Macauff(cm)
        mcff._calculate_cf_areas()
        assert_allclose(cm.cf_areas, np.ones((6), float), rtol=0.02)

        cm.cross_match_extent = np.array([50, 55, 85, 90])
        cm.cf_region_points = np.array([[a, b] for a in 0.5+np.arange(50, 55, 1)
                                        for b in 0.5+np.arange(85, 90, 1)])
        mcff = Macauff(cm)
        mcff._calculate_cf_areas()
        calculated_areas = np.array(
            [(c[0]+0.5 - (c[0]-0.5))*180/np.pi * (np.sin(np.radians(c[1]+0.5)) -
             np.sin(np.radians(c[1]-0.5))) for c in cm.cf_region_points])
        assert_allclose(cm.cf_areas, calculated_areas, rtol=0.025)

    # pylint: disable-next=too-many-statements
    def test_csv_inputs(self):
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'))
        with open(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        old_line = 'make_output_csv = no'
        new_line = ''
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__),
                      'data/crossmatch_params.txt'), idx, new_line, out_file=os.path.join(
                      os.path.dirname(__file__), 'data/crossmatch_params_.txt'))

        with pytest.raises(ValueError, match="Missing key make_output_csv"):
            cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                 'data/crossmatch_params_.txt'),
                                 os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                                 os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))

        with open(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        old_line = 'make_output_csv = no'
        new_line = 'make_output_csv = yes\n'
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__),
                      'data/crossmatch_params.txt'), idx, new_line, out_file=os.path.join(
                      os.path.dirname(__file__), 'data/crossmatch_params_.txt'))

        old_line = 'make_output_csv = yes\n'
        with open(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params_.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        idx = np.where([old_line in line for line in f])[0][0]
        lines = ['make_output_csv = yes\n', 'output_csv_folder = output_csv_folder\n',
                 'match_out_csv_name = match.csv\n']
        for i, key in enumerate(['output_csv_folder', 'match_out_csv_name',
                                 'nonmatch_out_csv_name']):
            new_line = ''
            for j in range(i+1):
                new_line = new_line + lines[j]
            _replace_line(os.path.join(os.path.dirname(__file__),
                          'data/crossmatch_params_.txt'), idx, new_line,
                          out_file=os.path.join(os.path.dirname(__file__),
                                                'data/crossmatch_params__.txt'))
            with pytest.raises(ValueError, match=f"Missing key {key} from joint"):
                cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                     'data/crossmatch_params__.txt'),
                                     os.path.join(os.path.dirname(__file__),
                                     'data/cat_a_params.txt'),
                                     os.path.join(os.path.dirname(__file__),
                                     'data/cat_b_params.txt'))
        new_line = ''
        for line in lines:
            new_line = new_line + line
        new_line = new_line + 'nonmatch_out_csv_name = nonmatch.csv\n'
        _replace_line(os.path.join(os.path.dirname(__file__),
                      'data/crossmatch_params_.txt'), idx, new_line,
                      out_file=os.path.join(os.path.dirname(__file__),
                                            'data/crossmatch_params__.txt'))

        old_line = 'snr_mag_params_path = a_snr_mag\n'
        with open(os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'), encoding='utf-8') as file:
            f = file.readlines()
        idx = np.where([old_line in line for line in f])[0][0]
        lines = ['snr_mag_params_path = a_snr_mag\n\n', 'input_csv_folder = input_csv_folder\n',
                 'cat_csv_name = catalogue.csv\n', 'cat_col_names = A B C\n',
                 'cat_col_nums = 1 2 3\n', 'input_npy_folder = blah\n', 'csv_has_header = no\n',
                 'extra_col_names = None\n']
        for i, key in enumerate(['input_csv_folder', 'cat_csv_name', 'cat_col_names',
                                 'cat_col_nums', 'input_npy_folder', 'csv_has_header',
                                 'extra_col_names', 'extra_col_nums']):
            new_line = ''
            for j in range(i+1):
                new_line = new_line + lines[j]
            _replace_line(os.path.join(os.path.dirname(__file__),
                          'data/cat_a_params.txt'), idx, new_line,
                          out_file=os.path.join(os.path.dirname(__file__),
                                                'data/cat_a_params_.txt'))
            with pytest.raises(ValueError, match=f'Missing key {key} from catalogue "a"'):
                cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                     'data/crossmatch_params__.txt'),
                                     os.path.join(os.path.dirname(__file__),
                                     'data/cat_a_params_.txt'),
                                     os.path.join(os.path.dirname(__file__),
                                     'data/cat_b_params.txt'))

        new_line = ''
        for line in lines:
            new_line = new_line + line
        new_line = new_line + 'extra_col_nums = None\n'
        _replace_line(os.path.join(os.path.dirname(__file__),
                      'data/cat_a_params.txt'), idx, new_line,
                      out_file=os.path.join(os.path.dirname(__file__),
                                            'data/cat_a_params_.txt'))
        _replace_line(os.path.join(os.path.dirname(__file__),
                      'data/cat_b_params.txt'), idx, new_line,
                      out_file=os.path.join(os.path.dirname(__file__),
                                            'data/cat_b_params_.txt'))
        old_line = 'input_npy_folder = '
        new_line = 'input_npy_folder = None\n'
        with open(os.path.join(os.path.dirname(__file__), 'data/cat_a_params_.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__),
                      'data/cat_a_params_.txt'), idx, new_line,
                      out_file=os.path.join(os.path.dirname(__file__),
                                            'data/cat_a_params_.txt'))

        # Initially this will fail because we don't have input_csv_folder
        # created, then it will fail without input_npy_folder:
        for error_key, folder, cat_ in zip(['input_csv_folder', 'input_npy_folder'],
                                           ['input_csv_folder', 'blah'], ['a', 'b']):
            if os.path.exists(folder):
                os.rmdir(folder)
            with pytest.raises(OSError, match=f'{error_key} from catalogue "{cat_}" does '):
                cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                     'data/crossmatch_params__.txt'),
                                     os.path.join(os.path.dirname(__file__),
                                     'data/cat_a_params_.txt'),
                                     os.path.join(os.path.dirname(__file__),
                                     'data/cat_b_params_.txt'))
            os.makedirs(folder, exist_ok=True)

        # At this point we should successfully load the csv-related parameters.
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                             'data/crossmatch_params__.txt'),
                             os.path.join(os.path.dirname(__file__),
                             'data/cat_a_params_.txt'),
                             os.path.join(os.path.dirname(__file__),
                             'data/cat_b_params_.txt'))
        assert cm.output_csv_folder == os.path.abspath('output_csv_folder')
        assert cm.match_out_csv_name == 'match.csv'
        assert cm.b_nonmatch_out_csv_name == 'WISE_nonmatch.csv'

        assert cm.b_input_csv_folder == os.path.abspath('input_csv_folder')
        assert cm.a_cat_csv_name == 'catalogue.csv'
        assert np.all(cm.a_cat_col_names == np.array(['Gaia_A', 'Gaia_B', 'Gaia_C']))
        assert np.all(cm.b_cat_col_nums == np.array([1, 2, 3]))
        assert cm.a_input_npy_folder is None
        assert cm.b_input_npy_folder == os.path.abspath('blah')
        assert cm.a_csv_has_header is False
        assert cm.a_extra_col_names is None
        assert cm.a_extra_col_nums is None

        # Check for various input points of failure:
        for old_line, new_line, error_msg, err_type, cfg_type in zip(
                ['output_csv_folder = output_csv_folder',
                 'cat_col_nums = 1 2 3', 'cat_col_nums = 1 2 3', 'cat_col_nums = 1 2 3',
                 'extra_col_names = None'],
                ['output_csv_folder = /Volume/test/path/that/fails\n',
                 'cat_col_nums = 1 2 A\n', 'cat_col_nums = 1 2 3 4\n', 'cat_col_nums = 1 2 3.4\n',
                 'extra_col_names = D F G\n'],
                ['Error when trying to create ',
                 'cat_col_nums should be a list of integers in catalogue "b"',
                 'a_cat_col_names and a_cat_col_nums should contain the same',
                 'All elements of a_cat_col_nums', 'Both extra_col_names and extra_col_nums must '
                 'be None if either is None in catalogue "a"'],
                [OSError, ValueError, ValueError, ValueError, ValueError],
                ['j', 'b', 'a', 'a', 'a']):
            j = 'crossmatch_params_3' if cfg_type == 'j' else 'crossmatch_params__'
            a = 'cat_a_params_2' if cfg_type == 'a' else 'cat_a_params_'
            b = 'cat_b_params_2' if cfg_type == 'b' else 'cat_b_params_'
            if cfg_type == 'j':
                load_old = 'crossmatch_params__'
                load_new = 'crossmatch_params_3'
            elif cfg_type == 'a':
                load_old = 'cat_a_params_'
                load_new = 'cat_a_params_2'
            elif cfg_type == 'b':
                load_old = 'cat_b_params_'
                load_new = 'cat_b_params_2'
            with open(os.path.join(os.path.dirname(__file__), f'data/{load_old}.txt'),
                      encoding='utf-8') as file:
                f = file.readlines()
            idx = np.where([old_line in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__), f'data/{load_old}.txt'), idx, new_line,
                          out_file=os.path.join(os.path.dirname(__file__), f'data/{load_new}.txt'))
            with pytest.raises(err_type, match=error_msg):
                cm._initialise_chunk(os.path.join(os.path.dirname(__file__), f'data/{j}.txt'),
                                     os.path.join(os.path.dirname(__file__), f'data/{a}.txt'),
                                     os.path.join(os.path.dirname(__file__), f'data/{b}.txt'))

        # Finally, to check for extra_col_* issues, we need to set both
        # to not None first.
        old_line = 'extra_col_names = None'
        new_line = 'extra_col_names = D E F\n'
        with open(os.path.join(os.path.dirname(__file__), 'data/cat_a_params_.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_a_params_.txt'), idx,
                      new_line)
        old_line = 'extra_col_nums = None'
        new_line = 'extra_col_nums = 1 2 3\n'
        with open(os.path.join(os.path.dirname(__file__), 'data/cat_a_params_.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_a_params_.txt'), idx,
                      new_line)
        # First check this passes fine
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                             'data/crossmatch_params__.txt'),
                             os.path.join(os.path.dirname(__file__),
                             'data/cat_a_params_.txt'),
                             os.path.join(os.path.dirname(__file__),
                             'data/cat_b_params_.txt'))
        assert np.all(cm.a_extra_col_names == np.array(['Gaia_D', 'Gaia_E', 'Gaia_F']))
        assert np.all(cm.a_extra_col_nums == np.array([1, 2, 3]))
        # Then check for issues correctly being raised
        for old_line, new_line, error_msg in zip(
                ['extra_col_nums = 1 2 3', 'extra_col_nums = 1 2 3', 'extra_col_nums = 1 2 3'],
                ['extra_col_nums = 1 A 3\n', 'extra_col_nums = 1 2 3 4\n',
                 'extra_col_nums = 1 2 3.1\n'],
                ['extra_col_nums should be a list of integers in catalogue "a"',
                 'a_extra_col_names and a_extra_col_nums should contain the same number',
                 'All elements of a_extra_col_nums should be integers']):
            with open(os.path.join(os.path.dirname(__file__), 'data/cat_a_params_.txt'),
                      encoding='utf-8') as file:
                f = file.readlines()
            idx = np.where([old_line in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__),
                          'data/cat_a_params_.txt'), idx, new_line,
                          out_file=os.path.join(os.path.dirname(__file__),
                          'data/cat_a_params_2.txt'))
            with pytest.raises(ValueError, match=error_msg):
                cm._initialise_chunk(
                    os.path.join(os.path.dirname(__file__), 'data/crossmatch_params__.txt'),
                    os.path.join(os.path.dirname(__file__), 'data/cat_a_params_2.txt'),
                    os.path.join(os.path.dirname(__file__), 'data/cat_b_params_.txt'))

    @pytest.mark.remote_data
    # pylint: disable-next=too-many-lines,too-many-branches,too-many-statements,too-many-locals
    def test_crossmatch_correct_astrometry_inputs(self):
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'))
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        assert cm.n_pool == 2
        assert cm.a_correct_astrometry is False
        assert cm.b_correct_astrometry is False
        assert cm.a_compute_snr_mag_relation is False

        for cat_n in ['a', 'b']:
            old_line = "correct_astrometry = no"
            new_line = "correct_astrometry = yes\n"
            with open(os.path.join(os.path.dirname(__file__), f'data/cat_{cat_n}_params.txt'),
                      encoding='utf-8') as file:
                f = file.readlines()
            idx = np.where([old_line in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__),
                          f'data/cat_{cat_n}_params.txt'), idx, new_line,
                          out_file=os.path.join(os.path.dirname(__file__), f'data/cat_{cat_n}_params_.txt'))
            old_line = "compute_snr_mag_relation = no"
            new_line = "compute_snr_mag_relation = yes\n"
            idx = np.where([old_line in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__),
                          f'data/cat_{cat_n}_params_.txt'), idx, new_line)
            with pytest.raises(ValueError, match=f"Ambiguity in catalogue '{cat_n}' hav"):
                cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                                     os.path.join(os.path.dirname(__file__),
                                                  f"data/cat_a_params{'_' if 'a' in cat_n else ''}.txt"),
                                     os.path.join(os.path.dirname(__file__),
                                                  f"data/cat_b_params{'_' if 'b' in cat_n else ''}.txt"))

        old_line = "n_pool = 2"
        new_line = ""
        with open(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__),
                      'data/crossmatch_params.txt'), idx, new_line,
                      out_file=os.path.join(os.path.dirname(__file__),
                      'data/crossmatch_params_.txt'))
        with pytest.raises(ValueError, match="Missing key n_pool from joint"):
            cm._initialise_chunk(
                os.path.join(os.path.dirname(__file__), 'data/crossmatch_params_.txt'),
                os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        old_line = "correct_astrometry = no"
        new_line = ""
        with open(os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'), encoding='utf-8') as file:
            f = file.readlines()
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__),
                      'data/cat_b_params.txt'), idx, new_line,
                      out_file=os.path.join(os.path.dirname(__file__),
                      'data/cat_b_params_.txt'))
        with pytest.raises(ValueError, match='Missing key correct_astrometry from catalogue "b"'):
            cm._initialise_chunk(
                os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                os.path.join(os.path.dirname(__file__), 'data/cat_b_params_.txt'))

        old_line, new_line = "correct_astrometry = no", "correct_astrometry = yes\n"
        for x in ['a', 'b']:
            with open(os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'),
                      encoding='utf-8') as file:
                f = file.readlines()
            idx = np.where([old_line in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__), f'data/cat_{x}_params.txt'), idx, new_line,
                          out_file=os.path.join(os.path.dirname(__file__), f'data/cat_{x}_params_.txt'))

        old_line = "n_pool = 2"
        error_msg = "n_pool should be a single integer number."
        for new_line in ["n_pool = A\n", "n_pool = 1 2\n", "n_pool = 1.5\n"]:
            with open(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                      encoding='utf-8') as file:
                f = file.readlines()
            idx = np.where([old_line in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__),
                          'data/crossmatch_params.txt'), idx, new_line,
                          out_file=os.path.join(os.path.dirname(__file__),
                          'data/crossmatch_params_2.txt'))
            with pytest.raises(ValueError, match=error_msg):
                cm._initialise_chunk(
                    os.path.join(os.path.dirname(__file__), 'data/crossmatch_params_2.txt'),
                    os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                    os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))

        # Fake dd_params and l_cut
        ddp = np.ones((5, 15, 2), float)
        np.save('dd_params.npy', ddp)
        lc = np.ones(3, float)
        np.save('l_cut.npy', lc)

        with open(os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'), encoding='utf-8') as file:
            f = file.readlines()
        old_line = 'fit_gal_flag = no'
        new_line = ('gal_wavs = 0.513 0.641 0.778\n'
                    'gal_zmax = 4.5 4.5 5\ngal_nzs = 46 46 51\n'
                    'gal_aboffsets = 0.5 0.5 0.5\n'
                    'gal_filternames = gaiadr2-BP gaiadr2-G gaiadr2-RP\n'
                    'saturation_magnitudes = 5 5 5\n')
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_a_params_.txt'), idx,
                      new_line, out_file=os.path.join(os.path.dirname(__file__), 'data/cat_a_params_.txt'))
        with open(os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'), encoding='utf-8') as file:
            f = file.readlines()
        old_line = 'fit_gal_flag = no'
        new_line = ('gal_wavs = 3.37 4.62 12.08 22.19\n'
                    'gal_zmax = 3.2 4.0 1 4\ngal_nzs = 33 41 11 41\n'
                    'gal_aboffsets = 0.5 0.5 0.5 0.5\n'
                    'gal_filternames = wise2010-W1 wise2010-W2 wise2010-W3 wise2010-W4\n'
                    'saturation_magnitudes = 5 5 5 5\n')
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_b_params_.txt'), idx,
                      new_line, out_file=os.path.join(os.path.dirname(__file__),
                      'data/cat_b_params_.txt'))

        with open(os.path.join(os.path.dirname(__file__), 'data/cat_a_params_.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        old_line = 'auf_folder_path = gaia_auf_folder'
        new_line = ('auf_folder_path = gaia_auf_folder\ndens_hist_tri_location = None\n'
                    'tri_model_mags_location = None\ntri_model_mag_mids_location = None\n'
                    'tri_model_mags_interval_location = None\ntri_dens_uncert_location = None\n'
                    'tri_n_bright_sources_star_location = None\n')
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_a_params_.txt'), idx, new_line)
        with open(os.path.join(os.path.dirname(__file__), 'data/cat_b_params_.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        old_line = 'auf_folder_path = wise_auf_folder'
        new_line = ('auf_folder_path = wise_auf_folder\ndens_hist_tri_location = None\n'
                    'tri_model_mags_location = None\ntri_model_mag_mids_location = None\n'
                    'tri_model_mags_interval_location = None\ntri_dens_uncert_location = None\n'
                    'tri_n_bright_sources_star_location = None\n')
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_b_params_.txt'), idx, new_line)

        # Test all of the inputs being needed one by one loading into cat_a_params:
        dd_l_path = os.path.join(os.path.dirname(__file__), 'data')
        old_line = 'correct_astrometry = yes\n'
        with open(os.path.join(os.path.dirname(__file__), 'data/cat_a_params_.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        idx = np.where([old_line in line for line in f])[0][0]
        lines = [f'correct_astrometry = yes\n\ndd_params_path = {dd_l_path}\nl_cut_path = {dd_l_path}\n',
                 'correct_astro_save_folder = ac_folder\n', 'csv_cat_file_string = file_{}.csv\n',
                 'pos_and_err_indices = 0 1 2 0 1 2\n', 'mag_indices = 3 5 7\n', 'mag_unc_indices = 4 6 8\n',
                 'best_mag_index = 0\n', 'nn_radius = 30\n', 'ref_csv_cat_file_string = ref_{}.csv\n',
                 'correct_mag_array = 14.07 14.17 14.27 14.37 14.47\n',
                 'correct_mag_slice = 0.05 0.05 0.05 0.05 0.05\n',
                 'correct_sig_slice = 0.1 0.1 0.1 0.1 0.1\n', 'chunk_overlap_col = None\n',
                 'best_mag_index_col = 8\n', 'use_photometric_uncertainties = no\n',
                 'mn_fit_type = quadratic\n', 'seeing_ranges = 0.9 1.1\n']
        for i, key in enumerate(['correct_astro_save_folder', 'csv_cat_file_string',
                                 'pos_and_err_indices', 'mag_indices', 'mag_unc_indices',
                                 'best_mag_index', 'nn_radius', 'ref_csv_cat_file_string',
                                 'correct_mag_array', 'correct_mag_slice', 'correct_sig_slice',
                                 'chunk_overlap_col', 'best_mag_index_col',
                                 'use_photometric_uncertainties', 'mn_fit_type', 'seeing_ranges']):
            new_line = ''
            for j in range(i+1):
                new_line = new_line + lines[j]
            _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_a_params_.txt'), idx, new_line,
                          out_file=os.path.join(os.path.dirname(__file__), 'data/cat_a_params_2.txt'))
            with pytest.raises(ValueError, match=f'Missing key {key} from catalogue "a"'):
                cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                                     os.path.join(os.path.dirname(__file__), 'data/cat_a_params_2.txt'),
                                     os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        # Test use_photometric_uncertainties for failure.
        new_line = ''
        for line in lines:
            new_line = new_line + line
        _replace_line(os.path.join(os.path.dirname(__file__),
                      'data/cat_a_params_.txt'), idx, new_line,
                      out_file=os.path.join(os.path.dirname(__file__),
                                            'data/cat_a_params_2.txt'))
        old_line = 'use_photometric_uncertainties = no'
        new_line = 'use_photometric_uncertainties = something else\n'
        with open(os.path.join(os.path.dirname(__file__), 'data/cat_a_params_2.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__),
                      'data/cat_a_params_2.txt'), idx, new_line,
                      out_file=os.path.join(os.path.dirname(__file__),
                                            'data/cat_a_params_2c.txt'))
        with pytest.raises(ValueError, match='Boolean flag key not set to allowed'):
            cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                 'data/crossmatch_params.txt'),
                                 os.path.join(os.path.dirname(__file__),
                                 'data/cat_a_params_2c.txt'),
                                 os.path.join(os.path.dirname(__file__),
                                 'data/cat_b_params.txt'))
        old_line = 'mn_fit_type = quadratic'
        new_line = 'mn_fit_type = something else\n'
        with open(os.path.join(os.path.dirname(__file__), 'data/cat_a_params_2.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__),
                      'data/cat_a_params_2.txt'), idx, new_line,
                      out_file=os.path.join(os.path.dirname(__file__),
                                            'data/cat_a_params_2d.txt'))
        with pytest.raises(ValueError, match="mn_fit_type must be 'quadratic' or 'linear' in"):
            cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                 'data/crossmatch_params.txt'),
                                 os.path.join(os.path.dirname(__file__),
                                 'data/cat_a_params_2d.txt'),
                                 os.path.join(os.path.dirname(__file__),
                                 'data/cat_b_params.txt'))
        old_line = 'seeing_ranges = 0.9'
        new_line = 'seeing_ranges = a\n'
        with open(os.path.join(os.path.dirname(__file__), 'data/cat_a_params_2.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__),
                      'data/cat_a_params_2.txt'), idx, new_line,
                      out_file=os.path.join(os.path.dirname(__file__),
                                            'data/cat_a_params_2e.txt'))
        with pytest.raises(ValueError, match="seeing_ranges must be a 1-D list or array of ints, length 1, "):
            cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                 'data/crossmatch_params.txt'),
                                 os.path.join(os.path.dirname(__file__),
                                 'data/cat_a_params_2e.txt'),
                                 os.path.join(os.path.dirname(__file__),
                                 'data/cat_b_params.txt'))
        old_line = 'seeing_ranges = 0.9'
        new_line = 'seeing_ranges = 1 2 3 4\n'
        with open(os.path.join(os.path.dirname(__file__), 'data/cat_a_params_2.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__),
                      'data/cat_a_params_2.txt'), idx, new_line,
                      out_file=os.path.join(os.path.dirname(__file__),
                                            'data/cat_a_params_2f.txt'))
        with pytest.raises(ValueError, match="seeing_ranges must be a 1-D list or array of ints, length 1, "):
            cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                 'data/crossmatch_params.txt'),
                                 os.path.join(os.path.dirname(__file__),
                                 'data/cat_a_params_2f.txt'),
                                 os.path.join(os.path.dirname(__file__),
                                 'data/cat_b_params.txt'))
        # Set up a completely valid test of cat_a_params and cat_b_params
        for x in ['a', 'b']:
            old_line = 'correct_astrometry = yes\n'
            with open(os.path.join(os.path.dirname(__file__), f'data/cat_{x}_params_.txt'),
                      encoding='utf-8') as file:
                f = file.readlines()
            idx = np.where([old_line in line for line in f])[0][0]
            if x == 'b':
                lines[np.where(['mag_indices' in y for y in
                      lines])[0][0]] = 'mag_indices = 3 5 7 9\n'
                lines[np.where(['mag_unc_indices' in y for y in
                      lines])[0][0]] = 'mag_unc_indices = 4 6 8 10\n'
                lines[np.where(['best_mag_index_col' in y for y in
                      lines])[0][0]] = 'best_mag_index_col = 11\n'
                lines[np.where(['chunk_overlap_col' in y for y in
                      lines])[0][0]] = 'chunk_overlap_col = 12\n'
            new_line = ''
            for line in lines:
                new_line = new_line + line
            _replace_line(os.path.join(os.path.dirname(__file__), f'data/cat_{x}_params_.txt'), idx, new_line,
                          out_file=os.path.join(os.path.dirname(__file__), f'data/cat_{x}_params_2.txt'))
        # Fake some TRILEGAL downloads with random data.
        os.makedirs('wise_auf_folder/134.5/0.0', exist_ok=True)
        text = ('#area = 4.0 sq deg\n#Av at infinity = 1\n' +
                'Gc logAge [M/H] m_ini   logL   logTe logg  m-M0   Av    ' +
                'm2/m1 mbol   J      H      Ks     IRAC_3.6 IRAC_4.5 IRAC_5.8 IRAC_8.0 MIPS_24 ' +
                'MIPS_70 MIPS_160 W1     W2     W3     W4       Mact\n')
        self.rng = np.random.default_rng(seed=67235589)
        w1s = self.rng.uniform(13.5, 15.5, size=1000)
        for w1 in w1s:
            text = text + (
                '1   6.65 -0.39  0.02415 -2.701 3.397  4.057 14.00  8.354 0.00 25.523 25.839 ' +
                '24.409 23.524 22.583 22.387 22.292 22.015 21.144 19.380 20.878 '
                f'{w1} 22.391 21.637 21.342  0.024\n ')
        with open('wise_auf_folder/134.5/0.0/trilegal_auf_simulation_faint.dat', "w", encoding='utf-8') as f:
            f.write(text)
        # Fake some "real" csv data
        ax1_min, ax1_max, ax2_min, ax2_max = 100, 110, -3, 3
        cat_args = (1,)
        t_a_c = TAC()
        t_a_c.npy_or_csv = 'csv'
        t_a_c.n = 5000
        t_a_c.rng = self.rng
        choice = t_a_c.rng.choice(t_a_c.n, size=t_a_c.n, replace=False)
        t_a_c.true_ra = np.linspace(100, 110, t_a_c.n)[choice]
        t_a_c.true_dec = np.linspace(-3, 3, t_a_c.n)[choice]
        t_a_c.a_cat_name = 'ref_{}.csv'
        t_a_c.b_cat_name = 'file_{}.csv'
        t_a_c.fake_cata_cutout(ax1_min, ax1_max, ax2_min, ax2_max, *cat_args)
        t_a_c.fake_catb_cutout(ax1_min, ax1_max, ax2_min, ax2_max, *cat_args)
        # Re-fake data with multiple magnitude columns.
        x = np.loadtxt('ref_1.csv', delimiter=',')
        y = np.empty((len(x), 11), float)
        y[:, [0, 1, 2]] = x[:, [0, 1, 2]]
        y[:, [3, 4]] = x[:, [3, 4]]
        y[:, [5, 6]] = x[:, [3, 4]]
        y[:, [7, 8]] = x[:, [3, 4]]
        # Pad with both a best index and chunk overlap column
        y[:, 9] = 2
        y[:, 10] = 1
        np.savetxt('ref_1.csv', y, delimiter=',')
        x = np.loadtxt('file_1.csv', delimiter=',')
        y = np.empty((len(x), 13), float)
        y[:, [0, 1, 2]] = x[:, [0, 1, 2]]
        y[:, [3, 4]] = x[:, [3, 4]]
        y[:, [5, 6]] = x[:, [3, 4]]
        y[:, [7, 8]] = x[:, [3, 4]]
        y[:, [9, 10]] = x[:, [3, 4]]
        y[:, 11] = np.random.default_rng(seed=5673523).choice(4, size=len(x), replace=True)
        y[:, 12] = np.random.default_rng(seed=45645132234).choice(2, size=len(x), replace=True)
        np.savetxt('file_1.csv', y, delimiter=',')
        # Check for outputs, but first force the removal of ancillary checkpoints.
        if os.path.isfile('ac_folder/npy/snr_mag_params.npy'):
            os.remove('ac_folder/npy/snr_mag_params.npy')
        # Using the ORIGINAL cat_a_params means we don't fit for corrections
        # to catalogue 'a'.
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'))
        cm.chunk_id = 1

        old_line = 'real_hankel_points = 10000'
        new_line = 'real_hankel_points = 1000\n'
        with open(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__),
                      'data/crossmatch_params.txt'), idx, new_line,
                      out_file=os.path.join(os.path.dirname(__file__),
                                            'data/crossmatch_params_v2.txt'))
        old_line = 'four_max_rho = 100'
        new_line = 'four_max_rho = 30\n'
        with open(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params_v2.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__),
                      'data/crossmatch_params_v2.txt'), idx, new_line)
        old_line = 'four_hankel_points = 10000'
        new_line = 'four_hankel_points = 1000\n'
        with open(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params_v2.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__),
                      'data/crossmatch_params_v2.txt'), idx, new_line)

        cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                             'data/crossmatch_params_v2.txt'),
                             os.path.join(os.path.dirname(__file__),
                             'data/cat_a_params.txt'),
                             os.path.join(os.path.dirname(__file__),
                             'data/cat_b_params_2.txt'))
        # pylint: disable=no-member
        assert cm.b_best_mag_index == 0
        assert_allclose(cm.b_nn_radius, 30)
        assert cm.b_correct_astro_save_folder == os.path.abspath('ac_folder')
        assert cm.b_csv_cat_file_string == os.path.abspath('file_{}.csv')
        assert cm.b_ref_csv_cat_file_string == os.path.abspath('ref_{}.csv')
        assert_allclose(cm.b_correct_mag_array, np.array([14.07, 14.17, 14.27, 14.37, 14.47]))
        assert_allclose(cm.b_correct_mag_slice, np.array([0.05, 0.05, 0.05, 0.05, 0.05]))
        assert_allclose(cm.b_correct_sig_slice, np.array([0.1, 0.1, 0.1, 0.1, 0.1]))
        assert np.all(cm.b_pos_and_err_indices == np.array([[0, 1, 2], [0, 1, 2]]))
        assert np.all(cm.b_mag_indices == np.array([3, 5, 7, 9]))
        assert np.all(cm.b_mag_unc_indices == np.array([4, 6, 8, 10]))
        marray = np.load('ac_folder/npy/m_sigs_array.npy')
        narray = np.load('ac_folder/npy/n_sigs_array.npy')
        assert_allclose([marray[0], narray[0]], [2, 0], rtol=0.1, atol=0.01)
        # pylint: enable=no-member

        assert np.all(np.load('wise_folder/in_chunk_overlap.npy') == y[:, 12].astype(int))

        old_line = 'chunk_overlap_col = '
        new_line = 'chunk_overlap_col = None\n'
        with open(os.path.join(os.path.dirname(__file__), 'data/cat_b_params_2.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__),
                      'data/cat_b_params_2.txt'), idx, new_line)

        if os.path.isfile('ac_folder/npy/snr_mag_params.npy'):
            os.remove('ac_folder/npy/snr_mag_params.npy')
        # Swapped a+b to test a_* versions of things
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'))
        cm.chunk_id = 1
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                             'data/crossmatch_params_v2.txt'),
                             os.path.join(os.path.dirname(__file__),
                             'data/cat_b_params_2.txt'),
                             os.path.join(os.path.dirname(__file__),
                             'data/cat_a_params.txt'))
        # pylint: disable=no-member
        assert cm.a_best_mag_index == 0
        assert_allclose(cm.a_nn_radius, 30)
        assert cm.a_correct_astro_save_folder == os.path.abspath('ac_folder')
        assert cm.a_csv_cat_file_string == os.path.abspath('file_{}.csv')
        assert cm.a_ref_csv_cat_file_string == os.path.abspath('ref_{}.csv')
        assert_allclose(cm.a_correct_mag_array, np.array([14.07, 14.17, 14.27, 14.37, 14.47]))
        assert_allclose(cm.a_correct_mag_slice, np.array([0.05, 0.05, 0.05, 0.05, 0.05]))
        assert_allclose(cm.a_correct_sig_slice, np.array([0.1, 0.1, 0.1, 0.1, 0.1]))
        assert np.all(cm.a_pos_and_err_indices == np.array([[0, 1, 2], [0, 1, 2]]))
        assert np.all(cm.a_mag_indices == np.array([3, 5, 7, 9]))
        assert np.all(cm.a_mag_unc_indices == np.array([4, 6, 8, 10]))
        marray = np.load('ac_folder/npy/m_sigs_array.npy')
        narray = np.load('ac_folder/npy/n_sigs_array.npy')
        assert_allclose([marray[0], narray[0]], [2, 0], rtol=0.1, atol=0.01)
        # pylint: enable=no-member

        assert np.all(np.load('wise_folder/in_chunk_overlap.npy') == 0)

        # Set up a completely valid test of cat_a_params and cat_b_params
        # for compute_snr_mag_relation.
        lines = [
            'compute_snr_mag_relation = yes\n', 'correct_astro_save_folder = ac_folder\n',
            'csv_cat_file_string = file_{}.csv\n', 'pos_and_err_indices = 0 1 2\n',
            'mag_indices = 3 5 7\n', 'mag_unc_indices = 4 6 8\n']
        for x in ['a', 'b']:
            old_line = 'compute_snr_mag_relation = no'
            with open(os.path.join(os.path.dirname(__file__), f'data/cat_{x}_params.txt'),
                      encoding='utf-8') as file:
                f = file.readlines()
            idx = np.where([old_line in line for line in f])[0][0]
            if x == 'b':
                lines[np.where(['mag_indices' in y for y in
                      lines])[0][0]] = 'mag_indices = 3 5 7 9\n'
                lines[np.where(['mag_unc_indices' in y for y in
                      lines])[0][0]] = 'mag_unc_indices = 4 6 8 10\n'
            new_line = ''
            for line in lines:
                new_line = new_line + line
            _replace_line(os.path.join(os.path.dirname(__file__), f'data/cat_{x}_params.txt'), idx, new_line,
                          out_file=os.path.join(os.path.dirname(__file__), f'data/cat_{x}_params_2b.txt'))
        if os.path.isfile('ac_folder/npy/snr_mag_params.npy'):
            os.remove('ac_folder/npy/snr_mag_params.npy')
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'))
        cm.chunk_id = 1
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                             'data/crossmatch_params_v2.txt'),
                             os.path.join(os.path.dirname(__file__),
                             'data/cat_a_params.txt'),
                             os.path.join(os.path.dirname(__file__),
                             'data/cat_b_params_2b.txt'))
        # pylint: disable=no-member
        assert cm.b_compute_snr_mag_relation is True
        assert not hasattr(cm, 'b_correct_mag_slice')
        assert cm.b_correct_astro_save_folder == os.path.abspath('ac_folder')
        assert cm.b_csv_cat_file_string == os.path.abspath('file_{}.csv')
        assert np.all(cm.b_pos_and_err_indices == np.array([[0, 1, 2], [0, 1, 2]]))
        assert np.all(cm.b_mag_indices == np.array([3, 5, 7, 9]))
        assert np.all(cm.b_mag_unc_indices == np.array([4, 6, 8, 10]))
        marray = np.load('ac_folder/npy/m_sigs_array.npy')
        narray = np.load('ac_folder/npy/n_sigs_array.npy')
        assert_allclose([marray[0], narray[0]], [2, 0], rtol=0.1, atol=0.01)
        if os.path.isfile('ac_folder/npy/snr_mag_params.npy'):
            os.remove('ac_folder/npy/snr_mag_params.npy')
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'))
        cm.chunk_id = 1
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                             'data/crossmatch_params_v2.txt'),
                             os.path.join(os.path.dirname(__file__),
                             'data/cat_a_params_2b.txt'),
                             os.path.join(os.path.dirname(__file__),
                             'data/cat_b_params.txt'))
        assert cm.a_compute_snr_mag_relation is True
        assert not hasattr(cm, 'a_best_mag_index')
        assert cm.a_correct_astro_save_folder == os.path.abspath('ac_folder')
        assert cm.a_csv_cat_file_string == os.path.abspath('file_{}.csv')
        assert np.all(cm.a_pos_and_err_indices == np.array([[0, 1, 2], [0, 1, 2]]))
        assert np.all(cm.a_mag_indices == np.array([3, 5, 7]))
        assert np.all(cm.a_mag_unc_indices == np.array([4, 6, 8]))
        marray = np.load('ac_folder/npy/m_sigs_array.npy')
        narray = np.load('ac_folder/npy/n_sigs_array.npy')
        assert_allclose([marray[0], narray[0]], [2, 0], rtol=0.1, atol=0.01)

        # New test of the AC run, just with pre-made histograms.
        with open(os.path.join(os.path.dirname(__file__), 'data/cat_b_params_2.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        dens, tri_mags, tri_mags_mids, dtri_mags, uncert, num_bright_obj = make_tri_counts(
            'wise_auf_folder', '134.5/0.0/trilegal_auf_simulation', 'W1', 0.1, 13.5, 16)
        dhtl = 'ac_folder/npy/dhtl.npy'
        np.save(dhtl, [dens, dens, dens, dens])
        tmml = 'ac_folder/npy/tmml.npy'
        np.save(tmml, [tri_mags, tri_mags, tri_mags, tri_mags])
        tmmml = 'ac_folder/npy/tmmml.npy'
        np.save(tmmml, [tri_mags_mids, tri_mags_mids, tri_mags_mids, tri_mags_mids])
        tmmil = 'ac_folder/npy/tmmil.npy'
        np.save(tmmil, [dtri_mags, dtri_mags, dtri_mags, dtri_mags])
        tdul = 'ac_folder/npy/tdul.npy'
        np.save(tdul, [uncert, uncert, uncert, uncert])
        tnbssl = 'ac_folder/npy/tnbssl.npy'
        np.save(tnbssl, [num_bright_obj, num_bright_obj, num_bright_obj, num_bright_obj])

        ol, nl = 'auf_folder_path = ', 'auf_folder_path = None\n'
        idx = np.where([ol in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_b_params_2.txt'), idx, nl,
                      out_file=os.path.join(os.path.dirname(__file__), 'data/cat_b_params_2c.txt'))
        for ol, nl in zip(['tri_set_name = ', 'tri_filt_names = ', 'tri_filt_num = ', 'download_tri = ',
                           'tri_maglim_faint = ', 'tri_num_faint = ', 'dens_hist_tri_location = ',
                           'tri_model_mags_location = ', 'tri_model_mag_mids_location = ',
                           'tri_model_mags_interval_location = ', 'tri_dens_uncert_location = ',
                           'tri_n_bright_sources_star_location = '], [
                'tri_set_name = None\n', 'tri_filt_names = None\n', 'tri_filt_num = None\n',
                'download_tri = None\n', 'tri_maglim_faint = None\n', 'tri_num_faint = None\n',
                f'dens_hist_tri_location = {dhtl}\n', f'tri_model_mags_location = {tmml}\n',
                f'tri_model_mag_mids_location = {tmmml}\n', f'tri_model_mags_interval_location = {tmmil}\n',
                f'tri_dens_uncert_location = {tdul}\n', f'tri_n_bright_sources_star_location = {tnbssl}\n']):
            idx = np.where([ol in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_b_params_2c.txt'), idx, nl)
        if os.path.isfile('ac_folder/npy/snr_mag_params.npy'):
            os.remove('ac_folder/npy/snr_mag_params.npy')
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'))
        cm.chunk_id = 1
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                             'data/crossmatch_params_v2.txt'),
                             os.path.join(os.path.dirname(__file__),
                             'data/cat_b_params_2c.txt'),
                             os.path.join(os.path.dirname(__file__),
                             'data/cat_a_params.txt'))
        assert cm.a_best_mag_index == 0
        assert_allclose(cm.a_nn_radius, 30)
        assert cm.a_correct_astro_save_folder == os.path.abspath('ac_folder')
        assert cm.a_csv_cat_file_string == os.path.abspath('file_{}.csv')
        assert cm.a_ref_csv_cat_file_string == os.path.abspath('ref_{}.csv')
        assert_allclose(cm.a_correct_mag_array, np.array([14.07, 14.17, 14.27, 14.37, 14.47]))
        assert_allclose(cm.a_correct_mag_slice, np.array([0.05, 0.05, 0.05, 0.05, 0.05]))
        assert_allclose(cm.a_correct_sig_slice, np.array([0.1, 0.1, 0.1, 0.1, 0.1]))
        assert np.all(cm.a_pos_and_err_indices == np.array([[0, 1, 2], [0, 1, 2]]))
        assert np.all(cm.a_mag_indices == np.array([3, 5, 7, 9]))
        assert np.all(cm.a_mag_unc_indices == np.array([4, 6, 8, 10]))
        marray = np.load('ac_folder/npy/m_sigs_array.npy')
        narray = np.load('ac_folder/npy/n_sigs_array.npy')
        assert_allclose([marray[0], narray[0]], [2, 0], rtol=0.1, atol=0.01)
        # pylint: enable=no-member

        # Dummy folder that won't contain l_cut.npy
        os.makedirs('./l_cut_dummy_folder', exist_ok=True)

        for old_line, new_line, x, match_text in zip(
                ['best_mag_index = ', 'best_mag_index = ', 'best_mag_index = ',
                 'nn_radius = ', 'nn_radius = ', 'correct_mag_array = ',
                 'correct_mag_slice = ', 'correct_mag_slice = ', 'correct_sig_slice = ',
                 'correct_sig_slice = ', 'pos_and_err_indices = ', 'pos_and_err_indices = ',
                 'pos_and_err_indices = ', 'mag_indices =', 'mag_indices =', 'mag_indices =',
                 'mag_unc_indices =', 'mag_unc_indices =', 'mag_unc_indices =',
                 'chunk_overlap_col = ', 'chunk_overlap_col = ', 'chunk_overlap_col = ',
                 'best_mag_index_col = ', 'best_mag_index_col = ', 'dd_params_path = ',
                 'l_cut_path = '],
                ['best_mag_index = A\n', 'best_mag_index = 2.5\n', 'best_mag_index = 7\n',
                 'nn_radius = A\n', 'nn_radius = 1 2\n', 'correct_mag_array = 1 2 A 4 5\n',
                 'correct_mag_slice = 0.1 0.1 0.1 A 0.1\n', 'correct_mag_slice = 0.1 0.1 0.1\n',
                 'correct_sig_slice = 0.1 0.1 0.1 A 0.1\n', 'correct_sig_slice = 0.1 0.1 0.1\n',
                 'pos_and_err_indices = 1 2 3 4 5 A\n', 'pos_and_err_indices = 1 2 3 4 5.5 6\n',
                 'pos_and_err_indices = 1 2 3 4 5\n', 'mag_indices = A 1 2\n',
                 'mag_indices = 1 2\n', 'mag_indices = 1.2 2 3 4\n', 'mag_unc_indices = A 1 2\n',
                 'mag_unc_indices = 1 2\n', 'mag_unc_indices = 1.2 2 3\n',
                 'chunk_overlap_col = Non\n', 'chunk_overlap_col = A\n',
                 'chunk_overlap_col = 1.2\n', 'best_mag_index_col = A\n',
                 'best_mag_index_col = 1.2\n', 'dd_params_path = ./some_folder\n',
                 'l_cut_path = ./l_cut_dummy_folder\n'],
                ['a', 'b', 'a', 'b', 'a', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'a', 'a', 'b', 'b',
                 'b', 'a', 'a', 'a', 'b', 'a', 'a', 'b', 'b', 'a'],
                ['best_mag_index should be an integer in the catalogue "a"',
                 'best_mag_index should be an integer in the catalogue "b"',
                 'best_mag_index cannot be a larger index than the list of filters '
                 'in the catalogue "a', 'nn_radius must be a float in the catalogue "b"',
                 'nn_radius must be a float in the catalogue "a"',
                 'correct_mag_array should be a list of floats in the catalogue "a"',
                 'correct_mag_slice should be a list of floats in the catalogue "b"',
                 'a_correct_mag_array and a_correct_mag_slice should contain the same',
                 'correct_sig_slice should be a list of floats in the catalogue "b"',
                 'a_correct_mag_array and a_correct_sig_slice should contain the same',
                 'pos_and_err_indices should be a list of integers in the catalogue "b"',
                 'All elements of a_pos_and_err_indices should be integers',
                 'a_pos_and_err_indices should contain six elements when correct_astrometry',
                 'mag_indices should be a list of integers in the catalogue "a" ',
                 'b_filt_names and b_mag_indices should contain the',
                 'All elements of b_mag_indices should be integers.',
                 'mag_unc_indices should be a list of integers in the catalogue "b" ',
                 'a_mag_unc_indices and a_mag_indices should contain the',
                 'All elements of a_mag_unc_indices should be integers.',
                 'chunk_overlap_col should be an integer in the catalogue "a"',
                 'chunk_overlap_col should be an integer in the catalogue "b"',
                 'chunk_overlap_col should be an integer in the catalogue "a"',
                 'best_mag_index_col should be an integer in the catalogue "a"',
                 'best_mag_index_col should be an integer in the catalogue "b"',
                 'b_dd_params_path does not exist. Please ensure that path for catalogue "b"',
                 'l_cut file not found in catalogue "a" path. Please ensure PSF ']):
            with open(os.path.join(os.path.dirname(__file__), f'data/cat_{x}_params_2.txt'),
                      encoding='utf-8') as file:
                f = file.readlines()
            idx = np.where([old_line in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__), f'data/cat_{x}_params_2.txt'), idx,
                          new_line, out_file=os.path.join(os.path.dirname(__file__),
                                                          f'data/cat_{x}_params_3.txt'))

            if 'folder' not in new_line:
                type_of_error = ValueError
            elif 'dd_params' in new_line:
                type_of_error = OSError
            else:
                type_of_error = FileNotFoundError
            with pytest.raises(type_of_error, match=match_text):
                cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                     'data/crossmatch_params.txt'),
                                     os.path.join(os.path.dirname(__file__),
                                     f"data/cat_a_params{'_2' if x == 'b' else '_3'}.txt"),
                                     os.path.join(os.path.dirname(__file__),
                                     f"data/cat_b_params{'_2' if x == 'a' else '_3'}.txt"))

        x = 'b'
        old_line = 'correct_astrometry = '
        new_line = 'correct_astrometry = no\n'
        with open(os.path.join(os.path.dirname(__file__), f'data/cat_{x}_params_2.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), f'data/cat_{x}_params_2.txt'), idx, new_line,
                      out_file=os.path.join(os.path.dirname(__file__), f'data/cat_{x}_params_3.txt'))
        old_line = 'compute_snr_mag_relation = '
        new_line = 'compute_snr_mag_relation = yes\n'
        with open(os.path.join(os.path.dirname(__file__), f'data/cat_{x}_params_3.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), f'data/cat_{x}_params_3.txt'), idx, new_line)
        old_line = 'pos_and_err_indices = '
        new_line = 'pos_and_err_indices = 1 2 3 4 5\n'
        match_text = 'b_pos_and_err_indices should contain three elements when compute_snr_mag_relation '
        with open(os.path.join(os.path.dirname(__file__), f'data/cat_{x}_params_3.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), f'data/cat_{x}_params_3.txt'), idx, new_line)
        with pytest.raises(ValueError, match=match_text):
            cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                 'data/crossmatch_params.txt'),
                                 os.path.join(os.path.dirname(__file__),
                                 f"data/cat_a_params{'_2' if x == 'b' else '_3'}.txt"),
                                 os.path.join(os.path.dirname(__file__),
                                 f"data/cat_b_params{'_2' if x == 'a' else '_3'}.txt"))


class TestPostProcess:
    def setup_method(self):
        self.joint_folder_path = os.path.abspath('joint')
        self.a_cat_folder_path = os.path.abspath('a_cat')
        self.b_cat_folder_path = os.path.abspath('b_cat')

        os.makedirs(f'{self.joint_folder_path}', exist_ok=True)
        os.makedirs(self.a_cat_folder_path, exist_ok=True)
        os.makedirs(self.b_cat_folder_path, exist_ok=True)

        na, nb, nmatch = 10000, 7000, 4000
        self.na, self.nb, self.nmatch = na, nb, nmatch

        self.cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'))

        rng = np.random.default_rng(seed=7893467234)
        self.ac = rng.choice(na, size=nmatch, replace=False)
        self.cm.ac = self.ac
        self.bc = rng.choice(nb, size=nmatch, replace=False)
        self.cm.bc = self.bc

        self.af = np.delete(np.arange(na), self.ac)
        self.cm.af = self.af
        self.bf = np.delete(np.arange(nb), self.bc)
        self.cm.bf = self.bf

        np.save(f'{self.a_cat_folder_path}/in_chunk_overlap.npy', rng.choice(2, size=na).astype(bool))
        np.save(f'{self.b_cat_folder_path}/in_chunk_overlap.npy', rng.choice(2, size=nb).astype(bool))

        rng = np.random.default_rng(seed=13256)
        self.cm.pc = rng.uniform(0, 1, size=self.nmatch)
        self.cm.eta = rng.uniform(0, 1, size=self.nmatch)
        self.cm.xi = rng.uniform(0, 1, size=self.nmatch)
        self.cm.acontamflux = rng.uniform(0, 1, size=self.nmatch)
        self.cm.bcontamflux = rng.uniform(0, 1, size=self.nmatch)
        self.cm.pacontam = rng.uniform(0, 1, size=(2, self.nmatch))
        self.cm.pbcontam = rng.uniform(0, 1, size=(2, self.nmatch))
        self.cm.crptseps = rng.uniform(0, 1, size=self.nmatch)

        for t, n in zip(['a', 'b'], [self.na, self.nb]):
            setattr(self.cm, f'{t}fieldflux', rng.uniform(0, 1, size=n-self.nmatch))
            setattr(self.cm, f'pf{t}', rng.uniform(0, 1, size=n-self.nmatch))
            setattr(self.cm, f'{t}fieldseps', rng.uniform(0, 1, size=n-self.nmatch))
            setattr(self.cm, f'{t}fieldeta', rng.uniform(0, 1, size=n-self.nmatch))
            setattr(self.cm, f'{t}fieldxi', rng.uniform(0, 1, size=n-self.nmatch))

        self.cm.reject_a = None
        self.cm.reject_b = None

    def make_temp_catalogue(self, nrow, ncol, nnans, designation):
        # Fake a ID/ra/dec/err/[mags xN]/bestflag/inchunk csv file.
        rng = np.random.default_rng(seed=657234234)

        data = rng.standard_normal(size=(nrow, ncol))
        data[:, ncol-2] = np.round(data[:, ncol-2]).astype(int)
        nan_cols = [rng.choice(nrow, size=(nnans,), replace=False)]*(ncol-6)
        for i, nan_col in enumerate(nan_cols):
            data[nan_col, 4+i] = np.nan
        data[:, ncol-1] = rng.choice(2, size=(nrow,))

        data1 = data.astype(str)
        data1[data1 == 'nan'] = ''
        data1[:, 0] = [f'{designation}{i}' for i in data1[:, 0]]
        return data, data1

    def test_postprocess(self):
        self.cm.joint_folder_path = self.joint_folder_path
        self.cm.a_cat_folder_path = self.a_cat_folder_path
        self.cm.b_cat_folder_path = self.b_cat_folder_path

        self.cm.make_output_csv = False

        self.cm.chunk_id = 1

        self.cm._postprocess_chunk()

        aino = np.load(f'{self.a_cat_folder_path}/in_chunk_overlap.npy')
        bino = np.load(f'{self.b_cat_folder_path}/in_chunk_overlap.npy')
        ac = np.load(f'{self.joint_folder_path}/ac.npy')
        af = np.load(f'{self.joint_folder_path}/af.npy')
        bc = np.load(f'{self.joint_folder_path}/bc.npy')
        bf = np.load(f'{self.joint_folder_path}/bf.npy')

        assert np.all(~aino[ac] | ~bino[bc])
        assert np.all(~aino[af])
        assert np.all(~bino[bf])

        deleted_ac = np.delete(self.ac, np.array([np.argmin(np.abs(q - self.ac)) for q in ac]))
        deleted_bc = np.delete(self.bc, np.array([np.argmin(np.abs(q - self.bc)) for q in bc]))
        assert np.all((aino[deleted_ac] & bino[deleted_bc]))

        deleted_af = np.delete(self.af, np.array([np.argmin(np.abs(q - self.af)) for q in af]))
        deleted_bf = np.delete(self.bf, np.array([np.argmin(np.abs(q - self.bf)) for q in bf]))
        assert np.all(aino[deleted_af])
        assert np.all(bino[deleted_bf])

    # pylint: disable-next=too-many-statements,too-many-locals
    def test_postprocess_with_csv(self):
        self.cm.joint_folder_path = self.joint_folder_path
        self.cm.a_cat_folder_path = self.a_cat_folder_path
        self.cm.b_cat_folder_path = self.b_cat_folder_path

        self.cm.make_output_csv = True

        # Set a whole load of fake inputs
        self.cm.output_csv_folder = 'output_csv_folder'
        os.makedirs(self.cm.output_csv_folder, exist_ok=True)
        self.cm.a_input_csv_folder = 'a_input_csv_folder'
        os.makedirs(self.cm.a_input_csv_folder, exist_ok=True)
        self.cm.a_cat_csv_name = 'gaia_catalogue.csv'
        self.cm.a_csv_has_header = False
        acat, acatstring = self.make_temp_catalogue(self.na, 8, 100, 'Gaia ')
        np.savetxt(f'{self.cm.a_input_csv_folder}/{self.cm.a_cat_csv_name}', acatstring, delimiter=',',
                   fmt='%s', header='')
        self.cm.b_input_csv_folder = 'b_input_csv_folder'
        os.makedirs(self.cm.b_input_csv_folder, exist_ok=True)
        self.cm.b_cat_csv_name = 'wise_catalogue.csv'
        self.cm.b_csv_has_header = True
        bcat, bcatstring = self.make_temp_catalogue(self.nb, 10, 500, 'J')
        np.savetxt(f'{self.cm.b_input_csv_folder}/{self.cm.b_cat_csv_name}', bcatstring, delimiter=',',
                   fmt='%s', header='ID, RA, Dec, Err, W1, W2, W3, W4, bestflag, inchunk')
        self.cm.match_out_csv_name = 'match.csv'
        self.cm.a_nonmatch_out_csv_name = 'gaia_nonmatch.csv'
        self.cm.b_nonmatch_out_csv_name = 'wise_nonmatch.csv'
        # These would be ['Des', 'RA', 'Dec', 'G', 'RP'] as passed to CrossMatch,
        # but to avoid exactly this situation we prepend the catalogue name on the
        # front since RA and Dec are likely duplicated in all matches...
        self.cm.a_cat_col_names = ['Gaia_Des', 'Gaia_RA', 'Gaia_Dec', 'Gaia_G', 'Gaia_RP']
        self.cm.a_cat_col_nums = [0, 1, 2, 4, 5]
        self.cm.b_cat_col_names = ['WISE_ID', 'WISE_RA', 'WISE_Dec', 'WISE_W1', 'WISE_W2',
                                   'WISE_W3', 'WISE_W4']
        self.cm.b_cat_col_nums = [0, 1, 2, 4, 5, 6, 7]
        self.cm.a_cat_name = 'Gaia'
        self.cm.b_cat_name = 'WISE'
        self.cm.a_input_npy_folder = None
        self.cm.b_input_npy_folder = None
        self.cm.a_extra_col_names = None
        self.cm.a_extra_col_nums = None
        self.cm.b_extra_col_names = ['WErr']
        self.cm.b_extra_col_nums = [3]
        self.cm.chunk_id = 1

        self.cm._postprocess_chunk()

        aino = np.load(f'{self.a_cat_folder_path}/in_chunk_overlap.npy')
        bino = np.load(f'{self.b_cat_folder_path}/in_chunk_overlap.npy')
        ac = np.load(f'{self.joint_folder_path}/ac.npy')
        af = np.load(f'{self.joint_folder_path}/af.npy')
        bc = np.load(f'{self.joint_folder_path}/bc.npy')
        bf = np.load(f'{self.joint_folder_path}/bf.npy')

        assert np.all(~aino[ac] | ~bino[bc])
        assert np.all(~aino[af])
        assert np.all(~bino[bf])

        deleted_ac = np.delete(self.ac, np.array([np.argmin(np.abs(q - self.ac)) for q in ac]))
        deleted_bc = np.delete(self.bc, np.array([np.argmin(np.abs(q - self.bc)) for q in bc]))
        assert np.all((aino[deleted_ac] & bino[deleted_bc]))

        deleted_af = np.delete(self.af, np.array([np.argmin(np.abs(q - self.af)) for q in af]))
        deleted_bf = np.delete(self.bf, np.array([np.argmin(np.abs(q - self.bf)) for q in bf]))
        assert np.all(aino[deleted_af])
        assert np.all(bino[deleted_bf])

        # Check that the outputs make sense, treating this more like a
        # parse_catalogue test than anything else, but importantly
        # checking for correct lengths of produced outputs like pc.
        assert os.path.isfile(f'{self.cm.output_csv_folder}/{self.cm.match_out_csv_name}')
        assert os.path.isfile(f'{self.cm.output_csv_folder}/{self.cm.a_nonmatch_out_csv_name}')
        assert os.path.isfile(f'{self.cm.output_csv_folder}/{self.cm.b_nonmatch_out_csv_name}')

        pc = np.load(f'{self.joint_folder_path}/pc.npy')
        eta = np.load(f'{self.joint_folder_path}/eta.npy')
        xi = np.load(f'{self.joint_folder_path}/xi.npy')
        acf = np.load(f'{self.joint_folder_path}/acontamflux.npy')
        bcf = np.load(f'{self.joint_folder_path}/bcontamflux.npy')
        pac = np.load(f'{self.joint_folder_path}/pacontam.npy')
        pbc = np.load(f'{self.joint_folder_path}/pbcontam.npy')
        csep = np.load(f'{self.joint_folder_path}/crptseps.npy')
        pfa = np.load(f'{self.joint_folder_path}/pfa.npy')
        afs = np.load(f'{self.joint_folder_path}/afieldseps.npy')
        afeta = np.load(f'{self.joint_folder_path}/afieldeta.npy')
        afxi = np.load(f'{self.joint_folder_path}/afieldxi.npy')
        aff = np.load(f'{self.joint_folder_path}/afieldflux.npy')
        pfb = np.load(f'{self.joint_folder_path}/pfb.npy')
        bfs = np.load(f'{self.joint_folder_path}/bfieldseps.npy')
        bfeta = np.load(f'{self.joint_folder_path}/bfieldeta.npy')
        bfxi = np.load(f'{self.joint_folder_path}/bfieldxi.npy')
        bff = np.load(f'{self.joint_folder_path}/bfieldflux.npy')

        extra_cols = ['MATCH_P', 'SEPARATION', 'ETA', 'XI', 'A_AVG_CONT', 'B_AVG_CONT',
                      'A_CONT_F1', 'A_CONT_F10', 'B_CONT_F1', 'B_CONT_F10']
        names = np.append(np.append(self.cm.a_cat_col_names, self.cm.b_cat_col_names),
                          np.append(extra_cols, self.cm.b_extra_col_names))

        df = pd.read_csv(f'{self.cm.output_csv_folder}/{self.cm.match_out_csv_name}',
                         header=None, names=names)
        for i, col in zip([1, 2, 4, 5], self.cm.a_cat_col_names[1:]):
            assert_allclose(df[col], acat[ac, i])

        assert np.all([df[self.cm.a_cat_col_names[0]].iloc[i] == acatstring[ac[i], 0] for i in
                       range(len(ac))])

        for i, col in zip([1, 2, 4, 5, 6], self.cm.b_cat_col_names[1:]):
            assert_allclose(df[col], bcat[bc, i])
        assert np.all([df[self.cm.b_cat_col_names[0]].iloc[i] == bcatstring[bc[i], 0] for i in
                       range(len(bc))])

        for f, col in zip([pc, csep, eta, xi, acf, bcf, pac[0], pac[1], pbc[0], pbc[1]],
                          extra_cols):
            assert_allclose(df[col], f)

        names = np.append(self.cm.a_cat_col_names,
                          ['MATCH_P', 'NNM_SEPARATION', 'NNM_ETA', 'NNM_XI', 'A_AVG_CONT'])
        df = pd.read_csv(f'{self.cm.output_csv_folder}/{self.cm.a_nonmatch_out_csv_name}', header=None,
                         names=names)
        for i, col in zip([1, 2, 4, 5], self.cm.a_cat_col_names[1:]):
            assert_allclose(df[col], acat[af, i])
        assert np.all([df[self.cm.a_cat_col_names[0]].iloc[i] == acatstring[af[i], 0] for i in
                       range(len(af))])
        assert_allclose(df['MATCH_P'], pfa)
        assert_allclose(df['A_AVG_CONT'], aff)
        assert_allclose(df['NNM_SEPARATION'], afs)
        assert_allclose(df['NNM_ETA'], afeta)
        assert_allclose(df['NNM_XI'], afxi)
        names = np.append(np.append(self.cm.b_cat_col_names,
                          ['MATCH_P', 'NNM_SEPARATION', 'NNM_ETA', 'NNM_XI', 'B_AVG_CONT']),
                          self.cm.b_extra_col_names)
        df = pd.read_csv(f'{self.cm.output_csv_folder}/{self.cm.b_nonmatch_out_csv_name}', header=None,
                         names=names)
        for i, col in zip([1, 2, 4, 5, 6], self.cm.b_cat_col_names[1:]):
            assert_allclose(df[col], bcat[bf, i])
        assert np.all([df[self.cm.b_cat_col_names[0]].iloc[i] == bcatstring[bf[i], 0] for i in
                       range(len(bf))])
        assert_allclose(df['MATCH_P'], pfb)
        assert_allclose(df['B_AVG_CONT'], bff)
        assert_allclose(df['NNM_SEPARATION'], bfs)
        assert_allclose(df['NNM_ETA'], bfeta)
        assert_allclose(df['NNM_XI'], bfxi)
