# Licensed under a 3-clause BSD style license - see LICENSE
'''
Tests for the "matching" module.
'''

import pytest
import os
from configparser import ConfigParser
from numpy.testing import assert_allclose
import numpy as np

from ..matching import CrossMatch


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
    lines = open(file_name, 'r').readlines()
    lines[line_num] = text
    out = open(out_file, 'w')
    out.writelines(lines)
    out.close()


class TestInputs:
    def setup_class(self):
        joint_config = ConfigParser()
        with open(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt')) as f:
            joint_config.read_string('[config]\n' + f.read())
        joint_config = joint_config['config']
        cat_a_config = ConfigParser()
        with open(os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt')) as f:
            cat_a_config.read_string('[config]\n' + f.read())
        cat_a_config = cat_a_config['config']
        cat_b_config = ConfigParser()
        with open(os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt')) as f:
            cat_b_config.read_string('[config]\n' + f.read())
        cat_b_config = cat_b_config['config']
        self.a_cat_folder_path = os.path.abspath(cat_a_config['cat_folder_path'])
        self.b_cat_folder_path = os.path.abspath(cat_b_config['cat_folder_path'])

        os.makedirs(self.a_cat_folder_path, exist_ok=True)
        os.makedirs(self.b_cat_folder_path, exist_ok=True)

        np.save('{}/con_cat_astro.npy'.format(self.a_cat_folder_path), np.zeros((2, 3), float))
        np.save('{}/con_cat_photo.npy'.format(self.a_cat_folder_path), np.zeros((2, 3), float))
        np.save('{}/magref.npy'.format(self.a_cat_folder_path), np.zeros(2, float))

        np.save('{}/con_cat_astro.npy'.format(self.b_cat_folder_path), np.zeros((2, 3), float))
        np.save('{}/con_cat_photo.npy'.format(self.b_cat_folder_path), np.zeros((2, 4), float))
        np.save('{}/magref.npy'.format(self.b_cat_folder_path), np.zeros(2, float))

        mag_h_params = np.ones((3, 5), float)
        np.save('mag_h_params.npy', mag_h_params)

    def test_crossmatch_run_input(self):
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'), use_memmap_files=True)
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

        cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        assert cm.run_auf is False
        assert cm.run_group is False
        assert cm.run_cf is True
        assert cm.run_source is True

        # List of simple one line config file replacements for error message checking
        f = open(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt')).readlines()
        for old_line, new_line, match_text in zip(['run_cf = yes', 'run_auf = no', 'run_auf = no'],
                                                  ['', 'run_auf = aye\n', 'run_auf = yes\n'],
                                                  ['Missing key', 'Boolean flag key not set',
                                                   'Inconsistency between run/no run']):
            idx = np.where([old_line in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__),
                          'data/crossmatch_params.txt'), idx, new_line,
                          out_file=os.path.join(os.path.dirname(__file__),
                          'data/crossmatch_params_.txt'))

            with pytest.raises(ValueError, match=match_text):
                cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                     'data/crossmatch_params_.txt'),
                                     os.path.join(os.path.dirname(__file__),
                                     'data/cat_a_params.txt'),
                                     os.path.join(os.path.dirname(__file__),
                                     'data/cat_b_params.txt'))

    def test_crossmatch_auf_cf_input(self):
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'), use_memmap_files=True)
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        assert cm.cf_region_frame == 'equatorial'
        assert_allclose(cm.cf_region_points,
                        np.array([[131, -1], [132, -1], [133, -1], [134, -1],
                                  [131, 0], [132, 0], [133, 0], [134, 0],
                                  [131, 1], [132, 1], [133, 1], [134, 1]]))

        f = open(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt')).readlines()
        old_line = 'include_perturb_auf = no'
        new_line = 'include_perturb_auf = yes\n'
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                      idx, new_line, out_file=os.path.join(
                      os.path.dirname(__file__), 'data/crossmatch_params_.txt'))
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params_.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        assert cm.a_auf_region_frame == 'equatorial'
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
            in_file = 'crossmatch_params' if 'cf' in kind else 'cat_a_params'
            f = open(os.path.join(os.path.dirname(__file__),
                                  'data/{}.txt'.format(in_file))).readlines()
            # List of simple one line config file replacements for error message checking
            for old_line, new_line, match_text in zip(
                ['{}type = rectangle'.format(kind), '{}type = rectangle'.format(kind),
                 '{}points = 131 134 4 -1 1 3'.format(kind),
                 '{}points = 131 134 4 -1 1 3'.format(kind),
                 '{}frame = equatorial'.format(kind), '{}points = 131 134 4 -1 1 3'.format(kind)],
                ['', '{}type = triangle\n'.format(kind),
                 '{}points = 131 134 4 -1 1 a\n'.format(kind),
                 '{}points = 131 134 4 -1 1\n'.format(kind), '{}frame = ecliptic\n'.format(kind),
                 '{}points = 131 134 4 -1 1 3.4\n'.format(kind)],
                ['Missing key {}type'.format(kind),
                 "{}{}type should either be 'rectangle' or".format('' if 'cf' in kind
                                                                   else 'a_', kind),
                 '{}{}points should be 6 numbers'.format('' if 'cf' in kind else 'a_', kind),
                 '{}{}points should be 6 numbers'.format('' if 'cf' in kind else 'a_', kind),
                 "{}{}frame should either be 'equatorial' or".format(
                    '' if 'cf' in kind else 'a_', kind),
                 'start and stop values for {}{}points'.format('' if 'cf' in kind
                                                               else 'a_', kind)]):
                idx = np.where([old_line in line for line in f])[0][0]
                _replace_line(os.path.join(os.path.dirname(__file__),
                              'data/{}.txt'.format(in_file)), idx, new_line,
                              out_file=os.path.join(os.path.dirname(__file__),
                              'data/{}_.txt'.format(in_file)))

                with pytest.raises(ValueError, match=match_text):
                    cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                                 'data/crossmatch_params{}.txt'.format(
                                                 '_' if 'cf' in kind else '')),
                                         os.path.join(os.path.dirname(__file__),
                                                 'data/cat_a_params{}.txt'.format(
                                                 '_' if 'cf' not in kind else '')),
                                         os.path.join(os.path.dirname(__file__),
                                                 'data/cat_b_params.txt'))

            # Check correct and incorrect *_region_points when *_region_type is 'points'
            idx = np.where(['{}type = rectangle'.format(kind) in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__),
                          'data/{}.txt'.format(in_file)), idx, '{}type = points\n'.format(kind),
                          out_file=os.path.join(os.path.dirname(__file__),
                                                'data/{}_.txt'.format(in_file)))

            idx = np.where(['{}points = 131 134 4 -1 1 3'.format(kind) in line for
                            line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__),
                          'data/{}_.txt'.format(in_file)), idx,
                          '{}points = (131, 0), (133, 0), (132, -1)\n'.format(kind),
                          out_file=os.path.join(os.path.dirname(__file__),
                                                'data/{}_2.txt'.format(in_file)))

            cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                 'data/crossmatch_params{}.txt'
                                              .format('_2' if 'cf' in kind else '')),
                                 os.path.join(os.path.dirname(__file__),
                                 'data/cat_a_params{}.txt'
                                              .format('_2' if 'cf' not in kind else '')),
                                 os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
            assert_allclose(getattr(cm, '{}{}points'.format('' if 'cf' in kind
                            else 'a_', kind)), np.array([[131, 0], [133, 0], [132, -1]]))

            old_line = '{}points = 131 134 4 -1 1 3'.format(kind)
            for new_line in ['{}points = (131, 0), (131, )\n'.format(kind),
                             '{}points = (131, 0), (131, 1, 2)\n'.format(kind),
                             '{}points = (131, 0), (131, a)\n'.format(kind)]:
                idx = np.where([old_line in line for line in f])[0][0]
                _replace_line(os.path.join(os.path.dirname(__file__),
                              'data/{}_.txt'.format(in_file)), idx, new_line,
                              out_file=os.path.join(os.path.dirname(__file__),
                                                    'data/{}_2.txt'.format(in_file)))

                with pytest.raises(ValueError):
                    cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                                 'data/crossmatch_params{}.txt'.format(
                                                 '_2' if 'cf' in kind else '')),
                                         os.path.join(os.path.dirname(__file__),
                                                 'data/cat_a_params{}.txt'.format(
                                                 '_2' if 'cf' not in kind else '')),
                                         os.path.join(os.path.dirname(__file__),
                                                 'data/cat_b_params.txt'))

            # Check single-length point grids are fine
            idx = np.where(['{}points = 131 134 4 -1 1 3'.format(kind) in line
                            for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__),
                          'data/{}.txt'.format(in_file)), idx,
                          '{}points = 131 131 1 0 0 1\n'.format(kind),
                          out_file=os.path.join(os.path.dirname(__file__),
                                                'data/{}_.txt'.format(in_file)))

            cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                 'data/crossmatch_params{}.txt'
                                              .format('_' if 'cf' in kind else '')),
                                 os.path.join(os.path.dirname(__file__),
                                 'data/cat_a_params{}.txt'.format('_' if 'cf' not in kind else '')),
                                 os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
            assert_allclose(getattr(cm, '{}{}points'.format('' if 'cf' in kind
                            else 'a_', kind)), np.array([[131, 0]]))

            idx = np.where(['{}type = rectangle'.format(kind) in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__),
                          'data/{}.txt'.format(in_file)), idx,
                          '{}type = points\n'.format(kind),
                          out_file=os.path.join(os.path.dirname(__file__),
                                                'data/{}_.txt'.format(in_file)))

            idx = np.where(['{}points = 131 134 4 -1 1 3'.format(kind) in
                            line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__),
                          'data/{}_.txt'.format(in_file)), idx,
                          '{}points = (131, 0)\n'.format(kind),
                          out_file=os.path.join(os.path.dirname(__file__),
                                                'data/{}_2.txt'.format(in_file)))

            cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                 'data/crossmatch_params{}.txt'
                                              .format('_2' if 'cf' in kind else '')),
                                 os.path.join(os.path.dirname(__file__),
                                 'data/cat_a_params{}.txt'
                                              .format('_2' if 'cf' not in kind else '')),
                                 os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
            assert_allclose(getattr(cm, '{}{}points'.format('' if 'cf' in kind
                            else 'a_', kind)), np.array([[131, 0]]))

        # Check galactic run is also fine -- here we have to replace all 3 parameter
        # options with "galactic", however.
        for in_file in ['crossmatch_params', 'cat_a_params', 'cat_b_params']:
            kind = 'cf_region_' if 'h_p' in in_file else 'auf_region_'
            f = open(os.path.join(os.path.dirname(__file__),
                                  'data/{}.txt'.format(in_file))).readlines()
            idx = np.where(['{}frame = equatorial'.format(kind) in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__),
                          'data/{}.txt'.format(in_file)), idx, '{}frame = galactic\n'.format(kind),
                          out_file=os.path.join(os.path.dirname(__file__),
                                                'data/{}_.txt'.format(in_file)))

        cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params_.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params_.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params_.txt'))
        for kind in ['auf_region_', 'cf_region_']:
            assert getattr(cm, '{}{}frame'.format('' if 'cf' in kind
                                                  else 'a_', kind)) == 'galactic'
            assert_allclose(getattr(cm, '{}{}points'.format('' if 'cf' in kind
                                                            else 'a_', kind)),
                            np.array([[131, -1], [132, -1], [133, -1], [134, -1],
                                      [131, 0], [132, 0], [133, 0], [134, 0],
                                      [131, 1], [132, 1], [133, 1], [134, 1]]))

    def test_crossmatch_folder_path_inputs(self):
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'), use_memmap_files=True)
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
                ['Missing key', 'Error when trying to create temporary',
                 'Missing key auf_folder_path from catalogue "a"',
                 'folder for catalogue "b" AUF outputs. Please ensure that b_auf_folder_path'],
                [ValueError, OSError, ValueError, OSError],
                ['crossmatch_params', 'crossmatch_params', 'cat_a_params', 'cat_b_params']):
            f = open(os.path.join(os.path.dirname(__file__),
                     'data/{}.txt'.format(in_file))).readlines()
            idx = np.where([old_line in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__),
                          'data/{}.txt'.format(in_file)), idx, new_line, out_file=os.path.join(
                          os.path.dirname(__file__), 'data/{}_.txt'.format(in_file)))

            with pytest.raises(error, match=match_text):
                cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                     'data/crossmatch_params{}.txt'.format(
                                     '_' if 'h_p' in in_file else '')),
                                     os.path.join(os.path.dirname(__file__),
                                     'data/cat_a_params{}.txt'
                                                  .format('_' if '_a_' in in_file else '')),
                                     os.path.join(os.path.dirname(__file__),
                                     'data/cat_b_params{}.txt'
                                                  .format('_' if '_b_' in in_file else '')))

    def test_crossmatch_tri_inputs(self):
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'), use_memmap_files=True)
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        assert not hasattr(cm, 'a_tri_set_name')

        f = open(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt')).readlines()
        old_line = 'include_perturb_auf = no'
        new_line = 'include_perturb_auf = yes\n'
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__),
                      'data/crossmatch_params.txt'), idx, new_line, out_file=os.path.join(
                      os.path.dirname(__file__), 'data/crossmatch_params_.txt'))

        cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params_.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        assert cm.a_tri_set_name == 'gaiaDR2'
        assert np.all(cm.b_tri_filt_names == np.array(['W1', 'W2', 'W3', 'W4']))
        assert cm.a_tri_filt_num == 1
        assert not cm.b_download_tri

        # List of simple one line config file replacements for error message checking
        for old_line, new_line, match_text, in_file in zip(
                ['tri_set_name = gaiaDR2', 'tri_filt_num = 11', 'tri_filt_num = 11',
                 'download_tri = no', 'download_tri = no', 'tri_maglim_bright = 13.1',
                 'tri_maglim_faint = 32', 'tri_num_faint = 1500000', 'tri_num_bright = 500000',
                 'tri_num_faint = 1500000'],
                ['', 'tri_filt_num = a\n', 'tri_filt_num = 3.4\n', 'download_tri = aye\n',
                 'download_tri = yes\n', 'tri_maglim_bright = 13.1 14.5\n',
                 'tri_maglim_faint = a\n', 'tri_num_faint = 1500000.1\n', 'tri_num_bright = a\n',
                 'tri_num_faint = 1500000 15\n'],
                ['Missing key tri_set_name from catalogue "a"',
                 'tri_filt_num should be a single integer number in catalogue "b"',
                 'tri_filt_num should be a single integer number in catalogue "b"',
                 'Boolean flag key not set', 'a_download_tri is True and run_auf is False',
                 'tri_maglim_bright in catalogue "a" must be a float.',
                 'tri_maglim_faint in catalogue "b" must be a float.',
                 'tri_num_faint should be a single integer number in catalogue "b"',
                 'tri_num_bright should be a single integer number in catalogue "a" metadata file',
                 'tri_num_faint should be a single integer number in catalogue "a"'],
                ['cat_a_params', 'cat_b_params', 'cat_b_params', 'cat_a_params', 'cat_a_params',
                 'cat_a_params', 'cat_b_params', 'cat_b_params', 'cat_a_params', 'cat_a_params']):
            f = open(os.path.join(os.path.dirname(__file__),
                                  'data/{}.txt'.format(in_file))).readlines()
            idx = np.where([old_line in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__),
                          'data/{}.txt'.format(in_file)), idx, new_line, out_file=os.path.join(
                          os.path.dirname(__file__), 'data/{}_.txt'.format(in_file)))

            with pytest.raises(ValueError, match=match_text):
                cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                     'data/crossmatch_params_.txt'),
                                     os.path.join(os.path.dirname(__file__),
                                     'data/cat_a_params{}.txt'
                                                  .format('_' if '_a_' in in_file else '')),
                                     os.path.join(os.path.dirname(__file__),
                                     'data/cat_b_params{}.txt'
                                                  .format('_' if '_b_' in in_file else '')))

    def test_crossmatch_psf_param_inputs(self):
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'), use_memmap_files=True)
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        assert np.all(cm.b_filt_names == np.array(['W1', 'W2', 'W3', 'W4']))

        f = open(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt')).readlines()
        old_line = 'include_perturb_auf = no'
        new_line = 'include_perturb_auf = yes\n'
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__),
                      'data/crossmatch_params.txt'), idx, new_line, out_file=os.path.join(
                      os.path.dirname(__file__), 'data/crossmatch_params_.txt'))

        cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params_.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        assert np.all(cm.a_psf_fwhms == np.array([0.12, 0.12, 0.12]))

        # List of simple one line config file replacements for error message checking
        for old_line, new_line, match_text, in_file in zip(
                ['filt_names = G_BP G G_RP', 'filt_names = G_BP G G_RP',
                 'psf_fwhms = 6.08 6.84 7.36 11.99', 'psf_fwhms = 6.08 6.84 7.36 11.99'],
                ['', 'filt_names = G_BP G\n',
                 'psf_fwhms = 6.08 6.84 7.36\n', 'psf_fwhms = 6.08 6.84 7.36 word\n'],
                ['Missing key filt_names from catalogue "a"',
                 'a_tri_filt_names and a_filt_names should contain the same',
                 'b_psf_fwhms and b_filt_names should contain the same',
                 'psf_fwhms should be a list of floats in catalogue "b".'],
                ['cat_a_params', 'cat_a_params', 'cat_b_params', 'cat_b_params']):
            f = open(os.path.join(os.path.dirname(__file__),
                                  'data/{}.txt'.format(in_file))).readlines()
            idx = np.where([old_line in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__),
                          'data/{}.txt'.format(in_file)), idx, new_line, out_file=os.path.join(
                          os.path.dirname(__file__), 'data/{}_.txt'.format(in_file)))

            with pytest.raises(ValueError, match=match_text):
                cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                     'data/crossmatch_params_.txt'),
                                     os.path.join(os.path.dirname(__file__),
                                     'data/cat_a_params{}.txt'
                                                  .format('_' if '_a_' in in_file else '')),
                                     os.path.join(os.path.dirname(__file__),
                                     'data/cat_b_params{}.txt'
                                                  .format('_' if '_b_' in in_file else '')))

    def test_crossmatch_cat_name_inputs(self):
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'), use_memmap_files=True)
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        assert cm.b_cat_name == 'WISE'
        assert os.path.exists('{}/test_path/WISE'.format(os.getcwd()))

        f = open(os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt')).readlines()
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
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'), use_memmap_files=True)
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        assert cm.pos_corr_dist == 11
        assert not hasattr(cm, 'a_dens_dist')
        assert not hasattr(cm, 'b_dens_mags')
        f = open(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt')).readlines()
        old_line = 'include_perturb_auf = no'
        new_line = 'include_perturb_auf = yes\n'
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__),
                      'data/crossmatch_params.txt'), idx, new_line, out_file=os.path.join(
                      os.path.dirname(__file__), 'data/crossmatch_params_.txt'))
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params_.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        assert np.all(cm.a_dens_mags == np.array([20, 20, 20]))
        assert not hasattr(cm, 'b_dens_dist')

        f = open(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt')).readlines()
        old_line = 'compute_local_density = no'
        new_line = 'compute_local_density = yes\n'
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__),
                      'data/crossmatch_params_.txt'), idx, new_line, out_file=os.path.join(
                      os.path.dirname(__file__), 'data/crossmatch_params_2.txt'))
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                          'data/crossmatch_params_2.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        assert np.all(cm.a_dens_mags == np.array([20, 20, 20]))
        assert cm.b_dens_dist == 0.25

        # List of simple one line config file replacements for error message checking
        for old_line, new_line, match_text, in_file in zip(
                ['pos_corr_dist = 11', 'pos_corr_dist = 11', 'dens_dist = 0.25',
                 'dens_dist = 0.25', 'dens_mags = 20 20 20 20', 'dens_mags = 20 20 20 20',
                 'dens_mags = 20 20 20'],
                ['', 'pos_corr_dist = word\n', '', 'dens_dist = word\n', '',
                 'dens_mags = 20 20 20\n', 'dens_mags = word word word\n'],
                ['Missing key pos_corr_dist', 'pos_corr_dist must be a float',
                 'Missing key dens_dist from catalogue "b"', 'dens_dist in catalogue "a" must',
                 'Missing key dens_mags from catalogue "b"',
                 'b_dens_mags and b_filt_names should contain the same number',
                 'dens_mags should be a list of floats in catalogue "a'],
                ['crossmatch_params', 'crossmatch_params', 'cat_b_params', 'cat_a_params',
                 'cat_b_params', 'cat_b_params', 'cat_a_params']):
            f = open(os.path.join(os.path.dirname(__file__),
                                  'data/{}.txt'.format(in_file))).readlines()
            idx = np.where([old_line in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__),
                          'data/{}{}.txt'.format(in_file, '_2' if 'h_p' in in_file else '')), idx,
                          new_line, out_file=os.path.join(os.path.dirname(__file__),
                          'data/{}_{}.txt'.format(in_file, '3' if 'h_p' in in_file else '')))

            with pytest.raises(ValueError, match=match_text):
                cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                     'data/crossmatch_params{}.txt'.format(
                                     '_3' if 'h_p' in in_file else '_2')),
                                     os.path.join(os.path.dirname(__file__),
                                     'data/cat_a_params{}.txt'
                                                  .format('_' if '_a_' in in_file else '')),
                                     os.path.join(os.path.dirname(__file__),
                                     'data/cat_b_params{}.txt'
                                                  .format('_' if '_b_' in in_file else '')))

    def test_crossmatch_perturb_auf_inputs(self):
        f = open(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt')).readlines()
        old_line = 'include_perturb_auf = no'
        new_line = 'include_perturb_auf = yes\n'
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__),
                      'data/crossmatch_params.txt'), idx, new_line, out_file=os.path.join(
                      os.path.dirname(__file__), 'data/crossmatch_params_.txt'))

        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'), use_memmap_files=True)
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params_.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        assert cm.num_trials == 10000
        assert not cm.compute_local_density
        assert cm.d_mag == 0.1

        for old_line, new_line, match_text in zip(
                ['num_trials = 10000', 'num_trials = 10000', 'num_trials = 10000',
                 'd_mag = 0.1', 'd_mag = 0.1', 'compute_local_density = no',
                 'compute_local_density = no', 'compute_local_density = no'],
                ['', 'num_trials = word\n', 'num_trials = 10000.1\n', '', 'd_mag = word\n', '',
                 'compute_local_density = word\n', 'compute_local_density = 10\n'],
                ['Missing key num_trials from joint', 'num_trials should be an integer',
                 'num_trials should be an integer', 'Missing key d_mag from joint',
                 'd_mag must be a float', 'Missing key compute_local_density from joint',
                 'Boolean flag key not set to allowed', 'Boolean flag key not set to allowed']):
            # Make sure to keep the first edit of crossmatch_params, adding each
            # second change in turn.
            f = open(os.path.join(os.path.dirname(__file__),
                                  'data/crossmatch_params_.txt')).readlines()
            idx = np.where([old_line in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__),
                          'data/crossmatch_params_.txt'), idx, new_line, out_file=os.path.join(
                          os.path.dirname(__file__), 'data/crossmatch_params_2.txt'))
            f = open(os.path.join(os.path.dirname(__file__),
                                  'data/crossmatch_params_2.txt')).readlines()

            with pytest.raises(ValueError, match=match_text):
                cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                     'data/crossmatch_params_2.txt'),
                                     os.path.join(os.path.dirname(__file__),
                                     'data/cat_a_params.txt'),
                                     os.path.join(os.path.dirname(__file__),
                                     'data/cat_b_params.txt'))

        for old_line, var_name in zip(['fit_gal_flag = no', 'run_fw_auf = yes', 'run_psf_auf = no',
                                       'mag_h_params_path = .'],
                                      ['fit_gal_flag', 'run_fw_auf', 'run_psf_auf',
                                       'mag_h_params_path']):
            for cat_reg, cat_name in zip(['"a"', '"b"'], ['cat_a_params', 'cat_b_params']):
                f = open(os.path.join(os.path.dirname(__file__),
                         'data/{}.txt'.format(cat_name))).readlines()
                new_line = ''
                idx = np.where([old_line in line for line in f])[0][0]
                _replace_line(os.path.join(os.path.dirname(__file__),
                              'data/{}.txt'.format(cat_name)), idx, new_line, out_file=os.path.join(
                              os.path.dirname(__file__), 'data/{}_.txt'.format(cat_name)))
                with pytest.raises(ValueError, match='Missing key {} from catalogue {}'
                                   .format(var_name, cat_reg)):
                    cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                         'data/crossmatch_params_.txt'),
                                         os.path.join(os.path.dirname(__file__),
                                         'data/cat_a_params{}.txt'.format(
                                             '_' if 'a' in cat_reg else '')),
                                         os.path.join(os.path.dirname(__file__),
                                         'data/cat_b_params{}.txt'.format(
                                             '_' if 'b' in cat_reg else '')))

        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'), use_memmap_files=True)
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                             'data/crossmatch_params_.txt'),
                             os.path.join(os.path.dirname(__file__),
                             'data/cat_a_params.txt'),
                             os.path.join(os.path.dirname(__file__),
                             'data/cat_b_params.txt'))
        assert not hasattr(cm, 'a_dd_params_path')
        assert not hasattr(cm, 'b_l_cut_path')

        for cat_reg, cat_name in zip(['"a"', '"b"'], ['cat_a_params', 'cat_b_params']):
            f = open(os.path.join(os.path.dirname(__file__),
                                  'data/{}.txt'.format(cat_name))).readlines()
            old_line = 'run_psf_auf = no'
            new_line = 'run_psf_auf = yes\n'
            idx = np.where([old_line in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__),
                          'data/{}.txt'.format(cat_name)), idx, new_line,
                          out_file=os.path.join(os.path.dirname(__file__),
                          'data/{}_.txt'.format(cat_name)))
            f = open(os.path.join(os.path.dirname(__file__),
                     'data/{}_.txt'.format(cat_name))).readlines()
            old_line = 'mag_h_params_path = .'
            new_line = 'mag_h_params_path = .\ndd_params_path = .\nl_cut_path = .\n'
            idx = np.where([old_line in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__),
                          'data/{}_.txt'.format(cat_name)), idx, new_line)
            for old_line, var_name in zip(['dd_params_path = .', 'l_cut_path = .'],
                                          ['dd_params_path', 'l_cut_path']):
                f = open(os.path.join(os.path.dirname(__file__),
                         'data/{}_.txt'.format(cat_name))).readlines()
                new_line = ''
                idx = np.where([old_line in line for line in f])[0][0]
                _replace_line(os.path.join(os.path.dirname(__file__),
                              'data/{}_.txt'.format(cat_name)), idx,
                              new_line, out_file=os.path.join(
                              os.path.dirname(__file__), 'data/{}__.txt'.format(cat_name)))
                with pytest.raises(ValueError, match='Missing key {} from catalogue {}'
                                   .format(var_name, cat_reg)):
                    cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                         'data/crossmatch_params_.txt'),
                                         os.path.join(os.path.dirname(__file__),
                                         'data/cat_a_params{}.txt'.format(
                                             '__' if 'a' in cat_reg else '')),
                                         os.path.join(os.path.dirname(__file__),
                                         'data/cat_b_params{}.txt'.format(
                                             '__' if 'b' in cat_reg else '')))

        ddp = np.ones((5, 15, 2), float)
        np.save('dd_params.npy', ddp)
        lc = np.ones(3, float)
        np.save('l_cut.npy', lc)
        for cat_name in ['cat_a_params', 'cat_b_params']:
            f = open(os.path.join(os.path.dirname(__file__),
                                  'data/{}.txt'.format(cat_name))).readlines()
            old_line = 'run_psf_auf = no'
            new_line = 'run_psf_auf = yes\n'
            idx = np.where([old_line in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__),
                          'data/{}.txt'.format(cat_name)), idx, new_line,
                          out_file=os.path.join(os.path.dirname(__file__),
                          'data/{}_.txt'.format(cat_name)))
            f = open(os.path.join(os.path.dirname(__file__),
                     'data/{}_.txt'.format(cat_name))).readlines()
            old_line = 'mag_h_params_path = .'
            new_line = 'mag_h_params_path = .\ndd_params_path = .\nl_cut_path = .\n'
            idx = np.where([old_line in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__),
                          'data/{}_.txt'.format(cat_name)), idx, new_line)
        f = open(os.path.join(os.path.dirname(__file__),
                 'data/cat_b_params_.txt')).readlines()
        old_line = 'mag_h_params_path = .'
        new_line = 'mag_h_params_path = /some/path/or/other\n'
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__),
                      'data/cat_b_params_.txt'), idx, new_line,
                      out_file=os.path.join(os.path.dirname(__file__), 'data/cat_b_params__.txt'))
        with pytest.raises(OSError, match='b_mag_h_params_path does not exist.'):
            cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                 'data/crossmatch_params_.txt'),
                                 os.path.join(os.path.dirname(__file__),
                                 'data/cat_a_params_.txt'),
                                 os.path.join(os.path.dirname(__file__),
                                 'data/cat_b_params__.txt'))
        os.remove('mag_h_params.npy')
        with pytest.raises(FileNotFoundError,
                           match='mag_h_params file not found in catalogue "a" path'):
            cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                 'data/crossmatch_params_.txt'),
                                 os.path.join(os.path.dirname(__file__),
                                 'data/cat_a_params_.txt'),
                                 os.path.join(os.path.dirname(__file__),
                                 'data/cat_b_params_.txt'))
        for fn, array, err_msg in zip([
                'mag_h_params', 'mag_h_params', 'mag_h_params', 'dd_params', 'dd_params',
                'dd_params', 'dd_params', 'l_cut', 'l_cut'],
                [np.ones(4, float), np.ones((5, 3, 2), float), np.ones((4, 4), float),
                 np.ones(5, float), np.ones((5, 3), float), np.ones((4, 4, 2), float),
                 np.ones((5, 3, 1), float), np.ones((4, 2), float), np.ones(4, float)],
                [r'a_mag_h_params should be of shape \(X, 5\)',
                 r'a_mag_h_params should be of shape \(X, 5\)',
                 r'a_mag_h_params should be of shape \(X, 5\)',
                 r'a_dd_params should be of shape \(5, X, 2\)',
                 r'a_dd_params should be of shape \(5, X, 2\)',
                 r'a_dd_params should be of shape \(5, X, 2\)',
                 r'a_dd_params should be of shape \(5, X, 2\)',
                 r'a_l_cut should be of shape \(3,\) only.',
                 r'a_l_cut should be of shape \(3,\) only.']):
            np.save('{}.npy'.format(fn), array)
            with pytest.raises(ValueError, match=err_msg):
                cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                     'data/crossmatch_params_.txt'),
                                     os.path.join(os.path.dirname(__file__),
                                     'data/cat_a_params_.txt'),
                                     os.path.join(os.path.dirname(__file__),
                                     'data/cat_b_params_.txt'))
            # Re-make "good" fake arrays
            mag_h_params = np.ones((3, 5), float)
            np.save('mag_h_params.npy', mag_h_params)
            ddp = np.ones((5, 15, 2), float)
            np.save('dd_params.npy', ddp)
            lc = np.ones(3, float)
            np.save('l_cut.npy', lc)

        f = open(os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt')).readlines()
        old_line = 'fit_gal_flag = no'
        new_line = ('fit_gal_flag = yes\ngal_wavs = 0.513 0.641 0.778\n'
                    'gal_zmax = 4.5 4.5 5\ngal_nzs = 46 46 51\n'
                    'gal_aboffsets = 0.5 0.5 0.5\n'
                    'gal_filternames = gaiadr2-BP gaiadr2-G gaiadr2-RP\n'
                    'gal_al_avs = 0.589 0.789 1.002\n')
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'), idx,
                      new_line, out_file=os.path.join(os.path.dirname(__file__),
                      'data/cat_a_params_.txt'))
        f = open(os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt')).readlines()
        old_line = 'fit_gal_flag = no'
        new_line = ('fit_gal_flag = yes\ngal_wavs = 3.37 4.62 12.08 22.19\n'
                    'gal_zmax = 3.2 4.0 1 4\ngal_nzs = 33 41 11 41\n'
                    'gal_aboffsets = 0.5 0.5 0.5 0.5\n'
                    'gal_filternames = wise2010-W1 wise2010-W2 wise2010-W3 wise2010-W4\n'
                    'gal_al_avs = 0.039 0.026 0.015 0.005\n')
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'), idx,
                      new_line, out_file=os.path.join(os.path.dirname(__file__),
                      'data/cat_b_params_.txt'))

        cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                             'data/crossmatch_params_.txt'),
                             os.path.join(os.path.dirname(__file__),
                             'data/cat_a_params_.txt'),
                             os.path.join(os.path.dirname(__file__),
                             'data/cat_b_params_.txt'))
        assert_allclose(cm.a_gal_zmax, np.array([4.5, 4.5, 5.0]))
        assert np.all(cm.b_gal_nzs == np.array([33, 41, 11, 41]))
        assert np.all(cm.a_gal_filternames == ['gaiadr2-BP', 'gaiadr2-G', 'gaiadr2-RP'])

        f = open(os.path.join(os.path.dirname(__file__), 'data/cat_b_params_.txt')).readlines()
        for i, key in enumerate(['gal_wavs', 'gal_zmax', 'gal_nzs',
                                 'gal_aboffsets', 'gal_filternames', 'gal_al_avs']):
            idx = np.where([key in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_b_params_.txt'),
                          idx, '', out_file=os.path.join(os.path.dirname(__file__),
                          'data/cat_b_params__.txt'))
            with pytest.raises(ValueError,
                               match='Missing key {} from catalogue "b"'.format(key)):
                cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                     'data/crossmatch_params_.txt'),
                                     os.path.join(os.path.dirname(__file__),
                                     'data/cat_a_params_.txt'),
                                     os.path.join(os.path.dirname(__file__),
                                     'data/cat_b_params__.txt'))

        for old_line, new_line, match_text, file in zip(
                ['gal_wavs = 0.513 0.641 0.778', 'gal_aboffsets = 0.5 0.5 0.5 0.5',
                 'gal_nzs = 46 46 51', 'gal_nzs = 33 41 11 41', 'gal_nzs = 33 41 11 41',
                 'gal_filternames = gaiadr2-BP gaiadr2-G gaiadr2-RP',
                 'gal_al_avs = 0.589 0.789 1.002', 'gal_al_avs = 0.589 0.789 1.002'],
                ['gal_wavs = 0.513 0.641\n', 'gal_aboffsets = a 0.5 0.5 0.5\n',
                 'gal_nzs = 46 a 51\n', 'gal_nzs = 33.1 41 11 41\n', 'gal_nzs = 33 41 11\n',
                 'gal_filternames = gaiadr2-BP gaiadr2-G gaiadr2-RP wise2010-W1\n',
                 'gal_al_avs = words\n', 'gal_al_avs = 0.789 1.002\n'],
                ['a_gal_wavs and a_filt_names should contain the same number',
                 'gal_aboffsets should be a list of floats in catalogue "b"',
                 'gal_nzs should be a list of integers in catalogue "a"',
                 'All elements of b_gal_nzs should be integers.',
                 'b_gal_nzs and b_filt_names should contain the same number of entries.',
                 'a_gal_filternames and a_filt_names should contain the same number of entries.',
                 'gal_al_avs should be a list of floats in catalogue "a"',
                 'a_gal_al_avs and a_filt_names should contain the same number of entries.'],
                ['cat_a_params', 'cat_b_params', 'cat_a_params', 'cat_b_params', 'cat_b_params',
                 'cat_a_params', 'cat_a_params', 'cat_a_params']):
            f = open(os.path.join(os.path.dirname(__file__),
                                  'data/{}_.txt'.format(file))).readlines()
            idx = np.where([old_line in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__),
                          'data/{}_.txt'.format(file)), idx, new_line, out_file=os.path.join(
                          os.path.dirname(__file__), 'data/{}__.txt'.format(file)))

            with pytest.raises(ValueError, match=match_text):
                cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                     'data/crossmatch_params_.txt'),
                                     os.path.join(os.path.dirname(__file__),
                                     'data/cat_a_params{}.txt'
                                                  .format('_' if '_b_' in file else '__')),
                                     os.path.join(os.path.dirname(__file__),
                                     'data/cat_b_params{}.txt'
                                                  .format('_' if '_a_' in file else '__')))

    def test_crossmatch_fourier_inputs(self):
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'), use_memmap_files=True)
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        assert cm.real_hankel_points == 10000
        assert cm.four_hankel_points == 10000
        assert cm.four_max_rho == 100

        # List of simple one line config file replacements for error message checking
        for old_line, new_line, match_text in zip(
                ['real_hankel_points = 10000', 'four_hankel_points = 10000', 'four_max_rho = 100'],
                ['', 'four_hankel_points = 10000.1\n', 'four_max_rho = word\n'],
                ['Missing key real_hankel_points', 'four_hankel_points should be an integer.',
                 'four_max_rho should be an integer.']):
            f = open(os.path.join(os.path.dirname(__file__),
                                  'data/crossmatch_params.txt')).readlines()
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
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'), use_memmap_files=True)
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        assert cm.a_auf_region_frame == 'equatorial'
        assert cm.b_auf_region_frame == 'equatorial'
        assert cm.cf_region_frame == 'equatorial'

        # List of simple one line config file replacements for error message checking
        match_text = 'Region frames for c/f and AUF creation must all be the same.'
        for old_line, new_line, in_file in zip(
                ['cf_region_frame = equatorial', 'auf_region_frame = equatorial',
                 'auf_region_frame = equatorial'],
                ['cf_region_frame = galactic\n', 'auf_region_frame = galactic\n',
                 'auf_region_frame = galactic\n'],
                ['crossmatch_params', 'cat_a_params', 'cat_b_params']):
            f = open(os.path.join(os.path.dirname(__file__),
                                  'data/{}.txt'.format(in_file))).readlines()
            idx = np.where([old_line in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__),
                          'data/{}.txt'.format(in_file)), idx, new_line, out_file=os.path.join(
                          os.path.dirname(__file__), 'data/{}_.txt'.format(in_file)))

            with pytest.raises(ValueError, match=match_text):
                cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                     'data/crossmatch_params{}.txt'.format(
                                     '_' if 'h_p' in in_file else '')),
                                     os.path.join(os.path.dirname(__file__),
                                     'data/cat_a_params{}.txt'
                                                  .format('_' if '_a_' in in_file else '')),
                                     os.path.join(os.path.dirname(__file__),
                                     'data/cat_b_params{}.txt'
                                                  .format('_' if '_b_' in in_file else '')))

    def test_cross_match_extent(self):
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'), use_memmap_files=True)
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        assert np.all(cm.cross_match_extent == np.array([131, 138, -3, 3]))

        # List of simple one line config file replacements for error message checking
        in_file = 'crossmatch_params'
        f = open(os.path.join(os.path.dirname(__file__),
                              'data/{}.txt'.format(in_file))).readlines()
        old_line = 'cross_match_extent = 131 138 -3 3'
        for new_line, match_text in zip(
                ['', 'cross_match_extent = 131 138 -3 word\n', 'cross_match_extent = 131 138 -3\n',
                 'cross_match_extent = 131 138 -3 3 1'],
                ['Missing key cross_match_extent', 'All elements of cross_match_extent should be',
                 'cross_match_extent should contain.', 'cross_match_extent should contain']):
            idx = np.where([old_line in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__),
                          'data/{}.txt'.format(in_file)), idx, new_line, out_file=os.path.join(
                          os.path.dirname(__file__), 'data/{}_.txt'.format(in_file)))

            with pytest.raises(ValueError, match=match_text):
                cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                     'data/crossmatch_params{}.txt'.format(
                                     '_' if 'h_p' in in_file else '')),
                                     os.path.join(os.path.dirname(__file__),
                                     'data/cat_a_params{}.txt'
                                                  .format('_' if '_a_' in in_file else '')),
                                     os.path.join(os.path.dirname(__file__),
                                     'data/cat_b_params{}.txt'
                                                  .format('_' if '_b_' in in_file else '')))

    def test_int_fracs(self):
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'), use_memmap_files=True)
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        assert np.all(cm.int_fracs == np.array([0.63, 0.9, 0.99]))

        # List of simple one line config file replacements for error message checking
        in_file = 'crossmatch_params'
        f = open(os.path.join(os.path.dirname(__file__),
                              'data/{}.txt'.format(in_file))).readlines()
        old_line = 'int_fracs = 0.63 0.9 0.99'
        for new_line, match_text in zip(
                ['', 'int_fracs = 0.63 0.9 word\n', 'int_fracs = 0.63 0.9\n'],
                ['Missing key int_fracs', 'All elements of int_fracs should be',
                 'int_fracs should contain.']):
            idx = np.where([old_line in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__),
                          'data/{}.txt'.format(in_file)), idx, new_line, out_file=os.path.join(
                          os.path.dirname(__file__), 'data/{}_.txt'.format(in_file)))

            with pytest.raises(ValueError, match=match_text):
                cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                     'data/crossmatch_params{}.txt'.format(
                                     '_' if 'h_p' in in_file else '')),
                                     os.path.join(os.path.dirname(__file__),
                                     'data/cat_a_params{}.txt'
                                                  .format('_' if '_a_' in in_file else '')),
                                     os.path.join(os.path.dirname(__file__),
                                     'data/cat_b_params{}.txt'
                                                  .format('_' if '_b_' in in_file else '')))

    def test_crossmatch_chunk_num(self):
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'), use_memmap_files=True)
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        assert np.all(cm.mem_chunk_num == 10)

        # List of simple one line config file replacements for error message checking
        in_file = 'crossmatch_params'
        f = open(os.path.join(os.path.dirname(__file__),
                              'data/{}.txt'.format(in_file))).readlines()
        old_line = 'mem_chunk_num = 10'
        for new_line, match_text in zip(
                ['', 'mem_chunk_num = word\n', 'mem_chunk_num = 10.1\n'],
                ['Missing key mem_chunk_num', 'mem_chunk_num should be a single integer',
                 'mem_chunk_num should be a single integer']):
            idx = np.where([old_line in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__),
                          'data/{}.txt'.format(in_file)), idx, new_line, out_file=os.path.join(
                          os.path.dirname(__file__), 'data/{}_.txt'.format(in_file)))

            with pytest.raises(ValueError, match=match_text):
                cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                     'data/crossmatch_params{}.txt'.format(
                                     '_' if 'h_p' in in_file else '')),
                                     os.path.join(os.path.dirname(__file__),
                                     'data/cat_a_params{}.txt'
                                                  .format('_' if '_a_' in in_file else '')),
                                     os.path.join(os.path.dirname(__file__),
                                     'data/cat_b_params{}.txt'
                                                  .format('_' if '_b_' in in_file else '')))

    def test_crossmatch_shared_data(self):
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'), use_memmap_files=True)
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        assert np.all(cm.r == np.linspace(0, 11, 10000))
        assert_allclose(cm.dr, np.ones(9999, float) * 11/9999)
        assert np.all(cm.rho == np.linspace(0, 100, 10000))
        assert_allclose(cm.drho, np.ones(9999, float) * 100/9999)

    def test_cat_folder_path(self):
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'), use_memmap_files=True)
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        assert os.path.exists(self.a_cat_folder_path)
        assert os.path.exists(self.b_cat_folder_path)
        assert cm.a_cat_folder_path == self.a_cat_folder_path
        assert np.all(np.load('{}/con_cat_astro.npy'.format(
                      self.a_cat_folder_path)).shape == (2, 3))
        assert np.all(np.load('{}/con_cat_photo.npy'.format(
                      self.b_cat_folder_path)).shape == (2, 4))
        assert np.all(np.load('{}/magref.npy'.format(
                      self.b_cat_folder_path)).shape == (2,))

        os.system('rm -rf {}'.format(self.a_cat_folder_path))
        with pytest.raises(OSError, match="a_cat_folder_path does not exist."):
            cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                              'data/crossmatch_params.txt'),
                                 os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                                 os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        self.setup_class()

        os.system('rm -rf {}'.format(self.b_cat_folder_path))
        with pytest.raises(OSError, match="b_cat_folder_path does not exist."):
            cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                              'data/crossmatch_params.txt'),
                                 os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                                 os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        self.setup_class()

        for catpath, file in zip([self.a_cat_folder_path, self.b_cat_folder_path],
                                 ['con_cat_astro', 'magref']):
            os.system('rm {}/{}.npy'.format(catpath, file))
            with pytest.raises(FileNotFoundError,
                               match='{} file not found in catalogue '.format(file)):
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
            np.save('{}/{}.npy'.format(self.b_cat_folder_path, name), data)
            with pytest.raises(ValueError, match=match):
                cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                     'data/crossmatch_params.txt'),
                                     os.path.join(os.path.dirname(__file__),
                                                  'data/cat_a_params.txt'),
                                     os.path.join(os.path.dirname(__file__),
                                                  'data/cat_b_params.txt'))
            self.setup_class()

    def test_calculate_cf_areas(self):
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'), use_memmap_files=True)
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        cm.cross_match_extent = np.array([131, 134, -1, 1])
        cm.cf_region_points = np.array([[a, b] for a in [131.5, 132.5, 133.5]
                                        for b in [-0.5, 0.5]])
        cm._calculate_cf_areas()
        assert_allclose(cm.cf_areas, np.ones((6), float), rtol=0.02)

        cm.cross_match_extent = np.array([50, 55, 85, 90])
        cm.cf_region_points = np.array([[a, b] for a in 0.5+np.arange(50, 55, 1)
                                        for b in 0.5+np.arange(85, 90, 1)])
        cm._calculate_cf_areas()
        calculated_areas = np.array(
            [(c[0]+0.5 - (c[0]-0.5))*180/np.pi * (np.sin(np.radians(c[1]+0.5)) -
             np.sin(np.radians(c[1]-0.5))) for c in cm.cf_region_points])
        assert_allclose(cm.cf_areas, calculated_areas, rtol=0.025)


@pytest.mark.parametrize('use_memmap', [True, False])
class TestPostProcess:
    def setup_class(self):
        self.joint_folder_path = os.path.abspath('joint')
        self.a_cat_folder_path = os.path.abspath('a_cat')
        self.b_cat_folder_path = os.path.abspath('b_cat')

        os.makedirs('{}/pairing'.format(self.joint_folder_path), exist_ok=True)
        os.makedirs(self.a_cat_folder_path, exist_ok=True)
        os.makedirs(self.b_cat_folder_path, exist_ok=True)

        Na, Nb, Nmatch = 10000, 7000, 4000

        rng = np.random.default_rng(seed=7893467234)
        self.ac = rng.choice(Na, size=Nmatch, replace=False)
        np.save('{}/pairing/ac.npy'.format(self.joint_folder_path), self.ac)
        self.bc = rng.choice(Nb, size=Nmatch, replace=False)
        np.save('{}/pairing/bc.npy'.format(self.joint_folder_path), self.bc)

        self.af = np.delete(np.arange(Na), self.ac)
        np.save('{}/pairing/af.npy'.format(self.joint_folder_path), self.af)
        self.bf = np.delete(np.arange(Nb), self.bc)
        np.save('{}/pairing/bf.npy'.format(self.joint_folder_path), self.bf)

        np.save('{}/in_chunk_overlap.npy'.format(self.a_cat_folder_path),
                rng.choice(2, size=Na).astype(bool))
        np.save('{}/in_chunk_overlap.npy'.format(self.b_cat_folder_path),
                rng.choice(2, size=Nb).astype(bool))

    def test_postprocess(self, use_memmap):
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'),
                        use_memmap_files=use_memmap)
        cm.joint_folder_path = self.joint_folder_path
        cm.a_cat_folder_path = self.a_cat_folder_path
        cm.b_cat_folder_path = self.b_cat_folder_path

        cm._postprocess_chunk()

        aino = np.load('{}/in_chunk_overlap.npy'.format(self.a_cat_folder_path))
        bino = np.load('{}/in_chunk_overlap.npy'.format(self.b_cat_folder_path))
        ac = np.load('{}/pairing/ac.npy'.format(self.joint_folder_path))
        af = np.load('{}/pairing/af.npy'.format(self.joint_folder_path))
        bc = np.load('{}/pairing/bc.npy'.format(self.joint_folder_path))
        bf = np.load('{}/pairing/bf.npy'.format(self.joint_folder_path))

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
