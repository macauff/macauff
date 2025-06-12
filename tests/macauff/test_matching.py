# Licensed under a 3-clause BSD style license - see LICENSE
'''
Tests for the "matching" module.
'''

# pylint: disable=too-many-lines,duplicate-code

import os

import numpy as np
import pandas as pd
import pytest
import yaml
from numpy.testing import assert_allclose
from test_fit_astrometry import TestAstroCorrection as TAC
from test_utils import mock_filename

from macauff.macauff import Macauff
from macauff.matching import CrossMatch
from macauff.misc_functions import convex_hull_area
from macauff.perturbation_auf import make_tri_counts


class TestInputs:
    def setup_class(self):
        self.chunk_id = 9
        os.makedirs('a_cat', exist_ok=True)
        os.makedirs('b_cat', exist_ok=True)
        with open(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.yaml'),
                  encoding='utf-8') as f:
            joint_config = yaml.safe_load(f)
        with open(os.path.join(os.path.dirname(__file__), 'data/cat_a_params.yaml'), encoding='utf-8') as f:
            cat_a_config = yaml.safe_load(f)
        with open(os.path.join(os.path.dirname(__file__), 'data/cat_b_params.yaml'), encoding='utf-8') as f:
            cat_b_config = yaml.safe_load(f)
        self.a_cat_csv_file_path = os.path.abspath(cat_a_config['cat_csv_file_path'].format(self.chunk_id))
        self.b_cat_csv_file_path = os.path.abspath(cat_b_config['cat_csv_file_path'].format(self.chunk_id))

        os.makedirs(joint_config['output_save_folder'].format(self.chunk_id), exist_ok=True)
        os.makedirs(os.path.dirname(self.a_cat_csv_file_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.b_cat_csv_file_path), exist_ok=True)

        self.a_cat = '0, 0, 0, 0, 0, 0, 0, 1\n0, 0, 0, 0, 0, 0, 0, 2'
        self.b_cat = '0, 0, 0, 0, 0, 0, 0, 0, 0\n0, 0, 0, 0, 0, 0, 0, 0, 3'
        with open(self.a_cat_csv_file_path, "w", encoding='utf-8') as f:
            f.write(self.a_cat)
        with open(self.b_cat_csv_file_path, "w", encoding='utf-8') as f:
            f.write(self.b_cat)
        os.makedirs('data', exist_ok=True)

        with open(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.yaml'),
                  encoding='utf-8') as cm_p:
            self.cm_p_text = cm_p.read()
        with open(os.path.join(os.path.dirname(__file__), 'data/cat_a_params.yaml'),
                  encoding='utf-8') as ca_p:
            self.ca_p_text = ca_p.read()
        with open(os.path.join(os.path.dirname(__file__), 'data/cat_b_params.yaml'),
                  encoding='utf-8') as cb_p:
            self.cb_p_text = cb_p.read()

    def test_crossmatch_run_input(self):
        with pytest.raises(FileNotFoundError):
            CrossMatch(*[os.path.join(os.path.dirname(__file__), f'data/{x}') for x in [
                './file.yaml', './file2.yaml', './file3.yaml']])
        with pytest.raises(FileNotFoundError):
            CrossMatch(*[os.path.join(os.path.dirname(__file__), f'data/{x}') for x in [
                os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.yaml'), './file2.yaml',
                './file3.yaml']])
        with pytest.raises(FileNotFoundError):
            CrossMatch(*[os.path.join(os.path.dirname(__file__), f'data/{x}') for x in [
                os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.yaml'),
                os.path.join(os.path.dirname(__file__), 'data/cat_a_params.yaml'), './file3.yaml']])

    # pylint: disable-next=too-many-statements
    def test_crossmatch_auf_cf_input(self):
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.yaml'),
                        os.path.join(os.path.dirname(__file__), 'data/cat_a_params.yaml'),
                        os.path.join(os.path.dirname(__file__), 'data/cat_b_params.yaml'))
        cm._load_metadata_config(self.chunk_id)
        assert cm.cf_region_frame == 'equatorial'  # pylint: disable=no-member
        assert_allclose(cm.cf_region_points,
                        np.array([[131, -1], [132, -1], [133, -1], [134, -1],
                                  [131, 0], [132, 0], [133, 0], [134, 0],
                                  [131, 1], [132, 1], [133, 1], [134, 1]]))

        cm_p_ = self.cm_p_text.replace('include_perturb_auf: False', 'include_perturb_auf: True')
        ca_p_ = self.ca_p_text.replace(r'auf_file_path: gaia_auf_folder/trilegal_download_{}.dat',
                                       r'auf_file_path: gaia_auf_folder/trilegal_download_{}.dat'
                                       '\ndens_hist_tri_location: None\ntri_model_mags_location: None\n'
                                       'tri_model_mag_mids_location: None\n'
                                       'tri_model_mags_interval_location: None\n'
                                       'tri_dens_uncert_location: None\n'
                                       'tri_n_bright_sources_star_location: None')
        cb_p_ = self.cb_p_text.replace(r'auf_file_path: wise_auf_folder/trilegal_download_{}.dat',
                                       r'auf_file_path: wise_auf_folder/trilegal_download_{}.dat' + '\n'
                                       'dens_hist_tri_location: None\ntri_model_mags_location: None\n'
                                       'tri_model_mag_mids_location: None\n'
                                       'tri_model_mags_interval_location: None\n'
                                       'tri_dens_uncert_location: None\n'
                                       'tri_n_bright_sources_star_location: None')

        cm = CrossMatch(mock_filename(cm_p_.encode("utf-8")), mock_filename(ca_p_.encode("utf-8")),
                        mock_filename(cb_p_.encode("utf-8")))
        cm._load_metadata_config(self.chunk_id)
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

        c = os.path.join(os.path.dirname(__file__), 'data/cat_b_params.yaml')
        for kind in ['auf_region_', 'cf_region_']:
            in_file = self.cm_p_text if 'cf' in kind else self.ca_p_text
            # List of simple one line config file replacements for error message checking
            for old_line, new_line, match_text in zip(
                [f'{kind}type: rectangle', f'{kind}type: rectangle',
                 f'{kind}points_per_chunk:\n  - [131, 134, 4, -1, 1, 3]',
                 f'{kind}points_per_chunk:\n  - [131, 134, 4, -1, 1, 3]', f'{kind}frame: equatorial',
                 f'{kind}points_per_chunk:\n  - [131, 134, 4, -1, 1, 3]'],
                ['', f'{kind}type: triangle', f'{kind}points_per_chunk:\n  - [131, 134, 4, -1, 1, a]',
                 f'{kind}points_per_chunk:\n  - [131, 134, 4, -1, 1]', f'{kind}frame: ecliptic',
                 f'{kind}points_per_chunk:\n  - [131, 134, 4, -1, 1, 3.4]'],
                [f'Missing key {kind}type',
                 f"{kind}type should either be 'rectangle' or 'points' in the "
                 f"{'joint' if 'cf' in kind else 'catalogue a'}",
                 f"{'' if 'cf' in kind else 'a_'}{kind}points should be 6 numbers",
                 f"{'' if 'cf' in kind else 'a_'}{kind}points should be 6 numbers",
                 f"{kind}frame should either be 'equatorial' or 'galactic' in the "
                 f"{'joint' if 'cf' in kind else 'catalogue a'}",
                 f"start and stop values for {'' if 'cf' in kind else 'a_'}{kind}points"]):
                new_in_file = in_file.replace(old_line, new_line)

                a = (os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.yaml') if 'cf'
                     not in kind else mock_filename(new_in_file.encode('utf-8')))
                b = (os.path.join(os.path.dirname(__file__), 'data/cat_a_params.yaml') if 'cf'
                     in kind else mock_filename(new_in_file.encode('utf-8')))
                with pytest.raises(ValueError, match=match_text):
                    cm = CrossMatch(a, b, c)
                    cm._load_metadata_config(self.chunk_id)

            # Check correct and incorrect *_region_points when *_region_type is 'points'
            new_in_file = in_file.replace(f'{kind}type: rectangle', f'{kind}type: points')
            new_in_file = new_in_file.replace(f'{kind}points_per_chunk:\n  - [131, 134, 4, -1, 1, 3]',
                                              f'{kind}points_per_chunk:\n  - [[131, 0], [133, 0], [132, -1]]')
            a = (os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.yaml') if 'cf'
                 not in kind else mock_filename(new_in_file.encode('utf-8')))
            b = (os.path.join(os.path.dirname(__file__), 'data/cat_a_params.yaml') if 'cf'
                 in kind else mock_filename(new_in_file.encode('utf-8')))
            cm = CrossMatch(a, b, c)
            cm._load_metadata_config(self.chunk_id)
            assert_allclose(getattr(cm, f"{'' if 'cf' in kind else 'a_'}{kind}points"),
                            np.array([[131, 0], [133, 0], [132, -1]]))

            for new_line in [f'{kind}points_per_chunk:\n  - [[131, 0], [131, ]]',
                             f'{kind}points_per_chunk:\n  - [[131, 0], [131, 1, 2]]',
                             f'{kind}points_per_chunk:\n  - [[131, 0], [131, a]]']:
                new_in_file = in_file.replace(f'{kind}type: rectangle', f'{kind}type: points')
                new_in_file = new_in_file.replace(f'{kind}points_per_chunk:\n  - [131, 134, 4, -1, 1, 3]',
                                                  new_line)

                with pytest.raises(ValueError):
                    a = (os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.yaml') if 'cf'
                         not in kind else mock_filename(new_in_file.encode('utf-8')))
                    b = (os.path.join(os.path.dirname(__file__), 'data/cat_a_params.yaml') if 'cf'
                         in kind else mock_filename(new_in_file.encode('utf-8')))
                    cm = CrossMatch(a, b, c)
                    cm._load_metadata_config(self.chunk_id)

            # Check single-length point grids are fine
            new_in_file = in_file.replace(f'{kind}points_per_chunk:\n  - [131, 134, 4, -1, 1, 3]',
                                          f'{kind}points_per_chunk:\n  - [131, 131, 1, 0, 0, 1]')
            a = (os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.yaml') if 'cf'
                 not in kind else mock_filename(new_in_file.encode('utf-8')))
            b = (os.path.join(os.path.dirname(__file__), 'data/cat_a_params.yaml') if 'cf'
                 in kind else mock_filename(new_in_file.encode('utf-8')))
            cm = CrossMatch(a, b, c)
            cm._load_metadata_config(self.chunk_id)
            assert_allclose(getattr(cm, f"{'' if 'cf' in kind else 'a_'}{kind}points"), np.array([[131, 0]]))

            new_in_file = in_file.replace(f'{kind}type: rectangle', f'{kind}type: points')
            new_in_file = new_in_file.replace(f'{kind}points_per_chunk:\n  - [131, 134, 4, -1, 1, 3]',
                                              f'{kind}points_per_chunk:\n  - [[131, 0]]')
            a = (os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.yaml') if 'cf'
                 not in kind else mock_filename(new_in_file.encode('utf-8')))
            b = (os.path.join(os.path.dirname(__file__), 'data/cat_a_params.yaml') if 'cf'
                 in kind else mock_filename(new_in_file.encode('utf-8')))
            cm = CrossMatch(a, b, c)
            cm._load_metadata_config(self.chunk_id)
            assert_allclose(getattr(cm, f"{'' if 'cf' in kind else 'a_'}{kind}points"), np.array([[131, 0]]))

        # Check galactic run is also fine -- here we have to replace all 3 parameter
        # options with "galactic", however.
        new_in_files = [None, None, None]
        for i, in_file in enumerate([self.cm_p_text, self.ca_p_text, self.cb_p_text]):
            kind = 'cf_region_' if i == 0 else 'auf_region_'
            new_in_files[i] = in_file.replace(f'{kind}frame: equatorial', f'{kind}frame: galactic')
        cm = CrossMatch(*[mock_filename(x.encode("utf-8")) for x in new_in_files])
        cm._load_metadata_config(self.chunk_id)
        for kind in ['auf_region_', 'cf_region_']:
            assert getattr(cm, f"{'' if 'cf' in kind else 'a_'}{kind}frame") == 'galactic'
            assert_allclose(getattr(cm, f"{'' if 'cf' in kind else 'a_'}{kind}points"),
                            np.array([[131, -1], [132, -1], [133, -1], [134, -1],
                                      [131, 0], [132, 0], [133, 0], [134, 0],
                                      [131, 1], [132, 1], [133, 1], [134, 1]]))

        cm_p_ = self.cm_p_text.replace('include_phot_like: False',
                                       'include_phot_like: True\nwith_and_without_photometry: False')
        cm = CrossMatch(mock_filename(cm_p_.encode("utf-8")),
                        os.path.join(os.path.dirname(__file__), 'data/cat_a_params.yaml'),
                        os.path.join(os.path.dirname(__file__), 'data/cat_b_params.yaml'))
        cm._load_metadata_config(self.chunk_id)
        assert cm.include_phot_like
        assert not cm.with_and_without_photometry

        for old_line, new_line, match_text in zip(
                ['with_and_without_photometry: False', 'with_and_without_photometry: False'],
                ['', 'with_and_without_photometry: banana'],
                ['Missing key with_and_without',
                 'Boolean flag key with_and_without_photometry not set to allowed']):
            cm_p_ = self.cm_p_text.replace('include_phot_like: False',
                                           'include_phot_like: True\nwith_and_without_photometry: False')
            cm_p_ = cm_p_.replace(old_line, new_line)
            with pytest.raises(ValueError, match=match_text):
                cm = CrossMatch(mock_filename(cm_p_.encode("utf-8")),
                                os.path.join(os.path.dirname(__file__), 'data/cat_a_params.yaml'),
                                os.path.join(os.path.dirname(__file__), 'data/cat_b_params.yaml'))
                cm._load_metadata_config(self.chunk_id)

    def test_crossmatch_load_tri_hists(self):  # pylint: disable=too-many-statements
        cm_p_ = self.cm_p_text.replace('include_perturb_auf: False', 'include_perturb_auf: True')
        ca_p_ = self.ca_p_text.replace(r'auf_file_path: gaia_auf_folder/trilegal_download_{}.dat',
                                       r'auf_file_path: None'
                                       '\ndens_hist_tri_location: None\ntri_model_mags_location: None\n'
                                       'tri_model_mag_mids_location: None\n'
                                       'tri_model_mags_interval_location: None\n'
                                       'tri_dens_uncert_location: None\n'
                                       'tri_n_bright_sources_star_location: None')
        with pytest.raises(ValueError,
                           match="Either all flags related to running TRILEGAL histogram generation within"):
            cm = CrossMatch(mock_filename(cm_p_.encode("utf-8")),
                            mock_filename(ca_p_.encode("utf-8")),
                            os.path.join(os.path.dirname(__file__), 'data/cat_b_params.yaml'))
            cm._load_metadata_config(self.chunk_id)

        cb_p_ = self.cb_p_text.replace(r'auf_file_path: wise_auf_folder/trilegal_download_{}.dat',
                                       r'auf_file_path: None'
                                       '\ndens_hist_tri_location: None\ntri_model_mags_location: None\n'
                                       'tri_model_mag_mids_location: None\n'
                                       'tri_model_mags_interval_location: None\n'
                                       'tri_dens_uncert_location: None\n'
                                       'tri_n_bright_sources_star_location: None')
        for flag in ['a', 'b']:
            for name in ['tri_set_name', 'tri_filt_names', 'tri_filt_num', 'download_tri',
                         'tri_maglim_faint', 'tri_num_faint']:
                lines = ca_p_.split('\n') if flag == 'a' else cb_p_.split('\n')
                ind = np.where([name in x for x in lines])[0][0]
                if flag == 'a':
                    ca_p_ = ca_p_.replace(lines[ind], f'{lines[ind].split(":")[0]}: None')
                else:
                    cb_p_ = cb_p_.replace(lines[ind], f'{lines[ind].split(":")[0]}: None')
        # With everything set to None we hit the "can't have anything set" error:
        with pytest.raises(ValueError, match="Ambiguity in whether TRILEGAL histogram generation is being "):
            cm = CrossMatch(mock_filename(cm_p_.encode("utf-8")),
                            mock_filename(ca_p_.encode("utf-8")),
                            mock_filename(cb_p_.encode("utf-8")))
            cm._load_metadata_config(self.chunk_id)
        for location, error, match_text in zip(
                ['some_fake_folder', 'data/dens_hist_tri.npy',
                 os.path.join(os.path.dirname(__file__), 'data/cat_a_params.yaml'), 'data/dens_hist_tri.npy'],
                [FileNotFoundError, ValueError, ValueError, ValueError],
                ['File not found for dens_hist_tri. Please verify',
                 'Either all flags related to running TRILEGAL histogram generation externa',
                 'File could not be loaded from dens_hist_tri',
                 'number of filters in a_filt_names and a_dens_hist_tri']):
            if 'npy' in location and 'number of filters in' in match_text:
                np.save(location, np.ones((5, 10), float))
            elif 'npy' in location:
                np.save(location, np.ones((3, 10), float))
            lines = ca_p_.split('\n')
            ind = np.where(['dens_hist_tri_location' in x for x in lines])[0][0]
            ca_p_2 = ca_p_.replace(lines[ind], f'dens_hist_tri_location: {location}')
            with pytest.raises(error, match=match_text):
                cm = CrossMatch(mock_filename(cm_p_.encode("utf-8")),
                                mock_filename(ca_p_2.encode("utf-8")),
                                mock_filename(cb_p_.encode("utf-8")))
        np.save('data/dens_hist_tri.npy', np.ones((3, 10), float))
        lines = ca_p_.split('\n')
        ind = np.where(['dens_hist_tri_location' in x for x in lines])[0][0]
        ca_p_2 = ca_p_.replace(lines[ind], 'dens_hist_tri_location: data/dens_hist_tri.npy')
        for file, match_text in zip([np.ones((5, 10), float), np.ones((3, 4), float)],
                                    ['The number of filter-elements in dens_hist_tri and tri_model_mags',
                                     'The number of magnitude-elements in dens_hist_tri and tri_model_mags']):
            np.save('tri_model_mags.npy', file)
            lines = ca_p_2.split('\n')
            ind = np.where(['tri_model_mags_location' in x for x in lines])[0][0]
            ca_p_2 = ca_p_2.replace(lines[ind], 'tri_model_mags_location: tri_model_mags.npy')
            with pytest.raises(ValueError, match=match_text):
                cm = CrossMatch(mock_filename(cm_p_.encode("utf-8")),
                                mock_filename(ca_p_2.encode("utf-8")),
                                mock_filename(cb_p_.encode("utf-8")))
        for catname, nfilts in zip(['a', 'b'], [3, 4]):
            for file, file_name in zip(
                    [np.ones((nfilts, 10), float), np.ones((nfilts, 10), float), np.ones((nfilts, 10), float),
                     np.ones((nfilts, 10), float), np.ones((nfilts, 10), float), np.ones((nfilts,), float)],
                    ['dens_hist_tri', 'tri_model_mags', 'tri_model_mag_mids', 'tri_model_mags_interval',
                     'tri_dens_uncert', 'tri_n_bright_sources_star']):
                np.save(f'data/{catname}_{file_name}.npy', file)
                lines = ca_p_.split('\n') if catname == 'a' else cb_p_.split('\n')
                ind = np.where([file_name in x for x in lines])[0][0]
                location = f'data/{catname}_{file_name}.npy'
                if catname == 'a':
                    ca_p_ = ca_p_.replace(lines[ind], f'{file_name}_location: {location}')
                else:
                    cb_p_ = cb_p_.replace(lines[ind], f'{file_name}_location: {location}')
        cm = CrossMatch(mock_filename(cm_p_.encode("utf-8")),
                        mock_filename(ca_p_.encode("utf-8")),
                        mock_filename(cb_p_.encode("utf-8")))
        cm._load_metadata_config(self.chunk_id)
        assert cm.b_auf_file_path is None
        assert np.all([b is None for b in cm.a_tri_filt_names])
        assert np.all(cm.a_dens_hist_tri_list == np.ones((3, 10), float))  # pylint: disable=no-member
        assert np.all(cm.a_tri_dens_uncert_list == np.ones((3, 10), float))  # pylint: disable=no-member
        # pylint: disable-next=no-member
        assert np.all(cm.b_tri_n_bright_sources_star_list == np.ones((4,), float))

    def test_crossmatch_folder_path_inputs(self):
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.yaml'),
                        os.path.join(os.path.dirname(__file__), 'data/cat_a_params.yaml'),
                        os.path.join(os.path.dirname(__file__), 'data/cat_b_params.yaml'))
        cm._load_metadata_config(self.chunk_id)
        assert cm.output_save_folder == os.path.join(os.getcwd(), 'test_path')
        assert os.path.isdir(os.path.join(os.getcwd(), 'test_path'))
        assert cm.a_auf_file_path == os.path.join(os.getcwd(),
                                                  r'gaia_auf_folder/trilegal_download_9_{:.2f}_{:.2f}.dat')
        assert cm.b_auf_file_path == os.path.join(os.getcwd(),
                                                  r'wise_auf_folder/trilegal_download_9_{:.2f}_{:.2f}.dat')

        # List of simple one line config file replacements for error message checking
        for old_line, new_line, match_text, error, fn in zip(
                ['output_save_folder: test_path', 'output_save_folder: test_path',
                 r'auf_file_path: gaia_auf_folder/trilegal_download_{}.dat',
                 r'auf_file_path: wise_auf_folder/trilegal_download_{}.dat'],
                ['', 'output_save_folder: /User/test/some/path/\n', '',
                 'auf_file_path: /User/test/some/path\n'],
                ['Missing key', 'Error when trying to create folder',
                 'Missing key auf_file_path from catalogue "a"',
                 'folder for catalogue "b" AUF outputs. Please ensure that b_auf_file_path'],
                [ValueError, OSError, ValueError, OSError], ['c', 'c', 'a', 'b']):
            _cmp = self.cm_p_text.replace(old_line, new_line) if fn == 'c' else self.cm_p_text
            _cap = self.ca_p_text.replace(old_line, new_line) if fn == 'a' else self.ca_p_text
            _cbp = self.cb_p_text.replace(old_line, new_line) if fn == 'b' else self.cb_p_text
            with pytest.raises(error, match=match_text):
                cm = CrossMatch(mock_filename(_cmp.encode("utf-8")),
                                mock_filename(_cap.encode("utf-8")),
                                mock_filename(_cbp.encode("utf-8")))
                cm._load_metadata_config(self.chunk_id)

    def test_crossmatch_tri_inputs(self):
        cm_p_ = self.cm_p_text.replace('include_perturb_auf: False', 'include_perturb_auf: True')
        ca_p_ = self.ca_p_text.replace(r'auf_file_path: gaia_auf_folder/trilegal_download_{}.dat',
                                       r'auf_file_path: gaia_auf_folder/trilegal_download_{}.dat'
                                       '\ndens_hist_tri_location: None\ntri_model_mags_location: None\n'
                                       'tri_model_mag_mids_location: None\n'
                                       'tri_model_mags_interval_location: None\n'
                                       'tri_dens_uncert_location: None\n'
                                       'tri_n_bright_sources_star_location: None')
        cb_p_ = self.cb_p_text.replace(r'auf_file_path: wise_auf_folder/trilegal_download_{}.dat',
                                       r'auf_file_path: wise_auf_folder/trilegal_download_{}.dat'
                                       '\ndens_hist_tri_location: None\ntri_model_mags_location: None\n'
                                       'tri_model_mag_mids_location: None\n'
                                       'tri_model_mags_interval_location: None\n'
                                       'tri_dens_uncert_location: None\n'
                                       'tri_n_bright_sources_star_location: None')
        cm = CrossMatch(mock_filename(cm_p_.encode("utf-8")),
                        mock_filename(ca_p_.encode("utf-8")),
                        mock_filename(cb_p_.encode("utf-8")))
        cm._load_metadata_config(self.chunk_id)
        assert cm.a_tri_set_name == 'gaiaDR2'
        assert np.all(cm.b_tri_filt_names == np.array(['W1', 'W2', 'W3', 'W4']))  # pylint: disable=no-member
        assert cm.a_tri_filt_num == 1
        assert not cm.b_download_tri

        # List of simple one line config file replacements for error message checking
        for old_line, new_line, match_text, in_file in zip(
                ['tri_set_name: gaiaDR2', 'tri_filt_num: 11', 'tri_filt_num: 11',
                 'download_tri: False', 'tri_maglim_faint: 32',
                 'tri_maglim_faint: 32', 'tri_num_faint: 1500000', 'tri_num_faint: 1500000',
                 'tri_num_faint: 1500000'],
                ['', 'tri_filt_num: a', 'tri_filt_num: 3.4', 'download_tri: aye',
                 'tri_maglim_faint: [32, 33.5]', 'tri_maglim_faint: a',
                 'tri_num_faint: 1500000.1', 'tri_num_faint: a',
                 'tri_num_faint: [1500000, 15]'],
                ['Missing key tri_set_name from catalogue "a"',
                 'tri_filt_num should be a single integer number in catalogue "b"',
                 'tri_filt_num should be a single integer number in catalogue "b"',
                 'Boolean key download_tri not set',
                 'tri_maglim_faint in catalogue "a" must be a float.',
                 'tri_maglim_faint in catalogue "b" must be a float.',
                 'tri_num_faint should be a single integer number in catalogue "b"',
                 'tri_num_faint should be a single integer number in catalogue "a" metadata file',
                 'tri_num_faint should be a single integer number in catalogue "a"'],
                ['cat_a_params', 'cat_b_params', 'cat_b_params', 'cat_a_params', 'cat_a_params',
                 'cat_b_params', 'cat_b_params', 'cat_a_params', 'cat_a_params']):
            _cap = ca_p_.replace(old_line, new_line) if '_a_' in in_file else ca_p_
            _cbp = cb_p_.replace(old_line, new_line) if '_b_' in in_file else cb_p_
            with pytest.raises(ValueError, match=match_text):
                cm = CrossMatch(mock_filename(cm_p_.encode("utf-8")),
                                mock_filename(_cap.encode("utf-8")),
                                mock_filename(_cbp.encode("utf-8")))
                cm._load_metadata_config(self.chunk_id)

    def test_crossmatch_psf_param_inputs(self):
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.yaml'),
                        os.path.join(os.path.dirname(__file__), 'data/cat_a_params.yaml'),
                        os.path.join(os.path.dirname(__file__), 'data/cat_b_params.yaml'))
        cm._load_metadata_config(self.chunk_id)
        assert np.all(cm.b_filt_names == np.array(['W1', 'W2', 'W3', 'W4']))

        cm_p_ = self.cm_p_text.replace('include_perturb_auf: False', 'include_perturb_auf: True')
        ca_p_ = self.ca_p_text.replace(r'auf_file_path: gaia_auf_folder/trilegal_download_{}.dat',
                                       r'auf_file_path: gaia_auf_folder/trilegal_download_{}.dat'
                                       '\ndens_hist_tri_location: None\ntri_model_mags_location: None\n'
                                       'tri_model_mag_mids_location: None\n'
                                       'tri_model_mags_interval_location: None\n'
                                       'tri_dens_uncert_location: None\n'
                                       'tri_n_bright_sources_star_location: None')
        cb_p_ = self.cb_p_text.replace(r'auf_file_path: wise_auf_folder/trilegal_download_{}.dat',
                                       r'auf_file_path: wise_auf_folder/trilegal_download_{}.dat'
                                       '\ndens_hist_tri_location: None\ntri_model_mags_location: None\n'
                                       'tri_model_mag_mids_location: None\n'
                                       'tri_model_mags_interval_location: None\n'
                                       'tri_dens_uncert_location: None\n'
                                       'tri_n_bright_sources_star_location: None')
        cm = CrossMatch(mock_filename(cm_p_.encode("utf-8")),
                        mock_filename(ca_p_.encode("utf-8")),
                        mock_filename(cb_p_.encode("utf-8")))
        cm._load_metadata_config(self.chunk_id)

        cm = CrossMatch(mock_filename(cm_p_.encode("utf-8")),
                        mock_filename(ca_p_.encode("utf-8")),
                        mock_filename(cb_p_.encode("utf-8")))
        cm._load_metadata_config(self.chunk_id)
        assert np.all(cm.a_psf_fwhms == np.array([0.12, 0.12, 0.12]))  # pylint: disable=no-member

        # List of simple one line config file replacements for error message checking
        for old_line, new_line, match_text, in_file in zip(
                ['created\nfilt_names: [G_BP, G, G_RP]', 'created\nfilt_names: [G_BP, G, G_RP]',
                 'psf_fwhms: [6.08, 6.84, 7.36, 11.99]', 'psf_fwhms: [6.08, 6.84, 7.36, 11.99]'],
                ['created\n', 'created\nfilt_names: [G_BP, G]', 'psf_fwhms: [6.08, 6.84, 7.36]',
                 'psf_fwhms: [6.08, 6.84, 7.36, word]'],
                ['Missing key filt_names from catalogue "a"',
                 'a_gal_al_avs and a_filt_names should contain the same',
                 'b_psf_fwhms and b_filt_names should contain the same',
                 'psf_fwhms should be a list of floats in catalogue "b".'],
                ['cat_a_params', 'cat_a_params', 'cat_b_params', 'cat_b_params']):
            _cap = ca_p_.replace(old_line, new_line) if '_a_' in in_file else ca_p_
            _cbp = cb_p_.replace(old_line, new_line) if '_b_' in in_file else cb_p_
            if 'gal_al_avs' in match_text:
                _cap = _cap.replace('mag_indices: [3, 4, 5]', 'mag_indices: [3, 4]')
                _cap = _cap.replace('snr_indices: [8, 9, 10]', 'snr_indices: [8, 9]')
            with pytest.raises(ValueError, match=match_text):
                cm = CrossMatch(mock_filename(cm_p_.encode("utf-8")),
                                mock_filename(_cap.encode("utf-8")),
                                mock_filename(_cbp.encode("utf-8")))
                cm._load_metadata_config(self.chunk_id)

    def test_crossmatch_cat_name_inputs(self):
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.yaml'),
                        os.path.join(os.path.dirname(__file__), 'data/cat_a_params.yaml'),
                        os.path.join(os.path.dirname(__file__), 'data/cat_b_params.yaml'))
        cm._load_metadata_config(self.chunk_id)
        assert cm.b_cat_name == 'WISE'

        ca_p_ = self.ca_p_text.replace('cat_name: Gaia', '')
        match_text = 'Missing key cat_name from catalogue "a"'
        with pytest.raises(ValueError, match=match_text):
            cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.yaml'),
                            mock_filename(ca_p_.encode('utf-8')),
                            os.path.join(os.path.dirname(__file__), 'data/cat_b_params.yaml'))

    def test_crossmatch_search_inputs(self):
        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.yaml'),
                        os.path.join(os.path.dirname(__file__), 'data/cat_a_params.yaml'),
                        os.path.join(os.path.dirname(__file__), 'data/cat_b_params.yaml'))
        cm._load_metadata_config(self.chunk_id)
        assert cm.pos_corr_dist == 11

        cm_p_ = self.cm_p_text.replace('include_perturb_auf: False', 'include_perturb_auf: True')
        ca_p_ = self.ca_p_text.replace(r'auf_file_path: gaia_auf_folder/trilegal_download_{}.dat',
                                       r'auf_file_path: gaia_auf_folder/trilegal_download_{}.dat'
                                       '\ndens_hist_tri_location: None\ntri_model_mags_location: None\n'
                                       'tri_model_mag_mids_location: None\n'
                                       'tri_model_mags_interval_location: None\n'
                                       'tri_dens_uncert_location: None\n'
                                       'tri_n_bright_sources_star_location: None')
        cb_p_ = self.cb_p_text.replace(r'auf_file_path: wise_auf_folder/trilegal_download_{}.dat',
                                       r'auf_file_path: wise_auf_folder/trilegal_download_{}.dat'
                                       '\ndens_hist_tri_location: None\ntri_model_mags_location: None\n'
                                       'tri_model_mag_mids_location: None\n'
                                       'tri_model_mags_interval_location: None\n'
                                       'tri_dens_uncert_location: None\n'
                                       'tri_n_bright_sources_star_location: None')

        cm = CrossMatch(mock_filename(cm_p_.encode("utf-8")), mock_filename(ca_p_.encode("utf-8")),
                        mock_filename(cb_p_.encode("utf-8")))
        cm._load_metadata_config(self.chunk_id)
        assert np.all(cm.a_psf_fwhms == np.array([0.12, 0.12, 0.12]))  # pylint: disable=no-member
        assert cm.b_dens_dist == 0.25

        # List of simple one line config file replacements for error message checking
        for old_line, new_line, match_text, fn in zip(
                ['pos_corr_dist: 11', 'pos_corr_dist: 11', 'dens_dist: 0.25',
                 'dens_dist: 0.25'], ['', 'pos_corr_dist: word\n', '', 'dens_dist: word\n'],
                ['Missing key pos_corr_dist', 'pos_corr_dist must be a float',
                 'Missing key dens_dist from catalogue "b"', 'dens_dist in catalogue "a" must'],
                ['c', 'c', 'b', 'a']):
            _cmp = cm_p_.replace(old_line, new_line) if fn == 'c' else cm_p_
            _cap = ca_p_.replace(old_line, new_line) if fn == 'a' else ca_p_
            _cbp = cb_p_.replace(old_line, new_line) if fn == 'b' else cb_p_
            with pytest.raises(ValueError, match=match_text):
                cm = CrossMatch(mock_filename(_cmp.encode("utf-8")),
                                mock_filename(_cap.encode("utf-8")),
                                mock_filename(_cbp.encode("utf-8")))
                cm._load_metadata_config(self.chunk_id)

    # pylint: disable-next=too-many-statements
    def test_crossmatch_perturb_auf_inputs(self):
        cm_p_ = self.cm_p_text.replace('include_perturb_auf: False', 'include_perturb_auf: True')
        ca_p_ = self.ca_p_text.replace(r'auf_file_path: gaia_auf_folder/trilegal_download_{}.dat',
                                       r'auf_file_path: gaia_auf_folder/trilegal_download_{}.dat'
                                       '\ndens_hist_tri_location: None\ntri_model_mags_location: None\n'
                                       'tri_model_mag_mids_location: None\n'
                                       'tri_model_mags_interval_location: None\n'
                                       'tri_dens_uncert_location: None\n'
                                       'tri_n_bright_sources_star_location: None')
        cb_p_ = self.cb_p_text.replace(r'auf_file_path: wise_auf_folder/trilegal_download_{}.dat',
                                       r'auf_file_path: wise_auf_folder/trilegal_download_{}.dat'
                                       '\ndens_hist_tri_location: None\ntri_model_mags_location: None\n'
                                       'tri_model_mag_mids_location: None\n'
                                       'tri_model_mags_interval_location: None\n'
                                       'tri_dens_uncert_location: None\n'
                                       'tri_n_bright_sources_star_location: None')

        cm = CrossMatch(mock_filename(cm_p_.encode("utf-8")), mock_filename(ca_p_.encode("utf-8")),
                        mock_filename(cb_p_.encode("utf-8")))
        cm._load_metadata_config(self.chunk_id)
        assert cm.num_trials == 10000
        assert cm.d_mag == 0.1  # pylint: disable=no-member

        for old_line, new_line, match_text in zip(
                ['num_trials: 10000', 'num_trials: 10000', 'num_trials: 10000',
                 'd_mag: 0.1', 'd_mag: 0.1'],
                ['', 'num_trials: word\n', 'num_trials: 10000.1\n', '', 'd_mag: word\n'],
                ['Missing key num_trials from joint', 'num_trials should be an integer',
                 'num_trials should be an integer', 'Missing key d_mag from joint', 'd_mag must be a float']):
            # Make sure to keep the first edit of crossmatch_params, adding each
            # second change in turn.
            _cmp = cm_p_.replace(old_line, new_line)
            with pytest.raises(ValueError, match=match_text):
                cm = CrossMatch(mock_filename(_cmp.encode("utf-8")),
                                mock_filename(ca_p_.encode("utf-8")),
                                mock_filename(cb_p_.encode("utf-8")))

        for old_line, var_name in zip(['fit_gal_flag: False', 'run_fw_auf: True', 'run_psf_auf: False'],
                                      ['fit_gal_flag', 'run_fw_auf', 'run_psf_auf']):
            for cat_reg, fn in zip(['"a"', '"b"'], ['a', 'b']):
                _cap = ca_p_.replace(old_line, '') if fn == 'a' else ca_p_
                _cbp = cb_p_.replace(old_line, '') if fn == 'b' else cb_p_
                with pytest.raises(ValueError, match=f'Missing key {var_name} from catalogue {cat_reg}'):
                    cm = CrossMatch(mock_filename(cm_p_.encode("utf-8")),
                                    mock_filename(_cap.encode("utf-8")),
                                    mock_filename(_cbp.encode("utf-8")))

        cm = CrossMatch(mock_filename(cm_p_.encode("utf-8")), mock_filename(ca_p_.encode("utf-8")),
                        mock_filename(cb_p_.encode("utf-8")))
        cm._load_metadata_config(self.chunk_id)
        assert not hasattr(cm, 'a_dd_params_path')
        assert not hasattr(cm, 'b_l_cut_path')

        for cat_reg in ['"a"', '"b"']:
            if cat_reg[1] == 'a':
                x = ca_p_.replace('run_psf_auf: False', 'run_psf_auf: True\ndd_params_path: .\nl_cut_path: .')
            else:
                x = cb_p_.replace('run_psf_auf: False', 'run_psf_auf: True\ndd_params_path: .\nl_cut_path: .')
            for old_line, var_name in zip(['dd_params_path: .', 'l_cut_path: .'],
                                          ['dd_params_path', 'l_cut_path']):
                x2 = x.replace(old_line, '')
                b, c = (x2, cb_p_) if cat_reg[1] == 'a' else (ca_p_, x2)
                with pytest.raises(ValueError, match=f'Missing key {var_name} from catalogue {cat_reg}'):
                    cm = CrossMatch(mock_filename(cm_p_.encode("utf-8")),
                                    mock_filename(b.encode("utf-8")),
                                    mock_filename(c.encode("utf-8")))
                    cm._load_metadata_config(self.chunk_id)

        ddp = np.ones((5, 15, 2), float)
        np.save('dd_params.npy', ddp)
        lc = np.ones(3, float)
        np.save('l_cut.npy', lc)

        ca_p_2 = ca_p_.replace('run_psf_auf: False', 'run_psf_auf: True\ndd_params_path: .\nl_cut_path: .')
        cb_p_2 = cb_p_.replace('run_psf_auf: False', 'run_psf_auf: True\ndd_params_path: .\nl_cut_path: .')

        for fn, array, err_msg in zip([
                'dd_params', 'dd_params', 'dd_params', 'dd_params', 'l_cut', 'l_cut'],
                [np.ones(5, float), np.ones((5, 3), float), np.ones((4, 4, 2), float),
                 np.ones((5, 3, 1), float), np.ones((4, 2), float), np.ones(4, float)],
                [r'a_dd_params should be of shape \(5, X, 2\)',
                 r'a_dd_params should be of shape \(5, X, 2\)',
                 r'a_dd_params should be of shape \(5, X, 2\)',
                 r'a_dd_params should be of shape \(5, X, 2\)',
                 r'a_l_cut should be of shape \(3,\) only.',
                 r'a_l_cut should be of shape \(3,\) only.']):
            np.save(f"{fn}.npy", array)
            with pytest.raises(ValueError, match=err_msg):
                cm = CrossMatch(mock_filename(cm_p_.encode("utf-8")),
                                mock_filename(ca_p_2.encode("utf-8")),
                                mock_filename(cb_p_2.encode("utf-8")))
                cm._load_metadata_config(self.chunk_id)
            # Re-make "good" fake arrays
            ddp = np.ones((5, 15, 2), float)
            np.save('dd_params.npy', ddp)
            lc = np.ones(3, float)
            np.save('l_cut.npy', lc)

        ca_p_2 = ca_p_.replace('fit_gal_flag: False', 'fit_gal_flag: True\ngal_wavs: [0.513, 0.641, 0.778]\n'
                               'gal_zmax: [4.5, 4.5, 5]\ngal_nzs: [46, 46, 51]\n'
                               'gal_aboffsets: [0.5, 0.5, 0.5]\n'
                               'gal_filternames: [gaiadr2-BP, gaiadr2-G, gaiadr2-RP]\n'
                               'saturation_magnitudes: [5, 5, 5]\n')
        cb_p_2 = cb_p_.replace('fit_gal_flag: False', 'fit_gal_flag: True\n'
                               'gal_wavs: [3.37, 4.62, 12.08, 22.19]\ngal_zmax: [3.2, 4.0, 1, 4]\n'
                               'gal_nzs: [33, 41, 11, 41]\ngal_aboffsets: [0.5, 0.5, 0.5, 0.5]\n'
                               'gal_filternames: [wise2010-W1, wise2010-W2, wise2010-W3, wise2010-W4]\n'
                               'saturation_magnitudes: [5, 5, 5, 5]\n')
        cm = CrossMatch(mock_filename(cm_p_.encode("utf-8")),
                        mock_filename(ca_p_2.encode("utf-8")),
                        mock_filename(cb_p_2.encode("utf-8")))
        cm._load_metadata_config(self.chunk_id)
        assert_allclose(cm.a_gal_zmax, np.array([4.5, 4.5, 5.0]))
        assert np.all(cm.b_gal_nzs == np.array([33, 41, 11, 41]))
        assert np.all(cm.a_gal_filternames == ['gaiadr2-BP', 'gaiadr2-G', 'gaiadr2-RP'])

        for key in ['gal_wavs', 'gal_zmax', 'gal_nzs', 'gal_aboffsets', 'gal_filternames', 'gal_al_avs']:
            lines = cb_p_2.split('\n')
            ind = np.where([key in x for x in lines])[0][0]
            cb_p_3 = cb_p_2.replace(lines[ind], '')
            with pytest.raises(ValueError,
                               match=f'Missing key {key} from catalogue "b"'):
                cm = CrossMatch(mock_filename(cm_p_.encode("utf-8")),
                                mock_filename(ca_p_2.encode("utf-8")),
                                mock_filename(cb_p_3.encode("utf-8")))
                cm._load_metadata_config(self.chunk_id)

        for old_line, new_line, match_text, in_file in zip(
                ['gal_wavs: [0.513, 0.641, 0.778]', 'gal_aboffsets: [0.5, 0.5, 0.5, 0.5]',
                 'gal_nzs: [46, 46, 51]', 'gal_nzs: [33, 41, 11, 41]', 'gal_nzs: [33, 41, 11, 41]',
                 'gal_filternames: [gaiadr2-BP, gaiadr2-G, gaiadr2-RP]',
                 'gal_al_avs: [1.002, 0.789, 0.589]', 'gal_al_avs: [1.002, 0.789, 0.589]',
                 'saturation_magnitudes: [5, 5, 5, 5]', 'saturation_magnitudes: [5, 5, 5, 5]'],
                ['gal_wavs: [0.513, 0.641]', 'gal_aboffsets: [a, 0.5, 0.5, 0.5]',
                 'gal_nzs: [46, a, 51]', 'gal_nzs: [33.1, 41, 11, 41]', 'gal_nzs: [33, 41, 11]',
                 'gal_filternames: [gaiadr2-BP, gaiadr2-G, gaiadr2-RP, wise2010-W1]',
                 'gal_al_avs: words', 'gal_al_avs: [0.789, 1.002]',
                 'saturation_magnitudes: [words]', 'saturation_magnitudes: [4, 4]'],
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
                ['a', 'b', 'a', 'b', 'b', 'a', 'a', 'a', 'b', 'b']):
            b = ca_p_2 if in_file == 'b' else ca_p_2.replace(old_line, new_line)
            c = cb_p_2 if in_file == 'a' else cb_p_2.replace(old_line, new_line)
            with pytest.raises(ValueError, match=match_text):
                cm = CrossMatch(mock_filename(cm_p_.encode("utf-8")),
                                mock_filename(b.encode("utf-8")),
                                mock_filename(c.encode("utf-8")))
                cm._load_metadata_config(self.chunk_id)

    def test_crossmatch_fourier_inputs(self):
        cm = CrossMatch(mock_filename(self.cm_p_text.encode("utf-8")),
                        mock_filename(self.ca_p_text.encode("utf-8")),
                        mock_filename(self.cb_p_text.encode("utf-8")))
        cm._load_metadata_config(self.chunk_id)
        assert cm.real_hankel_points == 10000  # pylint: disable=no-member
        assert cm.four_hankel_points == 10000  # pylint: disable=no-member
        assert cm.four_max_rho == 100  # pylint: disable=no-member

        # List of simple one line config file replacements for error message checking
        for old_line, new_line, match_text in zip(
                ['real_hankel_points: 10000', 'four_hankel_points: 10000', 'four_max_rho: 100'],
                ['', 'four_hankel_points: 10000.1\n', 'four_max_rho: word\n'],
                ['Missing key real_hankel_points', 'four_hankel_points should be an integer.',
                 'four_max_rho should be an integer.']):
            cm_p_ = self.cm_p_text.replace(old_line, new_line)

            with pytest.raises(ValueError, match=match_text):
                cm = CrossMatch(mock_filename(cm_p_.encode("utf-8")),
                                mock_filename(self.ca_p_text.encode("utf-8")),
                                mock_filename(self.cb_p_text.encode("utf-8")))
                cm._load_metadata_config(self.chunk_id)

    def test_crossmatch_frame_equality(self):
        cm = CrossMatch(mock_filename(self.cm_p_text.encode("utf-8")),
                        mock_filename(self.ca_p_text.encode("utf-8")),
                        mock_filename(self.cb_p_text.encode("utf-8")))
        cm._load_metadata_config(self.chunk_id)
        assert cm.a_auf_region_frame == 'equatorial'  # pylint: disable=no-member
        assert cm.b_auf_region_frame == 'equatorial'  # pylint: disable=no-member
        assert cm.cf_region_frame == 'equatorial'  # pylint: disable=no-member

        # List of simple one line config file replacements for error message checking
        match_text = 'Region frames for c/f and AUF creation must all be the same.'
        for old_line, new_line, in_file in zip(
                ['cf_region_frame: equatorial', 'auf_region_frame: equatorial',
                 'auf_region_frame: equatorial'],
                ['cf_region_frame: galactic', 'auf_region_frame: galactic',
                 'auf_region_frame: galactic'], ['c', 'a', 'b']):
            a = self.cm_p_text.replace(old_line, new_line) if in_file == 'c' else self.cm_p_text
            b = self.ca_p_text.replace(old_line, new_line) if in_file == 'a' else self.ca_p_text
            c = self.cb_p_text.replace(old_line, new_line) if in_file == 'b' else self.cb_p_text
            with pytest.raises(ValueError, match=match_text):
                cm = CrossMatch(mock_filename(a.encode("utf-8")),
                                mock_filename(b.encode("utf-8")),
                                mock_filename(c.encode("utf-8")))
                cm._load_metadata_config(self.chunk_id)

    def test_int_fracs(self):
        cm = CrossMatch(mock_filename(self.cm_p_text.encode("utf-8")),
                        mock_filename(self.ca_p_text.encode("utf-8")),
                        mock_filename(self.cb_p_text.encode("utf-8")))
        cm._load_metadata_config(self.chunk_id)
        assert np.all(cm.int_fracs == np.array([0.63, 0.9, 0.999]))  # pylint: disable=no-member

        # List of simple one line config file replacements for error message checking
        old_line = 'int_fracs: [0.63, 0.9, 0.999]'
        for new_line, match_text in zip(
                ['', 'int_fracs: [0.63, 0.9, word]', 'int_fracs: [0.63, 0.9]'],
                ['Missing key int_fracs', 'All elements of int_fracs should be',
                 'int_fracs should contain.']):
            a = self.cm_p_text.replace(old_line, new_line)
            with pytest.raises(ValueError, match=match_text):
                cm = CrossMatch(mock_filename(a.encode("utf-8")),
                                mock_filename(self.ca_p_text.encode("utf-8")),
                                mock_filename(self.cb_p_text.encode("utf-8")))
                cm._load_metadata_config(self.chunk_id)

    def test_crossmatch_shared_data(self):
        cm = CrossMatch(mock_filename(self.cm_p_text.encode("utf-8")),
                        mock_filename(self.ca_p_text.encode("utf-8")),
                        mock_filename(self.cb_p_text.encode("utf-8")))
        cm._load_metadata_config(self.chunk_id)
        cm._initialise_chunk()
        assert np.all(cm.r == np.linspace(0, 11, 10000))
        assert_allclose(cm.dr, np.ones(9999, float) * 11/9999)
        assert np.all(cm.rho == np.linspace(0, 100, 10000))
        assert_allclose(cm.drho, np.ones(9999, float) * 100/9999)

    def test_cat_csv_file_path(self):
        cm = CrossMatch(mock_filename(self.cm_p_text.encode("utf-8")),
                        mock_filename(self.ca_p_text.encode("utf-8")),
                        mock_filename(self.cb_p_text.encode("utf-8")))
        cm._load_metadata_config(self.chunk_id)
        assert os.path.exists(os.path.dirname(self.a_cat_csv_file_path))
        assert os.path.exists(os.path.dirname(self.b_cat_csv_file_path))
        assert os.path.isfile(self.a_cat_csv_file_path)
        assert os.path.isfile(self.b_cat_csv_file_path)
        assert cm.a_cat_csv_file_path == self.a_cat_csv_file_path
        f = pd.read_csv(self.a_cat_csv_file_path, header=None)
        assert np.all(f.shape == (2, 8))
        assert np.all(f.values[:, :-1] == 0)
        assert np.all(f.values[:, -1] == [1, 2])
        f = pd.read_csv(self.b_cat_csv_file_path, header=None)
        assert np.all(f.shape == (2, 9))
        assert np.all(f.values[:, :-1] == 0)
        assert np.all(f.values[:, -1] == [0, 3])

        os.system(f'rm -rf {self.a_cat_csv_file_path}')
        with pytest.raises(OSError, match="a_cat_csv_file_path does not exist."):
            cm = CrossMatch(mock_filename(self.cm_p_text.encode("utf-8")),
                            mock_filename(self.ca_p_text.encode("utf-8")),
                            mock_filename(self.cb_p_text.encode("utf-8")))
            cm._load_metadata_config(self.chunk_id)
        self.setup_class()

        os.system(f'rm -rf {self.b_cat_csv_file_path}')
        with pytest.raises(OSError, match="b_cat_csv_file_path does not exist."):
            cm = CrossMatch(mock_filename(self.cm_p_text.encode("utf-8")),
                            mock_filename(self.ca_p_text.encode("utf-8")),
                            mock_filename(self.cb_p_text.encode("utf-8")))
            cm._load_metadata_config(self.chunk_id)
        self.setup_class()

        for catpath, f in zip([self.a_cat_csv_file_path, self.b_cat_csv_file_path], ['a', 'b']):
            os.system(f'rm {catpath}')
            with pytest.raises(OSError, match=f'{f}_cat_csv_file_path does not exist. '):
                cm = CrossMatch(mock_filename(self.cm_p_text.encode("utf-8")),
                                mock_filename(self.ca_p_text.encode("utf-8")),
                                mock_filename(self.cb_p_text.encode("utf-8")))
                cm._load_metadata_config(self.chunk_id)
                cm._initialise_chunk()
            self.setup_class()

    @pytest.mark.parametrize("shape", ["rectangle", "circle"])
    @pytest.mark.parametrize("shifted", [True, False])
    # pylint: disable-next=too-many-branches,too-many-statements
    def test_calculate_cf_areas(self, shape, shifted):
        def circle_integral(R, x):
            if 0 < x < 2*R:
                return 0.5 * ((2 * R**2 * np.arctan(
                    np.sqrt(x / (2 * R - x)))) + (x - R) * np.sqrt(x * (2 * R - x)))
            if x == 0:
                # arctan(0) = 0, sqrt(x (2R - x)) = 0
                return 0
            if x == 2 * R:
                # arctan(1/0) = pi/2, sqrt(x (2R - x)) = 0
                return R**2 * np.pi / 2
            return None
        cm = CrossMatch(mock_filename(self.cm_p_text.encode("utf-8")),
                        mock_filename(self.ca_p_text.encode("utf-8")),
                        mock_filename(self.cb_p_text.encode("utf-8")))
        cm._load_metadata_config(self.chunk_id)
        cm.cf_region_points = np.array([[a, b] for a in [131.5, 132.5, 133.5]
                                        for b in [-0.5, 0.5]])
        cm.chunk_id = 1
        mcff = Macauff(cm)
        if shape == 'rectangle':
            if shifted:
                a_astro = np.array([[131.5, 134, 131.5, 134], [-1, -1, 1, 1]]).T
                b_astro = np.array([[131, 133.5, 131, 133.5], [-1, -1, 1, 1]]).T
            else:
                a_astro = np.array([[131, 134, 131, 134], [-1, -1, 1, 1]]).T
                b_astro = np.array([[131, 134, 131, 134], [-1, -1, 1, 1]]).T
        else:
            t = np.linspace(0, 2*np.pi, 31)
            r = 1
            a_astro = np.array([r * np.cos(t) + 132.5, r * np.sin(t)]).T
            b_astro = np.array([r * np.cos(t) + 132.5, r * np.sin(t)]).T
            if shifted:
                b_astro[:, 0] = b_astro[:, 0] - 0.5
        _, cm.a_hull_points, cm.a_hull_x_shift = convex_hull_area(
            a_astro[:, 0], a_astro[:, 1], return_hull=True)
        _, cm.b_hull_points, cm.b_hull_x_shift = convex_hull_area(
            b_astro[:, 0], b_astro[:, 1], return_hull=True)
        mcff._calculate_cf_areas()
        if shape == "rectangle":
            if shifted:
                assert_allclose(cm.cf_areas, np.array([0.5, 0.5, 1, 1, 0.5, 0.5]), rtol=0.02)
            else:
                assert_allclose(cm.cf_areas, np.ones((6), float), rtol=0.02)
        else:
            if shifted:
                calculated_areas = np.zeros(len(cm.cf_region_points), float)
                xs = np.linspace(-0.5, 0.5, 1001)
                xs = xs[:-1] + np.diff(xs)/2
                for i, c in enumerate(cm.cf_region_points):
                    dx = (xs[1] - xs[0]) * np.cos(np.radians(c[1]))
                    for _x in xs:
                        # Get lon.
                        x = _x + c[0]
                        # Get boundaries in lat of circle, skipping lons outside R.
                        if np.abs(x - 132.5) <= r and np.abs(x - (132.5 - 0.5)) <= r:
                            lat1 = np.sqrt(r**2 - (x - 132.5)**2) * np.array([1, -1]) + 0
                            lat2 = np.sqrt(r**2 - (x - (132.5 - 0.5))**2) * np.array([1, -1]) + 0
                            min_lat = max(lat1[1], lat2[1], c[1] - 0.5)
                            max_lat = min(lat1[0], lat2[0], c[1] + 0.5)
                            calculated_areas[i] += max(0, max_lat - min_lat) * dx
            else:
                chop = circle_integral(r, 0.5) - circle_integral(r, 0)
                semi = circle_integral(r, 1.5) - circle_integral(r, 0.5)
                calculated_areas = np.array([chop, chop, semi, semi, chop, chop])
            assert_allclose(cm.cf_areas, calculated_areas, rtol=0.02)

        cm.cf_region_points = np.array([[a, b] for a in 0.5+np.arange(50, 55, 1)
                                        for b in 0.5+np.arange(70, 75, 1)])
        mcff = Macauff(cm)
        if shape == 'rectangle':
            if shifted:
                a_astro = np.array([[50, 54.5, 50, 54.5], [70, 70, 75, 75]]).T
                b_astro = np.array([[50.5, 55, 50.5, 55], [70, 70, 75, 75]]).T
            else:
                a_astro = np.array([[50, 55, 50, 55], [70, 70, 75, 75]]).T
                b_astro = np.array([[50, 55, 50, 55], [70, 70, 75, 75]]).T
        else:
            t = np.linspace(0, 2*np.pi, 361)
            r = 2.5
            a_astro = np.array([r * np.cos(t) + 52.5, r * np.sin(t) + 72.5]).T
            b_astro = np.array([r * np.cos(t) + 52.5, r * np.sin(t) + 72.5]).T
            if shifted:
                b_astro[:, 0] = b_astro[:, 0] - 0.3
        _, cm.a_hull_points, cm.a_hull_x_shift = convex_hull_area(
            a_astro[:, 0], a_astro[:, 1], return_hull=True)
        _, cm.b_hull_points, cm.b_hull_x_shift = convex_hull_area(
            b_astro[:, 0], b_astro[:, 1], return_hull=True)
        mcff._calculate_cf_areas()
        if shape == 'rectangle':
            if shifted:
                plusses = np.ones((5, 5), float) * 0.5
                plusses[-1, :] = 0
                plusses = plusses.flatten()
                minuses = np.ones((5, 5), float) * 0.5
                minuses[0, :] = 0
                minuses = minuses.flatten()
                calculated_areas = np.array(
                    [(c[0]+p - (c[0]-m))*180/np.pi * (np.sin(np.radians(c[1]+0.5)) -
                     np.sin(np.radians(c[1]-0.5))) for c, p, m in zip(cm.cf_region_points, plusses, minuses)])
            else:
                calculated_areas = np.array(
                    [(c[0]+0.5 - (c[0]-0.5))*180/np.pi * (np.sin(np.radians(c[1]+0.5)) -
                     np.sin(np.radians(c[1]-0.5))) for c in cm.cf_region_points])
        else:
            if shifted:
                calculated_areas = np.zeros(len(cm.cf_region_points), float)
                xs = np.linspace(-0.5, 0.5, 1001)
                xs = xs[:-1] + np.diff(xs)/2
                for i, c in enumerate(cm.cf_region_points):
                    dx = (xs[1] - xs[0]) * np.cos(np.radians(c[1]))
                    for _x in xs:
                        # Get lon.
                        x = _x + c[0]
                        # Get boundaries in lat of circle, skipping lons outside R.
                        if np.abs(x - 52.5) <= r and np.abs(x - (52.5 - 0.3)) <= r:
                            lat1 = np.sqrt(r**2 - (x - 52.5)**2) * np.array([1, -1]) + 72.5
                            lat2 = np.sqrt(r**2 - (x - (52.5 - 0.3))**2) * np.array([1, -1]) + 72.5
                            min_lat = max(lat1[1], lat2[1], c[1] - 0.5)
                            max_lat = min(lat1[0], lat2[0], c[1] + 0.5)
                            calculated_areas[i] += max(0, max_lat - min_lat) * dx
            else:
                calculated_areas = np.zeros(len(cm.cf_region_points), float)
                xs = np.linspace(-0.5, 0.5, 1001)
                xs = xs[:-1] + np.diff(xs)/2
                for i, c in enumerate(cm.cf_region_points):
                    dx = (xs[1] - xs[0]) * np.cos(np.radians(c[1]))
                    for _x in xs:
                        # Get lat.
                        x = _x + c[1]
                        # Get boundaries in lon of circle.
                        lon = np.sqrt(r**2 - (x - 72.5)**2) * np.array([1, -1]) + 52.5
                        min_lon = max(lon[1], c[0] - 0.5)
                        max_lon = min(lon[0], c[0] + 0.5)
                        calculated_areas[i] += max(0, max_lon - min_lon) * dx

        assert_allclose(cm.cf_areas, calculated_areas, rtol=0.03 if shifted else 0.025)

    # pylint: disable-next=too-many-statements
    def test_csv_inputs(self):
        cm_p_ = self.cm_p_text.replace('make_output_csv: False', '')

        with pytest.raises(ValueError, match="Missing key make_output_csv"):
            cm = CrossMatch(mock_filename(cm_p_.encode("utf-8")),
                            mock_filename(self.ca_p_text.encode("utf-8")),
                            mock_filename(self.cb_p_text.encode("utf-8")))
            cm._load_metadata_config(self.chunk_id)

        lines = ['make_output_csv: True', '\nmatch_out_csv_name: match.csv']
        for i, key in enumerate(['match_out_csv_name', 'nonmatch_out_csv_name']):
            new_line = ''
            for j in range(i+1):
                new_line = new_line + lines[j]
            cm_p_ = self.cm_p_text.replace('make_output_csv: False', new_line)
            with pytest.raises(ValueError, match=f"Missing key {key} from joint"):
                cm = CrossMatch(mock_filename(cm_p_.encode("utf-8")),
                                mock_filename(self.ca_p_text.encode("utf-8")),
                                mock_filename(self.cb_p_text.encode("utf-8")))
                cm._load_metadata_config(self.chunk_id)

        new_line = ''
        for line in lines:
            new_line = new_line + line
        new_line = new_line + '\nnonmatch_out_csv_name: nonmatch.csv'
        cm_p_ = self.cm_p_text.replace('make_output_csv: False', new_line)

        old_line = 'csv_has_header: False'
        lines = ['csv_has_header: False\n', '\ncat_col_names: [A, B, C]', '\ncat_col_nums: [1, 2, 3]',
                 '\nextra_col_names: None']
        for i, key in enumerate(['cat_col_names', 'cat_col_nums', 'extra_col_names', 'extra_col_nums']):
            new_line = ''
            for j in range(i+1):
                new_line = new_line + lines[j]
            ca_p_ = self.ca_p_text.replace(old_line, new_line)
            with pytest.raises(ValueError, match=f'Missing key {key} from catalogue "a"'):
                cm = CrossMatch(mock_filename(cm_p_.encode("utf-8")),
                                mock_filename(ca_p_.encode("utf-8")),
                                mock_filename(self.cb_p_text.encode("utf-8")))
                cm._load_metadata_config(self.chunk_id)

        new_line = ''
        for line in lines:
            new_line = new_line + line
        new_line = new_line + '\nextra_col_nums: None'
        ca_p_ = self.ca_p_text.replace(old_line, new_line)
        cb_p_ = self.cb_p_text.replace(old_line, new_line)

        # At this point we should successfully load the csv-related parameters.
        cm = CrossMatch(mock_filename(cm_p_.encode("utf-8")),
                        mock_filename(ca_p_.encode("utf-8")),
                        mock_filename(cb_p_.encode("utf-8")))
        cm._load_metadata_config(self.chunk_id)
        assert cm.output_save_folder == os.path.abspath('test_path')
        assert cm.match_out_csv_name == 'match.csv'
        assert cm.b_nonmatch_out_csv_name == 'WISE_nonmatch.csv'

        assert np.all(cm.a_cat_col_names == np.array(['Gaia_A', 'Gaia_B', 'Gaia_C']))
        assert np.all(cm.b_cat_col_nums == np.array([1, 2, 3]))
        assert cm.a_csv_has_header is False
        assert cm.a_extra_col_names is None
        assert cm.a_extra_col_nums is None

        # Check for various input points of failure:
        for old_line, new_line, error_msg, err_type, cfg_type in zip(
                ['cat_col_nums: [1, 2, 3]', 'cat_col_nums: [1, 2, 3]', 'cat_col_nums: [1, 2, 3]',
                 'extra_col_names: None'],
                ['cat_col_nums: [1, 2, A]', 'cat_col_nums: [1, 2, 3, 4]', 'cat_col_nums: [1, 2, 3.4]',
                 'extra_col_names: [D, F, G]'],
                ['cat_col_nums should be a list of integers in catalogue "b"',
                 'a_cat_col_names and a_cat_col_nums should contain the same',
                 'All elements of a_cat_col_nums', 'Both extra_col_names and extra_col_nums must '
                 'be None if either is None in catalogue "a"'],
                [ValueError, ValueError, ValueError, ValueError], ['b', 'a', 'a', 'a']):
            j = cm_p_.replace(old_line, new_line) if cfg_type == 'j' else cm_p_
            a = ca_p_.replace(old_line, new_line) if cfg_type == 'a' else ca_p_
            b = cb_p_.replace(old_line, new_line) if cfg_type == 'b' else cb_p_
            with pytest.raises(err_type, match=error_msg):
                cm = CrossMatch(mock_filename(j.encode("utf-8")),
                                mock_filename(a.encode("utf-8")),
                                mock_filename(b.encode("utf-8")))
                cm._load_metadata_config(self.chunk_id)

        # Finally, to check for extra_col_* issues, we need to set both
        # to not None first.
        ca_p_2 = ca_p_.replace('extra_col_names: None', 'extra_col_names: [D, E, F]')
        ca_p_2 = ca_p_2.replace('extra_col_nums: None', 'extra_col_nums: [1, 2, 3]')
        # First check this passes fine
        cm = CrossMatch(mock_filename(cm_p_.encode("utf-8")),
                        mock_filename(ca_p_2.encode("utf-8")),
                        mock_filename(cb_p_.encode("utf-8")))
        cm._load_metadata_config(self.chunk_id)
        assert np.all(cm.a_extra_col_names == np.array(['Gaia_D', 'Gaia_E', 'Gaia_F']))
        assert np.all(cm.a_extra_col_nums == np.array([1, 2, 3]))
        # Then check for issues correctly being raised
        for old_line, new_line, error_msg in zip(
                ['extra_col_nums: [1, 2, 3]', 'extra_col_nums: [1, 2, 3]', 'extra_col_nums: [1, 2, 3]'],
                ['extra_col_nums: [1, A, 3]', 'extra_col_nums: [1, 2, 3, 4]', 'extra_col_nums: [1, 2, 3.1]'],
                ['extra_col_nums should be a list of integers in catalogue "a"',
                 'a_extra_col_names and a_extra_col_nums should contain the same number',
                 'All elements of a_extra_col_nums should be integers']):
            ca_p_3 = ca_p_2.replace(old_line, new_line)
            with pytest.raises(ValueError, match=error_msg):
                cm = CrossMatch(mock_filename(cm_p_.encode("utf-8")),
                                mock_filename(ca_p_3.encode("utf-8")),
                                mock_filename(cb_p_.encode("utf-8")))
                cm._load_metadata_config(self.chunk_id)

    @pytest.mark.remote_data
    @pytest.mark.filterwarnings("ignore:.*contains more than one AUF sampling point, .*")
    # pylint: disable-next=too-many-statements,too-many-locals
    def test_crossmatch_correct_astrometry_inputs(self):
        cm = CrossMatch(mock_filename(self.cm_p_text.encode("utf-8")),
                        mock_filename(self.ca_p_text.encode("utf-8")),
                        mock_filename(self.cb_p_text.encode("utf-8")))
        cm._load_metadata_config(self.chunk_id)
        assert cm.n_pool == 2  # pylint: disable=no-member
        assert cm.a_correct_astrometry is False  # pylint: disable=no-member
        assert cm.b_correct_astrometry is False  # pylint: disable=no-member

        cm_p_ = self.cm_p_text.replace('n_pool: 2', '')
        with pytest.raises(ValueError, match="Missing key n_pool from joint"):
            cm = CrossMatch(mock_filename(cm_p_.encode("utf-8")),
                            mock_filename(self.ca_p_text.encode("utf-8")),
                            mock_filename(self.cb_p_text.encode("utf-8")))
            cm._load_metadata_config(self.chunk_id)
        cb_p_ = self.cb_p_text.replace('correct_astrometry: False', '')
        with pytest.raises(ValueError, match='Missing key correct_astrometry from catalogue "b"'):
            cm = CrossMatch(mock_filename(self.cm_p_text.encode("utf-8")),
                            mock_filename(self.ca_p_text.encode("utf-8")),
                            mock_filename(cb_p_.encode("utf-8")))
            cm._load_metadata_config(self.chunk_id)

        ca_p_ = self.ca_p_text.replace('correct_astrometry: False', 'correct_astrometry: True')
        cb_p_ = self.cb_p_text.replace('correct_astrometry: False', 'correct_astrometry: True')

        for new_line in ["n_pool: A", "n_pool: [1, 2]", "n_pool: 1.5"]:
            cm_p_ = self.cm_p_text.replace('n_pool: 2', new_line)
            with pytest.raises(ValueError, match="n_pool should be a single integer number."):
                cm = CrossMatch(mock_filename(cm_p_.encode("utf-8")),
                                mock_filename(self.ca_p_text.encode("utf-8")),
                                mock_filename(self.cb_p_text.encode("utf-8")))
                cm._load_metadata_config(self.chunk_id)

        # Fake dd_params and l_cut
        ddp = np.ones((5, 15, 2), float)
        np.save('dd_params.npy', ddp)
        lc = np.ones(3, float)
        np.save('l_cut.npy', lc)

        ca_p_ = self.ca_p_text.replace(
            'fit_gal_flag: False', 'gal_wavs: [0.513, 0.641, 0.778]\ngal_zmax: [4.5, 4.5, 5]\n'
            'gal_nzs: [46, 46, 51]\ngal_aboffsets: [0.5, 0.5, 0.5]\n'
            'gal_filternames: [gaiadr2-BP, gaiadr2-G, gaiadr2-RP]\nsaturation_magnitudes: [5, 5, 5]\n')
        cb_p_ = self.cb_p_text.replace(
            'fit_gal_flag: False', 'gal_wavs: [3.37, 4.62, 12.08, 22.19]\ngal_zmax: [3.2, 4.0, 1, 4]\n'
            'gal_nzs: [33, 41, 11, 41]\ngal_aboffsets: [0.5, 0.5, 0.5, 0.5]\n'
            'gal_filternames: [wise2010-W1, wise2010-W2, wise2010-W3, wise2010-W4]\n'
            'saturation_magnitudes: [5, 5, 5, 5]\n')
        ca_p_ = ca_p_.replace(r'auf_file_path: gaia_auf_folder/trilegal_download_{}.dat',
                              r'auf_file_path: gaia_auf_folder/trilegal_download_{}.dat'
                              '\ndens_hist_tri_location: None\ntri_model_mags_location: None\n'
                              'tri_model_mag_mids_location: None\ntri_model_mags_interval_location: None\n'
                              'tri_dens_uncert_location: None\ntri_n_bright_sources_star_location: None')
        cb_p_ = cb_p_.replace(r'auf_file_path: wise_auf_folder/trilegal_download_{}.dat',
                              r'auf_file_path: wise_auf_folder/trilegal_download_{}.dat'
                              '\ndens_hist_tri_location: None\ntri_model_mags_location: None\n'
                              'tri_model_mag_mids_location: None\ntri_model_mags_interval_location: None\n'
                              'tri_dens_uncert_location: None\ntri_n_bright_sources_star_location: None')
        ca_p_ = ca_p_.replace('pos_and_err_indices: [0, 1, 2]', 'pos_and_err_indices: [0, 1, 2, 0, 1, 2]')
        ca_p_ = ca_p_.replace('snr_indices: [8, 9, 10]', '')

        # Test all of the inputs being needed one by one loading into cat_a_params:
        dd_l_path = os.path.join(os.path.dirname(__file__), 'data')
        lines = [f'correct_astrometry: True\n\ndd_params_path: {dd_l_path}\nl_cut_path: {dd_l_path}',
                 '\nsnr_indices: [4, 6, 8]', '\ncorrect_astro_save_folder: ac_folder',
                 '\ncorrect_astro_mag_indices_index: 0', '\nnn_radius: 30',
                 '\nref_cat_csv_file_path: ref_{}.csv',
                 '\ncorrect_mag_array: [[14.07, 14.17, 14.27, 14.37, 14.47]]',
                 '\ncorrect_mag_slice: [[0.05, 0.05, 0.05, 0.05, 0.05]]',
                 '\ncorrect_sig_slice: [[0.1, 0.1, 0.1, 0.1, 0.1]]', '\nuse_photometric_uncertainties: False',
                 '\nmn_fit_type: quadratic', '\nseeing_ranges: [0.9, 1.1]']
        for i, key in enumerate(['snr_indices', 'correct_astro_save_folder',
                                 'correct_astro_mag_indices_index', 'nn_radius', 'ref_cat_csv_file_path',
                                 'correct_mag_array', 'correct_mag_slice', 'correct_sig_slice',
                                 'use_photometric_uncertainties', 'mn_fit_type', 'seeing_ranges']):
            new_line = ''
            for j in range(i+1):
                new_line = new_line + lines[j]
            ca_p_2 = ca_p_.replace('correct_astrometry: False', new_line)
            with pytest.raises(ValueError, match=f'Missing key {key} from catalogue "a"'):
                cm = CrossMatch(mock_filename(self.cm_p_text.encode("utf-8")),
                                mock_filename(ca_p_2.encode("utf-8")),
                                mock_filename(self.cb_p_text.encode("utf-8")))
                cm._load_metadata_config(self.chunk_id)
        # Test use_photometric_uncertainties for failure.
        new_line = ''
        for line in lines:
            new_line = new_line + line
        ca_p_2 = ca_p_.replace('correct_astrometry: False', new_line)
        ca_p_3 = ca_p_2.replace('use_photometric_uncertainties: False',
                                'use_photometric_uncertainties: something else')
        with pytest.raises(ValueError, match='Boolean flag key use_photometric_uncertainties not set to '):
            cm = CrossMatch(mock_filename(self.cm_p_text.encode("utf-8")),
                            mock_filename(ca_p_3.encode("utf-8")),
                            mock_filename(self.cb_p_text.encode("utf-8")))
            cm._load_metadata_config(self.chunk_id)
        ca_p_3 = ca_p_2.replace('mn_fit_type: quadratic', 'mn_fit_type: something else')
        with pytest.raises(ValueError, match="mn_fit_type must be 'quadratic' or 'linear' in"):
            cm = CrossMatch(mock_filename(self.cm_p_text.encode("utf-8")),
                            mock_filename(ca_p_3.encode("utf-8")),
                            mock_filename(self.cb_p_text.encode("utf-8")))
            cm._load_metadata_config(self.chunk_id)
        ca_p_3 = ca_p_2.replace('seeing_ranges: [0.9, 1.1]', 'seeing_ranges: a')
        with pytest.raises(ValueError, match="seeing_ranges must be a 1-D list or array of ints, length 1, "):
            cm = CrossMatch(mock_filename(self.cm_p_text.encode("utf-8")),
                            mock_filename(ca_p_3.encode("utf-8")),
                            mock_filename(self.cb_p_text.encode("utf-8")))
            cm._load_metadata_config(self.chunk_id)
        ca_p_3 = ca_p_2.replace('seeing_ranges: [0.9, 1.1]', 'seeing_ranges: [1, 2, 3, 4]')
        with pytest.raises(ValueError, match="seeing_ranges must be a 1-D list or array of ints, length 1, "):
            cm = CrossMatch(mock_filename(self.cm_p_text.encode("utf-8")),
                            mock_filename(ca_p_3.encode("utf-8")),
                            mock_filename(self.cb_p_text.encode("utf-8")))
            cm._load_metadata_config(self.chunk_id)
        # Set up a completely valid test of cat_a_params and cat_b_params
        cb_p_2 = cb_p_.replace('correct_astrometry: False', new_line)
        cb_p_2 = cb_p_2.replace('snr_indices: [9, 10, 11, 12]', '')
        cb_p_2 = cb_p_2.replace('snr_indices: [4, 6, 8]', 'snr_indices: [4, 6, 8, 10]')
        cb_p_2 = cb_p_2.replace('pos_and_err_indices: [0, 1, 2]', 'pos_and_err_indices: [0, 1, 2, 0, 1, 2]')
        cb_p_2 = cb_p_2.replace(r'cat_csv_file_path: wise_folder/wise_{}.csv',
                                r'cat_csv_file_path: file_{}.csv')
        cb_p_2 = cb_p_2.replace('[3, 4, 5, 6]', '[3, 5, 7, 9]')
        cb_p_2 = cb_p_2.replace('chunk_overlap_col: 7', 'chunk_overlap_col: 12')
        # Fake some TRILEGAL downloads with random data.
        os.makedirs('wise_auf_folder', exist_ok=True)
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
        with open('wise_auf_folder/trilegal_download_9_131.0_-1.0_faint.dat', "w",
                  encoding='utf-8') as f:
            f.write(text)
        # Fake some "real" csv data
        ax1_min, ax1_max, ax2_min, ax2_max = 100, 110, -3, 3
        cat_args = (self.chunk_id,)
        t_a_c = TAC()
        t_a_c.npy_or_csv = 'csv'
        t_a_c.n = 5000
        t_a_c.rng = self.rng
        # Fake sources for fake_catb_cutout. We need all N-25 objects
        # to be within 0.25 degrees of each of the first 25 data points,
        # except the 26-100th objects, which can be wherever.
        x_25 = self.rng.uniform(100, 110, size=25)
        y_25 = self.rng.uniform(-3, 3, size=25)
        x_25_100 = self.rng.uniform(100, 110, size=75)
        y_25_100 = self.rng.uniform(-3, 3, size=75)
        spawn_choice = self.rng.choice(25, size=t_a_c.n-100, replace=True)
        spawn_radius = np.sqrt(self.rng.uniform(0, 1, size=t_a_c.n-100)) * 0.25
        spawn_angle = self.rng.uniform(0, 2*np.pi, size=t_a_c.n-100)
        spawn_x = x_25[spawn_choice] + spawn_radius * np.cos(spawn_angle)
        spawn_y = y_25[spawn_choice] + spawn_radius * np.sin(spawn_angle)
        t_a_c.true_ra = np.append(np.append(x_25, x_25_100), spawn_x)
        t_a_c.true_dec = np.append(np.append(y_25, y_25_100), spawn_y)
        t_a_c.a_cat_name = 'ref_{}.csv'
        t_a_c.b_cat_name = 'file_{}.csv'
        t_a_c.fake_cata_cutout(ax1_min, ax1_max, ax2_min, ax2_max, *cat_args)
        t_a_c.fake_catb_cutout(ax1_min, ax1_max, ax2_min, ax2_max, *cat_args)
        # Re-fake data with multiple magnitude columns.
        x = np.loadtxt('ref_9.csv', delimiter=',')
        y = np.empty((len(x), 11), float)
        y[:, [0, 1, 2]] = x[:, [0, 1, 2]]
        y[:, [3, 4]] = x[:, [3, 4]]
        y[:, [5, 6]] = x[:, [3, 4]]
        y[:, [7, 8]] = x[:, [3, 4]]
        # Pad with both a best index and chunk overlap column
        y[:, 9] = 2
        y[:, 10] = 1
        np.savetxt('ref_9.csv', y, delimiter=',')
        x = np.loadtxt('file_9.csv', delimiter=',')
        y = np.empty((len(x), 13), float)
        y[:, [0, 1, 2]] = x[:, [0, 1, 2]]
        y[:, [3, 4]] = x[:, [3, 4]]
        y[:, [5, 6]] = x[:, [3, 4]]
        y[:, [7, 8]] = x[:, [3, 4]]
        y[:, [9, 10]] = x[:, [3, 4]]
        y[:, 11] = np.random.default_rng(seed=5673523).choice(4, size=len(x), replace=True)
        y[:, 12] = np.random.default_rng(seed=45645132234).choice(2, size=len(x), replace=True)
        np.savetxt('file_9.csv', y, delimiter=',')
        cm_p_ = self.cm_p_text.replace('real_hankel_points: 10000', 'real_hankel_points: 1000')
        cm_p_ = cm_p_.replace('four_max_rho: 100', 'four_max_rho: 30')
        cm_p_ = cm_p_.replace('four_hankel_points: 10000', 'four_hankel_points: 1000')
        # Using the ORIGINAL cat_a_params means we don't fit for corrections
        # to catalogue 'a'.
        cm = CrossMatch(mock_filename(cm_p_.encode("utf-8")),
                        mock_filename(self.ca_p_text.encode("utf-8")),
                        mock_filename(cb_p_2.encode("utf-8")))
        cm._load_metadata_config(self.chunk_id)
        cm.chunk_id = self.chunk_id
        cm._initialise_chunk()
        # pylint: disable=no-member
        assert cm.b_correct_astro_mag_indices_index == 0
        assert_allclose(cm.b_nn_radius, 30)
        assert cm.b_correct_astro_save_folder == os.path.abspath('ac_folder')
        assert cm.b_cat_csv_file_path == os.path.abspath('file_9.csv')
        assert cm.b_ref_cat_csv_file_path == os.path.abspath('ref_9.csv')
        assert_allclose(cm.b_correct_mag_array, np.array([[14.07, 14.17, 14.27, 14.37, 14.47]]))
        assert_allclose(cm.b_correct_mag_slice, np.array([[0.05, 0.05, 0.05, 0.05, 0.05]]))
        assert_allclose(cm.b_correct_sig_slice, np.array([[0.1, 0.1, 0.1, 0.1, 0.1]]))
        assert np.all(cm.b_pos_and_err_indices == np.array([[0, 1, 2], [0, 1, 2]]))
        assert np.all(cm.b_mag_indices == np.array([3, 5, 7, 9]))
        assert np.all(cm.b_snr_indices == np.array([4, 6, 8, 10]))
        mnarray = np.load('ac_folder/npy/mn_sigs_array.npy')
        assert_allclose([mnarray[0, 0], mnarray[0, 1]], [2, 0], rtol=0.1, atol=0.01)
        # pylint: enable=no-member
        assert np.all(cm.b_in_overlaps == y[:, 12].astype(int))

        cb_p_2 = cb_p_2.replace('chunk_overlap_col: 12', 'chunk_overlap_col: None')
        cb_p_2 = cb_p_2.replace('best_mag_index_col: 8', 'best_mag_index_col: 11')

        # Swapped a+b to test a_* versions of things, but also test using the
        # photometric information.
        cm_p_2 = cm_p_.replace('  - 9', '  - 100')
        ca_p_3 = self.ca_p_text.replace('  - 9', '  - 100')
        ca_p_3 = ca_p_3.replace(r'cat_csv_file_path: gaia_folder/gaia_{}.csv',
                                r'cat_csv_file_path: file_{}.csv')
        cb_p_3 = cb_p_2.replace('  - 9', '  - 100')
        cb_p_3 = cb_p_3.replace(r'cat_csv_file_path: wise_folder/wise_{}.csv',
                                r'cat_csv_file_path: file_{}.csv')
        cb_p_3 = cb_p_3.replace('use_photometric_uncertainties: False', 'use_photometric_uncertainties: True')
        cb_p_3 = cb_p_3.replace('pos_and_err_indices: [0, 1, 2, 0, 1, 2]',
                                'pos_and_err_indices: [0, 1, 2, 13, 14, 15, 0, 1, 2]')
        cb_p_3 = cb_p_3.replace('correct_mag_array: [[14.07, 14.17, 14.27, 14.37, 14.47]]',
                                'correct_mag_array: [[14.07, 14.17, 14.27, 14.37, 14.47], [14.07, 14.17, '
                                '14.27, 14.37, 14.47], [14.07, 14.17, 14.27, 14.37, 14.47], [14.07, 14.17, '
                                '14.27, 14.37, 14.47]]')
        cb_p_3 = cb_p_3.replace('correct_mag_slice: [[0.05, 0.05, 0.05, 0.05, 0.05]]',
                                'correct_mag_slice: [[0.05, 0.05, 0.05, 0.05, 0.05], [0.05, 0.05, 0.05, '
                                '0.05, 0.05], [0.05, 0.05, 0.05, 0.05, 0.05], [0.05, 0.05, 0.05, 0.05, '
                                '0.05]]')
        cb_p_3 = cb_p_3.replace('correct_sig_slice: [[0.1, 0.1, 0.1, 0.1, 0.1]]',
                                'correct_sig_slice: [[0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1], '
                                '[0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1]]')

        os.system('cp -r gaia_folder_9 gaia_folder_100')
        os.system('cp -r wise_folder_9 wise_folder_100')
        os.system('cp -r wise_auf_folder_9 wise_auf_folder_100')
        os.system('cp ref_9.csv ref_100.csv')
        os.system('cp file_9.csv file_100.csv')
        x = np.loadtxt('file_100.csv', delimiter=',')
        y = np.empty((len(x), 16), float)
        y[:, :13] = x
        y[:, [13, 14, 15]] = x[:, [2, 2, 2]]
        np.savetxt('file_100.csv', y, delimiter=',')
        cm = CrossMatch(mock_filename(cm_p_2.encode("utf-8")),
                        mock_filename(cb_p_3.encode("utf-8")),
                        mock_filename(ca_p_3.encode("utf-8")))
        cm._load_metadata_config(100)
        cm.chunk_id = 100
        cm._initialise_chunk()
        # pylint: disable=no-member
        assert cm.a_correct_astro_mag_indices_index == 0
        assert_allclose(cm.a_nn_radius, 30)
        assert cm.a_correct_astro_save_folder == os.path.abspath('ac_folder')
        assert cm.a_cat_csv_file_path == os.path.abspath('file_100.csv')
        assert cm.a_ref_cat_csv_file_path == os.path.abspath('ref_100.csv')
        assert_allclose(cm.a_correct_mag_array, np.array([[14.07, 14.17, 14.27, 14.37, 14.47]] * 4))
        assert_allclose(cm.a_correct_mag_slice, np.array([[0.05, 0.05, 0.05, 0.05, 0.05]] * 4))
        assert_allclose(cm.a_correct_sig_slice, np.array([[0.1, 0.1, 0.1, 0.1, 0.1]] * 4))
        assert np.all(cm.a_pos_and_err_indices[0] == np.array([0, 1, 2, 13, 14, 15]))
        assert np.all(cm.a_pos_and_err_indices[1] == np.array([0, 1, 2]))
        assert np.all(cm.a_mag_indices == np.array([3, 5, 7, 9]))
        assert np.all(cm.a_snr_indices == np.array([4, 6, 8, 10]))
        mnarray = np.load('ac_folder/npy/mn_sigs_array.npy')
        assert np.all(mnarray.shape == (1, 4, 4))
        assert_allclose([mnarray[0, 0, 0], mnarray[0, 0, 1]], [2, 0], rtol=0.1, atol=0.01)
        assert_allclose([mnarray[0, 3, 0], mnarray[0, 3, 1]], [2, 0], rtol=0.1, atol=0.01)
        # pylint: enable=no-member
        assert np.all(cm.a_in_overlaps == 0)

        # New test of the AC run, just with pre-made histograms.
        dens, tri_mags, tri_mags_mids, dtri_mags, uncert, num_bright_obj = make_tri_counts(
            'wise_auf_folder/trilegal_download_9_131.0_-1.0.dat', 'W1', 0.1, 13.5, 16)
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

        cb_p_3 = cb_p_2.replace('auf_file_path: wise_auf_folder/trilegal_download_{}.dat',
                                'auf_file_path: None')
        lines = cb_p_3.split('\n')
        for ol, nl in zip(['tri_set_name: ', 'tri_filt_names: ', 'tri_filt_num: ', 'download_tri: ',
                           'tri_maglim_faint: ', 'tri_num_faint: ', 'dens_hist_tri_location: ',
                           'tri_model_mags_location: ', 'tri_model_mag_mids_location: ',
                           'tri_model_mags_interval_location: ', 'tri_dens_uncert_location: ',
                           'tri_n_bright_sources_star_location: '], [
                'tri_set_name: None', 'tri_filt_names: None', 'tri_filt_num: None',
                'download_tri: None', 'tri_maglim_faint: None', 'tri_num_faint: None',
                f'dens_hist_tri_location: {dhtl}', f'tri_model_mags_location: {tmml}',
                f'tri_model_mag_mids_location: {tmmml}', f'tri_model_mags_interval_location: {tmmil}',
                f'tri_dens_uncert_location: {tdul}', f'tri_n_bright_sources_star_location: {tnbssl}']):
            ind = np.where([ol in x for x in lines])[0][0]
            cb_p_3 = cb_p_3.replace(lines[ind], nl)
        cm = CrossMatch(mock_filename(cm_p_.encode("utf-8")),
                        mock_filename(cb_p_3.encode("utf-8")),
                        mock_filename(self.ca_p_text.encode("utf-8")))
        cm._load_metadata_config(self.chunk_id)
        cm.chunk_id = self.chunk_id
        cm._initialise_chunk()
        assert cm.a_correct_astro_mag_indices_index == 0
        assert_allclose(cm.a_nn_radius, 30)
        assert cm.a_correct_astro_save_folder == os.path.abspath('ac_folder')
        assert cm.a_cat_csv_file_path == os.path.abspath('file_9.csv')
        assert cm.a_ref_cat_csv_file_path == os.path.abspath('ref_9.csv')
        assert_allclose(cm.a_correct_mag_array, np.array([[14.07, 14.17, 14.27, 14.37, 14.47]]))
        assert_allclose(cm.a_correct_mag_slice, np.array([[0.05, 0.05, 0.05, 0.05, 0.05]]))
        assert_allclose(cm.a_correct_sig_slice, np.array([[0.1, 0.1, 0.1, 0.1, 0.1]]))
        assert np.all(cm.a_pos_and_err_indices == np.array([[0, 1, 2], [0, 1, 2]]))
        assert np.all(cm.a_mag_indices == np.array([3, 5, 7, 9]))
        assert np.all(cm.a_snr_indices == np.array([4, 6, 8, 10]))
        mnarray = np.load('ac_folder/npy/mn_sigs_array.npy')
        assert_allclose([mnarray[0, 0], mnarray[0, 1]], [2, 0], rtol=0.1, atol=0.01)
        # pylint: enable=no-member

        # Dummy folder that won't contain l_cut.npy
        os.makedirs('./l_cut_dummy_folder', exist_ok=True)

        lines_a, lines_b = ca_p_2.split('\n'), cb_p_2.split('\n')
        for old_line, new_line, x, match_text in zip(
                ['correct_astro_mag_indices_index: ', 'correct_astro_mag_indices_index: ',
                 'correct_astro_mag_indices_index: ', 'nn_radius: ', 'nn_radius: ',
                 'correct_mag_array: ', 'correct_mag_slice: ', 'correct_mag_slice: ', 'correct_sig_slice: ',
                 'correct_sig_slice: ', 'correct_sig_slice: ', 'pos_and_err_indices: ',
                 'pos_and_err_indices: ', 'pos_and_err_indices: ', 'pos_and_err_indices: ',
                 'pos_and_err_indices: ', 'mag_indices:', 'mag_indices:', 'mag_indices:', 'snr_indices:',
                 'snr_indices:', 'snr_indices:', 'chunk_overlap_col: ', 'chunk_overlap_col: ',
                 'chunk_overlap_col: ', 'best_mag_index_col: ', 'best_mag_index_col: ', 'dd_params_path: ',
                 'l_cut_path: '],
                ['correct_astro_mag_indices_index: A', 'correct_astro_mag_indices_index: 2.5',
                 'correct_astro_mag_indices_index: 7', 'nn_radius: A', 'nn_radius: [1, 2]',
                 'correct_mag_array: [[1, 2, A, 4, 5]]', 'correct_mag_slice: [[0.1, 0.1, 0.1, A, 0.1]]',
                 'correct_mag_slice: [[0.1, 0.1, 0.1]]', 'correct_sig_slice: [[0.1, 0.1, 0.1, A, 0.1]]',
                 'correct_sig_slice: [[0.1, 0.1, 0.1, 0.1, 0.1], [0.1]]',
                 'correct_sig_slice: [[0.1, 0.1, 0.1]]', 'pos_and_err_indices: [1 2 3 4 5 A]',
                 'pos_and_err_indices: [1, 2, 3, 4, 5.5, 6]', 'pos_and_err_indices: [1, 2, 3, 4, 5]',
                 'pos_and_err_indices: [1, 2, 3, 4, 5]', 'pos_and_err_indices: [1, 2, 3, 4, 5, 6, 7]',
                 'mag_indices: [A, 1, 2]', 'mag_indices: [1, 2]', 'mag_indices: [1.2, 2, 3, 4]',
                 'snr_indices: [A, 1, 2]', 'snr_indices: [1, 2]', 'snr_indices: [1.2, 2, 3]',
                 'chunk_overlap_col: Non', 'chunk_overlap_col: A', 'chunk_overlap_col: 1.2',
                 'best_mag_index_col: A', 'best_mag_index_col: 1.2', 'dd_params_path: ./some_folder',
                 'l_cut_path: ./l_cut_dummy_folder'],
                ['a', 'b', 'a', 'b', 'a', 'a', 'b', 'a', 'b', 'a', 'a', 'b', 'a', 'a', 'b', 'a', 'a', 'b',
                 'b', 'b', 'a', 'a', 'a', 'b', 'a', 'a', 'b', 'b', 'a'],
                ['correct_astro_mag_indices_index should be an integer in the catalogue "a"',
                 'correct_astro_mag_indices_index should be an integer in the catalogue "b"',
                 'correct_astro_mag_indices_index cannot be a larger index than the list of filters '
                 'in the catalogue "a', 'nn_radius must be a float in the catalogue "b"',
                 'nn_radius must be a float in the catalogue "a"',
                 'correct_mag_array should be a list of list of floats in the catalogue "a"',
                 'correct_mag_slice should be a list of list of floats in the catalogue "b"',
                 'a_correct_mag_array and a_correct_mag_slice should contain the same',
                 'correct_sig_slice should be a list of list of floats in the catalogue "b"',
                 'a_correct_mag_array and a_correct_sig_slice should contain the same',
                 'a_correct_mag_array and a_correct_sig_slice should contain the same',
                 'pos_and_err_indices should be a list of integers in the catalogue "b"',
                 'All elements of a_pos_and_err_indices should be integers',
                 'a_pos_and_err_indices should contain six elements when correct_astrometry',
                 'b_pos_and_err_indices should contain at least six elements when correct_astrometry',
                 'a_pos_and_err_indices should contain the same number of non-reference, non-position',
                 'mag_indices should be a list of integers in the catalogue "a" ',
                 'b_filt_names and b_mag_indices should contain the',
                 'All elements of b_mag_indices should be integers.',
                 'snr_indices should be a list of integers in catalogue "b" ',
                 'a_snr_indices and a_mag_indices should contain the',
                 'All elements of a_snr_indices should be integers.',
                 'chunk_overlap_col should be an integer in the catalogue "a"',
                 'chunk_overlap_col should be an integer in the catalogue "b"',
                 'chunk_overlap_col should be an integer in the catalogue "a"',
                 'best_mag_index_col should be an integer in the catalogue "a"',
                 'best_mag_index_col should be an integer in the catalogue "b"',
                 'b_dd_params_path does not exist. Please ensure that path for catalogue "b"',
                 'l_cut file not found in catalogue "a" path. Please ensure PSF ']):
            z, lines = (ca_p_2, lines_a) if x == 'a' else (cb_p_2, lines_b)
            ind = np.where([old_line in x for x in lines])[0][0]
            z = z.replace(lines[ind], new_line)
            b, c = (ca_p_2, z) if x == 'b' else (z, cb_p_2)

            if 'folder' not in new_line:
                type_of_error = ValueError
            elif 'dd_params' in new_line:
                type_of_error = OSError
            else:
                type_of_error = FileNotFoundError
            if "contain at least six" in match_text:
                c = c.replace('use_photometric_uncertainties: False', 'use_photometric_uncertainties: True')
            if "contain the same number of non-reference" in match_text:
                b = b.replace('use_photometric_uncertainties: False', 'use_photometric_uncertainties: True')
            with pytest.raises(type_of_error, match=match_text):
                cm = CrossMatch(mock_filename(self.cm_p_text.encode("utf-8")),
                                mock_filename(b.encode("utf-8")),
                                mock_filename(c.encode("utf-8")))
                cm._load_metadata_config(self.chunk_id)

        cb_p_3 = cb_p_2.replace('correct_astrometry: True', 'correct_astrometry: False')
        cb_p_3 = cb_p_3.replace('pos_and_err_indices: [0, 1, 2, 0, 1, 2]',
                                'pos_and_err_indices: [1, 2, 3, 4, 5]')
        match_text = 'b_pos_and_err_indices should contain three elements when correct_astrometry is F'
        with pytest.raises(ValueError, match=match_text):
            cm = CrossMatch(mock_filename(self.cm_p_text.encode("utf-8")),
                            mock_filename(ca_p_2.encode("utf-8")),
                            mock_filename(cb_p_3.encode("utf-8")))
            cm._load_metadata_config(self.chunk_id)


class TestPostProcess:
    def setup_method(self):
        self.output_save_folder = os.path.abspath('joint')
        self.a_cat_csv_file_path = os.path.abspath('a_input_csv_folder/gaia_catalogue.csv')
        self.b_cat_csv_file_path = os.path.abspath('b_input_csv_folder/wise_catalogue.csv')

        os.makedirs(f'{self.output_save_folder}', exist_ok=True)
        os.makedirs(os.path.dirname(self.a_cat_csv_file_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.b_cat_csv_file_path), exist_ok=True)

        na, nb, nmatch = 10000, 7000, 4000
        self.na, self.nb, self.nmatch = na, nb, nmatch

        with open(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.yaml'),
                  encoding='utf-8') as cm_p:
            self.cm_p_text = cm_p.read()
        with open(os.path.join(os.path.dirname(__file__), 'data/cat_a_params.yaml'),
                  encoding='utf-8') as ca_p:
            self.ca_p_text = ca_p.read()
        with open(os.path.join(os.path.dirname(__file__), 'data/cat_b_params.yaml'),
                  encoding='utf-8') as cb_p:
            self.cb_p_text = cb_p.read()

        self.cm = CrossMatch(mock_filename(self.cm_p_text.encode("utf-8")),
                             mock_filename(self.ca_p_text.encode("utf-8")),
                             mock_filename(self.cb_p_text.encode("utf-8")))

        rng = np.random.default_rng(seed=7893467234)
        self.ac = rng.choice(na, size=nmatch, replace=False)
        self.cm.ac = self.ac
        self.bc = rng.choice(nb, size=nmatch, replace=False)
        self.cm.bc = self.bc

        self.af = np.delete(np.arange(na), self.ac)
        self.cm.af = self.af
        self.bf = np.delete(np.arange(nb), self.bc)
        self.cm.bf = self.bf

        self.cm.a_in_overlaps = rng.choice(2, size=na).astype(bool)
        self.cm.b_in_overlaps = rng.choice(2, size=nb).astype(bool)

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

    @pytest.mark.parametrize("include_phot_like", [True, False])
    def test_postprocess(self, include_phot_like):
        self.cm.output_save_folder = self.output_save_folder
        self.cm.a_cat_csv_file_path = self.a_cat_csv_file_path
        self.cm.b_cat_csv_file_path = self.b_cat_csv_file_path

        self.cm.include_phot_like = include_phot_like
        if include_phot_like:
            self.cm.with_and_without_photometry = True
        self.cm.make_output_csv = False

        if include_phot_like:
            for n in ['ac', 'af']:
                setattr(self.cm, f'{n}_without_photometry', getattr(self.cm, f'{n}')[::-1])
            for n in ['bc', 'bf']:
                setattr(self.cm, f'{n}_without_photometry', getattr(self.cm, f'{n}'))

            for n in ['pc', 'eta', 'xi', 'acontamflux', 'bcontamflux', 'pacontam', 'pbcontam', 'crptseps']:
                setattr(self.cm, f'{n}_without_photometry', 2*getattr(self.cm, f'{n}'))

            for t in ['a', 'b']:
                setattr(self.cm, f'pf{t}_without_photometry', 2*getattr(self.cm, f'pf{t}'))
                for n in ['fieldflux', 'fieldseps', 'fieldeta', 'fieldxi']:
                    setattr(self.cm, f'{t}{n}_without_photometry', 2*getattr(self.cm, f'{t}{n}'))

        self.cm.chunk_id = 1

        self.cm._postprocess_chunk()

        if include_phot_like:
            exts = ['', '_without_photometry']
        else:
            exts = ['']

        for ext in exts:
            aino = self.cm.a_in_overlaps
            bino = self.cm.b_in_overlaps
            ac = np.load(f'{self.output_save_folder}/ac_{self.cm.chunk_id}{ext}.npy')
            af = np.load(f'{self.output_save_folder}/af_{self.cm.chunk_id}{ext}.npy')
            bc = np.load(f'{self.output_save_folder}/bc_{self.cm.chunk_id}{ext}.npy')
            bf = np.load(f'{self.output_save_folder}/bf_{self.cm.chunk_id}{ext}.npy')

            assert np.all(~aino[ac] | ~bino[bc])
            assert np.all(~aino[af])
            assert np.all(~bino[bf])

            if ext == '_without_photometry':
                self_ac, self_af = self.ac[::-1], self.af[::-1]
            else:
                self_ac, self_af = self.ac, self.af

            deleted_ac = np.delete(self_ac, np.array([np.argmin(np.abs(q - self_ac)) for q in ac]))
            deleted_bc = np.delete(self.bc, np.array([np.argmin(np.abs(q - self.bc)) for q in bc]))
            assert np.all((aino[deleted_ac] & bino[deleted_bc]))

            deleted_af = np.delete(self_af, np.array([np.argmin(np.abs(q - self_af)) for q in af]))
            deleted_bf = np.delete(self.bf, np.array([np.argmin(np.abs(q - self.bf)) for q in bf]))
            assert np.all(aino[deleted_af])
            assert np.all(bino[deleted_bf])

    @pytest.mark.parametrize("include_phot_like", [True, False])
    # pylint: disable-next=too-many-statements,too-many-locals
    def test_postprocess_with_csv(self, include_phot_like):
        self.cm.output_save_folder = self.output_save_folder
        self.cm.a_cat_csv_file_path = self.a_cat_csv_file_path
        self.cm.b_cat_csv_file_path = self.b_cat_csv_file_path

        self.cm.include_phot_like = include_phot_like
        if include_phot_like:
            self.cm.with_and_without_photometry = True
        self.cm.make_output_csv = True

        # Set a whole load of fake inputs
        self.cm.output_save_folder = 'output_save_folder'
        os.makedirs(self.cm.output_save_folder, exist_ok=True)
        self.cm.a_csv_has_header = False
        acat, acatstring = self.make_temp_catalogue(self.na, 8, 100, 'Gaia ')
        np.savetxt(self.cm.a_cat_csv_file_path, acatstring, delimiter=',', fmt='%s', header='')
        self.cm.b_csv_has_header = True
        bcat, bcatstring = self.make_temp_catalogue(self.nb, 10, 500, 'J')
        np.savetxt(self.cm.b_cat_csv_file_path, bcatstring, delimiter=',', fmt='%s',
                   header='ID, RA, Dec, Err, W1, W2, W3, W4, bestflag, inchunk')
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
        self.cm.a_extra_col_names = None
        self.cm.a_extra_col_nums = None
        self.cm.b_extra_col_names = ['WErr']
        self.cm.b_extra_col_nums = [3]
        self.cm.a_correct_astrometry = False
        self.cm.b_correct_astrometry = False
        self.cm.chunk_id = 1

        if include_phot_like:
            for n in ['ac', 'af']:
                setattr(self.cm, f'{n}_without_photometry', getattr(self.cm, f'{n}')[::-1])
            for n in ['bc', 'bf']:
                setattr(self.cm, f'{n}_without_photometry', getattr(self.cm, f'{n}'))

            for n in ['pc', 'eta', 'xi', 'acontamflux', 'bcontamflux', 'pacontam', 'pbcontam', 'crptseps']:
                setattr(self.cm, f'{n}_without_photometry', 2*getattr(self.cm, f'{n}'))

            for t in ['a', 'b']:
                setattr(self.cm, f'pf{t}_without_photometry', 2*getattr(self.cm, f'pf{t}'))
                for n in ['fieldflux', 'fieldseps', 'fieldeta', 'fieldxi']:
                    setattr(self.cm, f'{t}{n}_without_photometry', 2*getattr(self.cm, f'{t}{n}'))

        self.cm._postprocess_chunk()

        if include_phot_like:
            exts = ['', '_without_photometry']
        else:
            exts = ['']

        for ext in exts:
            aino = self.cm.a_in_overlaps
            bino = self.cm.b_in_overlaps
            ac = getattr(self.cm, f'ac{ext}')
            af = getattr(self.cm, f'af{ext}')
            bc = getattr(self.cm, f'bc{ext}')
            bf = getattr(self.cm, f'bf{ext}')

            if ext == '_without_photometry':
                self_ac, self_af = self.ac[::-1], self.af[::-1]
            else:
                self_ac, self_af = self.ac, self.af

            deleted_ac = np.delete(self_ac, np.array([np.argmin(np.abs(q - self_ac)) for q in ac]))
            deleted_bc = np.delete(self.bc, np.array([np.argmin(np.abs(q - self.bc)) for q in bc]))
            assert np.all((aino[deleted_ac] & bino[deleted_bc]))

            deleted_af = np.delete(self_af, np.array([np.argmin(np.abs(q - self_af)) for q in af]))
            deleted_bf = np.delete(self.bf, np.array([np.argmin(np.abs(q - self.bf)) for q in bf]))
            assert np.all(aino[deleted_af])
            assert np.all(bino[deleted_bf])

            # Check that the outputs make sense, treating this more like a
            # parse_catalogue test than anything else, but importantly
            # checking for correct lengths of produced outputs like pc.
            assert os.path.isfile(f'{self.cm.output_save_folder}/{self.cm.match_out_csv_name}')
            assert os.path.isfile(f'{self.cm.output_save_folder}/{self.cm.a_nonmatch_out_csv_name}')
            assert os.path.isfile(f'{self.cm.output_save_folder}/{self.cm.b_nonmatch_out_csv_name}')

            pc = getattr(self.cm, f'pc{ext}')
            eta = getattr(self.cm, f'eta{ext}')
            xi = getattr(self.cm, f'xi{ext}')
            acf = getattr(self.cm, f'acontamflux{ext}')
            bcf = getattr(self.cm, f'bcontamflux{ext}')
            pac = getattr(self.cm, f'pacontam{ext}')
            pbc = getattr(self.cm, f'pbcontam{ext}')
            csep = getattr(self.cm, f'crptseps{ext}')
            pfa = getattr(self.cm, f'pfa{ext}')
            afs = getattr(self.cm, f'afieldseps{ext}')
            afeta = getattr(self.cm, f'afieldeta{ext}')
            afxi = getattr(self.cm, f'afieldxi{ext}')
            aff = getattr(self.cm, f'afieldflux{ext}')
            pfb = getattr(self.cm, f'pfb{ext}')
            bfs = getattr(self.cm, f'bfieldseps{ext}')
            bfeta = getattr(self.cm, f'bfieldeta{ext}')
            bfxi = getattr(self.cm, f'bfieldxi{ext}')
            bff = getattr(self.cm, f'bfieldflux{ext}')

            extra_cols = ['MATCH_P', 'SEPARATION', 'ETA', 'XI', 'A_AVG_CONT', 'B_AVG_CONT',
                          'A_CONT_F1', 'A_CONT_F10', 'B_CONT_F1', 'B_CONT_F10']
            names = np.append(np.append(self.cm.a_cat_col_names, self.cm.b_cat_col_names),
                              np.append(extra_cols, self.cm.b_extra_col_names))

            _name = self.cm.match_out_csv_name[:-4] + ext + self.cm.match_out_csv_name[-4:]
            df = pd.read_csv(f'{self.cm.output_save_folder}/{_name}', header=None, names=names)
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
            _name = self.cm.a_nonmatch_out_csv_name[:-4] + ext + self.cm.a_nonmatch_out_csv_name[-4:]
            df = pd.read_csv(f'{self.cm.output_save_folder}/{_name}', header=None, names=names)
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
            _name = self.cm.b_nonmatch_out_csv_name[:-4] + ext + self.cm.b_nonmatch_out_csv_name[-4:]
            df = pd.read_csv(f'{self.cm.output_save_folder}/{_name}', header=None, names=names)
            for i, col in zip([1, 2, 4, 5, 6], self.cm.b_cat_col_names[1:]):
                assert_allclose(df[col], bcat[bf, i])
            assert np.all([df[self.cm.b_cat_col_names[0]].iloc[i] == bcatstring[bf[i], 0] for i in
                           range(len(bf))])
            assert_allclose(df['MATCH_P'], pfb)
            assert_allclose(df['B_AVG_CONT'], bff)
            assert_allclose(df['NNM_SEPARATION'], bfs)
            assert_allclose(df['NNM_ETA'], bfeta)
            assert_allclose(df['NNM_XI'], bfxi)
