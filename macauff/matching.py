# Licensed under a 3-clause BSD style license - see LICENSE
'''
This module provides the high-level framework for performing catalogue-catalogue cross-matches.
'''

import os
from configparser import ConfigParser
import numpy as np

__all__ = ['CrossMatch']


class CrossMatch():
    '''
    A class to cross-match two photometric catalogues with one another, producing
    a composite catalogue of merged sources.

    The class takes a path, the location of a metadata file containing all of the
    necessary parameters for the cross-match, and outputs a file containing the
    appropriate columns of the datasets plus additional derived parameters.

    Parameters
    ----------
    file_path : string
        A path to the location of the file containing the cross-match metadata.
    '''

    def __init__(self, file_path):
        '''
        Initialisation function for cross-match class.
        '''
        if not os.path.isfile(file_path):
            raise FileNotFoundError("Metadata file could not be found at specified location.")

        self.file_path = file_path

        self.read_metadata()

        # Important steps that can be save points in the match process are:
        # AUF creation, island splitting, c/f creation, star pairing. We have
        # to check if any later stages are flagged to not run (i.e., they are
        # the starting point) than earlier stages, and raise an error.
        flags = np.array([self.run_auf, self.run_group, self.run_cf, self.run_star])
        for i in range(3):
            if flags[i] and np.any(~flags[i+1:]):
                raise ValueError("Inconsistency between run/no run flags; please ensure that "
                                 "if a sub-process is set to run that all subsequent "
                                 "processes are also set to run.")

        # Ensure that we can create the folder for outputs.
        try:
            os.makedirs(self.folder_path, exist_ok=True)
        except OSError:
            raise OSError("Error when trying to create temporary folder for outputs. "
                          "Please ensure that folder_path is correct.")

        for folder in [self.a_cat_name, self.b_cat_name]:
            try:
                os.makedirs('{}/{}'.format(self.folder_path, folder), exist_ok=True)
            except OSError:
                raise OSError("Error when trying to create temporary folder for catalogue-level "
                              "outputs. Please ensure that catalogue folder names are correct.")

    def _replace_line(self, file_name, line_num, text, out_file=None):
        '''
        Helper function to update the metadata file on-the-fly, allowing for
        "run" flags to be set from run to no run once they have finished.
        '''
        if out_file is None:
            out_file = file_name
        lines = open(file_name, 'r').readlines()
        lines[line_num] = text
        out = open(out_file, 'w')
        out.writelines(lines)
        out.close()

    def _str2bool(self, v):
        '''
        Convenience function to convert strings to boolean values.
        '''
        val = v.lower()
        if val not in ("yes", "true", "t", "1", "no", "false", "f", "0"):
            raise ValueError('Boolean flag key not set to allowed value.')
        else:
            return v.lower() in ("yes", "true", "t", "1")

    def _make_regions_points(self, region_type, region_frame, region_points):
        '''
        Wrapper function for the creation of "region" coordinate tuples,
        given either a set of rectangular points or a list of coordinates.
        '''
        rt = region_type[1].lower()
        if rt == 'rectangle':
            try:
                a = region_points[1].split()
                a = [float(point) for point in a]
            except ValueError:
                raise ValueError("{} should be 6 numbers separated "
                                 "by spaces.".format(region_points[0]))
            if len(a) == 6:
                if not a[2].is_integer() or not a[5].is_integer():
                    raise ValueError("Number of steps between start and stop values for "
                                     "{} should be integers.".format(region_points[0]))
                ax1_p = np.linspace(a[0], a[1], int(a[2]))
                ax2_p = np.linspace(a[3], a[4], int(a[5]))
                points = np.stack(np.meshgrid(ax1_p, ax2_p), -1).reshape(-1, 2)
            else:
                raise ValueError("{} should be 6 numbers separated "
                                 "by spaces.".format(region_points[0]))
        elif rt == 'points':
            try:
                a = region_points[1].replace('(', ')').split('), )')
                # Remove the first ( and final ) that weren't split by "), (" -> "), )"
                a[0] = a[0][1:]
                a[-1] = a[-1][:-1]
                b = [q.split(', ') for q in a]
                points = np.array(b, dtype=float)
            except ValueError:
                raise ValueError("{} should be a list of '(a, b), (c, d)' tuples, "
                                 "separated by a comma.".format(region_points[0]))
        else:
            raise ValueError("{} should either be 'rectangle' or 'points'.".format(region_type[0]))

        setattr(self, region_points[0], points)

        rf = region_frame[1].lower()
        if rf == 'equatorial' or rf == 'galactic':
            setattr(self, region_frame[0], region_frame[1])
        else:
            raise ValueError("{} should either be 'equatorial' or 'galactic'.".format(
                             region_frame[0]))

    def read_metadata(self):
        '''
        Helper function to read in metadata and set various class attributes.
        '''
        config = ConfigParser()
        with open(self.file_path) as f:
            config.read_string('[config]\n' + f.read())
        config = config['config']

        for check_flag in ['include_perturb_auf', 'include_phot_like', 'run_auf', 'run_group',
                           'run_cf', 'run_star', 'auf_region_type', 'auf_region_frame',
                           'auf_region_points', 'cf_region_type', 'cf_region_frame',
                           'cf_region_points', 'folder_path', 'a_filt_names', 'b_filt_names',
                           'a_cat_name', 'b_cat_name']:
            if check_flag not in config:
                raise ValueError("Missing key {} from metadata file.".format(check_flag))

        for run_flag in ['include_perturb_auf', 'include_phot_like', 'run_auf', 'run_group',
                         'run_cf', 'run_star']:
            setattr(self, run_flag, self._str2bool(config[run_flag]))

        self._make_regions_points(['auf_region_type', config['auf_region_type']],
                                  ['auf_region_frame', config['auf_region_frame']],
                                  ['auf_region_points', config['auf_region_points']])

        self._make_regions_points(['cf_region_type', config['cf_region_type']],
                                  ['cf_region_frame', config['cf_region_frame']],
                                  ['cf_region_points', config['cf_region_points']])

        self.folder_path = os.path.abspath(config['folder_path'])

        # Only have to check for the existence of Pertubation AUF-related
        # parameters if we are using the perturbation AUF component.
        if self.include_perturb_auf:
            for check_flag in ['a_tri_set_name', 'b_tri_set_name', 'a_tri_filt_names',
                               'b_tri_filt_names', 'a_psf_fwhms', 'a_norm_scale_laws',
                               'b_psf_fwhms', 'b_norm_scale_laws']:
                if check_flag not in config:
                    raise ValueError("Missing key {} from metadata file.".format(check_flag))
            self.a_tri_set_name = config['a_tri_set_name']
            self.b_tri_set_name = config['b_tri_set_name']
            self.a_tri_filt_names = np.array(config['a_tri_filt_names'].split())
            self.b_tri_filt_names = np.array(config['b_tri_filt_names'].split())
            for flag in ['a_tri_filt_num', 'b_tri_filt_num']:
                try:
                    a = config[flag]
                    if float(a).is_integer():
                        setattr(self, flag, int(a))
                    else:
                        raise ValueError("{} should be a single integer number.".format(flag))
                except ValueError:
                    raise ValueError("{} should be a single integer number.".format(flag))

            # These parameters are also only used if we are using the
            # Perturbation AUF component.
            for flag in ['a_', 'b_']:
                for name in ['psf_fwhms', 'norm_scale_laws']:
                    a = config['{}{}'.format(flag, name)].split()
                    try:
                        b = np.array([float(f) for f in a])
                    except ValueError:
                        raise ValueError('{}{} should be a list of floats.'.format(flag, name))
                    if len(b) != len(getattr(self, '{}tri_filt_names'.format(flag))):
                        raise ValueError('{}{} and {}tri_filt_names should contain the '
                                         'same number of entries.'.format(flag, name, flag))
                    setattr(self, '{}{}'.format(flag, name), b)

        for flag in ['a_', 'b_']:
            a = config['{}filt_names'.format(flag)].split()
            if self.include_perturb_auf:
                if len(a) != len(getattr(self, '{}tri_filt_names'.format(flag))):
                    raise ValueError('{}filt_names and {}tri_filt_names should contain '
                                     'the same number of entries.'.format(flag, flag))
            setattr(self, '{}filt_names'.format(flag), a)

        self.a_cat_name = config['a_cat_name']
        self.b_cat_name = config['b_cat_name']
