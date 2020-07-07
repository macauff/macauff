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
                           'auf_region_points']:
            if check_flag not in config:
                raise ValueError("Missing key {} from metadata file.".format(check_flag))

        for run_flag in ['include_perturb_auf', 'include_phot_like', 'run_auf', 'run_group',
                         'run_cf', 'run_star']:
            setattr(self, run_flag, self._str2bool(config[run_flag]))

        art = config['auf_region_type'].lower()
        if art == 'rectangle':
            try:
                a = config['auf_region_points'].split()
                a = [float(point) for point in a]
            except ValueError:
                raise ValueError("auf_region_points should be 6 numbers separated "
                                 "by spaces.")
            if len(a) == 6:
                if not a[2].is_integer() or not a[5].is_integer():
                    raise ValueError("Number of steps between start and stop values for "
                                     "auf_region_points should be integers.")
                ax1_p = np.linspace(a[0], a[1], int(a[2]))
                ax2_p = np.linspace(a[3], a[4], int(a[5]))
                points = np.stack(np.meshgrid(ax1_p, ax2_p), -1).reshape(-1, 2)
            else:
                raise ValueError("auf_region_points should be 6 numbers separated "
                                 "by spaces.")
        elif art == 'points':
            try:
                a = config['auf_region_points'].replace('(', ')').split('), )')
                # Remove the first ( and final ) that weren't split by "), (" -> "), )"
                a[0] = a[0][1:]
                a[-1] = a[-1][:-1]
                b = [q.split(', ') for q in a]
                points = np.array(b, dtype=float)
            except ValueError:
                raise ValueError("auf_region_points should be a list of '(a, b), (c, d)' tuples, "
                                 "separated by a comma.")
        else:
            raise ValueError("auf_region_type should either be 'rectangle' or 'points'.")

        self.auf_region_points = points

        arf = config['auf_region_frame'].lower()
        if arf == 'equatorial' or arf == 'galactic':
            self.auf_region_frame = arf
        else:
            raise ValueError("auf_region_frame should either be 'equatorial' or 'galactic'.")
