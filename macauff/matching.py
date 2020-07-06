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

        for run_flag in ['include_perturb_auf', 'include_phot_like']:
            if run_flag in config:
                setattr(self, run_flag, self._str2bool(config[run_flag]))
            else:
                raise ValueError("Missing key {} from metadata file.".format(run_flag))

        for run_flag in ['run_auf', 'run_group', 'run_cf', 'run_star']:
            if run_flag in config:
                setattr(self, run_flag, self._str2bool(config[run_flag]))
            else:
                raise ValueError("Missing key {} from metadata file.".format(run_flag))
