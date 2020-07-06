# Licensed under a 3-clause BSD style license - see LICENSE
'''
This module provides the high-level framework for performing catalogue-catalogue cross-matches.
'''

import os

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
        if not os.path.isfile(file_path):
            raise FileNotFoundError("Metadata file could not be found at "
                                    "specified location.")
