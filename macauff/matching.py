# Licensed under a 3-clause BSD style license - see LICENSE
'''
This module provides the high-level framework for performing catalogue-catalogue cross-matches.
'''

import os
import sys
import warnings
from configparser import ConfigParser
import numpy as np

from .perturbation_auf import make_perturb_aufs
from .group_sources import make_island_groupings
from .misc_functions_fortran import misc_functions_fortran as mff

__all__ = ['CrossMatch']


class CrossMatch():
    '''
    A class to cross-match two photometric catalogues with one another, producing
    a composite catalogue of merged sources.

    The class takes three paths, the locations of the metadata files containing
    all of the necessary parameters for the cross-match, and outputs a file
    containing the appropriate columns of the datasets plus additional derived
    parameters.

    Parameters
    ----------
    joint_file_path : string
        A path to the location of the file containing the cross-match metadata.
    cat_a_file_path : string
        A path to the location of the file containing the catalogue "a" specific
        metadata.
    cat_b_file_path : string
        A path to the location of the file containing the catalogue "b" specific
        metadata.
    '''

    def __init__(self, joint_file_path, cat_a_file_path, cat_b_file_path):
        '''
        Initialisation function for cross-match class.
        '''
        for f in [joint_file_path, cat_a_file_path, cat_b_file_path]:
            if not os.path.isfile(f):
                raise FileNotFoundError("Input parameter file {} could not be found.".format(f))

        self.joint_file_path = joint_file_path
        self.cat_a_file_path = cat_a_file_path
        self.cat_b_file_path = cat_b_file_path

        self.read_metadata()

        # Important steps that can be save points in the match process are:
        # AUF creation, island splitting, c/f creation, star pairing. We have
        # to check if any later stages are flagged to not run (i.e., they are
        # the starting point) than earlier stages, and raise an error.
        flags = np.array([self.run_auf, self.run_group, self.run_cf, self.run_source])
        for i in range(3):
            if flags[i] and np.any(~flags[i+1:]):
                raise ValueError("Inconsistency between run/no run flags; please ensure that "
                                 "if a sub-process is set to run that all subsequent "
                                 "processes are also set to run.")

        # Ensure that we can create the folders for outputs.
        for path in ['group', 'reject']:
            try:
                os.makedirs('{}/{}'.format(self.joint_folder_path, path), exist_ok=True)
            except OSError:
                raise OSError("Error when trying to create temporary folder for joint outputs. "
                              "Please ensure that joint_folder_path is correct.")

        for path, catname, flag in zip([self.a_auf_folder_path, self.b_auf_folder_path],
                                       ['"a"', '"b"'], ['a_', 'b_']):
            try:
                os.makedirs(path, exist_ok=True)
            except OSError:
                raise OSError("Error when trying to create temporary folder for catalogue {} AUF "
                              "outputs. Please ensure that {}auf_folder_path is correct."
                              .format(catname, flag))

        # Unlike the AUF folder paths, which are allowed to not exist at
        # runtime, we simply check that cat_folder_path exists for both
        # input catalogues, with three appropriately shaped arrays in it,
        # and error if not.
        for path, catname, flag in zip([self.a_cat_folder_path, self.b_cat_folder_path],
                                       ['"a"', '"b"'], ['a_', 'b_']):
            if not os.path.exists(path):
                raise OSError('{}cat_folder_path does not exist. Please ensure that '
                              'path for catalogue {} is correct.'.format(flag, catname))
            else:
                # Currently forcing hard-coded three-part numpy array names,
                # to come out of "skinny table" consolidated catalogue
                # generation.
                for file_name in ['con_cat_astro', 'con_cat_photo', 'magref']:
                    if not os.path.isfile('{}/{}.npy'.format(path, file_name)):
                        raise FileNotFoundError('{} file not found in catalogue {} path. '
                                                'Please run catalogue consolidation'.format(
                                                    file_name, catname))
                # Shape, mapped to each of astro/photo/magref respectively,
                # should map to 3, number of magnitudes, and 1, where magref is
                # a 1-D array but the other two are 2-D.
                fn_a = np.load('{}/con_cat_astro.npy'.format(path), mmap_mode='r')
                fn_p = np.load('{}/con_cat_photo.npy'.format(path), mmap_mode='r')
                fn_m = np.load('{}/magref.npy'.format(path), mmap_mode='r')
                if len(fn_a.shape) != 2 or len(fn_p.shape) != 2 or len(fn_m.shape) != 1:
                    raise ValueError("Incorrect number of dimensions in consolidated "
                                     "catalogue {} files.".format(catname))
                if fn_a.shape[1] != 3:
                    raise ValueError("Second dimension of con_cat_astro in catalogue {} "
                                     "should be 3.".format(catname))
                if fn_p.shape[1] != len(getattr(self, '{}filt_names'.format(flag))):
                    raise ValueError("Second dimension of con_cat_photo in catalogue {} "
                                     "should be the same as the number of filters listed "
                                     "in {}filt_names.".format(catname, flag))
                if fn_m.shape[0] != fn_a.shape[0] or fn_p.shape[0] != fn_a.shape[0]:
                    raise ValueError("Consolidated catalogue arrays for catalogue {} should "
                                     "all be consistent lengths.".format(catname))

        for folder in [self.a_cat_name, self.b_cat_name]:
            try:
                os.makedirs('{}/{}'.format(self.joint_folder_path, folder), exist_ok=True)
            except OSError:
                raise OSError("Error when trying to create temporary folder for catalogue-level "
                              "outputs. Please ensure that catalogue folder names are correct.")

        if self.include_perturb_auf:
            for tri_flag, catname in zip([self.a_download_tri, self.b_download_tri], ['a_', 'b_']):
                if tri_flag and not self.run_auf:
                    raise ValueError("{}download_tri is True and run_auf is False. Please ensure "
                                     "that run_auf is True if new TRILEGAL simulations are to be "
                                     "downloaded.".format(catname))

        self.make_shared_data()

    def __call__(self):
        '''
        Call function for CrossMatch, to run the various stages of cross-matching
        two photometric catalogues.
        '''
        # The first step is to create the perturbation AUF components, if needed.
        # If run_auf is set to True or if there are not the appropriate number of
        # pre-saved outputs from a previous run then run perturbation AUF creation.
        # TODO: generalise the number of files per AUF simulation as input arg.
        self.create_perturb_auf(7)

        # Once AUF components are assembled, we now group sources based on
        # convolved AUF integration lengths, to get "common overlap" sources
        # and merge such overlaps into distinct "islands" of sources to match.
        self.group_sources(5)

    def _str2bool(self, v):
        '''
        Convenience function to convert strings to boolean values.

        Parameters
        ----------
        v : string
            String entry to be converted to ``True`` or ``False``.

        Returns
        -------
        flag_val : boolean
            Boolean-converted value that ``v`` represents.
        '''
        val = v.lower()
        if val not in ("yes", "true", "t", "1", "no", "false", "f", "0"):
            raise ValueError('Boolean flag key not set to allowed value.')
        else:
            flag_val = v.lower() in ("yes", "true", "t", "1")
            return flag_val

    def _make_regions_points(self, region_type, region_frame, region_points):
        '''
        Wrapper function for the creation of "region" coordinate tuples,
        given either a set of rectangular points or a list of coordinates.

        Parameters
        ----------
        region_type : string
            String containing the kind of system the region pointings are in.
            Should be "rectangle", regularly sampled points in the two sky
            coordinates, or "points", individually specified sky coordinates.
        region_Frame : string
            String containing the coordinate system the points are in. Should
            be either "equatorial" or "galactic".
        region_points : string
            String containing the evaluation points. If ``region_type`` is
            "rectangle", should be six values, the start and stop values and
            number of entries of the respective sky coordinates; and if
            ``region_type`` is "points", ``region_points`` should be tuples
            of the form ``(a, b)`` separated by whitespace.
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
        joint_config = ConfigParser()
        with open(self.joint_file_path) as f:
            joint_config.read_string('[config]\n' + f.read())
        joint_config = joint_config['config']
        cat_a_config = ConfigParser()
        with open(self.cat_a_file_path) as f:
            cat_a_config.read_string('[config]\n' + f.read())
        cat_a_config = cat_a_config['config']
        cat_b_config = ConfigParser()
        with open(self.cat_b_file_path) as f:
            cat_b_config.read_string('[config]\n' + f.read())
        cat_b_config = cat_b_config['config']

        for check_flag in ['include_perturb_auf', 'include_phot_like', 'run_auf', 'run_group',
                           'run_cf', 'run_source', 'cf_region_type', 'cf_region_frame',
                           'cf_region_points', 'joint_folder_path', 'pos_corr_dist',
                           'real_hankel_points', 'four_hankel_points', 'four_max_rho',
                           'cross_match_extent', 'mem_chunk_num']:
            if check_flag not in joint_config:
                raise ValueError("Missing key {} from joint metadata file.".format(check_flag))

        for config, catname in zip([cat_a_config, cat_b_config], ['"a"', '"b"']):
            for check_flag in ['auf_region_type', 'auf_region_frame', 'auf_region_points',
                               'filt_names', 'cat_name', 'dens_dist', 'auf_folder_path',
                               'cat_folder_path']:
                if check_flag not in config:
                    raise ValueError("Missing key {} from catalogue {} metadata file.".format(
                                     check_flag, catname))

        for run_flag in ['include_perturb_auf', 'include_phot_like', 'run_auf', 'run_group',
                         'run_cf', 'run_source']:
            setattr(self, run_flag, self._str2bool(joint_config[run_flag]))

        for config, catname in zip([cat_a_config, cat_b_config], ['a_', 'b_']):
            self._make_regions_points(['{}auf_region_type'.format(catname),
                                       config['auf_region_type']],
                                      ['{}auf_region_frame'.format(catname),
                                       config['auf_region_frame']],
                                      ['{}auf_region_points'.format(catname),
                                       config['auf_region_points']])

        self._make_regions_points(['cf_region_type', joint_config['cf_region_type']],
                                  ['cf_region_frame', joint_config['cf_region_frame']],
                                  ['cf_region_points', joint_config['cf_region_points']])

        # If the frame of the two AUF parameter files and the 'cf' frame are
        # not all the same then we have to raise an error.
        if (self.a_auf_region_frame != self.b_auf_region_frame or
                self.a_auf_region_frame != self.cf_region_frame):
            raise ValueError("Region frames for c/f and AUF creation must all be the same.")

        self.joint_folder_path = os.path.abspath(joint_config['joint_folder_path'])
        self.a_auf_folder_path = os.path.abspath(cat_a_config['auf_folder_path'])
        self.b_auf_folder_path = os.path.abspath(cat_b_config['auf_folder_path'])

        self.a_cat_folder_path = os.path.abspath(cat_a_config['cat_folder_path'])
        self.b_cat_folder_path = os.path.abspath(cat_b_config['cat_folder_path'])

        self.a_filt_names = np.array(cat_a_config['filt_names'].split())
        self.b_filt_names = np.array(cat_b_config['filt_names'].split())

        # Only have to check for the existence of Pertubation AUF-related
        # parameters if we are using the perturbation AUF component.
        if self.include_perturb_auf:
            for config, catname in zip([cat_a_config, cat_b_config], ['"a"', '"b"']):
                for check_flag in ['tri_set_name', 'tri_filt_names', 'psf_fwhms',
                                   'norm_scale_laws', 'download_tri']:
                    if check_flag not in config:
                        raise ValueError("Missing key {} from catalogue {} metadata file.".format(
                                         check_flag, catname))
            self.a_download_tri = self._str2bool(cat_a_config['download_tri'])
            self.b_download_tri = self._str2bool(cat_b_config['download_tri'])
            self.a_tri_set_name = cat_a_config['tri_set_name']
            self.b_tri_set_name = cat_b_config['tri_set_name']
            for config, flag in zip([cat_a_config, cat_b_config], ['a_', 'b_']):
                a = config['tri_filt_names'].split()
                if len(a) != len(getattr(self, '{}filt_names'.format(flag))):
                    raise ValueError('{}tri_filt_names and {}filt_names should contain the '
                                     'same number of entries.'.format(flag, flag))
                setattr(self, '{}tri_filt_names'.format(flag),
                        np.array(config['tri_filt_names'].split()))

            for config, catname, flag in zip([cat_a_config, cat_b_config], ['"a"', '"b"'],
                                             ['a_', 'b_']):
                try:
                    a = config['tri_filt_num']
                    if float(a).is_integer():
                        setattr(self, '{}tri_filt_num'.format(flag), int(a))
                    else:
                        raise ValueError("tri_filt_num should be a single integer number in "
                                         "catalogue {} metadata file.".format(catname))
                except ValueError:
                    raise ValueError("tri_filt_num should be a single integer number in "
                                     "catalogue {} metadata file.".format(catname))

            # These parameters are also only used if we are using the
            # Perturbation AUF component.
            for config, catname, flag in zip([cat_a_config, cat_b_config], ['"a"', '"b"'],
                                             ['a_', 'b_']):
                a = config['norm_scale_laws'].split()
                try:
                    b = np.array([float(f) for f in a])
                except ValueError:
                    raise ValueError('norm_scale_laws should be a list of floats in '
                                     'catalogue {} metadata file.'.format(catname))
                if len(b) != len(getattr(self, '{}filt_names'.format(flag))):
                    raise ValueError('{}norm_scale_laws and {}filt_names should contain the '
                                     'same number of entries.'.format(flag, flag))
                setattr(self, '{}norm_scale_laws'.format(flag), b)

        for config, catname, flag in zip([cat_a_config, cat_b_config], ['"a"', '"b"'],
                                         ['a_', 'b_']):
            a = config['psf_fwhms'].split()
            try:
                b = np.array([float(f) for f in a])
            except ValueError:
                raise ValueError('psf_fwhms should be a list of floats in catalogue {} metadata '
                                 'file.'.format(catname))
            if len(b) != len(getattr(self, '{}filt_names'.format(flag))):
                raise ValueError('{}psf_fwhms and {}filt_names should contain the '
                                 'same number of entries.'.format(flag, flag))
            setattr(self, '{}psf_fwhms'.format(flag), b)

        self.a_cat_name = cat_a_config['cat_name']
        self.b_cat_name = cat_b_config['cat_name']

        try:
            self.pos_corr_dist = float(joint_config['pos_corr_dist'])
        except ValueError:
            raise ValueError("pos_corr_dist must be a float.")

        for config, catname, flag in zip([cat_a_config, cat_b_config], ['"a"', '"b"'],
                                         ['a_', 'b_']):
            try:
                setattr(self, '{}dens_dist'.format(flag), float(config['dens_dist']))
            except ValueError:
                raise ValueError("dens_dist in catalogue {} must be a float.".format(catname))

        for flag in ['real_hankel_points', 'four_hankel_points', 'four_max_rho']:
            a = joint_config[flag]
            try:
                a = float(a)
            except ValueError:
                raise ValueError("{} should be an integer.".format(flag))
            if not a.is_integer():
                raise ValueError("{} should be an integer.".format(flag))
            setattr(self, flag, int(a))

        a = joint_config['cross_match_extent'].split()
        try:
            b = np.array([float(f) for f in a])
        except ValueError:
            raise ValueError("All elements of cross_match_extent should be floats.")
        if len(b) != 4:
            raise ValueError("cross_match_extent should contain four elements.")
        self.cross_match_extent = b

        try:
            a = joint_config['mem_chunk_num']
            if float(a).is_integer():
                self.mem_chunk_num = int(a)
            else:
                raise ValueError("mem_chunk_num should be a single integer number.")
        except ValueError:
            raise ValueError("mem_chunk_num should be a single integer number.")

    def make_shared_data(self):
        """
        Function to initialise the shared variables used in the cross-match process.
        """

        self.r = np.linspace(0, self.pos_corr_dist, self.real_hankel_points)
        self.dr = np.diff(self.r)
        self.rho = np.linspace(0, self.four_max_rho, self.four_hankel_points)
        self.drho = np.diff(self.rho)
        # Only need to calculate this the first time we need it, so buffer for now.
        self.j0s = None

    def create_perturb_auf(self, files_per_auf_sim, perturb_auf_func=make_perturb_aufs):
        '''
        Function wrapping the main perturbation AUF component creation routines.

        Parameters
        ----------
        files_per_auf_sim : integer
            The number of output files for each individual perturbation simulation.
        perturb_auf_func : callable, optional
            ``perturb_auf_func`` should create the perturbation AUF output files
            for each filter-pointing combination.
        '''
        # Each catalogue has in its auf_folder_path a single file, a local
        # normalising density, plus -- per AUF "pointing" -- a simulation file
        # and N simulation files per filter. Additionally, it will contain a
        # single file with each source's index reference into the cube of AUFs,
        # and a convenience cube for each N-m combination array's length.
        a_expected_files = 3 + len(self.a_auf_region_points) + (
            files_per_auf_sim * len(self.a_filt_names) * len(self.a_auf_region_points))
        a_file_number = np.sum([len(files) for _, _, files in
                                os.walk(self.a_auf_folder_path)])
        a_correct_file_number = a_expected_files == a_file_number

        a_n_sources = len(np.load('{}/magref.npy'.format(self.a_cat_folder_path), mmap_mode='r'))

        if self.run_auf or not a_correct_file_number:
            # Only warn if we did NOT choose to run AUF, but DID hit wrong file
            # number.
            if not a_correct_file_number and not self.run_auf:
                warnings.warn('Incorrect number of files in catalogue "a" perturbation'
                              'AUF simulation folder. Deleting all files and re-running '
                              'cross-match process.')
                # Once run AUf flag is updated, all other flags need to be set to run
                self.run_group, self.run_cf, self.run_source = True, True, True
            os.system("rm -rf {}/*".format(self.a_auf_folder_path))
            if self.include_perturb_auf:
                _kwargs = {'psf_fwhms': self.a_psf_fwhms, 'tri_download_flag': self.a_download_tri}
            else:
                _kwargs = {}
            perturb_auf_func(self.a_auf_folder_path, self.a_cat_folder_path, self.a_filt_names,
                             self.a_auf_region_points, self.cross_match_extent, self.r, self.dr,
                             self.rho, self.drho, 'a', self.include_perturb_auf, a_n_sources,
                             self.mem_chunk_num, **_kwargs)
        else:
            print('Loading empirical crowding AUFs for catalogue "a"...')
            sys.stdout.flush()

        b_expected_files = 3 + len(self.b_auf_region_points) + (
            files_per_auf_sim * len(self.b_filt_names) * len(self.b_auf_region_points))
        b_file_number = np.sum([len(files) for _, _, files in
                                os.walk(self.b_auf_folder_path)])
        b_correct_file_number = b_expected_files == b_file_number

        b_n_sources = len(np.load('{}/magref.npy'.format(self.b_cat_folder_path), mmap_mode='r'))

        if self.run_auf or not b_correct_file_number:
            if not b_correct_file_number and not self.run_auf:
                warnings.warn('Incorrect number of files in catalogue "b" perturbation'
                              'AUF simulation folder. Deleting all files and re-running '
                              'cross-match process.')
                self.run_group, self.run_cf, self.run_source = True, True, True
            os.system("rm -rf {}/*".format(self.b_auf_folder_path))
            if self.include_perturb_auf:
                _kwargs = {'psf_fwhms': self.b_psf_fwhms, 'tri_download_flag': self.b_download_tri}
            else:
                _kwargs = {}
            perturb_auf_func(self.b_auf_folder_path, self.b_cat_folder_path, self.b_filt_names,
                             self.b_auf_region_points, self.cross_match_extent, self.r, self.dr,
                             self.rho, self.drho, 'b', self.include_perturb_auf, b_n_sources,
                             self.mem_chunk_num, **_kwargs)
        else:
            print('Loading empirical crowding AUFs for catalogue "b"...')
            sys.stdout.flush()

    def group_sources(self, files_per_grouping, group_func=make_island_groupings):
        '''
        Function to handle the creation of catalogue "islands" and potential
        astrometrically related sources across the two catalogues.

        Parameters
        ----------
        files_per_grouping : integer
            The number of output files from each catalogue, made during the
            island and overlap creation process.
        group_func : callable, optional
            ``group_func`` should create the various island- and overlap-related
            files by which objects across the two catalogues are assigned as
            potentially counterparts to one another.
        '''

        # Each catalogue should expect 5 files in "group/" or "reject/": island
        # lengths, indices into the opposite catalogue for each source, the
        # indices of sources in this catalogue in each island, the number of
        # opposing catalogue overlaps for each source, and the list of any
        # "rejected" source indices. However, there may be no "reject" arrays,
        # so we might expect two fewer files.
        if (np.all(['reject_a' not in f for f in
                    os.listdir('{}/reject'.format(self.joint_folder_path))]) and
            np.all(['reject_b' not in f for f in
                    os.listdir('{}/reject'.format(self.joint_folder_path))])):
            expected_files = (files_per_grouping - 1) * 2
        else:
            expected_files = files_per_grouping * 2
        file_number = np.sum([len(files) for _, _, files in
                              os.walk('{}/group'.format(self.joint_folder_path))]) + np.sum(
            [len(files) for _, _, files in os.walk('{}/reject'.format(self.joint_folder_path))])
        correct_file_number = expected_files == file_number

        # Currently hard-code the integral fractions used in group_sources:
        int_fracs = np.array([0.63, 0.9, 0.99])

        # First check whether we actually need to dip into the group sources
        # routine or not.
        if self.run_group or not correct_file_number:
            if self.j0s is None:
                self.j0s = mff.calc_j0(self.rho[:-1]+self.drho/2, self.r[:-1]+self.dr/2)
            # Only worry about the warning if we didn't choose to run the grouping
            # but hit incorrect file numbers.
            if not correct_file_number and not self.run_group:
                warnings.warn('Incorrect number of grouping files. Deleting all '
                              'grouping files and re-running cross-match process.')
                self.run_cf, self.run_source = True, True
            os.system('rm -rf {}/group/*'.format(self.joint_folder_path))
            os.system('rm -rf {}/reject/*'.format(self.joint_folder_path))
            group_func(self.joint_folder_path, self.a_cat_folder_path, self.b_cat_folder_path,
                       self.a_auf_folder_path, self.b_auf_folder_path, self.a_auf_region_points,
                       self.b_auf_region_points, self.a_filt_names, self.b_filt_names,
                       self.a_cat_name, self.b_cat_name, self.r, self.dr, self.rho, self.drho,
                       self.j0s, self.pos_corr_dist, self.cross_match_extent, int_fracs,
                       self.mem_chunk_num, self.include_phot_like)
        else:
            print('Loading catalogue islands and overlaps...')
            sys.stdout.flush()
