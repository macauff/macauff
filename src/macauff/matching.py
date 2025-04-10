# Licensed under a 3-clause BSD style license - see LICENSE
'''
This module provides the high-level framework for performing catalogue-catalogue cross-matches.
'''
# pylint: disable=too-many-lines
# pylint: disable=duplicate-code

import datetime
import os
import sys
import warnings
import yaml
from time import sleep

import numpy as np

try:
    from mpi4py import MPI  # pylint:disable=import-error
except ModuleNotFoundError:
    MPI = None

# pylint: disable=import-error,no-name-in-module
from macauff.counterpart_pairing import source_pairing
from macauff.fit_astrometry import AstrometricCorrections, SNRMagnitudeRelationship
from macauff.group_sources import make_island_groupings
from macauff.macauff import Macauff
from macauff.parse_catalogue import csv_to_npy, npy_to_csv
from macauff.perturbation_auf import make_perturb_aufs
from macauff.photometric_likelihood import compute_photometric_likelihoods

# pylint: enable=import-error,no-name-in-module

__all__ = ['CrossMatch']


# Dynamic attribute assignment causes pylint some headaches, so just disable it.
# This means that attribute use and/or assignment should be checked carefully!
# pylint: disable=no-member
# pylint: disable-next=too-many-instance-attributes
class CrossMatch():
    '''
    A class to cross-match two photometric catalogues with one another, producing
    a composite catalogue of merged sources.

    Parameters
    ----------
    parameter_file_folder_path : string
        A path to the location of the folder containing input parameter files.
    resume_file_path : string, optional
        A path to the location of the file containing resume information for the
        cross match.
    use_mpi : boolean, optional
        Enable/disable the use of MPI parallelisation (enabled by default).
    walltime : string, optional
        Maximum runtime of the cross-match job in format 'HH:MM:SS' (hours, minutes
        and seconds). Controls checkpointing.
    end_within : string, optional
        End time in 'HH:MM:SS' format (hours, minutes and seconds). Default is
        '00:10:00', i.e. end within 10 minutes of the given walltime.
    polling_rate : integer, optional
        Rate in seconds at which manager process checks for new work requests and
        monitors walltime. Default is 1 second.
    '''

    def __init__(self, crossmatch_params_file_path, cat_a_params_file_path, cat_b_params_file_path,
                 resume_file_path=None, use_mpi=True, walltime=None, end_within='00:10:00', polling_rate=1):
        '''
        Initialisation function for cross-match class.
        '''
        for file in [crossmatch_params_file_path, cat_a_params_file_path, cat_b_params_file_path]:
            if not os.path.exists(file):
                raise FileNotFoundError(f"File {file} not found on disk.")
        self.crossmatch_params_file_path = crossmatch_params_file_path
        self.cat_a_params_file_path = cat_a_params_file_path
        self.cat_b_params_file_path = cat_b_params_file_path

        # Initialise MPI if available and enabled
        if MPI is not None and use_mpi:
            self.comm = MPI.COMM_WORLD  # pylint: disable=c-extension-no-member
            self.rank = self.comm.Get_rank()
            self.comm_size = self.comm.Get_size()
            # Set MPI error handling to return exceptions rather than MPI_Abort the
            # application. Allows for recovery of crashed workers.
            self.comm.Set_errhandler(MPI.ERRORS_RETURN)  # pylint: disable=c-extension-no-member
        else:
            if use_mpi:
                print("Warning: MPI initialisation failed. Check mpi4py is correctly installed. "
                      "Falling back to serial mode.")
            self.rank = 0
            self.comm_size = 1

        # Special signals for MPI processes
        #   'NO_MORE_WORK' - manager uses to signal workers to shut down
        #   'WORK_REQUEST' - manager uses to signal new chunk for processing.
        #                    worker uses to request initial chunk from manager.
        #   'WORK_COMPLETE' - worker uses to report successfully processed given chunk
        #   'WORK_ERROR' - worker uses to report failed processing of given chunk
        self.worker_signals = {'NO_MORE_WORK': 0,
                               'WORK_REQUEST': 1,
                               'WORK_COMPLETE': 2,
                               'WORK_ERROR': 3}
        # Only manager process needs to set up the resume file and queue
        if self.rank == 0:
            completed_chunks = set()
            try:
                # Open and read existing resume file
                # pylint: disable-next=consider-using-with
                self.resume_file = open(resume_file_path, 'r+', encoding='utf-8')
                for line in self.resume_file:
                    completed_chunks.add(line.rstrip())
            except FileNotFoundError:
                # Resume file specified but does not exist. Create new one.
                # pylint: disable-next=consider-using-with
                self.resume_file = open(resume_file_path, 'w', encoding='utf-8')
            except TypeError:
                # Resume file was not specified. Disable checkpointing
                self.resume_file = None
            # Chunk queue will not contain chunks recorded as completed in the
            # resume file
            self.crossmatch_params_dict, self.cat_a_params_dict, self.cat_b_params_dict = self.read_metadata()
            self.chunk_queue = self._make_chunk_queue(completed_chunks)
            # Used to keep track of progress to completion
            self.num_chunks_to_process = len(self.chunk_queue)

            # In seconds, how often the manager checks for new work requests
            self.polling_rate = polling_rate

            if walltime is not None:
                # Expect job walltime and "end within" time in Hours:Minutes:Seconds (%H:%M:%S)
                # format, e.g. 02:44:12 for 2 hours, 44 minutes, 12 seconds
                # Calculate job end time from start time + walltime
                hour, minute, second = walltime.split(':')
                self.end_time = datetime.datetime.now() + \
                    datetime.timedelta(hours=int(hour), minutes=int(minute), seconds=int(second))
                # Keep track of "end within" time as a timedelta for easy comparison
                hour, minute, second = end_within.split(':')
                self.end_within = \
                    datetime.timedelta(hours=int(hour), minutes=int(minute), seconds=int(second))
            else:
                self.end_time = None
                self.end_within = None

    def _initialise_chunk(self):
        '''
        Initialisation function for a single chunk of sky.

        The function takes three paths, the locations of the metadata files containing
        all of the necessary parameters for the cross-match, and outputs a file
        containing the appropriate columns of the datasets plus additional derived
        parameters.
        '''

        # If astrometry of either catalogue needs fixing, do that now.
        if self.a_correct_astrometry or self.a_compute_snr_mag_relation:
            # Generate from current data: just need the singular chunk mid-points
            # and to leave all other parameters as they are.
            if len(self.a_auf_region_points) > 1:
                warnings.warn("a_auf_region_points contains more than one AUF sampling point, but either "
                              "a_correct_astrometry or a_compute_snr_mag_relation is True. Check results "
                              "carefully.")
            ax1_mids = np.array([self.a_auf_region_points[0, 0]])
            ax2_mids = np.array([self.a_auf_region_points[0, 1]])
            ax_dimension = 2
            a_npy_or_csv = 'csv'
            a_coord_or_chunk = 'chunk'
        if self.a_correct_astrometry:
            acbi = self.a_best_mag_index
            a_correct_astro_tri_name = '{}/{}/trilegal_auf_simulation'
            t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{t} Rank {self.rank}, chunk {self.chunk_id}: Calculating catalogue 'a' "
                  "uncertainty corrections...")
            ac = AstrometricCorrections(
                self.a_psf_fwhms[acbi], self.num_trials, self.a_nn_radius, self.a_dens_dist,
                self.a_correct_astro_save_folder, self.a_gal_wavs[acbi], self.a_gal_aboffsets[acbi],
                self.a_gal_filternames[acbi], self.a_gal_al_avs[acbi], self.d_mag, self.a_dd_params,
                # pylint: disable-next=possibly-used-before-assignment
                self.a_l_cut, ax1_mids, ax2_mids, ax_dimension, self.a_correct_mag_array,
                # pylint: disable-next=possibly-used-before-assignment
                self.a_correct_mag_slice, self.a_correct_sig_slice, self.n_pool, a_npy_or_csv,
                # pylint: disable-next=possibly-used-before-assignment
                a_coord_or_chunk, self.a_pos_and_err_indices, self.a_mag_indices, self.a_mag_unc_indices,
                self.a_filt_names, self.a_best_mag_index, self.a_auf_region_frame,
                self.a_saturation_magnitudes, trifolder=self.a_auf_folder_path,
                triname=a_correct_astro_tri_name, maglim_f=self.a_tri_maglim_faint,
                magnum=self.a_tri_filt_num, tri_num_faint=self.a_tri_num_faint,
                trifilterset=self.a_tri_set_name, trifiltname=self.a_tri_filt_names[acbi],
                tri_hist=self.a_dens_hist_tri_list[acbi], tri_mags=self.a_tri_model_mags_list[acbi],
                dtri_mags=self.a_tri_model_mags_interval_list[acbi],
                tri_uncert=self.a_tri_dens_uncert_list[acbi],
                use_photometric_uncertainties=self.a_use_photometric_uncertainties, pregenerate_cutouts=True,
                chunks=[self.chunk_id], n_r=self.real_hankel_points, n_rho=self.four_hankel_points,
                max_rho=self.four_max_rho, mn_fit_type=self.a_mn_fit_type)
            ac(a_cat_name=self.a_ref_csv_cat_file_string, b_cat_name=self.a_csv_cat_file_string,
               tri_download=self.a_download_tri, make_plots=True, overwrite_all_sightlines=True,
               seeing_ranges=self.a_seeing_ranges)

            # Having corrected the astrometry, we have to call csv_to_npy
            # now, rather than pre-generating our binary input catalogues.
            csv_folder, csv_filename = os.path.split(
                self.a_csv_cat_file_string.format(self.chunk_id))
            csv_to_npy(csv_folder, csv_filename, self.a_cat_folder_path,
                       self.a_pos_and_err_indices[1], self.a_mag_indices, self.a_best_mag_index_col,
                       self.a_chunk_overlap_col, header=False, process_uncerts=True,
                       astro_sig_fits_filepath=f'{self.a_correct_astro_save_folder}/npy',
                       cat_in_radec=self.a_auf_region_frame == 'equatorial',
                       mn_in_radec=self.a_auf_region_frame == 'equatorial')
        if self.a_compute_snr_mag_relation:
            t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{t} Rank {self.rank}, chunk {self.chunk_id}: Calculating catalogue 'a' "
                  "SNR-mag relations...")
            smr = SNRMagnitudeRelationship(
                self.a_correct_astro_save_folder, ax1_mids, ax2_mids, ax_dimension, a_npy_or_csv,
                a_coord_or_chunk, self.a_pos_and_err_indices, self.a_mag_indices,
                self.a_mag_unc_indices, self.a_filt_names, self.a_auf_region_frame,
                chunks=[self.chunk_id])
            smr(b_cat_name=self.a_csv_cat_file_string, make_plots=True, overwrite_all_sightlines=True)
        if self.a_correct_astrometry or self.a_compute_snr_mag_relation:
            # If we re-made either side's astrometry then we need to load its
            # SNR-mag relation now.
            os.system(f'cp {self.a_correct_astro_save_folder}/npy/snr_mag_params.npy '
                      f'{self.a_snr_mag_params_file_path}')
            if not os.path.isfile(self.a_snr_mag_params_file_path):
                raise FileNotFoundError(f'{self.a_snr_mag_params_file_path} file not found in catalogue '
                                        '"a" path. Please ensure astrometry corrections are pre-generated.')
            a = np.load(self.a_snr_mag_params_file_path)
            if not (len(a.shape) == 3 and a.shape[2] == 5 and
                    a.shape[0] == len(self.a_filt_names)):
                raise ValueError('a_snr_mag_params should be of shape (X, Y, 5)')
            self.a_snr_mag_params = a

        if self.b_correct_astrometry or self.b_compute_snr_mag_relation:
            # Generate from current data: just need the singular chunk mid-points
            # and to leave all other parameters as they are.
            if len(self.b_auf_region_points) > 1:
                warnings.warn("b_auf_region_points contains more than one AUF sampling point, but either "
                              "b_correct_astrometry or b_compute_snr_mag_relation is True. Check results "
                              "carefully.")
            ax1_mids = np.array([self.b_auf_region_points[0, 0]])
            ax2_mids = np.array([self.b_auf_region_points[0, 1]])
            ax_dimension = 2
            b_npy_or_csv = 'csv'
            b_coord_or_chunk = 'chunk'
        if self.b_correct_astrometry:
            bcbi = self.b_best_mag_index
            b_correct_astro_tri_name = '{}/{}/trilegal_auf_simulation'
            t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{t} Rank {self.rank}, chunk {self.chunk_id}: Calculating catalogue 'b' "
                  "uncertainty corrections...")
            ac = AstrometricCorrections(
                self.b_psf_fwhms[bcbi], self.num_trials, self.b_nn_radius, self.b_dens_dist,
                self.b_correct_astro_save_folder, self.b_gal_wavs[bcbi], self.b_gal_aboffsets[bcbi],
                self.b_gal_filternames[bcbi], self.b_gal_al_avs[bcbi], self.d_mag, self.b_dd_params,
                self.b_l_cut, ax1_mids, ax2_mids, ax_dimension, self.b_correct_mag_array,
                # pylint: disable-next=possibly-used-before-assignment
                self.b_correct_mag_slice, self.b_correct_sig_slice, self.n_pool, b_npy_or_csv,
                # pylint: disable-next=possibly-used-before-assignment
                b_coord_or_chunk, self.b_pos_and_err_indices, self.b_mag_indices, self.b_mag_unc_indices,
                self.b_filt_names, self.b_best_mag_index, self.b_auf_region_frame,
                self.b_saturation_magnitudes, trifolder=self.b_auf_folder_path,
                triname=b_correct_astro_tri_name, maglim_f=self.b_tri_maglim_faint,
                magnum=self.b_tri_filt_num, tri_num_faint=self.b_tri_num_faint,
                trifilterset=self.b_tri_set_name, trifiltname=self.b_tri_filt_names[bcbi],
                tri_hist=self.b_dens_hist_tri_list[bcbi], tri_mags=self.b_tri_model_mags_list[bcbi],
                dtri_mags=self.b_tri_model_mags_interval_list[bcbi],
                tri_uncert=self.b_tri_dens_uncert_list[bcbi],
                use_photometric_uncertainties=self.b_use_photometric_uncertainties,
                pregenerate_cutouts=True, chunks=[self.chunk_id],
                n_r=self.real_hankel_points, n_rho=self.four_hankel_points, max_rho=self.four_max_rho,
                mn_fit_type=self.b_mn_fit_type)
            ac(a_cat_name=self.b_ref_csv_cat_file_string, b_cat_name=self.b_csv_cat_file_string,
               tri_download=self.b_download_tri, make_plots=True, overwrite_all_sightlines=True,
               seeing_ranges=self.b_seeing_ranges)

            csv_folder, csv_filename = os.path.split(
                self.b_csv_cat_file_string.format(self.chunk_id))
            csv_to_npy(csv_folder, csv_filename, self.b_cat_folder_path,
                       self.b_pos_and_err_indices[1], self.b_mag_indices, self.b_best_mag_index_col,
                       self.b_chunk_overlap_col, header=False, process_uncerts=True,
                       astro_sig_fits_filepath=f'{self.b_correct_astro_save_folder}/npy',
                       cat_in_radec=self.b_auf_region_frame == 'equatorial',
                       mn_in_radec=self.b_auf_region_frame == 'equatorial')
        if self.b_compute_snr_mag_relation:
            t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{t} Rank {self.rank}, chunk {self.chunk_id}: Calculating catalogue 'b' "
                  "SNR-mag relations...")
            smr = SNRMagnitudeRelationship(
                self.b_correct_astro_save_folder, ax1_mids, ax2_mids, ax_dimension, b_npy_or_csv,
                b_coord_or_chunk, self.b_pos_and_err_indices, self.b_mag_indices,
                self.b_mag_unc_indices, self.b_filt_names, self.b_auf_region_frame,
                chunks=[self.chunk_id])
            smr(b_cat_name=self.b_csv_cat_file_string, make_plots=True, overwrite_all_sightlines=True)
        if self.b_correct_astrometry or self.b_compute_snr_mag_relation:
            os.system(f'cp {self.b_correct_astro_save_folder}/npy/snr_mag_params.npy '
                      f'{self.b_snr_mag_params_file_path}')
            f = 'snr_mag_params'
            if not os.path.isfile(self.b_snr_mag_params_file_path):
                raise FileNotFoundError(f'{f} file not found in catalogue "b" path. '
                                        'Please ensure astrometry corrections are pre-generated.')
            a = np.load(self.b_snr_mag_params_file_path)
            if not (len(a.shape) == 3 and a.shape[2] == 5 and
                    a.shape[0] == len(self.b_filt_names)):
                raise ValueError('b_snr_mag_params should be of shape (X, Y, 5)')
            self.b_snr_mag_params = a

        # Ensure that we can save to the folders for outputs.
        if not os.path.exists(self.joint_folder_path):
            raise OSError("Error when trying to check temporary folder for joint outputs. "
                          "Please ensure that joint_folder_path is correct.")

        for path, catname, flag in zip([self.a_auf_folder_path, self.b_auf_folder_path],
                                       ['"a"', '"b"'], ['a_', 'b_']):
            if path is not None:
                try:
                    os.makedirs(path, exist_ok=True)
                except OSError as exc:
                    raise OSError(f"Error when trying to create temporary folder for catalogue {catname} AUF "
                                  f"outputs. Please ensure that {flag}auf_folder_path is correct.") from exc

        # Unlike the AUF folder paths, which are allowed to not exist at
        # runtime, we simply check that cat_folder_path exists for both
        # input catalogues, with three appropriately shaped arrays in it,
        # and error if not.
        for path, catname, flag in zip([self.a_cat_folder_path, self.b_cat_folder_path],
                                       ['"a"', '"b"'], ['a_', 'b_']):
            if not os.path.exists(path):
                raise OSError(f'{flag}cat_folder_path does not exist. Please ensure that '
                              f'path for catalogue {catname} is correct.')
            # Currently forcing hard-coded three-part numpy array names,
            # to come out of "skinny table" consolidated catalogue
            # generation.
            for file_name in ['con_cat_astro', 'con_cat_photo', 'magref']:
                if not os.path.isfile(f'{path}/{file_name}.npy'):
                    raise FileNotFoundError(f'{file_name} file not found in catalogue {catname} path. '
                                            'Please run catalogue consolidation')
            # Shape, mapped to each of astro/photo/magref respectively,
            # should map to 3, number of magnitudes, and 1, where magref is
            # a 1-D array but the other two are 2-D.
            fn_a = np.load(f'{path}/con_cat_astro.npy')
            fn_p = np.load(f'{path}/con_cat_photo.npy')
            fn_m = np.load(f'{path}/magref.npy')
            if len(fn_a.shape) != 2 or len(fn_p.shape) != 2 or len(fn_m.shape) != 1:
                raise ValueError("Incorrect number of dimensions in consolidated "
                                 f"catalogue {catname} files.")
            if fn_a.shape[1] != 3:
                raise ValueError(f"Second dimension of con_cat_astro in catalogue {catname} "
                                 "should be 3.")
            if fn_p.shape[1] != len(getattr(self, f'{flag}filt_names')):
                raise ValueError(f"Second dimension of con_cat_photo in catalogue {catname} "
                                 "should be the same as the number of filters listed "
                                 f"in {flag}filt_names.")
            if fn_m.shape[0] != fn_a.shape[0] or fn_p.shape[0] != fn_a.shape[0]:
                raise ValueError(f"Consolidated catalogue arrays for catalogue {catname} should "
                                 "all be consistent lengths.")
            setattr(self, f'{flag}astro', fn_a)
            setattr(self, f'{flag}photo', fn_p)
            setattr(self, f'{flag}magref', fn_m)

        self.make_shared_data()

    def __call__(self):
        '''
        Call function for CrossMatch, performs cross-matching two photometric catalogues.
        '''

        # Special case for single process, i.e. serial version of code.
        # Do not use manager-worker pattern. Instead, one process loops over all chunks
        if self.comm_size == 1:  # pylint: disable=too-many-nested-blocks
            for chunk_id in self.chunk_queue:
                t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f'{t} Rank {self.rank} processing chunk {chunk_id}')
                self.chunk_id = chunk_id
                self._load_metadata_config(chunk_id)
                self._process_chunk()
                if self.resume_file is not None:
                    self.resume_file.write(chunk_id+'\n')
        else:
            # Manager process:
            #   - assigns chunks to workers
            #   - receives notification of completed or failed cross-matches
            #   - writes completed chunk IDs to resume file
            #   - TODO handle crashed workers (segfaults in Fortran routines currently unrecoverable)
            #   - TODO handle crashed manager process
            #   - once queue is empty, workers are sent signal to stop
            if self.rank == 0:
                # Maintain count of active workers.
                # Initially every process other than manager.
                active_workers = self.comm_size - 1

                # Loop until all workers have been instructed there is no more work
                while active_workers > 0:
                    # If checkpointing disabled, simply wait for any worker to
                    # request a chunk and report completion of any previous chunk
                    if self.end_time is None:
                        (signal, worker_id, chunk_id) = self.comm.recv()
                    # Otherwise, use non-blocking recv to allow manager to keep
                    # track of job time via polling loop
                    else:
                        req = self.comm.irecv()
                        # Use an infinite loop with break rather than "while not req.Get_status()"
                        # to ensure walltime is checked even if request returns immediately, i.e.
                        # emulate a "do-while" loop
                        while True:
                            # Check if we're reaching the limit of job walltime. If so,
                            # empty the queue so no further work is issued
                            if self.end_time - datetime.datetime.now() < self.end_within:
                                t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                print(f'{t} Rank {self.rank}: reaching job walltime. Cancelling all further '
                                      f'work. {self.num_chunks_to_process} chunks remain unprocessed.')
                                self.chunk_queue.clear()
                                # Blank end time so we don't re-enter polling loop
                                self.end_time = None
                                break
                            if req.Get_status():
                                break
                            sleep(self.polling_rate)
                        # Request complete, extract data
                        (signal, worker_id, chunk_id) = req.wait()

                    # Record completed chunk
                    if signal == self.worker_signals['WORK_COMPLETE'] and self.resume_file is not None:
                        t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f'{t} Rank {self.rank}: chunk {chunk_id} processed by rank {worker_id}. '
                              'Adding to resume file.')
                        self.resume_file.write(chunk_id+'\n')
                        # Do not buffer. Immediately commit to disk for
                        # resilience against crashes and overrunning walltime
                        # flush() alone is not enough. See:
                        # https://docs.python.org/3/library/os.html#os.fsync
                        self.resume_file.flush()
                        os.fsync(self.resume_file)
                        # Update number of remaining chunks to process
                        self.num_chunks_to_process -= 1
                    # Handle failed chunk
                    elif signal == self.worker_signals['WORK_ERROR']:
                        t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f'{t} Rank {self.rank}: rank {worker_id} failed to process chunk {chunk_id}.')

                    # Assign chunks until no more work.
                    # Then count "no more work" signals until no more workers.
                    try:
                        new_chunk = self.chunk_queue.pop(0)
                        signal = self.worker_signals['WORK_REQUEST']
                        t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f'{t} Rank {self.rank}: sending rank {worker_id} chunk {new_chunk}')
                    except IndexError:
                        new_chunk = None
                        signal = self.worker_signals['NO_MORE_WORK']
                        active_workers -= 1

                    sys.stdout.flush()

                    self.comm.send((signal, new_chunk, self.crossmatch_params_dict,
                                    self.cat_a_params_dict, self.cat_b_params_dict), dest=worker_id)

            # Worker processes:
            #  - request chunk from manager
            #  - loop until given signal to terminate
            else:
                signal = self.worker_signals['WORK_REQUEST']
                completed_chunk_id = None
                # Infinite loop until given signal to break
                while True:
                    # Send own rank ID to manager so it knows which process to assign work
                    # in addition to signal and completed chunk id
                    self.comm.send((signal, self.rank, completed_chunk_id), dest=0)
                    (signal, chunk_id, joint_config_dict, cat_a_config_dict,
                     cat_b_config_dict) = self.comm.recv(source=0)

                    self.crossmatch_params_dict = joint_config_dict
                    self.cat_a_params_dict = cat_a_config_dict
                    self.cat_b_params_dict = cat_b_config_dict

                    # Handle received signal.
                    # Terminate when signalled there is no more work...
                    if signal == self.worker_signals['NO_MORE_WORK']:
                        break
                    # ...or process the given chunk
                    self.chunk_id = chunk_id
                    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f'{t} Rank {self.rank}: processing chunk {chunk_id}')

                    try:
                        self._load_metadata_config(chunk_id)
                        self._process_chunk()
                        signal = self.worker_signals['WORK_COMPLETE']
                    except Exception as e:  # pylint: disable=broad-exception-caught
                        # Recover worker on chunk processing error
                        signal = self.worker_signals['WORK_ERROR']
                        # pylint: disable-next=fixme
                        # TODO More granular error handling
                        t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"{t} Rank {self.rank}: failed to process chunk {chunk_id}. "
                              f"Exception: {e}")

                    completed_chunk_id = chunk_id

                    sys.stdout.flush()

        # Clean up and shut down
        self._cleanup()

    def _process_chunk(self, joint_file_path, cat_a_file_path, cat_b_file_path):
        '''
        Runs the various stages of cross-matching two photometric catalogues
        '''
        # pylint: disable-next=fixme
        # TODO: more correctly pass these into CrossMatch as arguments later on.
        self.perturb_auf_func = make_perturb_aufs
        self.group_func = make_island_groupings
        self.phot_like_func = compute_photometric_likelihoods
        self.count_pair_func = source_pairing

        mcff = Macauff(self)
        mcff()

        # Following cross-match completion, perform post-processing
        self._postprocess_chunk()

    def _postprocess_chunk(self):
        '''
        Runs the post-processing stage, resolving duplicate cross-matches and
        optionally creating output .csv files for use elsewhere.

        Duplicates are determined by match pairs (or singular non-matches) that
        are entirely outside of the "core" for a given chunk. This core/halo
        divide is defined by the ``in_chunk_overlap`` array; if only a singular
        chunk is being matched (i.e., there is no compartmentalisation of a
        larger region), then ``in_chunk_overlap`` should all be set to ``False``.
        '''
        t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{t} Rank {self.rank}, chunk {self.chunk_id}: Removing halo matches and non-matches...")

        a_in_overlaps = np.load(f'{self.a_cat_folder_path}/in_chunk_overlap.npy')
        b_in_overlaps = np.load(f'{self.b_cat_folder_path}/in_chunk_overlap.npy')

        if self.include_phot_like and self.with_and_without_photometry:
            loop_array_extensions = ['', '_without_photometry']
        else:
            loop_array_extensions = ['']

        for lae in loop_array_extensions:
            ac, bc = getattr(self, f'ac{lae}'), getattr(self, f'bc{lae}')
            core_matches = ~a_in_overlaps[ac] | ~b_in_overlaps[bc]
            np.save(f'{self.joint_folder_path}/ac{lae}.npy', ac[core_matches])
            np.save(f'{self.joint_folder_path}/bc{lae}.npy', bc[core_matches])
            for fname in ['pc', 'eta', 'xi', 'crptseps', 'acontamflux', 'bcontamflux']:
                np.save(f'{self.joint_folder_path}/{fname}{lae}.npy',
                        getattr(self, f'{fname}{lae}')[core_matches])
            for fname in ['pacontam', 'pbcontam']:
                np.save(f'{self.joint_folder_path}/{fname}{lae}.npy',
                        getattr(self, f'{fname}{lae}')[:, core_matches])

            af, bf = getattr(self, f'af{lae}'), getattr(self, f'bf{lae}')
            a_core_nonmatches = ~a_in_overlaps[af]
            b_core_nonmatches = ~b_in_overlaps[bf]
            np.save(f'{self.joint_folder_path}/af{lae}.npy', af[a_core_nonmatches])
            np.save(f'{self.joint_folder_path}/bf{lae}.npy', bf[b_core_nonmatches])
            for fnametype, cnm in zip(['a', 'b'], [a_core_nonmatches, b_core_nonmatches]):
                for fname_ in ['{}fieldflux', 'pf{}', '{}fieldeta', '{}fieldxi', '{}fieldseps']:
                    fname = fname_.format(fnametype)
                    np.save(f'{self.joint_folder_path}/{fname}{lae}.npy', getattr(self, f'{fname}{lae}')[cnm])

            if self.reject_a is not None:
                np.save(f'{self.joint_folder_path}/reject_a{lae}.npy',
                        np.append(np.append(self.reject_a, ac[~core_matches]), af[~a_core_nonmatches]))
            else:
                np.save(f'{self.joint_folder_path}/reject_a{lae}.npy',
                        np.append(ac[~core_matches], af[~a_core_nonmatches]))
            if self.reject_b is not None:
                np.save(f'{self.joint_folder_path}/reject_b{lae}.npy',
                        np.append(np.append(self.reject_b, bc[~core_matches]), bf[~b_core_nonmatches]))
            else:
                np.save(f'{self.joint_folder_path}/reject_b{lae}.npy',
                        np.append(bc[~core_matches], bf[~b_core_nonmatches]))

            if self.make_output_csv:
                npy_to_csv(
                    [self.a_input_csv_folder, self.b_input_csv_folder], self.joint_folder_path,
                    self.output_csv_folder, [self.a_cat_csv_name, self.b_cat_csv_name],
                    [self.match_out_csv_name, self.a_nonmatch_out_csv_name,
                     self.b_nonmatch_out_csv_name], [self.a_cat_col_names, self.b_cat_col_names],
                    [self.a_cat_col_nums, self.b_cat_col_nums], [self.a_cat_name, self.b_cat_name],
                    [self.a_input_npy_folder, self.b_input_npy_folder],
                    headers=[self.a_csv_has_header, self.b_csv_has_header],
                    extra_col_name_lists=[self.a_extra_col_names, self.b_extra_col_names],
                    extra_col_num_lists=[self.a_extra_col_nums, self.b_extra_col_nums],
                    file_extension=lae)

    def _make_chunk_queue(self, completed_chunks):
        '''
        Determines the order with which chunks will be processed.

        Parameters
        ----------
        completed_chunks : list of strings
            List of already completed chunks, to be removed from the set of
            all chunks to be run, avoiding re-doing complete parts during
            an intermediate stage.

        Returns
        -------
        chunk_queue_sorted : list of tuples of strings
            List with one element per chunk. Each element a tuple of chunk ID and
            paths to metadata files in order (ID, cross-match, catalogue "a", catalogue "b")
        '''
        chunk_queue = np.copy(self.crossmatch_params_dict['chunk_id_list'])

        chunk_sizes = np.empty(len(chunk_queue), dtype=float)
        for i, chunk_id in enumerate(chunk_queue):
            cat_a_file_path = self.cat_a_params_dict['cat_folder_path'].format(chunk_id)
            cat_b_file_path = self.cat_b_params_dict['cat_folder_path'].format(chunk_id)

            for cat_folder_path in [cat_a_file_path, cat_b_file_path]:
                for path, _, files in os.walk(cat_folder_path):
                    for f in files:
                        file_path = os.path.join(path, f)
                        chunk_sizes[i] += os.path.getsize(file_path)

        # Sort chunk list by size, largest to smallest
        chunk_queue_sorted = chunk_queue[np.argsort(chunk_sizes)[::-1]]

        return chunk_queue_sorted

    def _cleanup(self):
        '''
        Final clean up operations before application shut down
        '''
        if self.rank == 0 and self.resume_file is not None:
            self.resume_file.close()

    def _load_metadata_config(self, chunk_id):
        '''
        Generate per-chunk class variables from the three stored parameter
        metadata files.

        Parameters
        ----------
        chunk_id : string
            Identifier for extraction of single element of metadata parameters
            that vary on a per-chunk basis, rather than being fixed for the
            entire catalogue/cross-match run, across all regions.
        '''
        for key, item in self.crossmatch_params_dict.items():
            if "_per_chunk" in key:
                # If the key contains the (end-)string per_chunk then this
                # is a list of parameters, one per chunk. In this case, pick
                # from the correct element based on chunk_id_list from the
                # joint-catalogue config file.
                ind = np.where(chunk_id == np.array(self.crossmatch_params_dict['chunk_id_list']))[0][0]
                _item = np.array(item[ind]) if item[ind] is list else item[ind]
                setattr(self, key.replace("_per_chunk", ""), _item)
            elif isinstance(item, str) and r"_{}" in item:
                # If input variable contains _{} in a string, then we expect
                # and assume that it should be modulated with the chunk ID.
                setattr(self, key, item.format(chunk_id))
            else:
                # Otherwise we just add the item unchanged.
                _item = np.array(item) if item is list else item
                setattr(self, key, _item)
        for cat_prefix, cat_dict in zip(['a_', 'b_'], [self.cat_a_params_dict, self.cat_b_params_dict]):
            for key, item in cat_dict.items():
                if "_per_chunk" in key:
                    ind = np.where(chunk_id == np.array(cat_dict['chunk_id_list']))[0][0]
                    _item = np.array(item[ind]) if item[ind] is list else item[ind]
                    setattr(self, f'{cat_prefix}{key.replace("_per_chunk", "")}', _item)
                elif isinstance(item, str) and r"_{}" in item:
                    setattr(self, f'{cat_prefix}{key}', item.format(chunk_id))
                else:
                    _item = np.array(item) if item is list else item
                    setattr(self, f'{cat_prefix}{key}', _item)

        for config, catname in zip([self.cat_a_params_dict, self.cat_b_params_dict], ['a_', 'b_']):
            ind = np.where(chunk_id == np.array(config['chunk_id_list']))[0][0]
            self._make_regions_points([f'{catname}auf_region_type', config['auf_region_type']],
                                      [f'{catname}auf_region_points',
                                       config['auf_region_points_per_chunk'][ind]],
                                      config['chunk_id_list'][ind])

        ind = np.where(chunk_id == np.array(self.crossmatch_params_dict['chunk_id_list']))[0][0]
        self._make_regions_points(['cf_region_type', self.crossmatch_params_dict['cf_region_type']],
                                  ['cf_region_points',
                                   self.crossmatch_params_dict['cf_region_points_per_chunk'][ind]],
                                  self.crossmatch_params_dict['chunk_id_list'][ind])

        for config, flag in zip([self.cat_a_params_dict, self.cat_b_params_dict], ['a_', 'b_']):
            if (self.crossmatch_params_dict['include_perturb_auf'] or config['correct_astrometry'] or
                    config['compute_snr_mag_relation']):
                # SNR-mag file path is for the full path including file, but we
                # first need to check if the folder itself exists, and only if
                # no corrections are being made do we need to know the file
                # exists.
                if not os.path.exists(os.path.dirname(os.path.abspath(
                        config['snr_mag_params_file_path'].format(chunk_id)))):
                    raise OSError(f"{catname}snr_mag_params_file_path's folder does not exist. Please ensure "
                                  f"that path for catalogue {catname} is correct.")
                if self.crossmatch_params_dict['include_perturb_auf']:
                    if not os.path.isfile(os.path.abspath(
                            config['snr_mag_params_file_path'].format(chunk_id))):
                        raise OSError(f"The file in {catname}snr_mag_params_file_path does not exist. Please "
                                      f"ensure that path for catalogue {catname} is correct.")

                if not (config['correct_astrometry'] or config['compute_snr_mag_relation']):
                    # If we are correcting the astrometry, we will be
                    # re-making the SNR-mag relations so skip loading.
                    a = np.load(config['snr_mag_params_file_path'].format(chunk_id))
                    if not (len(a.shape) == 3 and a.shape[2] == 5 and
                            a.shape[0] == len(getattr(self, f'{flag}filt_names'))):
                        raise ValueError(f'{flag}snr_mag_params should be of shape (X, Y, 5).')
                    setattr(self, f'{flag}snr_mag_params', a)
                else:
                    setattr(self, f'{flag}snr_mag_params_file_path',
                            os.path.abspath(config['snr_mag_params_file_path']))

            # Only need dd_params or l_cut if we're using run_psf_auf or
            # correct_astrometry is True.
            if (self.crossmatch_params_dict['include_perturb_auf'] and
                    config['run_psf_auf']) or config['correct_astrometry']:
                for check_flag, f in zip(['dd_params_path', 'l_cut_path'], ['dd_params', 'l_cut']):
                    setattr(self, f'{flag}{f}', np.load(f'{config[check_flag]}/{f}.npy'))

        for config, catname, flag in zip([self.cat_a_params_dict, self.cat_b_params_dict], ['"a"', '"b"'],
                                         ['a_', 'b_']):
            if self.crossmatch_params_dict['include_perturb_auf'] or config['correct_astrometry']:
                for name in ['dens_hist_tri', 'tri_model_mags', 'tri_model_mag_mids',
                             'tri_model_mags_interval', 'tri_dens_uncert', 'tri_n_bright_sources_star']:
                    # If location variable was "None" in the first place we set
                    # {name}_list in config to a list of Nones and it got updated
                    # above already.
                    if config[f'{name}_location'] != "None":
                        setattr(self, f'{flag}{name}_list', np.load(config[f'{name}_location']))
        for config, catname, flag in zip(
                [self.cat_a_params_dict, self.cat_b_params_dict], ['"a"', '"b"'], ['a_', 'b_']):
            if config['correct_astrometry'] or config['compute_snr_mag_relation']:
                if config['correct_astrometry']:
                    # The reshape puts the first three elements in a[0], and hence
                    # those are ref_cat_inds, with a[1] this_cat_inds.
                    setattr(self, f'{flag}pos_and_err_indices', config['pos_and_err_indices'].reshape(2, 3))
                else:
                    # If we only want to compute the SNR-mag relation, then we've
                    # only got three elements, so we just store them in a (3,)
                    # shape array.
                    setattr(self, f'{flag}pos_and_err_indices', config['pos_and_err_indices'])

    def _make_regions_points(self, region_type, region_points, chunk_id):
        '''
        Wrapper function for the creation of "region" coordinate tuples,
        given either a set of rectangular points or a list of coordinates.

        Parameters
        ----------
        region_type : list of string and string
            String containing the kind of system the region pointings are in.
            Should be "rectangle", regularly sampled points in the two sky
            coordinates, or "points", individually specified sky coordinates,
            and the name into which to save the variable in the class.
        region_points : list of string and list
            String containing the evaluation points. If ``region_type`` is
            "rectangle", should be six values, the start and stop values and
            number of entries of the respective sky coordinates; and if
            ``region_type`` is "points", ``region_points`` should be tuples
            of the form ``(a, b)`` separated by whitespace, as well as the
            class variable name for storage.
        chunk_id : string
            Unique identifier for particular sub-region being loaded, used to
            inform of errors.
        '''
        rt = region_type[1].lower()
        if rt == 'rectangle':
            try:
                a = region_points[1]
                a = [float(point) for point in a]
            except ValueError as exc:
                raise ValueError(f"{region_points[0]} should be 6 numbers separated by spaces in chunk "
                                 f"{chunk_id}.") from exc
            if len(a) == 6:
                if not a[2].is_integer() or not a[5].is_integer():
                    raise ValueError("Number of steps between start and stop values for "
                                     f"{region_points[0]} should be integers in chunk {chunk_id}.")
                ax1_p = np.linspace(a[0], a[1], int(a[2]))
                ax2_p = np.linspace(a[3], a[4], int(a[5]))
                points = np.stack(np.meshgrid(ax1_p, ax2_p), -1).reshape(-1, 2)
            else:
                raise ValueError(f"{region_points[0]} should be 6 numbers separated by spaces in chunk "
                                 f"{chunk_id}.")
        elif rt == 'points':
            try:
                points = np.array(region_points[1], dtype=float)
            except ValueError as exc:
                raise ValueError(f"{region_points[0]} should be a list of two-element lists "
                                 f"'[[a, b], [c, d]]', separated by a comma in chunk {chunk_id}.") from exc

        setattr(self, region_points[0], points)

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
        joint_config = yaml.safe_load(open(self.crossmatch_params_file_path))
        cat_a_config = yaml.safe_load(open(self.cat_a_params_file_path))
        cat_b_config = yaml.safe_load(open(self.cat_b_params_file_path))

        for check_flag in ['include_perturb_auf', 'include_phot_like', 'use_phot_priors',
                           'cf_region_type', 'cf_region_frame', 'cf_region_points_per_chunk',
                           'joint_folder_path', 'pos_corr_dist', 'real_hankel_points', 'chunk_id_list',
                           'four_hankel_points', 'four_max_rho', 'int_fracs', 'make_output_csv', 'n_pool']:
            if check_flag not in joint_config:
                raise ValueError(f"Missing key {check_flag} from joint metadata file.")

        for config, catname in zip([cat_a_config, cat_b_config], ['"a"', '"b"']):
            for check_flag in ['auf_region_type', 'auf_region_frame', 'auf_region_points_per_chunk',
                               'filt_names', 'cat_name', 'auf_folder_path', 'cat_folder_path',
                               'correct_astrometry', 'compute_snr_mag_relation', 'chunk_id_list']:
                if check_flag not in config:
                    raise ValueError(f"Missing key {check_flag} from catalogue {catname} metadata file.")

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

        joint_config['joint_folder_path'] = os.path.abspath(joint_config['joint_folder_path'])
        if cat_a_config['auf_folder_path'] == "None":
            cat_a_config['auf_folder_path'] = None
        else:
            cat_a_config['auf_folder_path'] = os.path.abspath(cat_a_config['auf_folder_path'])
        if cat_b_config['auf_folder_path'] == "None":
            cat_b_config['auf_folder_path'] = None
        else:
            cat_b_config['auf_folder_path'] = os.path.abspath(cat_b_config['auf_folder_path'])

        cat_a_config['cat_folder_path'] = os.path.abspath(cat_a_config['cat_folder_path'])
        cat_b_config['cat_folder_path'] = os.path.abspath(cat_b_config['cat_folder_path'])

        # Only have to check for the existence of Pertubation AUF-related
        # parameters if we are using the perturbation AUF component.
        # However, calling AstrometricCorrections in its current form confuses
        # this, since it always uses the perturbation AUF component. We therefore
        # split out the items that are NOT required for AstrometricCorrections
        # first, if there are any.

        for n, config in zip(['a', 'b'], [cat_a_config, cat_b_config]):
            for p in ['correct_astrometry', 'compute_snr_mag_relation']:
                if config[p] not in (True, False):
                    raise ValueError(f"Boolean key {p} not set to allowed value in catalogue {n} "
                                     "metadata file.")

        if cat_a_config['correct_astrometry'] and cat_a_config['compute_snr_mag_relation']:
            raise ValueError("Ambiguity in catalogue 'a' having both correct_astrometry and "
                             "compute_snr_mag_relation both being True. Only set at most one "
                             "flag as 'True'.")
        if cat_b_config['correct_astrometry'] and cat_b_config['compute_snr_mag_relation']:
            raise ValueError("Ambiguity in catalogue 'b' having both correct_astrometry and "
                             "compute_snr_mag_relation both being True. Only set at most one "
                             "flag as 'True'.")

        if (joint_config['include_perturb_auf'] or cat_a_config['correct_astrometry'] or
                cat_b_config['correct_astrometry']):
            for check_flag in ['num_trials', 'd_mag']:
                if check_flag not in joint_config:
                    raise ValueError(f"Missing key {check_flag} from joint metadata file.")

            a = joint_config['num_trials']
            try:
                a = float(a)
            except ValueError as exc:
                raise ValueError("num_trials should be an integer.") from exc
            if not a.is_integer():
                raise ValueError("num_trials should be an integer.")

            for flag in ['d_mag']:
                try:
                    a = float(joint_config[flag])
                except ValueError as exc:
                    raise ValueError(f"{flag} must be a float.") from exc

        # Nominally these are all of the parameters required if include_perturb_auf
        # is True, but for the minute they're also forced if correct_astrometry
        # is also True instead of those aligning, so we bypass the if statement
        # if self.a_correct_astrometry or self.b_correct_astrometry respectively
        # have been set.
        # pylint: disable-next=too-many-nested-blocks
        for correct_astro, compute_snr_mag_relation, config, catname, flag in zip(
                [cat_a_config['correct_astrometry'], cat_b_config['correct_astrometry']],
                [cat_a_config['compute_snr_mag_relation'], cat_b_config['compute_snr_mag_relation']],
                [cat_a_config, cat_b_config], ['"a"', '"b"'], ['a_', 'b_']):
            if joint_config['include_perturb_auf'] or correct_astro:
                for check_flag in ['dens_dist']:
                    if check_flag not in config:
                        raise ValueError(f"Missing key {check_flag} from catalogue {catname} metadata file.")
                try:
                    a = float(config['dens_dist'])
                except ValueError as exc:
                    raise ValueError(f"dens_dist in catalogue {catname} must be a float.") from exc

            if joint_config['include_perturb_auf']:
                if 'fit_gal_flag' not in config:
                    raise ValueError(f"Missing key {check_flag} from catalogue {catname} metadata file.")
                if config['fit_gal_flag'] not in (True, False):
                    raise ValueError("Boolean key fit_gal_flag not set to allowed value in catalogue "
                                     f"{catname} metadata file.")

            # snr_mag_params_file_path is needed in any one of these three cases:
            if joint_config['include_perturb_auf'] or correct_astro or compute_snr_mag_relation:
                for check_flag in ['snr_mag_params_file_path']:
                    if check_flag not in config:
                        raise ValueError(f"Missing key {check_flag} from catalogue {catname} metadata file.")

            if joint_config['include_perturb_auf'] or correct_astro:
                for check_flag in ['tri_set_name', 'tri_filt_names', 'tri_filt_num',
                                   'download_tri', 'psf_fwhms', 'run_fw_auf', 'run_psf_auf',
                                   'tri_maglim_faint', 'tri_num_faint', 'gal_al_avs',
                                   'dens_hist_tri_location', 'tri_model_mags_location',
                                   'tri_model_mag_mids_location', 'tri_model_mags_interval_location',
                                   'tri_n_bright_sources_star_location']:
                    if check_flag not in config:
                        raise ValueError(f"Missing key {check_flag} from catalogue {catname} metadata file.")

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
                if len(b) != len(config['tri_filt_names']):
                    raise ValueError(f'{flag}psf_fwhms and {flag}filt_names should contain the '
                                     'same number of entries.')

                for auf_run_flag in ['run_fw_auf', 'run_psf_auf']:
                    if config[auf_run_flag] not in (True, False):
                        raise ValueError(f"Boolean key {auf_run_flag} not set to allowed value in catalogue "
                                         f"{catname} metadata file.")

                # Only need dd_params or l_cut if we're using run_psf_auf or
                # correct_astrometry is True.
                if config['run_psf_auf'] or correct_astro:
                    for check_flag, f in zip(['dd_params_path', 'l_cut_path'],
                                             ['dd_params', 'l_cut']):
                        if check_flag not in config:
                            raise ValueError(f"Missing key {check_flag} from catalogue {catname} "
                                             "metadata file.")

                    for check_flag, f in zip(['dd_params_path', 'l_cut_path'],
                                             ['dd_params', 'l_cut']):
                        if not os.path.exists(config[check_flag]):
                            raise OSError(f'{flag}{check_flag} does not exist. Please ensure that path for '
                                          f'catalogue {catname} is correct.')

                        if not os.path.isfile(f'{config[check_flag]}/{f}.npy'):
                            raise FileNotFoundError(f'{f} file not found in catalogue {catname} path. '
                                                    'Please ensure PSF photometry perturbations '
                                                    'are pre-generated.')

                        if 'dd_params' in check_flag:
                            dpp = config['dd_params_path']
                            a = np.load(f'{dpp}/dd_params.npy')
                            if not (len(a.shape) == 3 and a.shape[0] == 5 and a.shape[2] == 2):
                                raise ValueError(f'{flag}dd_params should be of shape (5, X, 2).')
                        if 'l_cut' in check_flag:
                            lcp = config['l_cut_path']
                            a = np.load(f'{lcp}/l_cut.npy')
                            if not (len(a.shape) == 1 and a.shape[0] == 3):
                                raise ValueError(f'{flag}l_cut should be of shape (3,) only.')

                try:
                    a = config['tri_filt_num']
                    if a == "None":
                        config['tri_filt_num'] = None
                    else:
                        if not float(a).is_integer():
                            raise ValueError("tri_filt_num should be a single integer number in "
                                             f"catalogue {catname} metadata file, or None.")
                except ValueError as exc:
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
                    except ValueError as exc:
                        raise ValueError(f"tri_num{suffix} should be a single integer number in "
                                         f"catalogue {catname} metadata file, or None.") from exc

                    try:
                        if config[f'tri_maglim{suffix}'] == "None":
                            config[f'tri_maglim{suffix}'] = None
                        else:
                            a = float(config[f'tri_maglim{suffix}'])
                    except ValueError as exc:
                        raise ValueError(f"tri_maglim{suffix} in catalogue {catname} must be a "
                                         "float, or None.") from exc

                # Assume that we input filenames, including full location, for each
                # pre-computed TRILEGAL histogram file, and that they are all shape
                # (len(filters), ...).
                for name in ['dens_hist_tri', 'tri_model_mags', 'tri_model_mag_mids',
                             'tri_model_mags_interval', 'tri_dens_uncert', 'tri_n_bright_sources_star']:
                    f = config[f'{name}_location']
                    if f == "None":
                        config[f'{name}_list'] = [None] * len(config['filt_names'])
                    else:
                        if not os.path.isfile(f):
                            raise FileNotFoundError(f"File not found for {name}. Please verify "
                                                    "the input location on disk.")
                        try:
                            g = np.load(f)
                        except Exception as exc:
                            raise ValueError(f"File could not be loaded from {name}.") from exc
                        if name == "dens_hist_tri":
                            shape_dht = g.shape
                            if g.shape[0] != len(getattr(self, f'{flag}filt_names')):
                                raise ValueError(f"The number of filters in {flag}filt_names and "
                                                 f"{flag}dens_hist_tri do not match.")
                        else:
                            if g.shape[0] != shape_dht[0]:
                                raise ValueError("The number of filter-elements in dens_hist_tri "
                                                 f"and {name} do not match.")
                            if name != "tri_n_bright_sources_star":
                                if len(g.shape) < 2 or len(shape_dht) < 2 or g.shape[1] != shape_dht[1]:
                                    raise ValueError("The number of magnitude-elements in "
                                                     f"dens_hist_tri and {name} do not match.")
                # Check for inter- and intra-TRILEGAL parameter compatibility.
                # If any one parameter from option A or B is None, they all
                # should be; and if any (all) options from A are None, zero
                # options from B should be None, and vice versa.
                run_internal_none_flag = [config[name] is None for name in
                                          ['auf_folder_path', 'tri_set_name', 'tri_maglim_faint',
                                          'tri_num_faint', 'download_tri', 'tri_filt_num']]
                run_internal_none_flag.append(np.all([b is None for b in config['tri_filt_names']]))
                if not (np.sum(run_internal_none_flag) == 0 or
                        np.sum(run_internal_none_flag) == len(run_internal_none_flag)):
                    raise ValueError("Either all flags related to running TRILEGAL histogram generation "
                                     f"within the catalogue {catname} cross-match call -- tri_filt_names, "
                                     "tri_set_name, etc. -- should be None or zero of them should be None.")
                run_external_none_flag = [np.all([b is None for b in config[name]]) for
                                          name in ['dens_hist_tri_list', 'tri_model_mags_list',
                                                   'tri_model_mag_mids_list', 'tri_model_mags_interval_list',
                                                   'tri_n_bright_sources_star_list']]
                if not (np.sum(run_external_none_flag) == 0 or
                        np.sum(run_external_none_flag) == len(run_external_none_flag)):
                    raise ValueError("Either all flags related to running TRILEGAL histogram generation "
                                     f"externally to the catalogue {catname} cross-match call -- "
                                     "dens_hist_tri, tri_model_mags, etc. -- should be None or zero of "
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
                        if len(b) != len(getattr(self, f'{flag}filt_names')):
                            raise ValueError(f'{flag}{var} and {flag}filt_names should contain the same '
                                             'number of entries.')
                    # galaxy_nzs should be a list of integers.
                    a = config['gal_nzs'].split(' ')
                    try:
                        b = np.array([float(f) for f in a])
                    except ValueError as exc:
                        raise ValueError('gal_nzs should be a list of integers '
                                         f'in catalogue {catname} metadata file') from exc
                    if len(b) != len(getattr(self, f'{flag}filt_names')):
                        raise ValueError(f'{flag}gal_nzs and {flag}filt_names should contain the same '
                                         'number of entries.')
                    if not np.all([c.is_integer() for c in b]):
                        raise ValueError(f'All elements of {flag}gal_nzs should be integers.')
                    # Filter names are simple lists of strings
                    b = config['gal_filternames'].split(' ')
                    if len(b) != len(config['filt_names']):
                        raise ValueError(f'{flag}gal_filternames and {flag}filt_names should contain the '
                                         'same number of entries.')

        try:
            a = float(joint_config['pos_corr_dist'])
        except ValueError as exc:
            raise ValueError("pos_corr_dist must be a float.") from exc

        for flag in ['real_hankel_points', 'four_hankel_points', 'four_max_rho']:
            a = joint_config[flag]
            try:
                a = float(a)
            except ValueError as exc:
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

        if joint_config["make_output_csv"] not in (True, False):
            raise ValueError('Boolean flag key make_output_csv not set to allowed value '
                             'in joint metadata file.')
        if joint_config["make_output_csv"]:
            joint_config, cat_a_config, cat_b_config = self._read_metadata_csv(
                joint_config, cat_a_config, cat_b_config)

        # Load the multiprocessing Pool count.
        try:
            a = joint_config['n_pool']
            if float(a).is_integer():
                a = int(a)
            else:
                raise ValueError("n_pool should be a single integer number.")
        except ValueError as exc:
            raise ValueError("n_pool should be a single integer number.") from exc

        for correct_astro, compute_snr_mag_relation, config, catname, flag in zip(
                [cat_a_config['correct_astrometry'], cat_b_config['correct_astrometry']],
                [cat_a_config['compute_snr_mag_relation'], cat_b_config['compute_snr_mag_relation']],
                [cat_a_config, cat_b_config], ['"a"', '"b"'], ['a_', 'b_']):
            # Have to split these parameters into two, as four of them are
            # required for the simpler case of just doing SNR-mag relation
            # calculations, instead of the full astrometry correction.
            if correct_astro or compute_snr_mag_relation:
                for check_flag in ['correct_astro_save_folder', 'csv_cat_file_string',
                                   'pos_and_err_indices', 'mag_indices', 'mag_unc_indices']:
                    if check_flag not in config:
                        raise ValueError(f"Missing key {check_flag} from catalogue {catname} metadata file.")

                config['correct_astro_save_folder'] = os.path.abspath(config['correct_astro_save_folder'])

                config['csv_cat_file_string'] = os.path.abspath(config['csv_cat_file_string'])

                a = config['mag_indices'].split(' ')
                try:
                    b = np.array([float(f) for f in a])
                except ValueError as exc:
                    raise ValueError('mag_indices should be a list of integers '
                                     f'in the catalogue {catname} metadata file') from exc
                if len(b) != len(getattr(self, f'{flag}filt_names')):
                    raise ValueError(f'{flag}filt_names and {flag}mag_indices should contain the '
                                     'same number of entries.')
                if not np.all([c.is_integer() for c in b]):
                    raise ValueError(f'All elements of {flag}mag_indices should be '
                                     'integers.')

                a = config['mag_unc_indices'].split(' ')
                try:
                    b = np.array([float(f) for f in a])
                except ValueError as exc:
                    raise ValueError('mag_unc_indices should be a list of integers '
                                     f'in the catalogue {catname} metadata file') from exc
                if len(b) != len(getattr(self, f'{flag}mag_indices')):
                    raise ValueError(f'{flag}mag_unc_indices and {flag}mag_indices should contain the '
                                     'same number of entries.')
                if not np.all([c.is_integer() for c in b]):
                    raise ValueError(f'All elements of {flag}mag_unc_indices should be integers.')

                # pos_and_err_indices should be a three- or six-integer list that
                # we then transform into [reference_cat_inds, this_cat_inds]
                # where each *_cat_inds is a three-element list [x, y, z],
                # or just this_cat_inds in the case of
                # compute_snr_mag_relation=True.
                a = config['pos_and_err_indices'].split(' ')
                try:
                    b = np.array([float(f) for f in a])
                except ValueError as exc:
                    raise ValueError('pos_and_err_indices should be a list of integers '
                                     f'in the catalogue {catname} metadata file') from exc
                if len(b) != 6 and correct_astro:
                    raise ValueError(f'{flag}pos_and_err_indices should contain six elements '
                                     'when correct_astrometry is True.')
                if len(b) != 3 and compute_snr_mag_relation:
                    raise ValueError(f'{flag}pos_and_err_indices should contain three elements '
                                     'when compute_snr_mag_relation is True.')
                if not np.all([c.is_integer() for c in b]):
                    raise ValueError(f'All elements of {flag}pos_and_err_indices should be integers.')

            if correct_astro:
                # If this particular catalogue requires a systematic correction
                # for astrometric biases from ensemble match distributions before
                # we can do a probability-based cross-match, then load some extra
                # pieces of information, over and above those already loaded
                # for just the SNR-mag case.
                for check_flag in ['best_mag_index', 'nn_radius', 'ref_csv_cat_file_string',
                                   'correct_mag_array', 'correct_mag_slice', 'correct_sig_slice',
                                   'chunk_overlap_col', 'best_mag_index_col',
                                   'use_photometric_uncertainties', 'mn_fit_type', 'seeing_ranges']:
                    if check_flag not in config:
                        raise ValueError(f"Missing key {check_flag} from catalogue {catname} metadata file.")

                if config['use_photometric_uncertainties'] not in (True, False):
                    raise ValueError('Boolean flag key use_photometric_uncertainties not set to allowed '
                                     f'value in catalogue {catname} metadata file.')

                mn_fit_type = config['mn_fit_type']
                if mn_fit_type not in ['quadratic', 'linear']:
                    raise ValueError(f"mn_fit_type must be 'quadratic' or 'linear' in catalogue {catname} "
                                     "metadata file.")

                # Since make_plots is always True, we always need seeing_ranges.
                a = config['seeing_ranges'].split(' ')
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

                a = config['best_mag_index']
                try:
                    a = float(a)
                except ValueError as exc:
                    raise ValueError(f"best_mag_index should be an integer in the catalogue {catname} "
                                     "metadata file.") from exc
                if not a.is_integer():
                    raise ValueError(f"best_mag_index should be an integer in the catalogue {catname} "
                                     "metadata file.")
                if int(a) >= len(getattr(self, f'{flag}filt_names')):
                    raise ValueError("best_mag_index cannot be a larger index than the list of "
                                     f"filters in the catalogue {catname} metadata file.")

                a = config['best_mag_index_col']
                try:
                    a = float(a)
                except ValueError as exc:
                    raise ValueError(f"best_mag_index_col should be an integer in the catalogue {catname} "
                                     "metadata file.") from exc
                if not a.is_integer():
                    raise ValueError(f"best_mag_index_col should be an integer in the catalogue {catname} "
                                     "metadata file.")

                a = config['chunk_overlap_col']
                if a == "None":
                    config['chunk_overlap_col'] = None
                else:
                    try:
                        a = float(a)
                    except ValueError as exc:
                        raise ValueError("chunk_overlap_col should be an integer in the "
                                         f"catalogue {catname} metadata file.") from exc
                    if not a.is_integer():
                        raise ValueError("chunk_overlap_col should be an integer in the "
                                         f"catalogue {catname} metadata file.")

                try:
                    a = float(config['nn_radius'])
                except ValueError as exc:
                    raise ValueError(f"nn_radius must be a float in the catalogue {catname} metadata "
                                     "file.") from exc

                config['ref_csv_cat_file_string'] = os.path.abspath(config['ref_csv_cat_file_string'])

                a = config['correct_mag_array'].split()
                try:
                    b = np.array([float(f) for f in a])
                except ValueError as exc:
                    raise ValueError('correct_mag_array should be a list of floats in the '
                                     f'catalogue {catname} metadata file.') from exc

                a = config['correct_mag_slice'].split()
                try:
                    b = np.array([float(f) for f in a])
                except ValueError as exc:
                    raise ValueError('correct_mag_slice should be a list of floats in the '
                                     f'catalogue {catname} metadata file.') from exc
                if len(b) != len(getattr(self, f'{flag}correct_mag_array')):
                    raise ValueError(f'{flag}correct_mag_array and {flag}correct_mag_slice should contain '
                                     'the same number of entries.')

                a = config['correct_sig_slice'].split()
                try:
                    b = np.array([float(f) for f in a])
                except ValueError as exc:
                    raise ValueError('correct_sig_slice should be a list of floats in the '
                                     f'catalogue {catname} metadata file.') from exc
                if len(b) != len(getattr(self, f'{flag}correct_mag_array')):
                    raise ValueError(f'{flag}correct_mag_array and {flag}correct_sig_slice should contain '
                                     'the same number of entries.')

        return joint_config, cat_a_config, cat_b_config

    def _read_metadata_csv(self, joint_config, cat_a_config, cat_b_config):
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

        for check_flag in ['output_csv_folder', 'match_out_csv_name', 'nonmatch_out_csv_name']:
            if check_flag not in joint_config:
                raise ValueError(f"Missing key {check_flag} from joint metadata file.")

        for config, catname in zip([cat_a_config, cat_b_config], ['"a"', '"b"']):
            for check_flag in ['input_csv_folder', 'cat_csv_name', 'cat_col_names', 'cat_col_nums',
                               'input_npy_folder', 'csv_has_header', 'extra_col_names',
                               'extra_col_nums']:
                if check_flag not in config:
                    raise ValueError(f"Missing key {check_flag} from catalogue {catname} metadata file.")

        joint_config['output_csv_folder'] = os.path.abspath(joint_config['output_csv_folder'])
        try:
            os.makedirs(joint_config['output_csv_folder'], exist_ok=True)
        except OSError as exc:
            raise OSError("Error when trying to create folder to store output csv files in. "
                          "Please ensure that output_csv_folder is correct in joint config file.") from exc

        for config, catname in zip([cat_a_config, cat_b_config], ['a_', 'b_']):
            # Non-match csv name should be of the format
            # [cat name]_[some indication this is a non-match], but note that
            # this is defined in joint_config, not each individual
            # catalogue config!
            nonmatch_out_name = joint_config['nonmatch_out_csv_name']
            joint_config['nonmatch_out_csv_name'] = f'{config["cat_name"]}_{nonmatch_out_name}'

            input_csv_folder = os.path.abspath(config['input_csv_folder'])
            if not os.path.exists(input_csv_folder):
                raise OSError(f'input_csv_folder from catalogue "{catname[0]}" does not exist.')

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
            if len(b) != len(getattr(self, f'{catname}cat_col_names')):
                raise ValueError(f'{catname}cat_col_names and {catname}cat_col_nums should contain the same '
                                 'number of entries.')
            if not np.all([c.is_integer() for c in b]):
                raise ValueError(f'All elements of {catname}cat_col_nums should be '
                                 'integers.')

            input_npy_folder = config['input_npy_folder']
            if input_npy_folder != 'None' and not os.path.exists(input_npy_folder):
                raise OSError(f'input_npy_folder from catalogue "{catname[0]}" does not exist.')
            if input_npy_folder == 'None':
                config['input_npy_folder'] = None
            else:
                config['input_npy_folder'] = os.path.abspath(input_npy_folder)

            if config['csv_has_header'] not in (True, False):
                raise ValueError('Boolean flag key csv_has_header not set to allowed value in catalogue '
                                 f'{catname[0]} metadata file.')

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
                if len(b) != len(getattr(self, f'{catname}extra_col_names')):
                    raise ValueError(f'{catname}extra_col_names and {catname}extra_col_nums should '
                                     'contain the same number of entries.')
                if not np.all([c.is_integer() for c in b]):
                    raise ValueError(f'All elements of {catname}extra_col_nums should be integers.')

        return joint_config, cat_a_config, cat_b_config

    def make_shared_data(self):
        """
        Function to initialise the shared variables used in the cross-match process.
        """

        self.r = np.linspace(0, self.pos_corr_dist, self.real_hankel_points)
        self.dr = np.diff(self.r)
        self.rho = np.linspace(0, self.four_max_rho, self.four_hankel_points)
        self.drho = np.diff(self.rho)
        # Only need to calculate these the first time we need them, so buffer for now.
        self.j0s = None
        self.j1s = None
