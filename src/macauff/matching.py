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
from time import sleep

import numpy as np
import yaml
from astropy.time import Time

try:
    from mpi4py import MPI
except ModuleNotFoundError:
    MPI = None

from macauff.counterpart_pairing import source_pairing
from macauff.fit_astrometry import AstrometricCorrections
from macauff.group_sources import make_island_groupings
from macauff.macauff import Macauff
from macauff.parse_catalogue import csv_to_npy, npy_to_csv
from macauff.perturbation_auf import make_perturb_aufs
from macauff.photometric_likelihood import compute_photometric_likelihoods

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
    crossmatch_params_file_path : string
        A path to the location of the joint-match parameter file.
    cat_a_params_file_path : string
        A path to the location of the file containing the left-hand catalogue
        parameters.
    cat_b_params_file_path : string
        A path to the location of the input parameter file containing the
        relevant information for the right-hand catalogue "b".
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
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.comm_size = self.comm.Get_size()
            # Set MPI error handling to return exceptions rather than MPI_Abort the
            # application. Allows for recovery of crashed workers.
            self.comm.Set_errhandler(MPI.ERRORS_RETURN)
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

    def _initialise_chunk(self):  # pylint: disable=too-many-branches,too-many-statements
        '''
        Initialisation function for a single chunk of sky.

        The function takes three paths, the locations of the metadata files containing
        all of the necessary parameters for the cross-match, and outputs a file
        containing the appropriate columns of the datasets plus additional derived
        parameters.
        '''

        # If astrometry of either catalogue needs fixing, do that now.
        if self.a_correct_astrometry:
            # Generate from current data: just need the singular chunk mid-points
            # and to leave all other parameters as they are.
            if len(self.a_auf_region_points) > 1:
                warnings.warn("a_auf_region_points contains more than one AUF sampling point, but "
                              "a_correct_astrometry is True. Check results carefully.")
            ax1_mids = np.array([self.a_auf_region_points[0, 0]])
            ax2_mids = np.array([self.a_auf_region_points[0, 1]])
            ax_dimension = 2
            a_npy_or_csv = 'csv'
            a_coord_or_chunk = 'chunk'
        if self.a_correct_astrometry:
            t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{t} Rank {self.rank}, chunk {self.chunk_id}: Calculating catalogue 'a' "
                  "uncertainty corrections...")
            ac = AstrometricCorrections(
                self.a_psf_fwhms, self.num_trials, self.a_nn_radius, self.a_dens_dist,
                self.a_correct_astro_save_folder, self.a_gal_wavs, self.a_gal_aboffsets,
                self.a_gal_filternames, self.a_gal_al_avs, self.d_mag, self.a_dd_params,
                self.a_l_cut, ax1_mids, ax2_mids, ax_dimension, self.a_correct_mag_array,
                self.a_correct_mag_slice, self.a_correct_sig_slice, self.n_pool, a_npy_or_csv,
                a_coord_or_chunk, self.a_pos_and_err_indices, self.a_mag_indices, self.a_snr_indices,
                self.a_filt_names, self.a_correct_astro_mag_indices_index, self.a_auf_region_frame,
                self.a_saturation_magnitudes, trifilepath=self.a_auf_file_path,
                maglim_f=self.a_tri_maglim_faint, magnum=self.a_tri_filt_num,
                tri_num_faint=self.a_tri_num_faint, trifilterset=self.a_tri_set_name,
                trifiltnames=self.a_tri_filt_names, tri_hists=self.a_dens_hist_tri_list,
                tri_magses=self.a_tri_model_mags_list, dtri_magses=self.a_tri_model_mags_interval_list,
                tri_uncerts=self.a_tri_dens_uncert_list,
                use_photometric_uncertainties=self.a_use_photometric_uncertainties, pregenerate_cutouts=True,
                chunks=[self.chunk_id], n_r=self.real_hankel_points, n_rho=self.four_hankel_points,
                max_rho=self.four_max_rho, mn_fit_type=self.a_mn_fit_type)
            ac(a_cat_name=self.a_ref_cat_csv_file_path, b_cat_name=self.a_cat_csv_file_path,
               tri_download=self.a_download_tri, make_plots=True, overwrite_all_sightlines=True,
               seeing_ranges=self.a_seeing_ranges)

            if not os.path.isfile(self.a_cat_csv_file_path):
                raise FileNotFoundError('Catalogue file not found in catalogue "a" path. '
                                        'Please ensure photometric catalogue is correctly saved.')

            if self.include_perturb_auf or self.a_correct_astrometry:
                snr_cols = self.a_snr_indices
            else:
                snr_cols = None
            if self.a_apply_proper_motion:
                if isinstance(self.a_ref_epoch_or_index, str):
                    pm_cols = self.a_pm_indices
                    pm_ref_epoch = self.a_ref_epoch_or_index
                else:
                    pm_cols = np.append(self.a_pm_indices, self.a_ref_epoch_or_index)
                    pm_ref_epoch = None
                pm_move_to_epoch = self.move_to_epoch
            else:
                pm_cols, pm_ref_epoch, pm_move_to_epoch = None, None, None
            # Having corrected the astrometry, we have to call csv_to_npy
            # now, rather than pre-generating our binary input catalogues.
            x = csv_to_npy(
                self.a_cat_csv_file_path, self.a_pos_and_err_indices[0], self.a_mag_indices,
                self.a_best_mag_index_col, self.a_chunk_overlap_col, snr_cols=snr_cols, header=False,
                process_uncerts=True, astro_sig_fits_filepath=f'{self.a_correct_astro_save_folder}/npy',
                cat_in_radec=self.a_auf_region_frame == 'equatorial',
                mn_in_radec=self.a_auf_region_frame == 'equatorial', pm_cols=pm_cols,
                pm_ref_epoch=pm_ref_epoch, pm_move_to_epoch=pm_move_to_epoch)
            if snr_cols is not None:
                # pylint: disable-next=unbalanced-tuple-unpacking
                self.a_astro, self.a_photo, self.a_magref, self.a_in_overlaps, self.a_snr = x
            else:
                # pylint: disable-next=unbalanced-tuple-unpacking
                self.a_astro, self.a_photo, self.a_magref, self.a_in_overlaps = x
        else:
            if not os.path.isfile(self.a_cat_csv_file_path):
                raise FileNotFoundError('Catalogue file not found in catalogue "a" path. '
                                        'Please ensure photometric catalogue is correctly saved.')

            if self.include_perturb_auf or self.a_correct_astrometry:
                snr_cols = self.a_snr_indices
            else:
                snr_cols = None
            if self.a_apply_proper_motion:
                if isinstance(self.a_ref_epoch_or_index, str):
                    pm_cols = self.a_pm_indices
                    pm_ref_epoch = self.a_ref_epoch_or_index
                else:
                    pm_cols = np.append(self.a_pm_indices, self.a_ref_epoch_or_index)
                    pm_ref_epoch = None
                pm_move_to_epoch = self.move_to_epoch
            else:
                pm_cols, pm_ref_epoch, pm_move_to_epoch = None, None, None
            # Otherwise, just load the files without correcting anything.
            x = csv_to_npy(
                self.a_cat_csv_file_path, self.a_pos_and_err_indices, self.a_mag_indices,
                self.a_best_mag_index_col, self.a_chunk_overlap_col, snr_cols=snr_cols, header=False,
                pm_cols=pm_cols, pm_ref_epoch=pm_ref_epoch, pm_move_to_epoch=pm_move_to_epoch)
            if snr_cols is not None:
                # pylint: disable-next=unbalanced-tuple-unpacking
                self.a_astro, self.a_photo, self.a_magref, self.a_in_overlaps, self.a_snr = x
            else:
                # pylint: disable-next=unbalanced-tuple-unpacking
                self.a_astro, self.a_photo, self.a_magref, self.a_in_overlaps = x

        if self.b_correct_astrometry:
            # Generate from current data: just need the singular chunk mid-points
            # and to leave all other parameters as they are.
            if len(self.b_auf_region_points) > 1:
                warnings.warn("b_auf_region_points contains more than one AUF sampling point, but "
                              "b_correct_astrometry is True. Check results carefully.")
            ax1_mids = np.array([self.b_auf_region_points[0, 0]])
            ax2_mids = np.array([self.b_auf_region_points[0, 1]])
            ax_dimension = 2
            b_npy_or_csv = 'csv'
            b_coord_or_chunk = 'chunk'
        if self.b_correct_astrometry:
            t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{t} Rank {self.rank}, chunk {self.chunk_id}: Calculating catalogue 'b' "
                  "uncertainty corrections...")
            ac = AstrometricCorrections(
                self.b_psf_fwhms, self.num_trials, self.b_nn_radius, self.b_dens_dist,
                self.b_correct_astro_save_folder, self.b_gal_wavs, self.b_gal_aboffsets,
                self.b_gal_filternames, self.b_gal_al_avs, self.d_mag, self.b_dd_params,
                self.b_l_cut, ax1_mids, ax2_mids, ax_dimension, self.b_correct_mag_array,
                self.b_correct_mag_slice, self.b_correct_sig_slice, self.n_pool, b_npy_or_csv,
                b_coord_or_chunk, self.b_pos_and_err_indices, self.b_mag_indices, self.b_snr_indices,
                self.b_filt_names, self.b_correct_astro_mag_indices_index, self.b_auf_region_frame,
                self.b_saturation_magnitudes, trifilepath=self.b_auf_file_path,
                maglim_f=self.b_tri_maglim_faint, magnum=self.b_tri_filt_num,
                tri_num_faint=self.b_tri_num_faint, trifilterset=self.b_tri_set_name,
                trifiltnames=self.b_tri_filt_names, tri_hists=self.b_dens_hist_tri_list,
                tri_magses=self.b_tri_model_mags_list, dtri_magses=self.b_tri_model_mags_interval_list,
                tri_uncerts=self.b_tri_dens_uncert_list,
                use_photometric_uncertainties=self.b_use_photometric_uncertainties,
                pregenerate_cutouts=True, chunks=[self.chunk_id],
                n_r=self.real_hankel_points, n_rho=self.four_hankel_points, max_rho=self.four_max_rho,
                mn_fit_type=self.b_mn_fit_type)
            ac(a_cat_name=self.b_ref_cat_csv_file_path, b_cat_name=self.b_cat_csv_file_path,
               tri_download=self.b_download_tri, make_plots=True, overwrite_all_sightlines=True,
               seeing_ranges=self.b_seeing_ranges)

            if not os.path.isfile(self.b_cat_csv_file_path):
                raise FileNotFoundError('Catalogue file not found in catalogue "b" path. '
                                        'Please ensure photometric catalogue is correctly saved.')

            if self.include_perturb_auf or self.a_correct_astrometry:
                snr_cols = self.b_snr_indices
            else:
                snr_cols = None
            if self.b_apply_proper_motion:
                if isinstance(self.b_ref_epoch_or_index, str):
                    pm_cols = self.b_pm_indices
                    pm_ref_epoch = self.b_ref_epoch_or_index
                else:
                    pm_cols = np.append(self.b_pm_indices, self.b_ref_epoch_or_index)
                    pm_ref_epoch = None
                pm_move_to_epoch = self.move_to_epoch
            else:
                pm_cols, pm_ref_epoch, pm_move_to_epoch = None, None, None
            x = csv_to_npy(
                self.b_cat_csv_file_path, self.b_pos_and_err_indices[0], self.b_mag_indices,
                self.b_best_mag_index_col, self.b_chunk_overlap_col, snr_cols=snr_cols, header=False,
                process_uncerts=True, astro_sig_fits_filepath=f'{self.b_correct_astro_save_folder}/npy',
                cat_in_radec=self.b_auf_region_frame == 'equatorial',
                mn_in_radec=self.b_auf_region_frame == 'equatorial', pm_cols=pm_cols,
                pm_ref_epoch=pm_ref_epoch, pm_move_to_epoch=pm_move_to_epoch)
            if snr_cols is not None:
                # pylint: disable-next=unbalanced-tuple-unpacking
                self.b_astro, self.b_photo, self.b_magref, self.b_in_overlaps, self.b_snr = x
            else:
                # pylint: disable-next=unbalanced-tuple-unpacking
                self.b_astro, self.b_photo, self.b_magref, self.b_in_overlaps = x
        else:
            if not os.path.isfile(self.b_cat_csv_file_path):
                raise FileNotFoundError('Catalogue file not found in catalogue "b" path. '
                                        'Please ensure photometric catalogue is correctly saved.')

            if self.include_perturb_auf or self.b_correct_astrometry:
                snr_cols = self.b_snr_indices
            else:
                snr_cols = None
            if self.b_apply_proper_motion:
                if isinstance(self.b_ref_epoch_or_index, str):
                    pm_cols = self.b_pm_indices
                    pm_ref_epoch = self.b_ref_epoch_or_index
                else:
                    pm_cols = np.append(self.b_pm_indices, self.b_ref_epoch_or_index)
                    pm_ref_epoch = None
                pm_move_to_epoch = self.move_to_epoch
            else:
                pm_cols, pm_ref_epoch, pm_move_to_epoch = None, None, None
            x = csv_to_npy(
                self.b_cat_csv_file_path, self.b_pos_and_err_indices, self.b_mag_indices,
                self.b_best_mag_index_col, self.b_chunk_overlap_col, snr_cols=snr_cols, header=False,
                pm_cols=pm_cols, pm_ref_epoch=pm_ref_epoch, pm_move_to_epoch=pm_move_to_epoch)
            if snr_cols is not None:
                # pylint: disable-next=unbalanced-tuple-unpacking
                self.b_astro, self.b_photo, self.b_magref, self.b_in_overlaps, self.b_snr = x
            else:
                # pylint: disable-next=unbalanced-tuple-unpacking
                self.b_astro, self.b_photo, self.b_magref, self.b_in_overlaps = x

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
                        new_chunk, self.chunk_queue = self.chunk_queue[0], self.chunk_queue[1:]
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

    def _process_chunk(self):
        '''
        Runs the various stages of cross-matching two photometric catalogues
        '''
        # pylint: disable-next=fixme
        # TODO: more correctly pass these into CrossMatch as arguments later on.
        self.perturb_auf_func = make_perturb_aufs
        self.group_func = make_island_groupings
        self.phot_like_func = compute_photometric_likelihoods
        self.count_pair_func = source_pairing

        self._initialise_chunk()

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

        a_in_overlaps = self.a_in_overlaps
        b_in_overlaps = self.b_in_overlaps

        if self.include_phot_like and self.with_and_without_photometry:
            loop_array_extensions = ['', '_without_photometry']
        else:
            loop_array_extensions = ['']

        for lae in loop_array_extensions:
            if self.make_output_csv:
                npy_to_csv(
                    [self.a_cat_csv_file_path, self.b_cat_csv_file_path], self, self.output_save_folder,
                    [self.match_out_csv_name, self.a_nonmatch_out_csv_name, self.b_nonmatch_out_csv_name],
                    [self.a_cat_col_names, self.b_cat_col_names], [self.a_cat_col_nums, self.b_cat_col_nums],
                    [self.a_cat_name, self.b_cat_name],
                    [self.a_correct_astrometry, self.b_correct_astrometry],
                    headers=[self.a_csv_has_header, self.b_csv_has_header],
                    extra_col_name_lists=[self.a_extra_col_names, self.b_extra_col_names],
                    extra_col_num_lists=[self.a_extra_col_nums, self.b_extra_col_nums], file_extension=lae)
            else:
                # Only need to save the outputs to binary files if we don't
                # want a set of .csv tables.
                ac, bc = getattr(self, f'ac{lae}'), getattr(self, f'bc{lae}')
                core_matches = ~a_in_overlaps[ac] | ~b_in_overlaps[bc]
                np.save(f'{self.output_save_folder}/ac_{self.chunk_id}{lae}.npy', ac[core_matches])
                np.save(f'{self.output_save_folder}/bc_{self.chunk_id}{lae}.npy', bc[core_matches])
                for fname in ['pc', 'eta', 'xi', 'crptseps', 'acontamflux', 'bcontamflux']:
                    np.save(f'{self.output_save_folder}/{fname}_{self.chunk_id}{lae}.npy',
                            getattr(self, f'{fname}{lae}')[core_matches])
                for fname in ['pacontam', 'pbcontam']:
                    np.save(f'{self.output_save_folder}/{fname}_{self.chunk_id}{lae}.npy',
                            getattr(self, f'{fname}{lae}')[:, core_matches])

                af, bf = getattr(self, f'af{lae}'), getattr(self, f'bf{lae}')
                a_core_nonmatches = ~a_in_overlaps[af]
                b_core_nonmatches = ~b_in_overlaps[bf]
                np.save(f'{self.output_save_folder}/af_{self.chunk_id}{lae}.npy', af[a_core_nonmatches])
                np.save(f'{self.output_save_folder}/bf_{self.chunk_id}{lae}.npy', bf[b_core_nonmatches])
                for fnametype, cnm in zip(['a', 'b'], [a_core_nonmatches, b_core_nonmatches]):
                    for fname_ in ['{}fieldflux', 'pf{}', '{}fieldeta', '{}fieldxi', '{}fieldseps']:
                        fname = fname_.format(fnametype)
                        np.save(f'{self.output_save_folder}/{fname}_{self.chunk_id}{lae}.npy',
                                getattr(self, f'{fname}{lae}')[cnm])

                if self.reject_a is not None:
                    np.save(f'{self.output_save_folder}/reject_a_{self.chunk_id}{lae}.npy',
                            np.append(np.append(self.reject_a, ac[~core_matches]), af[~a_core_nonmatches]))
                else:
                    np.save(f'{self.output_save_folder}/reject_a_{self.chunk_id}{lae}.npy',
                            np.append(ac[~core_matches], af[~a_core_nonmatches]))
                if self.reject_b is not None:
                    np.save(f'{self.output_save_folder}/reject_b_{self.chunk_id}{lae}.npy',
                            np.append(np.append(self.reject_b, bc[~core_matches]), bf[~b_core_nonmatches]))
                else:
                    np.save(f'{self.output_save_folder}/reject_b_{self.chunk_id}{lae}.npy',
                            np.append(bc[~core_matches], bf[~b_core_nonmatches]))

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

        chunk_sizes = np.zeros(len(chunk_queue), dtype=float)
        chunk_id_not_in_completed_chunks = np.empty(len(chunk_queue), dtype=bool)
        for i, chunk_id in enumerate(chunk_queue):
            # Skip completed chunks
            chunk_id_not_in_completed_chunks[i] = chunk_id not in completed_chunks
            cat_a_file_path = self.cat_a_params_dict['cat_csv_file_path'].format(chunk_id)
            cat_b_file_path = self.cat_b_params_dict['cat_csv_file_path'].format(chunk_id)

            for catname, flag, cfp in zip(['"a"', '"b"'], ['a_', 'b_'], [cat_a_file_path, cat_b_file_path]):
                if (not os.path.exists(cfp) or not os.path.isfile(cfp)):
                    raise OSError(f'{flag}cat_csv_file_path does not exist. Please ensure that '
                                  f'path for catalogue {catname} is correct.')

            for file_path in [cat_a_file_path, cat_b_file_path]:
                chunk_sizes[i] += os.path.getsize(file_path)

        # Sort chunk list by size, largest to smallest, removing already
        # completed chunks.
        chunk_queue_sorted = chunk_queue[chunk_id_not_in_completed_chunks][
            np.argsort(chunk_sizes[chunk_id_not_in_completed_chunks])[::-1]]

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

        # Ensure that we can save to the folders for outputs.
        try:
            os.makedirs(self.output_save_folder, exist_ok=True)
        except OSError as exc:
            raise OSError("Error when trying to create folder to store output csv files in. Please "
                          "ensure that output_save_folder is correct in joint config file.") from exc

        for catname, flag in zip(['"a"', '"b"'], ['a_', 'b_']):
            if (not os.path.exists(getattr(self, f'{flag[0]}_cat_csv_file_path')) or
                    not os.path.isfile(getattr(self, f'{flag[0]}_cat_csv_file_path'))):
                raise OSError(f'{flag}cat_csv_file_path does not exist. Please ensure that '
                              f'path for catalogue {catname} is correct.')
            if getattr(self, f'{flag[0]}_auf_file_path') is not None:
                try:
                    os.makedirs(os.path.dirname(getattr(self, f'{flag[0]}_auf_file_path')), exist_ok=True)
                except OSError as exc:
                    raise OSError(f"Error when trying to create temporary folder for catalogue {catname} AUF "
                                  f"outputs. Please ensure that {flag}auf_file_path is correct.") from exc
        # Force auf_file_path to have two ``_{}`` string formats in it, now
        # that we have filled in the original one with the chunk ID; these are
        # for inter-chunk AUF pointings, stored by coordinate in the filename.
        if self.a_auf_file_path is not None:  # pylint: disable=access-member-before-definition
            x, y = os.path.splitext(self.a_auf_file_path)  # pylint: disable=access-member-before-definition
            self.a_auf_file_path = x + r"_{:.2f}_{:.2f}" + y
        if self.b_auf_file_path is not None:  # pylint: disable=access-member-before-definition
            x, y = os.path.splitext(self.b_auf_file_path)  # pylint: disable=access-member-before-definition
            self.b_auf_file_path = x + r"_{:.2f}_{:.2f}" + y

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
            # Only need dd_params or l_cut if we're using run_psf_auf or
            # correct_astrometry is True.
            if (self.crossmatch_params_dict['include_perturb_auf'] and
                    config['run_psf_auf']) or config['correct_astrometry']:
                for check_flag, f in zip(['dd_params_path', 'l_cut_path'], ['dd_params', 'l_cut']):
                    setattr(self, f'{flag}{f}', np.load(f'{config[check_flag]}/{f}.npy'))

        for config, flag in zip([self.cat_a_params_dict, self.cat_b_params_dict], ['a_', 'b_']):
            if self.crossmatch_params_dict['include_perturb_auf'] or config['correct_astrometry']:
                for name in ['dens_hist_tri', 'tri_model_mags', 'tri_model_mag_mids',
                             'tri_model_mags_interval', 'tri_dens_uncert', 'tri_n_bright_sources_star']:
                    # If location variable was "None" in the first place we set
                    # {name}_list in config to a list of Nones and it got updated
                    # above already.
                    if config[f'{name}_location'] != "None":
                        setattr(self, f'{flag}{name}_list', np.load(config[f'{name}_location']))
        for config, flag in zip([self.cat_a_params_dict, self.cat_b_params_dict], ['a_', 'b_']):
            if config['correct_astrometry']:
                if not config['use_photometric_uncertainties']:
                    # The reshape puts the first three elements in a[0], and hence
                    # those are this_cat_inds, with a[1] ref_cat_inds.
                    setattr(self, f'{flag}pos_and_err_indices',
                            np.array(config['pos_and_err_indices']).reshape(2, 3))
                else:
                    # If use_photometric_uncertainties then we need to make a
                    # more generic two-list nested list. This is every index
                    # except the last three in the first list, the final three
                    # indices in a second nested list.
                    setattr(self, f'{flag}pos_and_err_indices',
                            [config['pos_and_err_indices'][:-3], config['pos_and_err_indices'][-3:]])
            else:
                # Otherwise we only need three elements, so we just store them
                # in a (3,) shape array.
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

        setattr(self, region_points[0], points)  # pylint: disable=possibly-used-before-assignment

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

        for config, flag, apply_pm in zip(
                [cat_a_config, cat_b_config], ['a_', 'b_'], [cat_a_config['apply_proper_motion'],
                                                             cat_b_config['apply_proper_motion']]):
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

        if cat_a_config['apply_proper_motion'] or cat_b_config['apply_proper_motion']:
            if 'move_to_epoch' not in joint_config:
                raise ValueError("Missing key move_to_epoch from joint metadata file.")

            a = joint_config['move_to_epoch']
            try:
                Time(a)
            except ValueError as exc:
                raise ValueError("move_to_epoch must be a string that astropy's Time "
                                 "function accepts, such as JYYYY or YYYY-MM-DD.") from exc

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

        # Only have to check for the existence of Pertubation AUF-related
        # parameters if we are using the perturbation AUF component.
        # However, calling AstrometricCorrections in its current form confuses
        # this, since it always uses the perturbation AUF component. We therefore
        # split out the items that are NOT required for AstrometricCorrections
        # first, if there are any.

        for n, config in zip(['a', 'b'], [cat_a_config, cat_b_config]):
            for p in ['correct_astrometry']:
                if config[p] not in (True, False):
                    raise ValueError(f"Boolean key {p} not set to allowed value in catalogue {n} "
                                     "metadata file.")

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
                                   'dens_hist_tri_location', 'tri_model_mags_location',
                                   'tri_model_mag_mids_location', 'tri_model_mags_interval_location',
                                   'tri_n_bright_sources_star_location']:
                    if check_flag not in config:
                        raise ValueError(f"Missing key {check_flag} from catalogue {catname} metadata file.")

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
                            if g.shape[0] != len(config['filt_names']):
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
                                          ['auf_file_path', 'tri_set_name', 'tri_maglim_faint',
                                          'tri_num_faint', 'download_tri', 'tri_filt_num']]
                run_internal_none_flag.append(np.all([b is None for b in config['tri_filt_names']]))
                if not (np.sum(run_internal_none_flag) == 0 or
                        np.sum(run_internal_none_flag) == len(run_internal_none_flag)):
                    raise ValueError("Either all flags related to running TRILEGAL histogram generation "
                                     f"within the catalogue {catname} cross-match call -- tri_filt_names, "
                                     "tri_set_name, etc. -- should be None or zero of them should be None.")
                run_external_none_flag = [config[name] == "None" for name in
                                          ['dens_hist_tri_location', 'tri_model_mags_location',
                                           'tri_model_mag_mids_location', 'tri_model_mags_interval_location',
                                           'tri_n_bright_sources_star_location']]
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
        except (ValueError, TypeError) as exc:
            raise ValueError("n_pool should be a single integer number.") from exc

        for correct_astro, config, catname, flag in zip(
                [cat_a_config['correct_astrometry'], cat_b_config['correct_astrometry']],
                [cat_a_config, cat_b_config], ['"a"', '"b"'], ['a_', 'b_']):
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
