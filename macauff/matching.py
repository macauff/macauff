# Licensed under a 3-clause BSD style license - see LICENSE
'''
This module provides the high-level framework for performing catalogue-catalogue cross-matches.
'''

import os
import sys
import datetime
from configparser import ConfigParser
from time import sleep
import numpy as np

try:
    from mpi4py import MPI
except:
    MPI = None

from .perturbation_auf import make_perturb_aufs
from .group_sources import make_island_groupings
from .group_sources_fortran import group_sources_fortran as gsf
from .misc_functions_fortran import misc_functions_fortran as mff
from .photometric_likelihood import compute_photometric_likelihoods
from .counterpart_pairing import source_pairing
from .parse_catalogue import npy_to_csv, csv_to_npy
from .fit_astrometry import AstrometricCorrections, SNRMagnitudeRelationship

__all__ = ['CrossMatch']


class CrossMatch():
    '''
    A class to cross-match two photometric catalogues with one another, producing
    a composite catalogue of merged sources.

    Parameters
    ----------
    chunks_folder_path : string
        A path to the location of the folder containing one subfolder per chunk.
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

    def __init__(self, chunks_folder_path, resume_file_path=None, use_mpi=True, walltime=None,
                 end_within='00:10:00', polling_rate=1):
        '''
        Initialisation function for cross-match class.
        '''
        self.chunks_folder_path = chunks_folder_path

        # Initialise MPI if available and enabled
        if MPI != None and use_mpi:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.comm_size = self.comm.Get_size()
            # Set MPI error handling to return exceptions rather than MPI_Abort the
            # application. Allows for recovery of crashed workers.
            self.comm.Set_errhandler(MPI.ERRORS_RETURN)
        else:
            if use_mpi:
                print("Warning: MPI initialisation failed. Check mpi4py is correctly installed. Falling back to serial mode.")
            self.rank = 0
            self.comm_size = 1

        # Special signals for MPI processes
        #   'NO_MORE_WORK' - manager uses to signal workers to shut down
        #   'WORK_REQUEST' - manager uses to signal new chunk for processing.
        #                    worker uses to request initial chunk from manager.
        #   'WORK_COMPLETE' - worker uses to report successfully processed given chunk
        #   'WORK_ERROR' - worker uses to report failed processing of given chunk
        self.worker_signals = { 'NO_MORE_WORK': 0,
                                'WORK_REQUEST': 1,
                                'WORK_COMPLETE': 2,
                                'WORK_ERROR': 3 }
        # Only manager process needs to set up the resume file and queue
        if self.rank == 0:
            completed_chunks = set()
            try:
                # Open and read existing resume file
                self.resume_file = open(resume_file_path, 'r+')
                for line in self.resume_file:
                    completed_chunks.add(line.rstrip())
            except FileNotFoundError:
                # Resume file specified but does not exist. Create new one.
                self.resume_file = open(resume_file_path, 'w')
            except TypeError:
                # Resume file was not specified. Disable checkpointing
                self.resume_file = None
            # Chunk queue will not contain chunks recorded as completed in the
            # resume file
            self.chunk_queue = self._make_chunk_queue(completed_chunks)
            # Used to keep track of progress to completion
            self.num_chunks_to_process = len(self.chunk_queue)

            # In seconds, how often the manager checks for new work requests
            self.polling_rate = polling_rate

            if walltime != None:
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


    def _initialise_chunk(self, joint_file_path, cat_a_file_path, cat_b_file_path):
        '''
        Initialisation function for a single chunk of sky.

        The function takes three paths, the locations of the metadata files containing
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
        for f in [joint_file_path, cat_a_file_path, cat_b_file_path]:
            if not os.path.isfile(f):
                raise FileNotFoundError("Input parameter file {} could not be found.".format(f))

        self.joint_file_path = joint_file_path
        self.cat_a_file_path = cat_a_file_path
        self.cat_b_file_path = cat_b_file_path

        self.read_metadata()

        # If astrometry of either catalogue needs fixing, do that now.
        if self.a_correct_astrometry or self.a_compute_snr_mag_relation:
            # Generate from current data: just need the singular chunk mid-points
            # and to leave all other parameters as they are.
            ax1_mids = np.array([0.1 * np.floor(0.5 * (self.cross_match_extent[0] +
                                                       self.cross_match_extent[1]) / 0.1)])
            ax2_mids = np.array([0.1 * np.floor(0.5 * (self.cross_match_extent[2] +
                                                       self.cross_match_extent[3]) / 0.1)])
            ax_dimension = 2
            a_npy_or_csv = 'csv'
            a_coord_or_chunk = 'chunk'
        if self.a_correct_astrometry:
            acbi = self.a_best_mag_index
            a_correct_astro_tri_name = '{}/{}/trilegal_auf_simulation'
            t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print("{} Rank {}, chunk {}: Calculating catalogue 'a' uncertainty corrections..."
                  .format(t, self.rank, self.chunk_id))
            ac = AstrometricCorrections(
                self.a_psf_fwhms[acbi], self.num_trials, self.a_nn_radius,
                self.a_dens_dist, self.a_correct_astro_save_folder, self.a_auf_folder_path,
                a_correct_astro_tri_name, self.a_tri_maglim_faint, self.a_tri_filt_num,
                self.a_tri_num_faint, self.a_tri_set_name, self.a_tri_filt_names[acbi],
                self.a_gal_wavs[acbi], self.a_gal_aboffsets[acbi], self.a_gal_filternames[acbi],
                self.a_gal_al_avs[acbi], self.d_mag, self.a_dd_params, self.a_l_cut, ax1_mids,
                ax2_mids, ax_dimension, self.a_correct_mag_array, self.a_correct_mag_slice,
                self.a_correct_sig_slice, self.n_pool, a_npy_or_csv, a_coord_or_chunk,
                self.a_pos_and_err_indices, self.a_mag_indices, self.a_mag_unc_indices,
                self.a_filt_names, self.a_best_mag_index, self.a_auf_region_frame,
                use_photometric_uncertainties=self.a_use_photometric_uncertainties,
                pregenerate_cutouts=True, chunks=[self.chunk_id])
            ac(self.a_ref_csv_cat_file_string, self.a_csv_cat_file_string,
               tri_download=self.a_download_tri, make_plots=True, overwrite_all_sightlines=True)

            # Having corrected the astrometry, we have to call csv_to_npy
            # now, rather than pre-generating our binary input catalogues.
            csv_folder, csv_filename = os.path.split(
                self.a_csv_cat_file_string.format(self.chunk_id))
            csv_to_npy(csv_folder, csv_filename, self.a_cat_folder_path,
                       self.a_pos_and_err_indices[1], self.a_mag_indices, self.a_best_mag_index_col,
                       self.a_chunk_overlap_col, header=False, process_uncerts=True,
                       astro_sig_fits_filepath='{}/npy'.format(self.a_correct_astro_save_folder),
                       cat_in_radec=self.a_auf_region_frame == 'equatorial',
                       mn_in_radec=self.a_auf_region_frame == 'equatorial')
        if self.a_compute_snr_mag_relation:
            t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print("{} Rank {}, chunk {}: Calculating catalogue 'a' SNR-mag relations..."
                  .format(t, self.rank, self.chunk_id))
            smr = SNRMagnitudeRelationship(
                self.a_correct_astro_save_folder, ax1_mids, ax2_mids, ax_dimension, a_npy_or_csv,
                a_coord_or_chunk, self.a_pos_and_err_indices, self.a_mag_indices,
                self.a_mag_unc_indices, self.a_filt_names, self.a_auf_region_frame,
                chunks=[self.chunk_id])
            smr(self.a_csv_cat_file_string, make_plots=True, overwrite_all_sightlines=True)
        if self.a_correct_astrometry or self.a_compute_snr_mag_relation:
            # If we re-made either side's astrometry then we need to load its
            # SNR-mag relation now.
            os.system('cp {}/npy/snr_mag_params.npy {}'.format(self.a_correct_astro_save_folder,
                      self.a_snr_mag_params_path))
            f = 'snr_mag_params'
            if not os.path.isfile('{}/{}.npy'.format(self.a_snr_mag_params_path, f)):
                raise FileNotFoundError('{} file not found in catalogue {} path. '
                                        'Please ensure {} are pre-generated.'
                                        .format(f, "a", 'astrometry corrections'))
            a = np.load('{}/snr_mag_params.npy'.format(self.a_snr_mag_params_path))
            if not (len(a.shape) == 3 and a.shape[2] == 5 and
                    a.shape[0] == len(self.a_filt_names)):
                raise ValueError('a_snr_mag_params should be of shape (X, Y, 5)')
            self.a_snr_mag_params = a

        if self.b_correct_astrometry or self.b_compute_snr_mag_relation:
            # Generate from current data: just need the singular chunk mid-points
            # and to leave all other parameters as they are.
            ax1_mids = np.array([0.1 * np.floor(0.5 * (self.cross_match_extent[0] +
                                                       self.cross_match_extent[1]) / 0.1)])
            ax2_mids = np.array([0.1 * np.floor(0.5 * (self.cross_match_extent[2] +
                                                       self.cross_match_extent[3]) / 0.1)])
            ax_dimension = 2
            b_npy_or_csv = 'csv'
            b_coord_or_chunk = 'chunk'
        if self.b_correct_astrometry:
            bcbi = self.b_best_mag_index
            b_correct_astro_tri_name = '{}/{}/trilegal_auf_simulation'
            t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print("{} Rank {}, chunk {}: Calculating catalogue 'b' uncertainty corrections..."
                  .format(t, self.rank, self.chunk_id))
            ac = AstrometricCorrections(
                self.b_psf_fwhms[bcbi], self.num_trials, self.b_nn_radius,
                self.b_dens_dist, self.b_correct_astro_save_folder, self.b_auf_folder_path,
                b_correct_astro_tri_name, self.b_tri_maglim_faint, self.b_tri_filt_num,
                self.b_tri_num_faint, self.b_tri_set_name, self.b_tri_filt_names[bcbi],
                self.b_gal_wavs[bcbi], self.b_gal_aboffsets[bcbi], self.b_gal_filternames[bcbi],
                self.b_gal_al_avs[bcbi], self.d_mag, self.b_dd_params, self.b_l_cut, ax1_mids,
                ax2_mids, ax_dimension, self.b_correct_mag_array, self.b_correct_mag_slice,
                self.b_correct_sig_slice, self.n_pool, b_npy_or_csv, b_coord_or_chunk,
                self.b_pos_and_err_indices, self.b_mag_indices, self.b_mag_unc_indices,
                self.b_filt_names, self.b_best_mag_index, self.b_auf_region_frame,
                use_photometric_uncertainties=self.b_use_photometric_uncertainties,
                pregenerate_cutouts=True, chunks=[self.chunk_id])
            ac(self.b_ref_csv_cat_file_string, self.b_csv_cat_file_string,
               tri_download=self.b_download_tri, make_plots=True, overwrite_all_sightlines=True)

            csv_folder, csv_filename = os.path.split(
                self.b_csv_cat_file_string.format(self.chunk_id))
            csv_to_npy(csv_folder, csv_filename, self.b_cat_folder_path,
                       self.b_pos_and_err_indices[1], self.b_mag_indices, self.b_best_mag_index_col,
                       self.b_chunk_overlap_col, header=False, process_uncerts=True,
                       astro_sig_fits_filepath='{}/npy'.format(self.b_correct_astro_save_folder),
                       cat_in_radec=self.b_auf_region_frame == 'equatorial',
                       mn_in_radec=self.b_auf_region_frame == 'equatorial')
        if self.b_compute_snr_mag_relation:
            t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print("{} Rank {}, chunk {}: Calculating catalogue 'b' SNR-mag relations..."
                  .format(t, self.rank, self.chunk_id))
            smr = SNRMagnitudeRelationship(
                self.b_correct_astro_save_folder, ax1_mids, ax2_mids, ax_dimension, b_npy_or_csv,
                b_coord_or_chunk, self.b_pos_and_err_indices, self.b_mag_indices,
                self.b_mag_unc_indices, self.b_filt_names, self.b_auf_region_frame,
                chunks=[self.chunk_id])
            smr(self.b_csv_cat_file_string, make_plots=True, overwrite_all_sightlines=True)
        if self.b_correct_astrometry or self.b_compute_snr_mag_relation:
            os.system('cp {}/npy/snr_mag_params.npy {}'.format(self.b_correct_astro_save_folder,
                      self.b_snr_mag_params_path))
            f = 'snr_mag_params'
            if not os.path.isfile('{}/{}.npy'.format(self.b_snr_mag_params_path, f)):
                raise FileNotFoundError('{} file not found in catalogue {} path. '
                                        'Please ensure {} are pre-generated.'
                                        .format(f, "b", 'astrometry corrections'))
            a = np.load('{}/snr_mag_params.npy'.format(self.b_snr_mag_params_path))
            if not (len(a.shape) == 3 and a.shape[2] == 5 and
                    a.shape[0] == len(self.b_filt_names)):
                raise ValueError('b_snr_mag_params should be of shape (X, Y, 5)')
            self.b_snr_mag_params = a

        # Ensure that we can create the folders for outputs.
        for path in ['reject', 'pairing']:
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

        self.make_shared_data()

    def __call__(self):
        '''
        Call function for CrossMatch, performs cross-matching two photometric catalogues.
        '''

        # Special case for single process, i.e. serial version of code.
        # Do not use manager-worker pattern. Instead, one process loops over all chunks
        if self.comm_size == 1:
            for (chunk_id, joint_file_path, cat_a_file_path, cat_b_file_path) in self.chunk_queue:
                t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print('{} Rank {} processing chunk {}'.format(t, self.rank, chunk_id))
                self.chunk_id = chunk_id
                self._process_chunk(joint_file_path, cat_a_file_path, cat_b_file_path)
                if self.resume_file != None:
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
                out_of_walltime = False
                while active_workers > 0:
                    # If checkpointing disabled, simply wait for any worker to
                    # request a chunk and report completion of any previous chunk
                    if self.end_time == None:
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
                                print('{} Rank {}: reaching job walltime. Cancelling all further '
                                      'work. {} chunks remain unprocessed.'
                                      .format(t, self.rank, self.num_chunks_to_process))
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
                    if signal == self.worker_signals['WORK_COMPLETE'] \
                    and self.resume_file != None:
                        t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print('{} Rank {}: chunk {} processed by rank {}. Adding to resume file.'
                              .format(t, self.rank, chunk_id, worker_id))
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
                        print('{} Rank {}: rank {} failed to process chunk {}.'
                              .format(t, self.rank, worker_id, chunk_id))

                    # Assign chunks until no more work.
                    # Then count "no more work" signals until no more workers.
                    try:
                        new_chunk = self.chunk_queue.pop(0)
                        signal = self.worker_signals['WORK_REQUEST']
                        t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print('{} Rank {}: sending rank {} chunk {}'.format(
                              t, self.rank, worker_id, new_chunk[0]))
                    except IndexError:
                        new_chunk = None
                        signal = self.worker_signals['NO_MORE_WORK']
                        active_workers -= 1

                    sys.stdout.flush()

                    self.comm.send((signal, new_chunk), dest=worker_id)

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
                    (signal, chunk) = self.comm.recv(source=0)

                    # Handle received signal.
                    # Terminate when signalled there is no more work...
                    if signal == self.worker_signals['NO_MORE_WORK']:
                        break
                    # ...or process the given chunk
                    else:
                        (chunk_id, joint_file_path, cat_a_file_path, cat_b_file_path) = chunk
                        self.chunk_id = chunk_id
                        t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print('{} Rank {}: processing chunk {}'.format(t, self.rank, chunk_id))

                        try:
                            self._process_chunk(joint_file_path, cat_a_file_path, cat_b_file_path)
                            signal = self.worker_signals['WORK_COMPLETE']
                        except Exception as e:
                            # Recover worker on chunk processing error
                            signal = self.worker_signals['WORK_ERROR']
                            # TODO More granular error handling
                            t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            print("{} Rank {}: failed to process chunk {}. Exception: {}"
                                  .format(t, self.rank, chunk_id, e))

                        completed_chunk_id = chunk_id

                    sys.stdout.flush()

        # Clean up and shut down
        self._cleanup()


    def _process_chunk(self, joint_file_path, cat_a_file_path, cat_b_file_path):
        '''
        Runs the various stages of cross-matching two photometric catalogues
        '''
        # Initialise using the current chunk data
        # TODO Move some initialisation into class constructor?
        # TODO Have manager perform file loads and broadcast result to reduce I/O
        self._initialise_chunk(joint_file_path, cat_a_file_path, cat_b_file_path)

        # The first step is to create the perturbation AUF components, if needed.
        self.create_perturb_auf()

        # Once AUF components are assembled, we now group sources based on
        # convolved AUF integration lengths, to get "common overlap" sources
        # and merge such overlaps into distinct "islands" of sources to match.
        self.group_sources()

        # The third step in this process is to, to some level, calculate the
        # photometry-related information necessary for the cross-match.
        self.calculate_phot_like()

        # The final stage of the cross-match process is that of putting together
        # the previous stages, and calculating the cross-match probabilities.
        self.pair_sources()

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
        print("{} Rank {}, chunk {}: Removing halo matches and non-matches..."
              .format(t, self.rank, self.chunk_id))

        ac = np.load('{}/pairing/ac.npy'.format(self.joint_folder_path))
        bc = np.load('{}/pairing/bc.npy'.format(self.joint_folder_path))

        af = np.load('{}/pairing/af.npy'.format(self.joint_folder_path))
        bf = np.load('{}/pairing/bf.npy'.format(self.joint_folder_path))

        a_in_overlaps = np.load('{}/in_chunk_overlap.npy'.format(self.a_cat_folder_path))
        b_in_overlaps = np.load('{}/in_chunk_overlap.npy'.format(self.b_cat_folder_path))

        core_matches = ~a_in_overlaps[ac] | ~b_in_overlaps[bc]
        np.save('{}/pairing/ac.npy'.format(self.joint_folder_path), ac[core_matches])
        np.save('{}/pairing/bc.npy'.format(self.joint_folder_path), bc[core_matches])
        for fname in ['pc', 'eta', 'xi', 'crptseps', 'acontamflux', 'bcontamflux']:
            np.save('{}/pairing/{}.npy'.format(self.joint_folder_path, fname),
                    np.load('{}/pairing/{}.npy'.format(self.joint_folder_path, fname))[
                        core_matches])
        for fname in ['pacontam', 'pbcontam']:
            np.save('{}/pairing/{}.npy'.format(self.joint_folder_path, fname),
                    np.load('{}/pairing/{}.npy'.format(self.joint_folder_path, fname))[
                        :, core_matches])

        a_core_nonmatches = ~a_in_overlaps[af]
        b_core_nonmatches = ~b_in_overlaps[bf]
        np.save('{}/pairing/af.npy'.format(self.joint_folder_path), af[a_core_nonmatches])
        np.save('{}/pairing/bf.npy'.format(self.joint_folder_path), bf[b_core_nonmatches])
        for fnametype, cnm in zip(['a', 'b'], [a_core_nonmatches, b_core_nonmatches]):
            for fname_ in ['{}fieldflux', 'pf{}', '{}fieldeta', '{}fieldxi', '{}fieldseps']:
                fname = fname_.format(fnametype)
                np.save('{}/pairing/{}.npy'.format(self.joint_folder_path, fname),
                        np.load('{}/pairing/{}.npy'.format(self.joint_folder_path, fname))[cnm])

        if self.make_output_csv:
            npy_to_csv(
                [self.a_input_csv_folder, self.b_input_csv_folder], self.joint_folder_path,
                self.output_csv_folder, [self.a_cat_csv_name, self.b_cat_csv_name],
                [self.match_out_csv_name, self.a_nonmatch_out_csv_name,
                 self.b_nonmatch_out_csv_name], [self.a_cat_col_names, self.b_cat_col_names],
                [self.a_cat_col_nums, self.b_cat_col_nums], [self.a_cat_name, self.b_cat_name],
                self.mem_chunk_num, [self.a_input_npy_folder, self.b_input_npy_folder],
                headers=[self.a_csv_has_header, self.b_csv_has_header],
                extra_col_name_lists=[self.a_extra_col_names, self.b_extra_col_names],
                extra_col_num_lists=[self.a_extra_col_nums, self.b_extra_col_nums])

    def _make_chunk_queue(self, completed_chunks):
        '''
        Determines the order with which chunks will be processed

        Returns
        -------
        chunk_queue : list of tuples of strings
            List with one element per chunk. Each element a tuple of chunk ID and
            paths to metadata files in order (ID, cross-match, catalogue "a", catalogue "b")
        '''
        # Each metadata file associated with a chunk assumed to be in a subfolder
        # e.g. Two chunks, "2017" and "2018", have structure:
        #   chunks_folder_path/2017/crossmatch_params_2017.txt
        #   chunks_folder_path/2017/cat_a_params_2017.txt
        #   chunks_folder_path/2017/cat_b_params_2017.txt
        #   chunks_folder_path/2018/crossmatch_params_2018.txt
        #   chunks_folder_path/2018/cat_a_params_2018.txt
        #   chunks_folder_path/2018/cat_b_params_2018.txt

        # Loop over subfolders in chunks folder, extracting paths to metadata files contained within
        chunk_queue = []
        for folder in os.listdir(self.chunks_folder_path):
            folder_path = os.path.join(self.chunks_folder_path, folder)

            # Skip non-directories
            if not os.path.isdir(folder_path):
                continue
            # Skip completed chunks
            if folder in completed_chunks:
                continue

            # Identify chunk by subfolder name
            chunk_id = folder
            joint_file_path = ""
            cat_a_file_path = ""
            cat_b_file_path = ""

            for filename in os.listdir(folder_path):
                # Ignore non-txt files
                if filename.endswith(".txt"):
                    # TODO Relying on particular naming convention for metadata files
                    if filename.startswith("crossmatch_params"):
                        joint_file_path = os.path.join(folder_path, filename)
                    elif filename.startswith("cat_a_params"):
                        cat_a_file_path = os.path.join(folder_path, filename)
                    elif filename.startswith("cat_b_params"):
                        cat_b_file_path = os.path.join(folder_path, filename)

            # Check results
            if joint_file_path == "":
                raise FileNotFoundError('Cross-match metadata file for chunk {} not found in directory {}'
                                        .format(chunk_id, folder_path))
            if cat_a_file_path == "":
                raise FileNotFoundError('Catalogue "a" metadata file for chunk {} not found in directory {}'
                                        .format(chunk_id, folder_path))
            if cat_b_file_path == "":
                raise FileNotFoundError('Catalogue "b" metadata file for chunk {} not found in directory {}'
                                        .format(chunk_id, folder_path))

            # Determine combined input file size for catalogues "a" and "b"
            # Used to sort queue
            chunk_size = 0
            for file_path in [cat_a_file_path, cat_b_file_path]:
                # Read input folder path from metadata file.
                # TODO: Use ConfigParser here?
                with open(file_path, 'r') as param_file:
                    cat_folder_path = ""
                    for line in param_file:
                        if line.startswith("cat_folder_path"):
                            cat_folder_path = line.split('=')[-1].strip()
                            break

                if not os.path.isdir(cat_folder_path):
                    raise FileNotFoundError('Catalogue directory {} not found'.format(cat_folder_path))

                # Get size of all files in input folder
                # Expected to be con_cat_astro.npy, con_cat_photo.npy and magref.npy
                for path, dirs, files in os.walk(cat_folder_path):
                    for f in files:
                        file_path = os.path.join(path, f)
                        chunk_size += os.path.getsize(file_path)

            # Append result as tuple of size, ID and all 3 paths
            chunk_queue.append((chunk_size, chunk_id, joint_file_path, cat_a_file_path, cat_b_file_path))

        # Sort chunk list by size, largest to smallest
        chunk_queue.sort(key=lambda x: x[0], reverse=True)
        # Remove chunk size from output list
        chunk_queue_sorted = [ (chunk_id, joint_file_path, cat_a_file_path, cat_b_file_path) for
                               (chunk_size, chunk_id, joint_file_path, cat_a_file_path, cat_b_file_path) in
                               chunk_queue ]

        return chunk_queue_sorted

    def _cleanup(self):
        '''
        Final clean up operations before application shut down
        '''
        if self.rank == 0 and self.resume_file != None:
            self.resume_file.close()

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

        for check_flag in ['include_perturb_auf', 'include_phot_like', 'use_phot_priors',
                           'cf_region_type', 'cf_region_frame', 'cf_region_points',
                           'joint_folder_path', 'pos_corr_dist', 'real_hankel_points',
                           'four_hankel_points', 'four_max_rho', 'cross_match_extent',
                           'mem_chunk_num', 'int_fracs', 'make_output_csv', 'n_pool']:
            if check_flag not in joint_config:
                raise ValueError("Missing key {} from joint metadata file.".format(check_flag))

        for config, catname in zip([cat_a_config, cat_b_config], ['"a"', '"b"']):
            for check_flag in ['auf_region_type', 'auf_region_frame', 'auf_region_points',
                               'filt_names', 'cat_name', 'auf_folder_path', 'cat_folder_path',
                               'correct_astrometry']:
                if check_flag not in config:
                    raise ValueError("Missing key {} from catalogue {} metadata file.".format(
                                     check_flag, catname))

        for run_flag in ['include_perturb_auf', 'include_phot_like', 'use_phot_priors']:
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
        # However, calling AstrometricCorrections in its current form confuses
        # this, since it always uses the perturbation AUF component. We therefore
        # split out the items that are NOT required for AstrometricCorrections
        # first, if there are any.

        self.a_correct_astrometry = self._str2bool(cat_a_config['correct_astrometry'])
        self.b_correct_astrometry = self._str2bool(cat_b_config['correct_astrometry'])
        self.a_compute_snr_mag_relation = self._str2bool(cat_a_config['compute_snr_mag_relation'])
        self.b_compute_snr_mag_relation = self._str2bool(cat_b_config['compute_snr_mag_relation'])
        if self.a_correct_astrometry and self.a_compute_snr_mag_relation:
            raise ValueError("Ambiguity in catalogue 'a' having both correct_astrometry and "
                             "compute_snr_mag_relation both being True. Only set at most one "
                             "flag as 'True'.")
        if self.b_correct_astrometry and self.b_compute_snr_mag_relation:
            raise ValueError("Ambiguity in catalogue 'b' having both correct_astrometry and "
                             "compute_snr_mag_relation both being True. Only set at most one "
                             "flag as 'True'.")

        if self.include_perturb_auf or self.a_correct_astrometry or self.b_correct_astrometry:
            for check_flag in ['num_trials', 'd_mag']:
                if check_flag not in joint_config:
                    raise ValueError("Missing key {} from joint metadata file.".format(check_flag))

            a = joint_config['num_trials']
            try:
                a = float(a)
            except ValueError:
                raise ValueError("num_trials should be an integer.")
            if not a.is_integer():
                raise ValueError("num_trials should be an integer.")
            self.num_trials = int(a)

            for flag in ['d_mag']:
                try:
                    setattr(self, flag, float(joint_config[flag]))
                except ValueError:
                    raise ValueError("{} must be a float.".format(flag))

        # Nominally these are all of the parameters required if include_perturb_auf
        # is True, but for the minute they're also forced if correct_astrometry
        # is also True instead of those aligning, so we bypass the if statement
        # if self.a_correct_astrometry or self.b_correct_astrometry respectively
        # have been set.
        for correct_astro, compute_snr_mag_relation, config, catname, flag in zip(
                [self.a_correct_astrometry, self.b_correct_astrometry],
                [self.a_compute_snr_mag_relation, self.b_compute_snr_mag_relation],
                [cat_a_config, cat_b_config], ['"a"', '"b"'], ['a_', 'b_']):
            if self.include_perturb_auf or correct_astro:
                for check_flag in ['dens_dist']:
                    if check_flag not in config:
                        raise ValueError("Missing key {} from catalogue {} metadata file."
                                         .format(check_flag, catname))
                try:
                    setattr(self, '{}dens_dist'.format(flag), float(config['dens_dist']))
                except ValueError:
                    raise ValueError("dens_dist in catalogue {} must be a float."
                                     .format(catname))

            if self.include_perturb_auf:
                for check_flag in ['fit_gal_flag']:
                    if check_flag not in config:
                        raise ValueError("Missing key {} from catalogue {} metadata file.".format(
                                         check_flag, catname))
                setattr(self, '{}fit_gal_flag'.format(flag), self._str2bool(config['fit_gal_flag']))

            # snr_mag_params_path is needed in any one of these three cases:
            if self.include_perturb_auf or correct_astro or compute_snr_mag_relation:
                for check_flag in ['snr_mag_params_path']:
                    if check_flag not in config:
                        raise ValueError("Missing key {} from catalogue {} metadata file.".format(
                                         check_flag, catname))

                if not os.path.exists(config['snr_mag_params_path']):
                    raise OSError('{}snr_mag_params_path does not exist. Please ensure that '
                                  'path for catalogue {} is correct.'.format(flag, catname))

                # If we are correcting the astrometry, we will be
                # re-making the SNR-mag relations so skip loading, and hence
                # checking the existence of, that for now.
                if not (correct_astro or compute_snr_mag_relation):
                    if not os.path.isfile('{}/snr_mag_params.npy'.format(
                                          config['snr_mag_params_path'])):
                        raise FileNotFoundError('snr_mag_params file not found in catalogue '
                                                '{} path. Please ensure astrometry corrections '
                                                'are pre-generated.'.format(catname))

                    a = np.load('{}/snr_mag_params.npy'.format(
                                config['snr_mag_params_path']))
                    if not (len(a.shape) == 3 and a.shape[2] == 5 and
                            a.shape[0] == len(getattr(self, '{}filt_names'.format(flag)))):
                        raise ValueError('{}snr_mag_params should be of shape (X, Y, 5).'
                                         .format(flag))
                    setattr(self, '{}snr_mag_params'.format(flag), a)
                else:
                    setattr(self, '{}snr_mag_params_path'.format(flag),
                            config['snr_mag_params_path'])
            if self.include_perturb_auf or correct_astro:
                for check_flag in ['tri_set_name', 'tri_filt_names', 'tri_filt_num',
                                   'download_tri', 'psf_fwhms', 'run_fw_auf', 'run_psf_auf',
                                   'tri_maglim_faint', 'tri_num_faint', 'gal_al_avs']:
                    if check_flag not in config:
                        raise ValueError("Missing key {} from catalogue {} metadata file.".format(
                                         check_flag, catname))

                # Set as a list of floats
                for var in ['gal_al_avs']:
                    a = config[var].split(' ')
                    try:
                        b = np.array([float(f) for f in a])
                    except ValueError:
                        raise ValueError('{} should be a list of floats in catalogue '
                                         '{} metadata file'.format(var, catname))
                    if len(b) != len(getattr(self, '{}filt_names'.format(flag))):
                        raise ValueError('{}{} and {}filt_names should contain the same '
                                         'number of entries.'.format(flag, var, flag))
                    setattr(self, '{}{}'.format(flag, var), b)

                setattr(self, '{}download_tri'.format(flag), self._str2bool(config['download_tri']))
                setattr(self, '{}tri_set_name'.format(flag), config['tri_set_name'])
                a = config['tri_filt_names'].split()
                if len(a) != len(getattr(self, '{}filt_names'.format(flag))):
                    raise ValueError('{}tri_filt_names and {}filt_names should contain the '
                                     'same number of entries.'.format(flag, flag))
                setattr(self, '{}tri_filt_names'.format(flag),
                        np.array(config['tri_filt_names'].split()))

                a = config['psf_fwhms'].split()
                try:
                    b = np.array([float(f) for f in a])
                except ValueError:
                    raise ValueError('psf_fwhms should be a list of floats in catalogue {} '
                                     'metadata file.'.format(catname))
                if len(b) != len(getattr(self, '{}filt_names'.format(flag))):
                    raise ValueError('{}psf_fwhms and {}filt_names should contain the '
                                     'same number of entries.'.format(flag, flag))
                setattr(self, '{}psf_fwhms'.format(flag), b)

                setattr(self, '{}run_fw_auf'.format(flag), self._str2bool(config['run_fw_auf']))
                setattr(self, '{}run_psf_auf'.format(flag), self._str2bool(config['run_psf_auf']))

                # Only need dd_params or l_cut if we're using run_psf_auf or
                # correct_astrometry is True.
                if getattr(self, '{}run_psf_auf'.format(flag)) or correct_astro:
                    for check_flag, f in zip(['dd_params_path', 'l_cut_path'],
                                             ['dd_params', 'l_cut']):
                        if check_flag not in config:
                            raise ValueError("Missing key {} from catalogue {} metadata file."
                                             .format(check_flag, catname))

                    for check_flag, f in zip(['dd_params_path', 'l_cut_path'],
                                             ['dd_params', 'l_cut']):
                        if not os.path.exists(config[check_flag]):
                            raise OSError('{}{} does not exist. Please ensure that path for '
                                          'catalogue {} is correct.'.format(flag, check_flag,
                                                                            catname))

                        if not os.path.isfile('{}/{}.npy'.format(config[check_flag], f)):
                            raise FileNotFoundError('{} file not found in catalogue {} path. '
                                                    'Please ensure PSF photometry perturbations '
                                                    'are pre-generated.'.format(f, catname))

                        if 'dd_params' in check_flag:
                            a = np.load('{}/dd_params.npy'.format(config['dd_params_path']))
                            if not (len(a.shape) == 3 and a.shape[0] == 5 and a.shape[2] == 2):
                                raise ValueError('{}dd_params should be of shape (5, X, 2).'
                                                 .format(flag))
                        if 'l_cut' in check_flag:
                            a = np.load('{}/l_cut.npy'.format(config['l_cut_path']))
                            if not (len(a.shape) == 1 and a.shape[0] == 3):
                                raise ValueError('{}l_cut should be of shape (3,) only.'
                                                 .format(flag))
                        setattr(self, '{}{}'.format(flag, f), a)

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

                for suffix in ['_faint']:
                    try:
                        a = config['tri_num{}'.format(suffix)]
                        if float(a).is_integer():
                            setattr(self, '{}tri_num{}'.format(flag, suffix), int(a))
                        else:
                            raise ValueError("tri_num{} should be a single integer number in "
                                             "catalogue {} metadata file.".format(suffix, catname))
                    except ValueError:
                        raise ValueError("tri_num{} should be a single integer number in "
                                         "catalogue {} metadata file.".format(suffix, catname))

                    try:
                        setattr(self, '{}tri_maglim{}'.format(flag, suffix),
                                float(config['tri_maglim{}'.format(suffix)]))
                    except ValueError:
                        raise ValueError("tri_maglim{} in catalogue {} must be a float."
                                         .format(suffix, catname))

                if not correct_astro:
                    if flag == "a_":
                        fit_gal_flag = self.a_fit_gal_flag
                    else:
                        fit_gal_flag = self.b_fit_gal_flag
                if correct_astro or fit_gal_flag:
                    for check_flag in ['gal_wavs', 'gal_zmax', 'gal_nzs',
                                       'gal_aboffsets', 'gal_filternames']:
                        if check_flag not in config:
                            raise ValueError("Missing key {} from catalogue {} metadata file."
                                             .format(check_flag, catname))
                    # Set all lists of floats
                    for var in ['gal_wavs', 'gal_zmax', 'gal_aboffsets']:
                        a = config[var].split(' ')
                        try:
                            b = np.array([float(f) for f in a])
                        except ValueError:
                            raise ValueError('{} should be a list of floats in catalogue '
                                             '{} metadata file'.format(var, catname))
                        if len(b) != len(getattr(self, '{}filt_names'.format(flag))):
                            raise ValueError('{}{} and {}filt_names should contain the same '
                                             'number of entries.'.format(flag, var, flag))
                        setattr(self, '{}{}'.format(flag, var), b)
                    # galaxy_nzs should be a list of integers.
                    a = config['gal_nzs'].split(' ')
                    try:
                        b = np.array([float(f) for f in a])
                    except ValueError:
                        raise ValueError('gal_nzs should be a list of integers '
                                         'in catalogue {} metadata file'.format(catname))
                    if len(b) != len(getattr(self, '{}filt_names'.format(flag))):
                        raise ValueError('{}gal_nzs and {}filt_names should contain the same '
                                         'number of entries.'.format(flag, flag))
                    if not np.all([c.is_integer() for c in b]):
                        raise ValueError('All elements of {}gal_nzs should be '
                                         'integers.'.format(flag))
                    setattr(self, '{}gal_nzs'.format(flag), np.array([int(c) for c in b]))
                    # Filter names are simple lists of strings
                    b = config['gal_filternames'].split(' ')
                    if len(b) != len(getattr(self, '{}filt_names'.format(flag))):
                        raise ValueError('{}gal_filternames and {}filt_names should contain the '
                                         'same number of entries.'.format(flag, flag))
                    setattr(self, '{}gal_filternames'.format(flag), np.array(b))

        self.a_cat_name = cat_a_config['cat_name']
        self.b_cat_name = cat_b_config['cat_name']

        try:
            self.pos_corr_dist = float(joint_config['pos_corr_dist'])
        except ValueError:
            raise ValueError("pos_corr_dist must be a float.")

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

        a = joint_config['int_fracs'].split()
        try:
            b = np.array([float(f) for f in a])
        except ValueError:
            raise ValueError("All elements of int_fracs should be floats.")
        if len(b) != 3:
            raise ValueError("int_fracs should contain three elements.")
        self.int_fracs = b

        self.make_output_csv = self._str2bool(joint_config['make_output_csv'])
        if self.make_output_csv:
            self._read_metadata_csv(joint_config, cat_a_config, cat_b_config)

        # Load the multiprocessing Pool count.
        try:
            a = joint_config['n_pool']
            if float(a).is_integer():
                self.n_pool = int(a)
            else:
                raise ValueError("n_pool should be a single integer number.")
        except ValueError:
            raise ValueError("n_pool should be a single integer number.")

        for correct_astro, compute_snr_mag_relation, config, catname, flag in zip(
                [self.a_correct_astrometry, self.b_correct_astrometry],
                [self.a_compute_snr_mag_relation, self.b_compute_snr_mag_relation],
                [cat_a_config, cat_b_config], ['"a"', '"b"'], ['a_', 'b_']):
            # Have to split these parameters into two, as four of them are
            # required for the simpler case of just doing SNR-mag relation
            # calculations, instead of the full astrometry correction.
            if correct_astro or compute_snr_mag_relation:
                for check_flag in ['correct_astro_save_folder', 'csv_cat_file_string',
                                   'pos_and_err_indices', 'mag_indices', 'mag_unc_indices']:
                    if check_flag not in config:
                        raise ValueError("Missing key {} from catalogue {} metadata file.".format(
                                         check_flag, catname))

                setattr(self, '{}correct_astro_save_folder'.format(flag),
                        os.path.abspath(config['correct_astro_save_folder']))

                setattr(self, '{}csv_cat_file_string'.format(flag),
                        os.path.abspath(config['csv_cat_file_string']))

                a = config['mag_indices'].split(' ')
                try:
                    b = np.array([float(f) for f in a])
                except ValueError:
                    raise ValueError('mag_indices should be a list of integers '
                                     'in the catalogue {} metadata file'.format(catname))
                if len(b) != len(getattr(self, '{}filt_names'.format(flag))):
                    raise ValueError('{}filt_names and {}mag_indices should contain the '
                                     'same number of entries.'.format(flag, flag))
                if not np.all([c.is_integer() for c in b]):
                    raise ValueError('All elements of {}mag_indices should be '
                                     'integers.'.format(flag))
                setattr(self, '{}mag_indices'.format(flag), np.array([int(c) for c in b]))

                a = config['mag_unc_indices'].split(' ')
                try:
                    b = np.array([float(f) for f in a])
                except ValueError:
                    raise ValueError('mag_unc_indices should be a list of integers '
                                     'in the catalogue {} metadata file'.format(catname))
                if len(b) != len(getattr(self, '{}mag_indices'.format(flag))):
                    raise ValueError('{}mag_unc_indices and {}mag_indices should contain the '
                                     'same number of entries.'.format(flag, flag))
                if not np.all([c.is_integer() for c in b]):
                    raise ValueError('All elements of {}mag_unc_indices should be '
                                     'integers.'.format(flag))
                setattr(self, '{}mag_unc_indices'.format(flag), np.array([int(c) for c in b]))

                # pos_and_err_indices should be a three- or six-integer list that
                # we then transform into [reference_cat_inds, this_cat_inds]
                # where each *_cat_inds is a three-element list [x, y, z],
                # or just this_cat_inds in the case of
                # compute_snr_mag_relation=True.
                a = config['pos_and_err_indices'].split(' ')
                try:
                    b = np.array([float(f) for f in a])
                except ValueError:
                    raise ValueError('pos_and_err_indices should be a list of integers '
                                     'in the catalogue {} metadata file'.format(catname))
                if len(b) != 6 and correct_astro:
                    raise ValueError('{}pos_and_err_indices should contain six elements '
                                     'when correct_astrometry is True.'.format(flag))
                if len(b) != 3 and compute_snr_mag_relation:
                    raise ValueError('{}pos_and_err_indices should contain three elements '
                                     'when compute_snr_mag_relation is True.'.format(flag))
                if not np.all([c.is_integer() for c in b]):
                    raise ValueError('All elements of {}pos_and_err_indices should be '
                                     'integers.'.format(flag))
                d = np.array([int(c) for c in b])
                if correct_astro:
                    # The reshape puts the first three elements in a[0], and hence
                    # those are ref_cat_inds, with a[1] this_cat_inds.
                    setattr(self, '{}pos_and_err_indices'.format(flag), d.reshape(2, 3))
                else:
                    # If we only want to compute the SNR-mag relation, then we've
                    # only got three elements, so we just store them in a (3,)
                    # shape array.
                    setattr(self, '{}pos_and_err_indices'.format(flag), d)

            if correct_astro:
                # If this particular catalogue requires a systematic correction
                # for astrometric biases from ensemble match distributions before
                # we can do a probability-based cross-match, then load some extra
                # pieces of information, over and above those already loaded
                # for just the SNR-mag case.
                for check_flag in ['best_mag_index', 'nn_radius', 'ref_csv_cat_file_string',
                                   'correct_mag_array', 'correct_mag_slice', 'correct_sig_slice',
                                   'chunk_overlap_col', 'best_mag_index_col',
                                   'use_photometric_uncertainties']:
                    if check_flag not in config:
                        raise ValueError("Missing key {} from catalogue {} metadata file.".format(
                                         check_flag, catname))

                setattr(self, '{}use_photometric_uncertainties'.format(flag),
                        self._str2bool(config['use_photometric_uncertainties']))

                a = config['best_mag_index']
                try:
                    a = float(a)
                except ValueError:
                    raise ValueError("best_mag_index should be an integer in the catalogue {} "
                                     "metadata file.".format(catname))
                if not a.is_integer():
                    raise ValueError("best_mag_index should be an integer in the catalogue {} "
                                     "metadata file.".format(catname))
                if int(a) >= len(getattr(self, '{}filt_names'.format(flag))):
                    raise ValueError("best_mag_index cannot be a larger index than the list of "
                                     "filters in the catalogue {} metadata file.".format(catname))
                setattr(self, '{}best_mag_index'.format(flag), int(a))

                a = config['best_mag_index_col']
                try:
                    a = float(a)
                except ValueError:
                    raise ValueError("best_mag_index_col should be an integer in the catalogue {} "
                                     "metadata file.".format(catname))
                if not a.is_integer():
                    raise ValueError("best_mag_index_col should be an integer in the catalogue {} "
                                     "metadata file.".format(catname))
                setattr(self, '{}best_mag_index_col'.format(flag), int(a))

                a = config['chunk_overlap_col']
                if a == "None":
                    setattr(self, '{}chunk_overlap_col'.format(flag), None)
                else:
                    try:
                        a = float(a)
                    except ValueError:
                        raise ValueError("chunk_overlap_col should be an integer in the "
                                         "catalogue {} metadata file.".format(catname))
                    if not a.is_integer():
                        raise ValueError("chunk_overlap_col should be an integer in the "
                                         "catalogue {} metadata file.".format(catname))
                    setattr(self, '{}chunk_overlap_col'.format(flag), int(a))

                try:
                    setattr(self, '{}nn_radius'.format(flag), float(config['nn_radius']))
                except ValueError:
                    raise ValueError("nn_radius must be a float in the catalogue {} "
                                     "metadata file.".format(catname))

                setattr(self, '{}ref_csv_cat_file_string'.format(flag),
                        os.path.abspath(config['ref_csv_cat_file_string']))

                a = config['correct_mag_array'].split()
                try:
                    b = np.array([float(f) for f in a])
                except ValueError:
                    raise ValueError('correct_mag_array should be a list of floats in the '
                                     'catalogue {} metadata file.'.format(catname))
                setattr(self, '{}correct_mag_array'.format(flag), b)

                a = config['correct_mag_slice'].split()
                try:
                    b = np.array([float(f) for f in a])
                except ValueError:
                    raise ValueError('correct_mag_slice should be a list of floats in the '
                                     'catalogue {} metadata file.'.format(catname))
                if len(b) != len(getattr(self, '{}correct_mag_array'.format(flag))):
                    raise ValueError('{}correct_mag_array and {}correct_mag_slice should contain '
                                     'the same number of entries.'.format(flag, flag))
                setattr(self, '{}correct_mag_slice'.format(flag), b)

                a = config['correct_sig_slice'].split()
                try:
                    b = np.array([float(f) for f in a])
                except ValueError:
                    raise ValueError('correct_sig_slice should be a list of floats in the '
                                     'catalogue {} metadata file.'.format(catname))
                if len(b) != len(getattr(self, '{}correct_mag_array'.format(flag))):
                    raise ValueError('{}correct_mag_array and {}correct_sig_slice should contain '
                                     'the same number of entries.'.format(flag, flag))
                setattr(self, '{}correct_sig_slice'.format(flag), b)

    def _read_metadata_csv(self, joint_config, cat_a_config, cat_b_config):
        """
        Read metadata from input config files relating to outputting combined
        .csv files during the post-process step, if requested.

        Parameters
        ----------
        joint_config : ConfigParser
            The pre-loaded set of input configuration parameters as related
            to the joint match between catalogues a and b.
        cat_a_config : ConfigParser
            Configuration parameters that are solely related to catalogue "a".
        cat_b_config : ConfigParser
            Configuration parameters that are just relate to catalogue "b".
        """

        for check_flag in ['output_csv_folder', 'match_out_csv_name', 'nonmatch_out_csv_name']:
            if check_flag not in joint_config:
                raise ValueError("Missing key {} from joint metadata file.".format(check_flag))

        for config, catname in zip([cat_a_config, cat_b_config], ['"a"', '"b"']):
            for check_flag in ['input_csv_folder', 'cat_csv_name', 'cat_col_names', 'cat_col_nums',
                               'input_npy_folder', 'csv_has_header', 'extra_col_names',
                               'extra_col_nums']:
                if check_flag not in config:
                    raise ValueError("Missing key {} from catalogue {} metadata file.".format(
                                     check_flag, catname))

        self.output_csv_folder = os.path.abspath(joint_config['output_csv_folder'])
        try:
            os.makedirs(self.output_csv_folder, exist_ok=True)
        except OSError:
            raise OSError("Error when trying to create folder to store output csv files in. "
                          "Please ensure that output_csv_folder is correct in joint config file.")

        self.match_out_csv_name = joint_config['match_out_csv_name']
        for config, catname in zip([cat_a_config, cat_b_config], ['a_', 'b_']):
            # Non-match csv name should be of the format
            # [cat name]_[some indication this is a non-match], but note that
            # this is defined in joint_config, not each individual
            # catalogue config!
            nonmatch_out_name = joint_config['nonmatch_out_csv_name']
            setattr(self, '{}nonmatch_out_csv_name'.format(catname),
                    '{}_{}'.format(getattr(self, '{}cat_name'.format(catname)), nonmatch_out_name))

            input_csv_folder = os.path.abspath(config['input_csv_folder'])
            if not os.path.exists(input_csv_folder):
                raise OSError('input_csv_folder from catalogue "{}" does not exist.'
                              .format(catname[0]))
            setattr(self, '{}input_csv_folder'.format(catname), input_csv_folder)

            setattr(self, '{}cat_csv_name'.format(catname), config['cat_csv_name'])

            # cat_col_names is simply a list/array of strings. However, to
            # avoid any issues with generic names like "RA" being added to the
            # output .csv file twice, we prepend the catalogue name to the front
            # of them all.
            catcolnames = config['cat_col_names'].split()
            setattr(self, '{}cat_col_names'.format(catname),
                    np.array(['{}_{}'.format(getattr(self, '{}cat_name'.format(catname)),
                              q) for q in catcolnames]))

            # But cat_col_nums is a list/array of integers, and should be of the
            # same length as cat_col_names.
            a = config['cat_col_nums'].split(' ')
            try:
                b = np.array([float(f) for f in a])
            except ValueError:
                raise ValueError('cat_col_nums should be a list of integers '
                                 'in catalogue "{}" metadata file'.format(catname[0]))
            if len(b) != len(getattr(self, '{}cat_col_names'.format(catname))):
                raise ValueError('{}cat_col_names and {}cat_col_nums should contain the same '
                                 'number of entries.'.format(catname, catname))
            if not np.all([c.is_integer() for c in b]):
                raise ValueError('All elements of {}cat_col_nums should be '
                                 'integers.'.format(catname))
            setattr(self, '{}cat_col_nums'.format(catname), np.array([int(c) for c in b]))

            input_npy_folder = config['input_npy_folder']
            if input_npy_folder != 'None' and not os.path.exists(input_npy_folder):
                raise OSError('input_npy_folder from catalogue "{}" does not exist.'
                              .format(catname[0]))
            if input_npy_folder == 'None':
                setattr(self, '{}input_npy_folder'.format(catname), None)
            else:
                setattr(self, '{}input_npy_folder'.format(catname),
                        os.path.abspath(input_npy_folder))

            setattr(self, '{}csv_has_header'.format(catname),
                    self._str2bool(config['csv_has_header']))

            # As above, extra_col_names is just strings but extra_col_names
            # is a list of integers.
            # However, both can be None (although if either is None both have to
            # be None), so check for that first
            a = config['extra_col_names']
            b = config['extra_col_nums']
            if a == 'None' and b == 'None':
                setattr(self, '{}extra_col_names'.format(catname), None)
                setattr(self, '{}extra_col_nums'.format(catname), None)
            else:
                if a == 'None' and b != 'None' or a != 'None' and b == 'None':
                    raise ValueError('Both extra_col_names and extra_col_nums must be None if '
                                     'either is None in catalogue "{}".'.format(catname[0]))
                catcolnames = config['extra_col_names'].split()
                setattr(self, '{}extra_col_names'.format(catname),
                        np.array(['{}_{}'.format(getattr(self, '{}cat_name'.format(catname)),
                                  q) for q in catcolnames]))
                a = config['extra_col_nums'].split(' ')
                try:
                    b = np.array([float(f) for f in a])
                except ValueError:
                    raise ValueError('extra_col_nums should be a list of integers '
                                     'in catalogue "{}" metadata file'.format(catname[0]))
                if len(b) != len(getattr(self, '{}extra_col_names'.format(catname))):
                    raise ValueError('{}extra_col_names and {}extra_col_nums should contain the '
                                     'same number of entries.'.format(catname, catname))
                if not np.all([c.is_integer() for c in b]):
                    raise ValueError('All elements of {}extra_col_nums should be '
                                     'integers.'.format(catname))
                setattr(self, '{}extra_col_nums'.format(catname), np.array([int(c) for c in b]))

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

    def create_perturb_auf(self, perturb_auf_func=make_perturb_aufs):
        '''
        Function wrapping the main perturbation AUF component creation routines.

        Parameters
        ----------
        perturb_auf_func : callable, optional
            ``perturb_auf_func`` should create the perturbation AUF output files
            for each filter-pointing combination.
        '''

        # Magnitude offsets corresponding to relative fluxes of perturbing sources; here
        # dm of 2.5 is 10% relative flux and dm = 5 corresponds to 1% relative flux. Used
        # to inform the fraction of simulations with a contaminant above these relative
        # fluxes.
        # TODO: allow as user input.
        self.delta_mag_cuts = np.array([2.5, 5])

        # TODO: allow as user input.
        self.gal_cmau_array = np.empty((5, 2, 4), float)
        # See Wilson (2022, RNAAS, 6, 60) for the meanings of the variables c, m,
        # a, and u. For each of M*/phi*/alpha/P/Q, for blue+red galaxies, 2-4
        # variables are derived as a function of wavelength, or Q(P).
        self.gal_cmau_array[0, :, :] = [[-24.286513, 1.141760, 2.655846, np.nan],
                                        [-23.192520, 1.778718, 1.668292, np.nan]]
        self.gal_cmau_array[1, :, :] = [[0.001487, 2.918841, 0.000510, np.nan],
                                        [0.000560, 7.691261, 0.003330, -0.065565]]
        self.gal_cmau_array[2, :, :] = [[-1.257761, 0.021362, np.nan, np.nan],
                                        [-0.309077, -0.067411, np.nan, np.nan]]
        self.gal_cmau_array[3, :, :] = [[-0.302018, 0.034203, np.nan, np.nan],
                                        [-0.713062, 0.233366, np.nan, np.nan]]
        self.gal_cmau_array[4, :, :] = [[1.233627, -0.322347, np.nan, np.nan],
                                        [1.068926, -0.385984, np.nan, np.nan]]
        self.gal_alpha0 = [[2.079, 3.524, 1.917, 1.992, 2.536], [2.461, 2.358, 2.568, 2.268, 2.402]]
        self.gal_alpha1 = [[2.265, 3.862, 1.921, 1.685, 2.480], [2.410, 2.340, 2.200, 2.540, 2.464]]
        self.gal_alphaweight = [[3.47e+09, 3.31e+06, 2.13e+09, 1.64e+10, 1.01e+09],
                                [3.84e+09, 1.57e+06, 3.91e+08, 4.66e+10, 3.03e+07]]

        t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print('{} Rank {}, chunk {}: Creating empirical perturbation AUFs for catalogue "a"...'
              .format(t, self.rank, self.chunk_id))
        sys.stdout.flush()
        if self.j0s is None:
            self.j0s = mff.calc_j0(self.rho[:-1]+self.drho/2, self.r[:-1]+self.dr/2)
        if self.include_perturb_auf:
            _kwargs = {'psf_fwhms': self.a_psf_fwhms, 'tri_download_flag': self.a_download_tri,
                       'delta_mag_cuts': self.delta_mag_cuts, 'num_trials': self.num_trials,
                       'j0s': self.j0s, 'd_mag': self.d_mag,
                       'density_radius': self.a_dens_dist,
                       'tri_filt_names': self.a_tri_filt_names,
                       'run_fw': self.a_run_fw_auf, 'run_psf': self.a_run_psf_auf,
                       'snr_mag_params': self.a_snr_mag_params,
                       'tri_maglim_faint': self.a_tri_maglim_faint,
                       'tri_num_faint': self.a_tri_num_faint,
                       'tri_set_name': self.a_tri_set_name,
                       'tri_filt_num': self.a_tri_filt_num,
                       'auf_region_frame': self.a_auf_region_frame,
                       'al_avs': self.a_gal_al_avs, 'fit_gal_flag': self.a_fit_gal_flag}
            if self.a_run_psf_auf:
                _kwargs = dict(_kwargs, **{'dd_params': self.a_dd_params,
                                           'l_cut': self.a_l_cut})
            if self.a_fit_gal_flag:
                _kwargs = dict(_kwargs,
                               **{'cmau_array': self.gal_cmau_array, 'wavs': self.a_gal_wavs,
                                  'z_maxs': self.a_gal_zmax, 'nzs': self.a_gal_nzs,
                                  'ab_offsets': self.a_gal_aboffsets,
                                  'filter_names': self.a_gal_filternames,
                                  'alpha0': self.gal_alpha0, 'alpha1': self.gal_alpha1,
                                  'alpha_weight': self.gal_alphaweight})
            else:
                _kwargs = dict(_kwargs, **{'fit_gal_flag': self.a_fit_gal_flag})
        else:
            _kwargs = {}
        self.a_modelrefinds, self.a_perturb_auf_outputs = perturb_auf_func(
            self.a_auf_folder_path, self.a_cat_folder_path, self.a_filt_names,
            self.a_auf_region_points, self.r, self.dr, self.rho, self.drho, 'a',
            self.include_perturb_auf, self.mem_chunk_num, **_kwargs)

        t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print('{} Rank {}, chunk {}: Creating empirical perturbation AUFs for catalogue "b"...'
              .format(t, self.rank, self.chunk_id))
        sys.stdout.flush()
        if self.j0s is None:
            self.j0s = mff.calc_j0(self.rho[:-1]+self.drho/2, self.r[:-1]+self.dr/2)
        if self.include_perturb_auf:
            _kwargs = {'psf_fwhms': self.b_psf_fwhms, 'tri_download_flag': self.b_download_tri,
                       'delta_mag_cuts': self.delta_mag_cuts, 'num_trials': self.num_trials,
                       'j0s': self.j0s, 'd_mag': self.d_mag,
                       'density_radius': self.b_dens_dist,
                       'tri_filt_names': self.b_tri_filt_names,
                       'run_fw': self.b_run_fw_auf, 'run_psf': self.b_run_psf_auf,
                       'snr_mag_params': self.b_snr_mag_params,
                       'tri_maglim_faint': self.b_tri_maglim_faint,
                       'tri_num_faint': self.b_tri_num_faint,
                       'tri_set_name': self.b_tri_set_name,
                       'tri_filt_num': self.b_tri_filt_num,
                       'auf_region_frame': self.b_auf_region_frame,
                       'al_avs': self.b_gal_al_avs, 'fit_gal_flag': self.b_fit_gal_flag}
            if self.b_run_psf_auf:
                _kwargs = dict(_kwargs, **{'dd_params': self.b_dd_params,
                                           'l_cut': self.b_l_cut})

            if self.b_fit_gal_flag:
                _kwargs = dict(_kwargs,
                               **{'cmau_array': self.gal_cmau_array, 'wavs': self.b_gal_wavs,
                                  'z_maxs': self.b_gal_zmax, 'nzs': self.b_gal_nzs,
                                  'ab_offsets': self.b_gal_aboffsets,
                                  'filter_names': self.b_gal_filternames,
                                  'alpha0': self.gal_alpha0, 'alpha1': self.gal_alpha1,
                                  'alpha_weight': self.gal_alphaweight})
            else:
                _kwargs = dict(_kwargs, **{'fit_gal_flag': self.b_fit_gal_flag})
        else:
            _kwargs = {}
        self.b_modelrefinds, self.b_perturb_auf_outputs = perturb_auf_func(
            self.b_auf_folder_path, self.b_cat_folder_path, self.b_filt_names,
            self.b_auf_region_points, self.r, self.dr, self.rho, self.drho, 'b',
            self.include_perturb_auf, self.mem_chunk_num, **_kwargs)

    def group_sources(self, group_func=make_island_groupings):
        '''
        Function to handle the creation of catalogue "islands" and potential
        astrometrically related sources across the two catalogues.

        Parameters
        ----------
        group_func : callable, optional
            ``group_func`` should create the various island- and overlap-related
            files by which objects across the two catalogues are assigned as
            potentially counterparts to one another.
        '''

        t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print('{} Rank {}, chunk {}: Creating catalogue islands and overlaps...'
              .format(t, self.rank, self.chunk_id))
        sys.stdout.flush()
        if self.j1s is None:
            self.j1s = gsf.calc_j1s(self.rho[:-1]+self.drho/2, self.r[:-1]+self.dr/2)
        os.system('rm -rf {}/reject/*'.format(self.joint_folder_path))
        self.group_sources_data = \
            group_func(self.joint_folder_path, self.a_cat_folder_path, self.b_cat_folder_path,
                       self.a_auf_region_points, self.b_auf_region_points, self.a_filt_names,
                       self.b_filt_names, self.a_cat_name, self.b_cat_name, self.a_modelrefinds,
                       self.b_modelrefinds, self.r, self.dr, self.rho, self.drho,
                       self.j1s, self.pos_corr_dist, self.cross_match_extent, self.int_fracs,
                       self.mem_chunk_num, self.include_phot_like, self.use_phot_priors,
                       self.n_pool, self.a_perturb_auf_outputs, self.b_perturb_auf_outputs)

    def calculate_phot_like(self, phot_like_func=compute_photometric_likelihoods):
        '''
        Create the photometric likelihood information used in the cross-match
        process.

        Parameters
        ----------
        phot_like_func : callable, optional
            The function that calls the overall computation of the counterpart
            and "field" star photometric likelihood-related information.
        '''

        t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print('{} Rank {}, chunk {}: Creating photometric priors and likelihoods...'
              .format(t, self.rank, self.chunk_id))
        sys.stdout.flush()
        self._calculate_cf_areas()
        if self.use_phot_priors or self.include_phot_like:
            bright_frac = self.int_fracs[0]
            field_frac = self.int_fracs[1]
        else:
            bright_frac = None
            field_frac = None
        self.phot_like_data = phot_like_func(
            self.joint_folder_path, self.a_cat_folder_path, self.b_cat_folder_path,
            self.a_filt_names, self.b_filt_names, self.mem_chunk_num, self.cf_region_points,
            self.cf_areas, self.include_phot_like, self.use_phot_priors, self.group_sources_data,
            bright_frac, field_frac)

    def _calculate_cf_areas(self):
        '''
        Convenience function to calculate the area around each
        ``cross_match_extent`` sky coordinate where it is defined as having the
        smallest on-sky separation.
        '''
        t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("{} Rank {}, chunk {}: Calculating photometric region areas..."
              .format(t, self.rank, self.chunk_id))
        dlon, dlat = 0.01, 0.01
        test_lons = np.arange(self.cross_match_extent[0], self.cross_match_extent[1], dlon)
        test_lats = np.arange(self.cross_match_extent[2], self.cross_match_extent[3], dlat)

        test_coords = np.array([[a, b] for a in test_lons for b in test_lats])

        inds = mff.find_nearest_point(test_coords[:, 0], test_coords[:, 1],
                                      self.cf_region_points[:, 0], self.cf_region_points[:, 1])

        cf_areas = np.zeros((len(self.cf_region_points)), float)

        # Unit area of a sphere is cos(theta) dtheta dphi if theta goes from -90
        # to +90 degrees (sin(theta) for 0 to 180 degrees). Note, however, that
        # dtheta and dphi have to be in radians, so we have to convert the entire
        # thing from degrees and re-convert at the end. Hence:
        for i, ind in enumerate(inds):
            theta = np.radians(test_coords[i, 1])
            dtheta, dphi = dlat / 180 * np.pi, dlon / 180 * np.pi
            # Remember to convert back to square degrees:
            cf_areas[ind] += (np.cos(theta) * dtheta * dphi) * (180 / np.pi)**2

        self.cf_areas = cf_areas

        return

    def pair_sources(self, count_pair_func=source_pairing):
        '''
        Assign sources in the two catalogues as either counterparts to one another
        or singly detected "field" sources.

        Parameters
        ----------
        count_pair_func : callable, optional
            The function that calls the counterpart determination routine.
        '''

        t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print('{} Rank {}, chunk {}: Determining counterparts...'
              .format(t, self.rank, self.chunk_id))
        sys.stdout.flush()
        os.system('rm -r {}/pairing/*'.format(self.joint_folder_path))
        count_pair_func(
            self.joint_folder_path, self.a_cat_folder_path, self.b_cat_folder_path,
            self.a_filt_names, self.b_filt_names, self.a_auf_region_points,
            self.b_auf_region_points, self.a_modelrefinds, self.b_modelrefinds, self.rho, self.drho,
            len(self.delta_mag_cuts), self.mem_chunk_num, self.group_sources_data,
            self.phot_like_data, self.a_perturb_auf_outputs, self.b_perturb_auf_outputs)
