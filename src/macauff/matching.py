# Licensed under a 3-clause BSD style license - see LICENSE
'''
This module provides the high-level framework for performing catalogue-catalogue cross-matches.
'''
# pylint: disable=too-many-lines
# pylint: disable=duplicate-code

import datetime
import os
import sys
from importlib import resources
from time import sleep

import numpy as np

try:
    from mpi4py import MPI
except ModuleNotFoundError:
    MPI = None

from macauff.counterpart_pairing import source_pairing
from macauff.fit_astrometry import derive_astrometric_corrections
from macauff.group_sources import make_island_groupings
from macauff.macauff import Macauff
from macauff.parse_catalogue import load_csv, npy_to_csv
from macauff.perturbation_auf import make_perturb_aufs
from macauff.photometric_likelihood import compute_photometric_likelihoods
from macauff.read_metadata import read_metadata

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

        self.load_psf_auf_params()

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
            self.crossmatch_params_dict, self.cat_a_params_dict, self.cat_b_params_dict = read_metadata(self)
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

    def load_psf_auf_params(self):
        '''
        Load PSF AUF parameters from package data to class attributes.
        '''
        # Only need dd_params or l_cut if we're using run_psf_auf or
        # correct_astrometry is True.
        for name in ['dd_params', 'l_cut']:
            with resources.files("macauff.data").joinpath(f"{name}.npy").open("rb") as f:
                a = np.load(f)
                if name == 'dd_params' and not (len(a.shape) == 3 and a.shape[0] == 5 and a.shape[2] == 2):
                    raise ValueError('dd_params should be of shape (5, X, 2).')
                if name == 'l_cut' and not (len(a.shape) == 1 and a.shape[0] == 3):
                    raise ValueError('l_cut should be of shape (3,) only.')
                setattr(self, name, a)

    def _initialise_chunk(self):  # pylint: disable=too-many-branches,too-many-statements
        '''
        Initialisation function for a single chunk of sky.
        '''

        # If astrometry of either catalogue needs fixing, do that now.
        if self.a_correct_astrometry:
            derive_astrometric_corrections(self, 'a')
        load_csv(self, 'a')
        if self.b_correct_astrometry:
            derive_astrometric_corrections(self, 'b')
        load_csv(self, 'b')

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

    def _load_metadata_config_files(self):
        '''
        Load per-chunk class variables from the paths in the stored parameter
        metadata files.
        '''
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

        for config, flag in zip([self.cat_a_params_dict, self.cat_b_params_dict], ['a_', 'b_']):
            if self.crossmatch_params_dict['include_perturb_auf'] or config['correct_astrometry']:
                for name in ['dens_hist_tri', 'tri_model_mags', 'tri_model_mags_interval',
                             'tri_n_bright_sources_star']:
                    # If location variable was "None" in the first place we set
                    # {name}_list in config to a list of Nones and it got updated
                    # above already.
                    if config[f'{name}_location'] != "None":
                        setattr(self, f'{flag}{name}_list', np.load(config[f'{name}_location']))

    def _load_metadata_config_params(self, chunk_id):
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
            p = _make_regions_points([f'{catname}auf_region_type', config['auf_region_type']],
                                     [f'{catname}auf_region_points',
                                      config['auf_region_points_per_chunk'][ind]],
                                     config['chunk_id_list'][ind])
            setattr(self, f'{catname}auf_region_points', p)  # pylint: disable=possibly-used-before-assignment

        ind = np.where(chunk_id == np.array(self.crossmatch_params_dict['chunk_id_list']))[0][0]
        p = _make_regions_points(['cf_region_type', self.crossmatch_params_dict['cf_region_type']],
                                 ['cf_region_points',
                                  self.crossmatch_params_dict['cf_region_points_per_chunk'][ind]],
                                 self.crossmatch_params_dict['chunk_id_list'][ind])
        setattr(self, 'cf_region_points', p)  # pylint: disable=possibly-used-before-assignment

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
        self._load_metadata_config_params(chunk_id)
        self._load_metadata_config_files()

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


def _make_regions_points(region_type, region_points, chunk_id):
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

    Returns
    -------
    points : numpy.ndarray
        An array of shape (N, 2), with each second-axis pair being a single
        sky coordinate for each of the N pointings.
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

    return points
