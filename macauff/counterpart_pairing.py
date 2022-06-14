# Licensed under a 3-clause BSD style license - see LICENSE
'''
This module provides the functionality for the final cross-match process, the
act of actually pairing sources across the two catalogues as counterparts.
'''

import os
import sys
import numpy as np
import warnings

from .misc_functions import load_small_ref_auf_grid
from .counterpart_pairing_fortran import counterpart_pairing_fortran as cpf

__all__ = ['source_pairing']


def source_pairing(joint_folder_path, a_cat_folder_path, b_cat_folder_path, a_auf_folder_path,
                   b_auf_folder_path, a_filt_names, b_filt_names, a_auf_pointings, b_auf_pointings,
                   a_modelrefinds, b_modelrefinds, alist_in, blist_in, agrplen_in, bgrplen_in, rho,
                   drho, n_fracs, mem_chunk_num, use_memmap_files=False):
    '''
    Function to iterate over all grouped islands of sources, calculating the
    probabilities of all permutations of matches and deriving the most likely
    counterparts for sources in the two catalogues.

    Parameters
    ----------
    joint_folder_path : string
        Folder in which all common match files are stored for this crossmatch.
    a_cat_folder_path : string
        Folder in which the "a" catalogue input catalogues are stored.
    b_cat_folder_path : string
        Folder where catalogue "b" files are located.
    a_auf_folder_path : string
        Folder where catalogue "a" perturbation AUF component files were saved
        previously.
    b_auf_folder_path : string
        Folder containing catalogue "b" perturbation AUF component files.
    a_filt_names : numpy.ndarray or list of strings
        Array or list containing names of the filters used in catalogue "a".
    b_filt_names : numpy.ndarray or list of strings
        Array or list of catalogue "b" filter names.
    a_auf_pointings : numpy.ndarray
        Array of celestial coordinates indicating the locations used in the
        simulations of perturbation AUFs for catalogue "a".
    b_auf_pointings : numpy.ndarray
        Sky coordinates of locations of catalogue "b" perturbation AUF
        component simulations.
    a_modelrefinds : numpy.ndarray
        Catalogue "a" modelrefinds array output from ``create_perturb_auf``. Used
        only when use_memmap_files is False.
        TODO Improve description
    b_modelrefinds : numpy.ndarray
        Catalogue "b" modelrefinds array output from ``create_perturb_auf``. Used
        only when use_memmap_files is False.
        TODO Improve description
    alist_in : numpy.ndarray
        Catalogue "a" alist output from ``make_island_groupings``. Used only
        when use_memmap_files is False.
        TODO Improve description
    blist_in : numpy.ndarray
        Catalogue "b" blist output from ``make_island_groupings``. Used only
        when use_memmap_files is False.
        TODO Improve description
    agrplen_in : numpy.ndarray
        Catalogue "a" agrplen output from ``make_island_groupings``. Used only
        when use_memmap_files is False.
        TODO Improve description
    bgrplen_in : numpy.ndarray
        Catalogue "b" bgrplen output from ``make_island_groupings``. Used only
        when use_memmap_files is False.
        TODO Improve description
    rho : numpy.ndarray
        Array of fourier-space values, used in the convolution of PDFs.
    drho : numpy.ndarray
        The spacings between the `rho` array elements.
    n_fracs : integer
        The number of relative contamination fluxes previously considered
        when calculating the probability of a source being contaminated by
        a perturbing source brighter than a given flux.
    mem_chunk_num : integer
        Number of sub-arrays to break loading of main catalogue into, to
        reduce the amount of memory used.
    use_memmap_files : boolean, optional
        When set to True, memory mapped files are used for several internal
        arrays. Reduces memory consumption at the cost of increased I/O
        contention.
    '''
    print("Creating catalogue matches...")
    sys.stdout.flush()

    print("Pairing sources...")
    sys.stdout.flush()

    if use_memmap_files:
        isle_len = np.load('{}/group/alist.npy'.format(joint_folder_path), mmap_mode='r').shape[1]
    else:
        isle_len = alist_in.shape[1]

    match_chunk_lengths = np.empty(mem_chunk_num, int)
    afield_chunk_lengths = np.empty(mem_chunk_num, int)
    bfield_chunk_lengths = np.empty(mem_chunk_num, int)

    match_chunk_lengths[0] = 0
    afield_chunk_lengths[0] = 0
    bfield_chunk_lengths[0] = 0
    cprt_max_len, len_a, len_b = 0, 0, 0
    for cnum in range(0, mem_chunk_num):
        lowind = np.floor(isle_len*cnum/mem_chunk_num).astype(int)
        highind = np.floor(isle_len*(cnum+1)/mem_chunk_num).astype(int)

        if use_memmap_files:
            agrplen = np.load('{}/group/agrplen.npy'.format(joint_folder_path),
                            mmap_mode='r')[lowind:highind]
            bgrplen = np.load('{}/group/bgrplen.npy'.format(joint_folder_path),
                            mmap_mode='r')[lowind:highind]
        else:
            agrplen = agrplen_in[lowind:highind]
            bgrplen = bgrplen_in[lowind:highind]

        sum_agrp, sum_bgrp = np.sum(agrplen), np.sum(bgrplen)
        match_lens = np.sum(np.minimum(agrplen, bgrplen))
        if cnum < mem_chunk_num-1:
            match_chunk_lengths[cnum+1] = match_chunk_lengths[cnum] + match_lens
            afield_chunk_lengths[cnum+1] = afield_chunk_lengths[cnum] + sum_agrp
            bfield_chunk_lengths[cnum+1] = bfield_chunk_lengths[cnum] + sum_bgrp

        len_a += sum_agrp
        len_b += sum_bgrp
        cprt_max_len += match_lens

    # Assume that the counterparts, at 100% match rate, can't be more than all
    # of the items in the smaller of the two *list arrays.
    acountinds = np.lib.format.open_memmap('{}/pairing/ac.npy'.format(joint_folder_path),
                                           mode='w+', dtype=int, shape=(cprt_max_len,))
    bcountinds = np.lib.format.open_memmap('{}/pairing/bc.npy'.format(joint_folder_path),
                                           mode='w+', dtype=int, shape=(cprt_max_len,))
    acontamprob = np.lib.format.open_memmap('{}/pairing/pacontam.npy'.format(joint_folder_path),
                                            mode='w+', dtype=float, shape=(n_fracs, cprt_max_len),
                                            fortran_order=True)
    bcontamprob = np.lib.format.open_memmap('{}/pairing/pbcontam.npy'.format(joint_folder_path),
                                            mode='w+', dtype=float, shape=(n_fracs, cprt_max_len),
                                            fortran_order=True)
    acontamflux = np.lib.format.open_memmap('{}/pairing/acontamflux.npy'.format(joint_folder_path),
                                            mode='w+', dtype=float, shape=(cprt_max_len,))
    bcontamflux = np.lib.format.open_memmap('{}/pairing/bcontamflux.npy'.format(joint_folder_path),
                                            mode='w+', dtype=float, shape=(cprt_max_len,))
    probcarray = np.lib.format.open_memmap('{}/pairing/pc.npy'.format(joint_folder_path),
                                           mode='w+', dtype=float, shape=(cprt_max_len,))
    etaarray = np.lib.format.open_memmap('{}/pairing/eta.npy'.format(joint_folder_path),
                                         mode='w+', dtype=float, shape=(cprt_max_len,))
    xiarray = np.lib.format.open_memmap('{}/pairing/xi.npy'.format(joint_folder_path),
                                        mode='w+', dtype=float, shape=(cprt_max_len,))
    crptseps = np.lib.format.open_memmap('{}/pairing/crptseps.npy'.format(joint_folder_path),
                                         mode='w+', dtype=float, shape=(cprt_max_len,))
    afieldinds = np.lib.format.open_memmap('{}/pairing/af.npy'.format(joint_folder_path),
                                           mode='w+', dtype=int, shape=(len_a,))
    probfaarray = np.lib.format.open_memmap('{}/pairing/pfa.npy'.format(joint_folder_path),
                                            mode='w+', dtype=float, shape=(len_a,))
    afieldflux = np.lib.format.open_memmap('{}/pairing/afieldflux.npy'.format(joint_folder_path),
                                           mode='w+', dtype=float, shape=(len_a,))
    afieldseps = np.lib.format.open_memmap('{}/pairing/afieldseps.npy'.format(joint_folder_path),
                                           mode='w+', dtype=float, shape=(len_a,))
    afieldeta = np.lib.format.open_memmap('{}/pairing/afieldeta.npy'.format(joint_folder_path),
                                          mode='w+', dtype=float, shape=(len_a,))
    afieldxi = np.lib.format.open_memmap('{}/pairing/afieldxi.npy'.format(joint_folder_path),
                                         mode='w+', dtype=float, shape=(len_a,))
    bfieldinds = np.lib.format.open_memmap('{}/pairing/bf.npy'.format(joint_folder_path),
                                           mode='w+', dtype=int, shape=(len_b,))
    probfbarray = np.lib.format.open_memmap('{}/pairing/pfb.npy'.format(joint_folder_path),
                                            mode='w+', dtype=float, shape=(len_b,))
    bfieldflux = np.lib.format.open_memmap('{}/pairing/bfieldflux.npy'.format(joint_folder_path),
                                           mode='w+', dtype=float, shape=(len_b,))
    bfieldseps = np.lib.format.open_memmap('{}/pairing/bfieldseps.npy'.format(joint_folder_path),
                                           mode='w+', dtype=float, shape=(len_b,))
    bfieldeta = np.lib.format.open_memmap('{}/pairing/bfieldeta.npy'.format(joint_folder_path),
                                          mode='w+', dtype=float, shape=(len_b,))
    bfieldxi = np.lib.format.open_memmap('{}/pairing/bfieldxi.npy'.format(joint_folder_path),
                                         mode='w+', dtype=float, shape=(len_b,))

    abinsarray = np.load('{}/phot_like/abinsarray.npy'.format(joint_folder_path), mmap_mode='r')
    abinlengths = np.load('{}/phot_like/abinlengths.npy'.format(joint_folder_path), mmap_mode='r')
    bbinsarray = np.load('{}/phot_like/bbinsarray.npy'.format(joint_folder_path), mmap_mode='r')
    bbinlengths = np.load('{}/phot_like/bbinlengths.npy'.format(joint_folder_path), mmap_mode='r')

    c_priors = np.load('{}/phot_like/c_priors.npy'.format(joint_folder_path), mmap_mode='r')
    c_array = np.load('{}/phot_like/c_array.npy'.format(joint_folder_path), mmap_mode='r')
    fa_priors = np.load('{}/phot_like/fa_priors.npy'.format(joint_folder_path), mmap_mode='r')
    fa_array = np.load('{}/phot_like/fa_array.npy'.format(joint_folder_path), mmap_mode='r')
    fb_priors = np.load('{}/phot_like/fb_priors.npy'.format(joint_folder_path), mmap_mode='r')
    fb_array = np.load('{}/phot_like/fb_array.npy'.format(joint_folder_path), mmap_mode='r')

    big_len_a = len(np.load('{}/con_cat_astro.npy'.format(a_cat_folder_path), mmap_mode='r'))
    big_len_b = len(np.load('{}/con_cat_astro.npy'.format(b_cat_folder_path), mmap_mode='r'))
    # large_len is the "safe" initialisation value for arrays, such that no index
    # can ever reach this value.
    large_len = max(big_len_a, big_len_b)

    for cnum in range(0, mem_chunk_num):
        lowind = np.floor(isle_len*cnum/mem_chunk_num).astype(int)
        highind = np.floor(isle_len*(cnum+1)/mem_chunk_num).astype(int)

        if use_memmap_files:
            alist_ = np.load('{}/group/alist.npy'.format(joint_folder_path),
                            mmap_mode='r')[:, lowind:highind]
            agrplen = np.load('{}/group/agrplen.npy'.format(joint_folder_path),
                            mmap_mode='r')[lowind:highind]
        else:
            alist_ = alist_in[:, lowind:highind]
            agrplen = agrplen_in[lowind:highind]

        alist_ = np.asfortranarray(alist_[:np.amax(agrplen), :])
        alistunique_flat = np.unique(alist_[alist_ > -1])
        a_astro = np.load('{}/con_cat_astro.npy'.format(a_cat_folder_path),
                          mmap_mode='r')[alistunique_flat]
        a_photo = np.load('{}/con_cat_photo.npy'.format(a_cat_folder_path),
                          mmap_mode='r')[alistunique_flat]
        amagref = np.load('{}/magref.npy'.format(a_cat_folder_path),
                          mmap_mode='r')[alistunique_flat]
        maparray = -1*np.ones(big_len_a+1).astype(int)
        maparray[alistunique_flat] = np.arange(0, len(a_astro), dtype=int)
        # *list maps the subarray indices, but *list_ keeps the full catalogue indices
        alist = np.asfortranarray(maparray[alist_.flatten()].reshape(alist_.shape))

        a_sky_inds = np.load('{}/phot_like/a_sky_inds.npy'.format(joint_folder_path),
                             mmap_mode='r')[alistunique_flat]

        if use_memmap_files:
            blist_ = np.load('{}/group/blist.npy'.format(joint_folder_path),
                            mmap_mode='r')[:, lowind:highind]
            bgrplen = np.load('{}/group/bgrplen.npy'.format(joint_folder_path),
                            mmap_mode='r')[lowind:highind]
        else:
            blist_ = blist_in[:, lowind:highind]
            bgrplen = bgrplen_in[lowind:highind]

        blist_ = np.asfortranarray(blist_[:np.amax(bgrplen), :])
        blistunique_flat = np.unique(blist_[blist_ > -1])
        b_astro = np.load('{}/con_cat_astro.npy'.format(b_cat_folder_path),
                          mmap_mode='r')[blistunique_flat]
        b_photo = np.load('{}/con_cat_photo.npy'.format(b_cat_folder_path),
                          mmap_mode='r')[blistunique_flat]
        bmagref = np.load('{}/magref.npy'.format(b_cat_folder_path),
                          mmap_mode='r')[blistunique_flat]
        maparray = -1*np.ones(big_len_b+1).astype(int)
        maparray[blistunique_flat] = np.arange(0, len(b_astro), dtype=int)
        blist = np.asfortranarray(maparray[blist_.flatten()].reshape(blist_.shape))

        b_sky_inds = np.load('{}/phot_like/b_sky_inds.npy'.format(joint_folder_path),
                             mmap_mode='r')[blistunique_flat]

        if use_memmap_files:
            amodrefind = np.load('{}/modelrefinds.npy'.format(a_auf_folder_path),
                                mmap_mode='r')[:, alistunique_flat]
        else:
            amodrefind = a_modelrefinds
        [afourier_grids, afrac_grids, aflux_grids], amodrefind = load_small_ref_auf_grid(
            amodrefind, a_auf_folder_path, ['fourier', 'frac', 'flux'])

        if use_memmap_files:
            bmodrefind = np.load('{}/modelrefinds.npy'.format(b_auf_folder_path),
                                mmap_mode='r')[:, blistunique_flat]
        else:
            bmodrefind = b_modelrefinds
        [bfourier_grids, bfrac_grids, bflux_grids], bmodrefind = load_small_ref_auf_grid(
            bmodrefind, b_auf_folder_path, ['fourier', 'frac', 'flux'])

        # Similar to crpts_max_len, mini_crpts_len is the maximum number of
        # counterparts at 100% match rate for this cutout.
        mini_crpts_len = np.sum(np.minimum(agrplen, bgrplen))

        (_acountinds, _bcountinds, _afieldinds, _bfieldinds, _acontamprob, _bcontamprob, _etaarray,
         _xiarray, _acontamflux, _bcontamflux, _probcarray, _crptseps, _probfaarray, _afieldfluxs,
         _afieldseps, _afieldetas, _afieldxis, _probfbarray, _bfieldfluxs, _bfieldseps, _bfieldetas,
         _bfieldxis) = cpf.find_island_probabilities(
            a_astro, a_photo, b_astro, b_photo, alist, alist_, blist, blist_, agrplen, bgrplen,
            c_array, fa_array, fb_array, c_priors, fa_priors, fb_priors, amagref, bmagref,
            amodrefind, bmodrefind, abinsarray, abinlengths, bbinsarray, bbinlengths, afrac_grids,
            aflux_grids, bfrac_grids, bflux_grids, afourier_grids, bfourier_grids, a_sky_inds,
            b_sky_inds, rho, drho, n_fracs, large_len, mini_crpts_len)

        ind_start, ind_end = match_chunk_lengths[cnum], match_chunk_lengths[cnum]+len(_acountinds)
        acountinds[ind_start:ind_end] = _acountinds
        bcountinds[ind_start:ind_end] = _bcountinds
        acontamprob[:, ind_start:ind_end] = _acontamprob
        bcontamprob[:, ind_start:ind_end] = _bcontamprob
        etaarray[ind_start:ind_end] = _etaarray
        xiarray[ind_start:ind_end] = _xiarray
        acontamflux[ind_start:ind_end] = _acontamflux
        bcontamflux[ind_start:ind_end] = _bcontamflux
        probcarray[ind_start:ind_end] = _probcarray
        crptseps[ind_start:ind_end] = _crptseps

        ind_start, ind_end = afield_chunk_lengths[cnum], afield_chunk_lengths[cnum]+len(_afieldinds)
        afieldinds[ind_start:ind_end] = _afieldinds
        probfaarray[ind_start:ind_end] = _probfaarray
        afieldflux[ind_start:ind_end] = _afieldfluxs
        afieldseps[ind_start:ind_end] = _afieldseps
        afieldeta[ind_start:ind_end] = _afieldetas
        afieldxi[ind_start:ind_end] = _afieldxis

        ind_start, ind_end = bfield_chunk_lengths[cnum], bfield_chunk_lengths[cnum]+len(_bfieldinds)
        bfieldinds[ind_start:ind_end] = _bfieldinds
        probfbarray[ind_start:ind_end] = _probfbarray
        bfieldflux[ind_start:ind_end] = _bfieldfluxs
        bfieldseps[ind_start:ind_end] = _bfieldseps
        bfieldeta[ind_start:ind_end] = _bfieldetas
        bfieldxi[ind_start:ind_end] = _bfieldxis

    countfilter = np.lib.format.open_memmap('{}/pairing/countfilt.npy'.format(joint_folder_path),
                                            mode='w+', dtype=bool, shape=(cprt_max_len,))
    afieldfilter = np.lib.format.open_memmap('{}/pairing/afieldfilt.npy'.format(joint_folder_path),
                                             mode='w+', dtype=bool, shape=(len_a,))
    bfieldfilter = np.lib.format.open_memmap('{}/pairing/bfieldfilt.npy'.format(joint_folder_path),
                                             mode='w+', dtype=bool, shape=(len_b,))
    for cnum in range(0, mem_chunk_num):
        lowind = np.floor(cprt_max_len*cnum/mem_chunk_num).astype(int)
        highind = np.floor(cprt_max_len*(cnum+1)/mem_chunk_num).astype(int)
        # *contamprob is (smalllen, nfracs) in shape and our check for correctness needs to check
        # all nfrac values, requiring an all check.
        countfilter[lowind:highind] = (
            (acountinds[lowind:highind] < large_len+1) &
            (bcountinds[lowind:highind] < large_len+1) &
            np.all(acontamprob[:, lowind:highind] >= 0, axis=0) &
            np.all(bcontamprob[:, lowind:highind] >= 0, axis=0) &
            (acontamflux[lowind:highind] >= 0) & (bcontamflux[lowind:highind] >= 0) &
            (probcarray[lowind:highind] >= 0) & (etaarray[lowind:highind] >= -10) &
            (xiarray[lowind:highind] >= -10))

        lowind = np.floor(len_a*cnum/mem_chunk_num).astype(int)
        highind = np.floor(len_a*(cnum+1)/mem_chunk_num).astype(int)
        afieldfilter[lowind:highind] = ((afieldinds[lowind:highind] < large_len+1) &
                                        (probfaarray[lowind:highind] >= 0))

        lowind = np.floor(len_b*cnum/mem_chunk_num).astype(int)
        highind = np.floor(len_b*(cnum+1)/mem_chunk_num).astype(int)
        bfieldfilter[lowind:highind] = ((bfieldinds[lowind:highind] < large_len+1) &
                                        (probfbarray[lowind:highind] >= 0))
    if os.path.isfile('{}/reject/reject_a.npy'.format(joint_folder_path)):
        lenrejecta = len(np.load('{}/reject/reject_a.npy'.format(joint_folder_path),
                                 mmap_mode='r'))
    else:
        lenrejecta = 0
    if os.path.isfile('{}/reject/reject_b.npy'.format(joint_folder_path)):
        lenrejectb = len(np.load('{}/reject/reject_b.npy'.format(joint_folder_path),
                                 mmap_mode='r'))
    else:
        lenrejectb = 0

    countsum = int(np.sum(countfilter))
    afieldsum = int(np.sum(afieldfilter))
    bfieldsum = int(np.sum(bfieldfilter))

    # Reduce size of output files, removing anything that doesn't meet the
    # criteria above from all saved numpy arrays.
    for file_name, variable, small_shape, typing, filter_variable in zip(
        ['ac', 'bc', 'pacontam', 'pbcontam', 'acontamflux', 'bcontamflux', 'af', 'bf', 'pc', 'eta',
         'xi', 'pfa', 'pfb', 'afieldflux', 'bfieldflux', 'crptseps', 'afieldseps', 'afieldeta',
         'afieldxi', 'bfieldseps', 'bfieldeta', 'bfieldxi'],
        [acountinds, bcountinds, acontamprob, bcontamprob, acontamflux, bcontamflux, afieldinds,
         bfieldinds, probcarray, etaarray, xiarray, probfaarray, probfbarray, afieldflux,
         bfieldflux, crptseps, afieldseps, afieldeta, afieldxi, bfieldseps, bfieldeta, bfieldxi],
        [(countsum,), (countsum,), (n_fracs, countsum), (n_fracs, countsum), (countsum,),
         (countsum,), (afieldsum,), (bfieldsum,), (countsum,), (countsum,), (countsum,),
         (afieldsum,), (bfieldsum,), (afieldsum,), (bfieldsum,), (countsum,), (afieldsum,),
         (afieldsum,), (afieldsum,), (bfieldsum,), (bfieldsum,), (bfieldsum,)],
        [int, int, float, float, float, float, int, int, float, float, float, float, float,
         float, float, float, float, float, float, float, float, float],
        [countfilter, countfilter, countfilter, countfilter, countfilter, countfilter,
         afieldfilter, bfieldfilter, countfilter, countfilter, countfilter, afieldfilter,
         bfieldfilter, afieldfilter, bfieldfilter, countfilter, afieldfilter, afieldfilter,
         afieldfilter, bfieldfilter, bfieldfilter, bfieldfilter]):
        temp_variable = np.lib.format.open_memmap('{}/pairing/{}2.npy'.format(
            joint_folder_path, file_name), mode='w+', dtype=typing, shape=small_shape)
        if file_name == 'pacontam' or file_name == 'pbcontam':
            large_shape = variable.shape[1]
        else:
            large_shape = variable.shape[0]
        di = max(1, large_shape // 20)
        temp_c = 0
        if file_name == 'pacontam' or file_name == 'pbcontam':
            for i in range(0, large_shape, di):
                n_extra = int(np.sum(filter_variable[i:i+di]))
                temp_variable[:, temp_c:temp_c+n_extra] = variable[:, i:i+di][:, filter_variable[i:i+di]]
                temp_c += n_extra
        else:
            for i in range(0, large_shape, di):
                n_extra = int(np.sum(filter_variable[i:i+di]))
                temp_variable[temp_c:temp_c+n_extra] = variable[i:i+di][filter_variable[i:i+di]]
                temp_c += n_extra
        os.system('mv {}/pairing/{}2.npy {}/pairing/{}.npy'.format(joint_folder_path, file_name,
                  joint_folder_path, file_name))
    del acountinds, bcountinds, acontamprob, bcontamprob, acontamflux, bcontamflux, afieldinds
    del bfieldinds, probcarray, etaarray, xiarray, probfaarray, probfbarray
    del afieldflux, bfieldflux
    del crptseps, afieldseps, afieldeta, afieldxi, bfieldseps, bfieldeta, bfieldxi
    tot = countsum + afieldsum + lenrejecta
    if tot < big_len_a:
        warnings.warn("{} catalogue a source{} not in either counterpart, field, or rejected "
                      "source lists.".format(big_len_a - tot, 's' if big_len_a - tot > 1 else ''))
    if tot > big_len_a:
        warnings.warn("{} additional catalogue a {} recorded, check results for duplications "
                      "carefully".format(tot - big_len_a, 'indices' if tot - big_len_a > 1 else
                                         'index'))
    tot = countsum + bfieldsum + lenrejectb
    if tot < big_len_b:
        warnings.warn("{} catalogue b source{} not in either counterpart, field, or rejected "
                      "source lists.".format(big_len_b - tot, 's' if big_len_b - tot > 1 else ''))
    if tot > big_len_b:
        warnings.warn("{} additional catalogue b {} recorded, check results for duplications "
                      "carefully".format(tot - big_len_b, 'indices' if tot - big_len_b > 1 else
                                         'index'))
    sys.stdout.flush()

    del countfilter, afieldfilter, bfieldfilter
    os.remove('{}/pairing/countfilt.npy'.format(joint_folder_path))
    os.remove('{}/pairing/afieldfilt.npy'.format(joint_folder_path))
    os.remove('{}/pairing/bfieldfilt.npy'.format(joint_folder_path))

    return
