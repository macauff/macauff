# Licensed under a 3-clause BSD style license - see LICENSE
'''
This module provides the functionality for the final cross-match process, the
act of actually pairing sources across the two catalogues as counterparts.
'''

import sys
import numpy as np

from .misc_functions import (create_auf_params_grid, load_small_ref_auf_grid)


def source_pairing(joint_folder_path, a_cat_folder_path, b_cat_folder_path, a_auf_folder_path,
                   b_auf_folder_path, a_filt_names, b_filt_names, a_auf_pointings, b_auf_pointings,
                   rho, delta_mag_cuts, mem_chunk_num, n_pool):
    print("Creating catalogue matches...")
    sys.stdout.flush()

    # Create the estimated levels of flux contamination and fraction of
    # contaminated source grids.
    n_fracs = len(delta_mag_cuts)
    create_auf_params_grid(a_auf_folder_path, a_auf_pointings, a_filt_names, 'frac', n_fracs)
    create_auf_params_grid(a_auf_folder_path, a_auf_pointings, a_filt_names, 'flux')
    create_auf_params_grid(b_auf_folder_path, b_auf_pointings, b_filt_names, 'frac', n_fracs)
    create_auf_params_grid(b_auf_folder_path, b_auf_pointings, b_filt_names, 'flux')

    print("Pairing stars...")
    sys.stdout.flush()
    len_a = len(np.load('{}/con_cat_astro.npy'.format(a_cat_folder_path), mmap_mode='r'))
    len_b = len(np.load('{}/con_cat_astro.npy'.format(b_cat_folder_path), mmap_mode='r'))
    small_len, large_len = min(len_a, len_b), max(len_a, len_b)

    acountinds = np.lib.format.open_memmap('{}/pairing/ac.npy'.format(joint_folder_path),
                                           mode='w+', dtype=int, shape=(small_len,))
    bcountinds = np.lib.format.open_memmap('{}/pairing/bc.npy'.format(joint_folder_path),
                                           mode='w+', dtype=int, shape=(small_len,))
    acontamprob = np.lib.format.open_memmap('{}/pairing/pacontam.npy'.format(joint_folder_path),
                                            mode='w+', dtype=float, shape=(small_len, n_fracs))
    bcontamprob = np.lib.format.open_memmap('{}/pairing/pbcontam.npy'.format(joint_folder_path),
                                            mode='w+', dtype=float, shape=(small_len, n_fracs))
    acontamflux = np.lib.format.open_memmap('{}/pairing/acontamflux.npy'.format(joint_folder_path),
                                            mode='w+', dtype=float, shape=(small_len,))
    bcontamflux = np.lib.format.open_memmap('{}/pairing/bcontamflux.npy'.format(joint_folder_path),
                                            mode='w+', dtype=float, shape=(small_len,))
    probcarray = np.lib.format.open_memmap('{}/pairing/pc.npy'.format(joint_folder_path),
                                           mode='w+', dtype=float, shape=(small_len,))
    etaarray = np.lib.format.open_memmap('{}/pairing/eta.npy'.format(joint_folder_path),
                                         mode='w+', dtype=float, shape=(small_len,))
    xiarray = np.lib.format.open_memmap('{}/pairing/xi.npy'.format(joint_folder_path),
                                        mode='w+', dtype=float, shape=(small_len,))
    for cnum in range(0, mem_chunk_num):
        lowind = np.floor(small_len*cnum/mem_chunk_num).astype(int)
        highind = np.floor(small_len*(cnum+1)/mem_chunk_num).astype(int)
        acountinds[lowind:highind] = large_len+1
        bcountinds[lowind:highind] = large_len+1
        acontamprob[lowind:highind, :] = -100
        bcontamprob[lowind:highind, :] = -100
        acontamflux[lowind:highind] = -100
        bcontamflux[lowind:highind] = -100
        probcarray[lowind:highind] = -100
        etaarray[lowind:highind] = -100
        xiarray[lowind:highind] = -100
    afieldinds = np.lib.format.open_memmap('{}/pairing/af.npy'.format(joint_folder_path),
                                           mode='w+', dtype=int, shape=(len_a,))
    probfaarray = np.lib.format.open_memmap('{}/pairing/pfa.npy'.format(joint_folder_path),
                                            mode='w+', dtype=float, shape=(len_a,))
    for cnum in range(0, mem_chunk_num):
        lowind = np.floor(len_a*cnum/mem_chunk_num).astype(int)
        highind = np.floor(len_a*(cnum+1)/mem_chunk_num).astype(int)
        afieldinds[lowind:highind] = large_len+1
        probfaarray[lowind:highind] = -100
    bfieldinds = np.lib.format.open_memmap('{}/pairing/bf.npy'.format(joint_folder_path),
                                           mode='w+', dtype=int, shape=(len_b,))
    probfbarray = np.lib.format.open_memmap('{}/pairing/pfb.npy'.format(joint_folder_path),
                                            mode='w+', dtype=float, shape=(len_b,))
    for cnum in range(0, mem_chunk_num):
        lowind = np.floor(len_b*cnum/mem_chunk_num).astype(int)
        highind = np.floor(len_b*(cnum+1)/mem_chunk_num).astype(int)
        bfieldinds[lowind:highind] = large_len+1
        probfbarray[lowind:highind] = -100

    counterpartticker = 0
    afieldticker = 0
    bfieldticker = 0

    isle_len = np.load('{}/group/alist.npy'.format(joint_folder_path), mmap_mode='r').shape[1]

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

    # The first thing we do is assign all non-grouped catalogue "b" sources
    # as being non-counterpart objects, as they have nothing to pair with in
    # catalogue "a", as per our definition in make_set_list.set_list.
    b_remain_inds = np.lib.format.open_memmap('{}/pairing/temp_b_remain.npy'.format(
        joint_folder_path), mode='w+', dtype=np.bool, shape=(len_b,))
    for cnum in range(0, mem_chunk_num):
        lowind = np.floor(len_b*cnum/mem_chunk_num).astype(int)
        highind = np.floor(len_b*(cnum+1)/mem_chunk_num).astype(int)
        b_remain_inds[lowind:highind] = 1
    blist = np.load('{}/group/blist.npy'.format(joint_folder_path), mmap_mode='r')
    for i in range(blist.shape[1]):
        for ind in blist[blist[:, i] > -1, i]:
            b_remain_inds[ind] = 0
    breject = np.load('{}/reject/breject.npy'.format(joint_folder_path), mmap_mode='r')
    for ind in breject:
        b_remain_inds[ind] = 0
    del blist, breject
    for i in range(0, len_b):
        if b_remain_inds[i]:
            bfieldinds[bfieldticker] = i
            probfbarray[bfieldticker] = 1
            bfieldticker += 1

    for cnum in range(0, mem_chunk_num):
        lowind = np.floor(isle_len*cnum/mem_chunk_num).astype(int)
        highind = np.floor(isle_len*(cnum+1)/mem_chunk_num).astype(int)

        alist_ = np.load('{}/group/alist.npy'.format(joint_folder_path),
                         mmap_mode='r')[:, lowind:highind]
        agrplen = np.load('{}/group/agrplen.npy'.format(joint_folder_path),
                          mmap_mode='r')[lowind:highind]
        alist_ = np.asfortranarray(alist_[:np.amax(agrplen), :])
        alistunique_flat = np.unique(alist_[alist_ > -1])
        a_astro = np.load('{}/con_cat_astro.npy'.format(a_cat_folder_path),
                          mmap_mode='r')[alistunique_flat]
        a_photo = np.load('{}/con_cat_photo.npy'.format(a_cat_folder_path),
                          mmap_mode='r')[alistunique_flat]
        amagref = np.load('{}/amagref.npy'.format(joint_folder_path), mmap_mode='r')[alistunique_flat]
        maparray = -1*np.ones(len_a+1).astype(int)
        maparray[alistunique_flat] = np.arange(0, len(a_astro), dtype=int)
        # *list maps the subarray indices, but *list_ keeps the full catalogue indices
        alist = np.asfortranarray(maparray[alist_.flatten()].reshape(alist_.shape))

        blist_ = np.load('{}/group/blist.npy'.format(joint_folder_path),
                         mmap_mode='r')[:, lowind:highind]
        bgrplen = np.load('{}/group/bgrplen.npy'.format(joint_folder_path),
                          mmap_mode='r')[lowind:highind]
        blist_ = np.asfortranarray(blist_[:np.amax(bgrplen), :])
        blistunique_flat = np.unique(blist_[blist_ > -1])
        b_astro = np.load('{}/con_cat_astro.npy'.format(b_cat_folder_path), mmap_mode='r')[blistunique_flat]
        b_photo = np.load('{}/con_cat_photo.npy'.format(b_cat_folder_path), mmap_mode='r')[blistunique_flat]
        bmagref = np.load('{}/magref.npy'.format(b_cat_folder_path), mmap_mode='r')[blistunique_flat]
        maparray = -1*np.ones(len_b+1).astype(int)
        maparray[blistunique_flat] = np.arange(0, len(b_astro), dtype=int)
        blist = np.asfortranarray(maparray[blist_.flatten()].reshape(blist_.shape))

        amodrefind = np.load('{}/modelrefinds.npy'.format(a_auf_folder_path),
                             mmap_mode='r')[:, alistunique_flat]
        [afourier_grids, afrac_grids, aflux_grids], amodrefind = load_small_ref_auf_grid(
            amodrefind, a_auf_folder_path, ['fourier', 'frac', 'flux'])

        bmodrefind = np.load('{}/modelrefinds.npy'.format(b_auf_folder_path),
                             mmap_mode='r')[:, blistunique_flat]
        [bfourier_grids, bfrac_grids, bflux_grids], bmodrefind = load_small_ref_auf_grid(
            bmodrefind, b_auf_folder_path, ['fourier', 'frac', 'flux'])
