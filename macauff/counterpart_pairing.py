# Licensed under a 3-clause BSD style license - see LICENSE
'''
This module provides the functionality for the final cross-match process, the
act of actually pairing sources across the two catalogues as counterparts.
'''

import sys
import numpy as np
import multiprocessing
import itertools

from .misc_functions import (create_auf_params_grid, load_small_ref_auf_grid)
from .misc_functions_fortran import misc_functions_fortran as mff
from .counterpart_pairing_fortran import counterpart_pairing_fortran as cpf


def source_pairing(joint_folder_path, a_cat_folder_path, b_cat_folder_path, a_auf_folder_path,
                   b_auf_folder_path, a_filt_names, b_filt_names, a_auf_pointings, b_auf_pointings,
                   rho, drho, delta_mag_cuts, mem_chunk_num, n_pool):
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

        # Initialise the multiprocessing loop setup:
        pool = multiprocessing.Pool(n_pool)
        counter = np.arange(0, alist.shape[1])
        expand_constants = [itertools.repeat(item) for item in [
            a_astro, a_photo, b_astro, b_photo, alist, alist_, blist, blist_, agrplen, bgrplen,
            c_array, fa_array, fb_array, c_priors, fa_priors, fb_priors, amagref, bmagref,
            amodelrefinds, bmodelrefinds, abinsarray, abinlengths, bbinsarray, bbinlengths,
            afrac_grids, aflux_grids, bfrac_grids, bflux_grids, afourier_grids, bfourier_grids,
            a_sky_inds, b_sky_inds, rho, drho, n_fracs]]
        iter_group = zip(counter, *expand_constants)
        for return_items in pool.imap_unordered(_individual_island_probability, iter_group,
                                                chunksize=max(1, len(counter) // n_pool)):
            # Use the quick-return check in _individual_island_probability
            # as a short-hand for zero-length "b" island -- i.e., "a" sources
            # only -- and update the probabilities of the afield sources
            # accordingly:
            if None in return_items:
                _, aperm, aperm_ = return_items
                afieldinds[afieldticker:afieldticker+len(aperm)] = aperm_
                probfaarray[afieldticker:afieldticker+len(aperm)] = 1
                afieldticker += len(aperm)
            else:
                [acrpts, bcrpts, acrptscontp, bcrptscontp, etacrpts, xicrpts, acrptflux, bcrptflux,
                 afield, bfield, prob, integral] = return_items
                acountinds[counterpartticker:counterpartticker+len(acrpts)] = acrpts
                bcountinds[counterpartticker:counterpartticker+len(bcrpts)] = bcrpts
                acontamprob[counterpartticker:counterpartticker+len(acrptscontp)] = acrptscontp
                bcontamprob[counterpartticker:counterpartticker+len(bcrptscontp)] = bcrptscontp
                etaarray[counterpartticker:counterpartticker+len(bcrptscontp)] = etacrpts
                xiarray[counterpartticker:counterpartticker+len(bcrptscontp)] = xicrpts
                acontamflux[counterpartticker:counterpartticker+len(acrptflux)] = acrptflux
                bcontamflux[counterpartticker:counterpartticker+len(bcrptflux)] = bcrptflux
                probcarray[counterpartticker:counterpartticker+len(acrpts)] = prob/integral
                counterpartticker += len(acrpts)

                afieldinds[afieldticker:afieldticker+len(afield)] = afield
                probfaarray[afieldticker:afieldticker+len(afield)] = prob/integral
                afieldticker += len(afield)

                bfieldinds[bfieldticker:bfieldticker+len(bfield)] = bfield
                probfbarray[bfieldticker:bfieldticker+len(bfield)] = prob/integral
                bfieldticker += len(bfield)
        pool.close()


def _individual_island_probability(iterable_wrapper):
    [i, a_astro, a_photo, b_astro, b_photo, alist, alist_, blist, blist_, agrplen, bgrplen,
     c_array, fa_array, fb_array, c_priors, fa_priors, fb_priors, amagref, bmagref, amodelrefinds,
     bmodelrefinds, abinsarray, abinlengths, bbinsarray, bbinlengths, afrac_grids, aflux_grids,
     bfrac_grids, bflux_grids, afourier_grids, bfourier_grids, a_sky_inds, b_sky_inds, rho, drho,
     n_fracs] = iterable_wrapper
    if bgrplen[i] == 0:
        return [None, alist[:agrplen[i], i], alist_[:agrplen[i], i]]
    else:
        aperm = alist[:agrplen[i], i]
        bperm = blist[:bgrplen[i], i]
        aperm_ = alist_[:agrplen[i], i]
        bperm_ = blist_[:bgrplen[i], i]

        acrpts = []
        bcrpts = []
        acrptscontp = []
        bcrptscontp = []
        etacrpts = []
        xicrpts = []
        acrptflux = []
        bcrptflux = []
        afield = []
        bfield = []

        aused = amagref[aperm]
        qa = a_sky_inds[aperm]

        bused = bmagref[bperm]
        qb = b_sky_inds[bperm]

        arefinds = amodelrefinds[:, aperm]
        brefinds = bmodelrefinds[:, bperm]

        counterpartgrid = np.empty((len(aperm), len(bperm)), float)
        etagrid = np.empty((len(aperm), len(bperm)), float)
        xigrid = np.empty((len(aperm), len(bperm)), float)
        acontamprobgrid = np.empty((len(aperm), len(bperm), n_fracs), float)
        bcontamprobgrid = np.empty((len(aperm), len(bperm), n_fracs), float)
        acontamfluxgrid = np.empty(len(aperm), float)
        bcontamfluxgrid = np.empty(len(bperm), float)
        afieldarray = np.empty(len(aperm), float)
        bfieldarray = np.empty(len(bperm), float)
        bina = np.empty((len(aperm)), int)
        binb = np.empty((len(bperm)), int)
        Nfa = np.zeros(len(aperm), float)
        Nfb = np.zeros(len(bperm), float)
        fa = np.empty((len(aperm)), float)
        fb = np.empty((len(bperm)), float)

        for j in range(0, len(aperm)):
            bina[j] = np.where(a_photo[aperm[j], amagref[aperm[j]]] - abinsarray[
                :abinlengths[aused[j], qa[j]], aused[j], qa[j]] >= 0)[0][-1]
            # For the field stars we don't know which other filter to use, so we
            # just default to using the first filter in the opposing catalogue,
            # but it shouldn't matter since it ought to be independent.
            Nfa[j] = fa_priors[0, aused[j], qa[j]]
            fa[j] = fa_array[bina[j], 0, aused[j], qa[j]]
            acontamfluxgrid[j] = aflux_grids[arefinds[0, j], arefinds[1, j], arefinds[2, j]]

        for j in range(0, len(bperm)):
            binb[j] = np.where(b_photo[bperm[j], bmagref[bperm[j]]] - bbinsarray[
                :bbinlengths[bused[j], qb[j]], bused[j], qb[j]] >= 0)[0][-1]
            Nfb[j] = fb_priors[bused[j], 0, qb[j]]
            fb[j] = fb_array[binb[j], bused[j], 0, qb[j]]
            bcontamfluxgrid[j] = bflux_grids[brefinds[0, j], brefinds[1, j], brefinds[2, j]]

        bfourgausses = np.empty((len(bperm), len(rho)-1), float)
        for k in range(0, len(bperm)):
            bsig = b_astro[bperm[k], 2]
            bfourgausses[k, :] = np.exp(-2 * np.pi**2 * (rho[:-1]+drho/2)**2 * bsig**2)

        afieldarray = Nfa*fa
        bfieldarray = Nfb*fb

        for j in range(0, len(aperm)):
            aF = afrac_grids[:, arefinds[0, j], arefinds[1, j], arefinds[2, j]]
            aoffs = afourier_grids[:, arefinds[0, j], arefinds[1, j], arefinds[2, j]]
            asig = a_astro[aperm[j], 2]
            afourgauss = np.exp(-2 * np.pi**2 * (rho[:-1]+drho/2)**2 * asig**2)
            for k in range(0, len(bperm)):
                sep = mff.haversine_wrapper(a_astro[aperm[j], 0], a_astro[aperm[j], 1],
                                            b_astro[bperm[k], 0], b_astro[bperm[k], 1])
                bF = bfrac_grids[:, brefinds[0, k], brefinds[1, k], brefinds[2, k]]
                boffs = bfourier_grids[:, brefinds[0, k], brefinds[1, k], brefinds[2, k]]

                # Calculate the probability densities of all four combinations
                # of perturbation and non-perturbation AUFs.
                G0nn = afourgauss*bfourgausses[k, :]
                G0cc = aoffs*boffs*G0nn
                G0cn = aoffs*G0nn
                G0nc = boffs*G0nn
                Gcc, Gcn, Gnc, Gnn = cpf.contamination_match_probabilities(
                    G0cc, G0cn, G0nc, G0nn, rho[:-1]+drho/2, drho, sep)
                # G would be in units of per square arcseconds, but we need it
                # in units of per square degree to compare to Nf.
                G = Gcc * 3600**2
                for ff in range(0, n_fracs):
                    pr = (aF[ff]*bF[ff]*Gcc + aF[ff]*(1-bF[ff])*Gcn +
                          (1-aF[ff])*bF[ff]*Gnc + (1-aF[ff])*(1-bF[ff])*Gnn)
                    # Marginalise over the opposite source contamination probability
                    # to calculate specific source's contamination chance.
                    if pr > 0:
                        acontamprobgrid[j, k, ff] = min(1, max(0, (aF[ff]*bF[ff]*Gcc +
                                                                   aF[ff]*(1-bF[ff])*Gcn)/pr))
                        bcontamprobgrid[j, k, ff] = min(1, max(0, (aF[ff]*bF[ff]*Gcc +
                                                                   (1-aF[ff])*bF[ff]*Gnc)/pr))
                    elif pr == 0:
                        # If source is so far out that G has truncated to zero,
                        # then we just set the chance to 50%, as it is very
                        # unlikely to affect the outcome.
                        acontamprobgrid[j, k, ff] = 0.5
                        bcontamprobgrid[j, k, ff] = 0.5

                Nc = c_priors[bused[k], aused[j], qb[k]]
                cdmdm = c_array[binb[k], bina[j], bused[k], aused[j], qb[k]]
                counterpartgrid[j, k] = Nc*G*cdmdm

                if fa[j]*fb[k] == 0:
                    etagrid[j, k] = 10
                elif cdmdm == 0:
                    etagrid[j, k] = -10
                else:
                    etagrid[j, k] = np.log10(cdmdm/(fa[j]*fb[k]))
                if Nfa[j]*Nfb[k] == 0:
                    xigrid[j, k] = 10
                elif Nc*G == 0:
                    xigrid[j, k] = -10
                else:
                    xigrid[j, k] = np.log10(Nc*G/(Nfa[j]*Nfb[k]))

        prob = 0
        integral = 0
        # Start with the case of no matches between any island objects.
        tempprob = np.prod(afieldarray) * np.prod(bfieldarray)
        integral = integral + tempprob
        if tempprob > prob:
            prob = tempprob
            acrpts = np.array([])
            bcrpts = np.array([])
            # With unknown blank array have to reshape the two two-dimensional
            # arrays to (0, n_fracs).
            acrptscontp = np.array([]).reshape(0, n_fracs)
            bcrptscontp = np.array([]).reshape(0, n_fracs)
            etacrpts = np.array([])
            xicrpts = np.array([])
            acrptflux = np.array([])
            bcrptflux = np.array([])
            afield = aperm_
            bfield = bperm_
        for N in range(1, min(len(aperm), len(bperm))+1):
            aiter = np.array(list(itertools.combinations(aperm, r=N)))
            biter = np.array(list(itertools.permutations(bperm, r=N)))
            for x in aiter:
                for y in biter:
                    # For paired stars, order matters, so we have to find the
                    # index of the array holding the star's overall catalogue
                    # index that matches that index in the permutation list.
                    ya = np.array([np.argmin(np.abs(j - aperm)) for j in x], int)
                    yb = np.array([np.argmin(np.abs(j - bperm)) for j in y], int)
                    ta = np.delete(np.arange(0, len(aperm)), ya)
                    tb = np.delete(np.arange(0, len(bperm)), yb)
                    tempprob = (np.prod(counterpartgrid[ya, yb]) *
                                np.prod(afieldarray[ta]) * np.prod(bfieldarray[tb]))
                    integral = integral + tempprob
                    if tempprob > prob:
                        prob = tempprob
                        acrpts = np.array(aperm_[ya])
                        bcrpts = np.array(bperm_[yb])
                        acrptscontp = np.array(acontamprobgrid[ya, yb])
                        bcrptscontp = np.array(bcontamprobgrid[ya, yb])
                        etacrpts = np.array(etagrid[ya, yb])
                        xicrpts = np.array(xigrid[ya, yb])
                        acrptflux = np.array(acontamfluxgrid[ya])
                        bcrptflux = np.array(bcontamfluxgrid[yb])
                        afield = np.array(aperm_[ta])
                        bfield = np.array(bperm_[tb])

        if integral <= 0:
            acrpts = np.array([])
            bcrpts = np.array([])
            acrptscontp = np.array([]).reshape(0, n_fracs)
            bcrptscontp = np.array([]).reshape(0, n_fracs)
            etacrpts = np.array([])
            xicrpts = np.array([])
            acrptflux = np.array([])
            bcrptflux = np.array([])
            afield = aperm_
            bfield = bperm_
            prob = 0
            integral = 1

        return [acrpts, bcrpts, acrptscontp, bcrptscontp, etacrpts, xicrpts, acrptflux, bcrptflux,
                afield, bfield, prob, integral]
