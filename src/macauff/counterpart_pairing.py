# Licensed under a 3-clause BSD style license - see LICENSE
'''
This module provides the functionality for the final cross-match process, the
act of actually pairing sources across the two catalogues as counterparts.
'''

import datetime
import sys
import warnings

import numpy as np

# pylint: disable-next=no-name-in-module,import-error
from macauff.counterpart_pairing_fortran import counterpart_pairing_fortran as cpf

__all__ = ['source_pairing']


# pylint: disable-next=too-many-locals
def source_pairing(cm):
    '''
    Function to iterate over all grouped islands of sources, calculating the
    probabilities of all permutations of matches and deriving the most likely
    counterparts for sources in the two catalogues.

    Parameters
    ----------
    cm : Class
        The cross-match wrapper, containing all of the necessary metadata to
        perform the cross-match and determine match islands.
    '''
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{t} Rank {cm.rank}, chunk {cm.chunk_id}: Pairing sources...")
    sys.stdout.flush()

    agrplen = cm.group_sources_data.agrplen
    bgrplen = cm.group_sources_data.bgrplen

    len_a, len_b = np.sum(agrplen), np.sum(bgrplen)

    abinsarray = cm.phot_like_data.abinsarray
    abinlengths = cm.phot_like_data.abinlengths
    bbinsarray = cm.phot_like_data.bbinsarray
    bbinlengths = cm.phot_like_data.bbinlengths

    c_priors = cm.phot_like_data.c_priors
    c_array = cm.phot_like_data.c_array
    fa_priors = cm.phot_like_data.fa_priors
    fa_array = cm.phot_like_data.fa_array
    fb_priors = cm.phot_like_data.fb_priors
    fb_array = cm.phot_like_data.fb_array

    alist = cm.group_sources_data.alist
    agrplen = cm.group_sources_data.agrplen

    a_astro = np.load(f'{cm.a_cat_folder_path}/con_cat_astro.npy')
    a_photo = np.load(f'{cm.a_cat_folder_path}/con_cat_photo.npy')
    amagref = np.load(f'{cm.a_cat_folder_path}/magref.npy')

    b_astro = np.load(f'{cm.b_cat_folder_path}/con_cat_astro.npy')
    b_photo = np.load(f'{cm.b_cat_folder_path}/con_cat_photo.npy')
    bmagref = np.load(f'{cm.b_cat_folder_path}/magref.npy')

    big_len_a = len(a_astro)
    big_len_b = len(b_astro)
    # large_len is the "safe" initialisation value for arrays, such that no index
    # can ever reach this value.
    large_len = max(big_len_a, big_len_b)

    a_sky_inds = cm.phot_like_data.a_sky_inds

    blist = cm.group_sources_data.blist
    bgrplen = cm.group_sources_data.bgrplen

    b_sky_inds = cm.phot_like_data.b_sky_inds

    afourier_grids = cm.a_perturb_auf_outputs['fourier_grid']
    afrac_grids = cm.a_perturb_auf_outputs['frac_grid']
    aflux_grids = cm.a_perturb_auf_outputs['flux_grid']
    bfourier_grids = cm.b_perturb_auf_outputs['fourier_grid']
    bfrac_grids = cm.b_perturb_auf_outputs['frac_grid']
    bflux_grids = cm.b_perturb_auf_outputs['flux_grid']

    # crpts_max_len is the maximum number of counterparts at 100% match rate.
    cprt_max_len = np.sum(np.minimum(agrplen, bgrplen))

    (acountinds, bcountinds, afieldinds, bfieldinds, acontamprob, bcontamprob, etaarray,
     xiarray, acontamflux, bcontamflux, probcarray, crptseps, probfaarray, afieldfluxs,
     afieldseps, afieldetas, afieldxis, probfbarray, bfieldfluxs, bfieldseps, bfieldetas,
     bfieldxis) = cpf.find_island_probabilities(
        a_astro, a_photo, b_astro, b_photo, alist, blist, agrplen, bgrplen,
        c_array, fa_array, fb_array, c_priors, fa_priors, fb_priors, amagref, bmagref,
        cm.a_modelrefinds, cm.b_modelrefinds, abinsarray, abinlengths, bbinsarray, bbinlengths,
        afrac_grids, aflux_grids, bfrac_grids, bflux_grids, afourier_grids, bfourier_grids,
        a_sky_inds, b_sky_inds, cm.rho, cm.drho, len(cm.delta_mag_cuts), large_len, cprt_max_len)

    afieldfilter = np.zeros(dtype=bool, shape=(len_a,))
    bfieldfilter = np.zeros(dtype=bool, shape=(len_b,))

    # *contamprob is (smalllen, nfracs) in shape and our check for correctness needs to check
    # all nfrac values, requiring an all check.
    countfilter = (
        (acountinds < large_len+1) & (bcountinds < large_len+1) &
        np.all(acontamprob >= 0, axis=0) & np.all(bcontamprob >= 0, axis=0) &
        (acontamflux >= 0) & (bcontamflux >= 0) & (probcarray >= 0) & (etaarray >= -30) &
        (xiarray >= -30))

    afieldfilter = (afieldinds < large_len+1) & (probfaarray >= 0)

    bfieldfilter = (bfieldinds < large_len+1) & (probfbarray >= 0)

    lenrejecta = cm.group_sources_data.lenrejecta
    lenrejectb = cm.group_sources_data.lenrejectb

    countsum = int(np.sum(countfilter))
    afieldsum = int(np.sum(afieldfilter))
    bfieldsum = int(np.sum(bfieldfilter))

    # Reduce size of output files, removing anything that doesn't meet the
    # criteria above from all saved numpy arrays.
    for file_name, variable, filter_variable in zip(
        ['ac', 'bc', 'pacontam', 'pbcontam', 'acontamflux', 'bcontamflux', 'af', 'bf', 'pc', 'eta',
         'xi', 'pfa', 'pfb', 'afieldflux', 'bfieldflux', 'crptseps', 'afieldseps', 'afieldeta',
         'afieldxi', 'bfieldseps', 'bfieldeta', 'bfieldxi'],
        [acountinds, bcountinds, acontamprob, bcontamprob, acontamflux, bcontamflux, afieldinds,
         bfieldinds, probcarray, etaarray, xiarray, probfaarray, probfbarray, afieldfluxs,
         bfieldfluxs, crptseps, afieldseps, afieldetas, afieldxis, bfieldseps, bfieldetas,
         bfieldxis],
        [countfilter, countfilter, countfilter, countfilter, countfilter, countfilter,
         afieldfilter, bfieldfilter, countfilter, countfilter, countfilter, afieldfilter,
         bfieldfilter, afieldfilter, bfieldfilter, countfilter, afieldfilter, afieldfilter,
         afieldfilter, bfieldfilter, bfieldfilter, bfieldfilter]):

        if file_name in ('pacontam', 'pbcontam'):
            temp_variable = variable[:, filter_variable]
        else:
            temp_variable = variable[filter_variable]
        np.save(f'{cm.joint_folder_path}/pairing/{file_name}.npy', temp_variable)

    tot = countsum + afieldsum + lenrejecta
    if tot < big_len_a:
        warnings.warn(f"{big_len_a - tot} catalogue a source{'s' if big_len_a - tot > 1 else ''} "
                      "not in either counterpart, field, or rejected source lists")
    if tot > big_len_a:
        warnings.warn(f"{tot - big_len_a} additional catalogue a "
                      f"{'indices' if tot - big_len_a > 1 else 'index'} recorded, check results "
                      "for duplications carefully")
    tot = countsum + bfieldsum + lenrejectb
    if tot < big_len_b:
        warnings.warn(f"{big_len_b - tot} catalogue b source{'s' if big_len_b - tot > 1 else ''} "
                      "not in either counterpart, field, or rejected source lists.")
    if tot > big_len_b:
        warnings.warn(f"{tot - big_len_b} additional catalogue b "
                      f"{'indices' if tot - big_len_b > 1 else 'index'} recorded, check results "
                      "for duplications carefully")
    sys.stdout.flush()
