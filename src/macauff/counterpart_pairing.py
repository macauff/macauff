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
def source_pairing(cm, force_no_phot_like=False):
    '''
    Function to iterate over all grouped islands of sources, calculating the
    probabilities of all permutations of matches and deriving the most likely
    counterparts for sources in the two catalogues.

    Parameters
    ----------
    cm : Class
        The cross-match wrapper, containing all of the necessary metadata to
        perform the cross-match and determine match islands.
    force_no_phot_like : boolean
        Flag for whether to override pre-generated photometric match and
        non-match likelihoods and create placeholder arrays, to simulate
        an astrometry-only match from a with-photometry match.
    '''
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if force_no_phot_like:
        print(f"{t} Rank {cm.rank}, chunk {cm.chunk_id}: Pairing sources, forcing no photometry...")
    else:
        print(f"{t} Rank {cm.rank}, chunk {cm.chunk_id}: Pairing sources...")
    sys.stdout.flush()

    len_a, len_b = np.sum(cm.agrplen), np.sum(cm.bgrplen)

    a_astro = cm.a_astro
    a_photo = cm.a_photo
    amagref = cm.a_magref

    b_astro = cm.b_astro
    b_photo = cm.b_photo
    bmagref = cm.b_magref

    big_len_a = len(a_astro)
    big_len_b = len(b_astro)
    # large_len is the "safe" initialisation value for arrays, such that no index
    # can ever reach this value.
    large_len = max(big_len_a, big_len_b)

    afourier_grids = cm.a_perturb_auf_outputs['fourier_grid']
    afrac_grids = cm.a_perturb_auf_outputs['frac_grid']
    aflux_grids = cm.a_perturb_auf_outputs['flux_grid']
    bfourier_grids = cm.b_perturb_auf_outputs['fourier_grid']
    bfrac_grids = cm.b_perturb_auf_outputs['frac_grid']
    bflux_grids = cm.b_perturb_auf_outputs['flux_grid']

    # crpts_max_len is the maximum number of counterparts at 100% match rate.
    cprt_max_len = np.sum(np.minimum(cm.agrplen, cm.bgrplen))

    if force_no_phot_like:
        c_array = np.ones_like(cm.c_array)
        fa_array = np.ones_like(cm.fa_array)
        fb_array = np.ones_like(cm.fb_array)
    else:
        c_array = cm.c_array
        fa_array = cm.fa_array
        fb_array = cm.fb_array

    (acountinds, bcountinds, afieldinds, bfieldinds, acontamprob, bcontamprob, etaarray,
     xiarray, acontamflux, bcontamflux, probcarray, crptseps, probfaarray, afieldfluxs,
     afieldseps, afieldetas, afieldxis, probfbarray, bfieldfluxs, bfieldseps, bfieldetas,
     bfieldxis) = cpf.find_island_probabilities(
        a_astro, a_photo, b_astro, b_photo, cm.alist, cm.blist, cm.agrplen, cm.bgrplen,
        c_array, fa_array, fb_array, cm.c_priors, cm.fa_priors, cm.fb_priors, amagref, bmagref,
        cm.a_modelrefinds, cm.b_modelrefinds, cm.abinsarray, cm.abinlengths, cm.bbinsarray, cm.bbinlengths,
        afrac_grids, aflux_grids, bfrac_grids, bflux_grids, afourier_grids, bfourier_grids,
        cm.a_sky_inds, cm.b_sky_inds, cm.rho, cm.drho, len(cm.delta_mag_cuts), large_len, cprt_max_len)

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

    countsum = int(np.sum(countfilter))
    afieldsum = int(np.sum(afieldfilter))
    bfieldsum = int(np.sum(bfieldfilter))

    if force_no_phot_like:
        file_extension = '_without_photometry'
    else:
        file_extension = ''

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
        setattr(cm, file_name + file_extension, temp_variable)

    tot = countsum + afieldsum + cm.lenrejecta
    if tot < big_len_a:
        warnings.warn(f"{big_len_a - tot} catalogue a source{'s' if big_len_a - tot > 1 else ''} "
                      "not in either counterpart, field, or rejected source lists")
    if tot > big_len_a:
        warnings.warn(f"{tot - big_len_a} additional catalogue a "
                      f"{'indices' if tot - big_len_a > 1 else 'index'} recorded, check results "
                      "for duplications carefully")
    tot = countsum + bfieldsum + cm.lenrejectb
    if tot < big_len_b:
        warnings.warn(f"{big_len_b - tot} catalogue b source{'s' if big_len_b - tot > 1 else ''} "
                      "not in either counterpart, field, or rejected source lists.")
    if tot > big_len_b:
        warnings.warn(f"{tot - big_len_b} additional catalogue b "
                      f"{'indices' if tot - big_len_b > 1 else 'index'} recorded, check results "
                      "for duplications carefully")
    sys.stdout.flush()

    print(f"{t} Rank {cm.rank}, chunk {cm.chunk_id}: Catalogue a/b match fraction: "
          f"{len(getattr(cm, 'ac' + file_extension)) / len(a_astro):.3f}/"
          f"{len(getattr(cm, 'bc' + file_extension)) / len(b_astro):.3f}")
    sys.stdout.flush()
