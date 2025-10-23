# Licensed under a 3-clause BSD style license - see LICENSE
'''
This module hosts the functions to create overlapping groups of "islands" of
objects, astrometrically in common based on relative sky separation compared to
their respective uncertainties across two catalogues.
'''

import itertools
import sys
import warnings

import numpy as np
import scipy.special

__all__ = ['set_list']

from macauff.misc_functions import make_pool


def set_list(aindices, bindices, aoverlap, boverlap, n_pool):
    '''
    Creates the inter-catalogue groupings between catalogues "a" and "b", based
    on previously determined individual source "overlaps" in astrometry.

    Parameters
    ----------
    aindices : numpy.ndarray
        The indices into catalogue "b", for each catalogue "a" source, that have
        been determined to be potentially positionally correlated.
    bindices : numpy.ndarray
        The equivalent to ``aindices``, but mapping the overlaps in catalogue "a"
        for each catalogue "b" source.
    aoverlap : numpy.ndarray
        The number of overlaps for each catalogue "a" source (i.e., the length
        of each row in ``aindices`` for each source).
    boverlap : numpy.ndarray
        The equivalent number of overlaps for each catalogue "b" object.
    n_pool : integer
        Number of multiprocessing pools to use when parallelising.

    Returns
    -------
    alist : numpy.ndarray
        The indices of all catalogue "a" sources that are in a common "island"
        group together. Each row of ``alist`` indicates all "a" sources
        potentially positionally correlated.
    blist : numpy.ndarray
        The indices of all catalogue "b" objects in the same groups (as other
        "b" sources, as well as mapping to catalogue "a" objects).
    agrouplengths : numpy.ndarray
        The number of catalogue "a" sources in each unique "island".
    bgrouplengths : numpy.ndarray
        The number of catalogue "b" sources in each island grouping.
    '''
    agroup, bgroup = _initial_group_numbering(aindices, bindices, aoverlap, boverlap)
    groupmax = max(np.amax(agroup), np.amax(bgroup))

    agrouplengths = np.zeros(dtype=int, shape=(groupmax,))
    bgrouplengths = np.zeros(dtype=int, shape=(groupmax,))

    for agrp in agroup:
        agrouplengths[agrp-1] += 1
    for bgrp in bgroup:
        bgrouplengths[bgrp-1] += 1

    # Search for any island groupings which are too large to calculate the
    # permutations of reasonably (limiting at 50,000). When considering a set,
    # the total number of matches considered is:
    # sum k from 0 to min(len(a), len(b)) [#] of k-combinations of a times
    # [#] of k-permutations of b, with
    # number of k-combinations = len(a)! / (k! (len(a) - k)!)
    # and number of k-permutations = len(b)! / (len(b) - k)!

    # However we can't do x! / (x-k)! so we do prod(range(x-k+1, x, 1)) for
    # computational simplicity.
    # Since 21! is larger than a 64-bit integer (see factorial maths in
    # counterpart_pairing_fortran.find_single_island_prob), we also filter for
    # islands with more than 20 objects in both catalogues within
    # _calc_group_length_exceeded.
    maxiters = 5000000
    grouplengthexceeded = np.zeros(dtype=bool, shape=(len(agrouplengths),))
    counter = np.arange(0, len(agrouplengths))
    iter_group = zip(counter, agrouplengths, bgrouplengths, itertools.repeat(maxiters))
    with make_pool(n_pool) as pool:
        for return_items in pool.imap_unordered(_calc_group_length_exceeded, iter_group,
                                                chunksize=max(1, len(counter) // n_pool)):
            i, len_exceeded_flag = return_items
            grouplengthexceeded[i] = len_exceeded_flag

    pool.join()

    if np.any(grouplengthexceeded):
        nremoveisland = np.sum(grouplengthexceeded)
        nacatremove = np.sum(agrouplengths[grouplengthexceeded])
        nbcatremove = np.sum(bgrouplengths[grouplengthexceeded])
        warnings.warn(f"{nremoveisland} island{'' if nremoveisland == 1 else 's'}, containing "
                      f"{nacatremove}/{len(aoverlap)} catalogue a and {nbcatremove}/{len(boverlap)} "
                      f"catalogue b stars, {'was' if nremoveisland == 1 else 'were'} removed for having "
                      f"more than {maxiters} possible iterations. Please check any results carefully.")
        sys.stdout.flush()
        rejectgroupnum = np.arange(1, groupmax+1)[grouplengthexceeded]
        areject = np.arange(0, len(aoverlap))[np.isin(agroup, rejectgroupnum)]
        breject = np.arange(0, len(boverlap))[np.isin(bgroup, rejectgroupnum)]
        reject_flag = True
    else:
        reject_flag = False

    # Keep track of which sources have "good" group sizes, and the size of each
    # group in the two catalogues (e.g., group 1 has 2 "a" and 3 "b" sources).
    goodlength = np.logical_not(grouplengthexceeded)
    acounters = np.zeros(dtype=int, shape=(groupmax,))
    bcounters = np.zeros(dtype=int, shape=(groupmax,))

    amaxlen = int(np.amax(agrouplengths[goodlength]))
    bmaxlen = int(np.amax(bgrouplengths[goodlength]))
    alist = np.full(dtype=int, shape=(amaxlen, groupmax), fill_value=-1, order='F')
    blist = np.full(dtype=int, shape=(bmaxlen, groupmax), fill_value=-1, order='F')
    # Remember that we started groups from one, so convert to zero-indexing.
    # Loop over each source in turn, skipping any which belong to an island
    # too large to run, updating alist or blist with the corresponding island
    # number the source belongs to.
    for i, agrp in enumerate(agroup):
        if goodlength[agrp-1]:
            alist[acounters[agrp-1], agrp-1] = i
            acounters[agrp-1] += 1
    for i, bgrp in enumerate(bgroup):
        if goodlength[bgrp-1]:
            blist[bcounters[bgrp-1], bgrp-1] = i
            bcounters[bgrp-1] += 1

    # Now, we simply want to remove any sources from the list with islands too
    # large to run.
    alist = alist[:, goodlength]
    blist = blist[:, goodlength]
    agrouplengths = agrouplengths[goodlength]
    bgrouplengths = bgrouplengths[goodlength]

    if reject_flag:
        # pylint: disable-next=possibly-used-before-assignment
        return alist, blist, agrouplengths, bgrouplengths, areject, breject
    return alist, blist, agrouplengths, bgrouplengths


def _initial_group_numbering(aindices, bindices, aoverlap, boverlap):
    '''
    Iterates over the indices mapping overlaps between the two catalogues,
    assigning initial group numbers to sources to be placed in "islands".

    Loops through all "lonely", single-source islands in each catalogue, and
    those with a one-to-one mapping of a single "a" object and one "b" source,
    before iteratively grouping all multi-object islands together.

    Parameters
    ----------
    aindices : numpy.ndarray
        The indices into catalogue "b", for each catalogue "a" source, that have
        been determined to be potentially positionally correlated.
    bindices : numpy.ndarray
        The equivalent to ``aindices``, but mapping the overlaps in catalogue "a"
        for each catalogue "b" source.
    aoverlap : numpy.ndarray
        The number of overlaps for each catalogue "a" source (i.e., the length
        of each row in ``aindices`` for each source).
    boverlap : numpy.ndarray
        The equivalent number of overlaps for each catalogue "b" object.

    Returns
    -------
    agroup : numpy.ndarray
        Array detailing the group number of each catalogue "a" source.
    bgroup : numpy.ndarray
        Array detailing the group number of each catalogue "b" source.
    '''
    agroup = np.zeros(dtype=int, shape=(len(aoverlap),))
    bgroup = np.zeros(dtype=int, shape=(len(boverlap),))

    group_num = 0
    # First search for catalogue "a" sources that are either lonely, with no
    # catalogue "b" object near them, or those with only one corresponding
    # object in catalogue "b", then iteratively group from a -> b -> a -> ...
    # any remaining objects.
    for i in range(0, len(agroup)):  # pylint: disable=consider-using-enumerate
        if aoverlap[i] == 0:
            group_num += 1
            agroup[i] = group_num
        elif aoverlap[i] == 1 and boverlap[aindices[0, i]] == 1:
            group_num += 1
            agroup[i] = group_num
            bgroup[aindices[0, i]] = group_num

    for i in range(0, len(bgroup)):  # pylint: disable=consider-using-enumerate
        if boverlap[i] == 0:
            group_num += 1
            bgroup[i] = group_num

    for i, agrp in enumerate(agroup):
        if agrp == 0:
            group_num += 1
            _a_to_b(i, group_num, aoverlap[i], aindices, bindices,
                    aoverlap, boverlap, agroup, bgroup)

    return agroup, bgroup


def _a_to_b(ind, grp, n, aindices, bindices, aoverlap, boverlap, agroup, bgroup):
    '''
    Iterative function, along with ``_b_to_a``, to assign all sources overlapping
    this catalogue "a" object in catalogue "b" as being in the same group as it.

    This subsequently calls ``_b_to_a`` for each of those "b" sources, to
    assign any "a" sources that overlap those objects to the same group, until
    there are no more overlaps.

    Parameters
    ----------
    ind : integer
        The index into ``agroup``, the specific source in question to be assigned
        this group number.
    grp : integer
        The group number of this "island" to be assigned.
    n : integer
        The number of sources that overlap this catalogue "a" source.
    aindices : numpy.ndarray
        The indices into catalogue "b", for each catalogue "a" source, that have
        been determined to be potentially positionally correlated.
    bindices : numpy.ndarray
        The equivalent to ``aindices``, but mapping the overlaps in catalogue "a"
        for each catalogue "b" source.
    aoverlap : numpy.ndarray
        The number of overlaps for each catalogue "a" source (i.e., the length
        of each row in ``aindices`` for each source).
    boverlap : numpy.ndarray
        The equivalent number of overlaps for each catalogue "b" object.
    agroup : numpy.ndarray
        Array detailing the group number of each catalogue "a" source.
    bgroup : numpy.ndarray
        Array detailing the group number of each catalogue "b" source.
    '''
    agroup[ind] = grp
    for q in aindices[:n, ind]:
        if bgroup[q] != grp:
            _b_to_a(q, grp, boverlap[q], aindices, bindices, aoverlap, boverlap, agroup, bgroup)


def _b_to_a(ind, grp, n, aindices, bindices, aoverlap, boverlap, agroup, bgroup):
    '''
    Iterative function, equivalent to ``_a_to_b``, to assign all sources
    overlapping this catalogue "b" object in catalogue "a" with its group number.

    Parameters
    ----------
    ind : integer
        The index into ``bgroup``, the specific source in question to be assigned
        this group number.
    grp : integer
        The group number of this "island" to be assigned.
    n : integer
        The number of sources that overlap this catalogue "b" source.
    aindices : numpy.ndarray
        The indices into catalogue "b", for each catalogue "a" source, that have
        been determined to be potentially positionally correlated.
    bindices : numpy.ndarray
        The equivalent to ``aindices``, but mapping the overlaps in catalogue "a"
        for each catalogue "b" source.
    aoverlap : numpy.ndarray
        The number of overlaps for each catalogue "a" source (i.e., the length
        of each row in ``aindices`` for each source).
    boverlap : numpy.ndarray
        The equivalent number of overlaps for each catalogue "b" object.
    agroup : numpy.ndarray
        Array detailing the group number of each catalogue "a" source.
    bgroup : numpy.ndarray
        Array detailing the group number of each catalogue "b" source.
    '''
    bgroup[ind] = grp
    for f in bindices[:n, ind]:
        if agroup[f] != grp:
            _a_to_b(f, grp, aoverlap[f], aindices, bindices, aoverlap, boverlap, agroup, bgroup)


def _calc_group_length_exceeded(iterable):
    i, n_a, n_b, maxiters = iterable
    if max(n_a, n_b) > 20:
        return i, 1
    counter = 0
    for k in np.arange(0, min(n_a, n_b)+1e-10, 1):
        kcomb = np.prod(np.arange(n_a-k+1, n_a+1e-10, 1)) / scipy.special.factorial(k)
        kperm = np.prod(np.arange(n_b-k+1, n_b+1e-10, 1))
        counter += kcomb*kperm
        if counter > maxiters:
            exceed_flag = 1
            return i, exceed_flag
    exceed_flag = 0
    return i, exceed_flag
