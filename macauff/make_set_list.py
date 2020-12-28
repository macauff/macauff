# Licensed under a 3-clause BSD style license - see LICENSE
'''
This module hosts the functions to create overlapping groups of "islands" of
objects, astrometrically in common based on relative sky separation compared to
their respective uncertainties across two catalogues.
'''

import sys
import os
import warnings
import numpy as np
import scipy.special

__all__ = ['set_list']


def set_list(aindices, bindices, aoverlap, boverlap, joint_folder_path):
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
    joint_folder_path : string
        Location of top-level folder containing the "group" folder in which
        index and overlap arrays are stored.

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
    agroup, bgroup = _initial_group_numbering(aindices, bindices, aoverlap, boverlap,
                                              joint_folder_path)
    groupmax = max(np.amax(agroup), np.amax(bgroup))
    agrouplengths = np.lib.format.open_memmap('{}/group/agrplen.npy'.format(joint_folder_path),
                                              mode='w+', dtype=int, shape=(groupmax,))
    agrouplengths[:] = 0
    bgrouplengths = np.lib.format.open_memmap('{}/group/bgrplen.npy'.format(joint_folder_path),
                                              mode='w+', dtype=int, shape=(groupmax,))
    bgrouplengths[:] = 0

    for i in range(0, len(agroup)):
        agrouplengths[agroup[i]-1] += 1
    for i in range(0, len(bgroup)):
        bgrouplengths[bgroup[i]-1] += 1

    # Search for any island groupings which are too large to calculate the
    # permutations of reasonably (limiting at 50,000). When considering a set,
    # the total number of matches considered is:
    # sum k from 0 to min(len(a), len(b)) [#] of k-combinations of a times
    # [#] of k-permutations of b, with
    # number of k-combinations = len(a)! / (k! (len(a) - k)!)
    # and number of k-permutations = len(b)! / (len(b) - k)!

    # However we can't do x! / (x-k)! so we do prod(range(x-k+1, x, 1)) for
    # computational simplicity.
    maxiters = 50000
    grouplengthexceeded = np.lib.format.open_memmap('{}/group/grplenexceed.npy'.format(
        joint_folder_path), mode='w+', dtype=np.bool, shape=(len(agrouplengths),))
    grouplengthexceeded[:] = 0
    for i in range(0, len(agrouplengths)):
        n_a = agrouplengths[i]
        n_b = bgrouplengths[i]
        counter = 0
        for k in np.arange(0, min(n_a, n_b)+1e-10, 1):
            kcomb = (np.prod([qq for qq in np.arange(n_a-k+1, n_a+1e-10, 1)]) /
                     scipy.special.factorial(k))
            kperm = np.prod([qq for qq in np.arange(n_b-k+1, n_b+1e-10, 1)])
            counter += kcomb*kperm
            if counter > maxiters:
                grouplengthexceeded[i] = 1
                break

    if np.any(grouplengthexceeded):
        nremoveisland = np.sum(grouplengthexceeded)
        nacatremove = np.sum(agrouplengths[grouplengthexceeded])
        nbcatremove = np.sum(bgrouplengths[grouplengthexceeded])
        warnings.warn("{} island{}, containing {}/{} catalogue a and {}/{} catalogue b stars, "
                      "{} removed for having more than {} possible iterations. Please check any "
                      "results carefully.".format(nremoveisland, '' if nremoveisland == 1 else 's',
                                                  nacatremove, len(aoverlap), nbcatremove,
                                                  len(boverlap), 'was' if nremoveisland == 1 else
                                                  'were', maxiters))
        sys.stdout.flush()
        rejectgroupnum = np.arange(1, groupmax+1)[grouplengthexceeded]
        reject_a = np.arange(0, len(aoverlap))[np.in1d(agroup, rejectgroupnum)]
        reject_b = np.arange(0, len(boverlap))[np.in1d(bgroup, rejectgroupnum)]
        np.save('{}/reject/areject.npy'.format(joint_folder_path), reject_a)
        np.save('{}/reject/breject.npy'.format(joint_folder_path), reject_b)
        del reject_a, reject_b, rejectgroupnum

    # Keep track of which sources have "good" group sizes, and the size of each
    # group in the two catalogues (e.g., group 1 has 2 "a" and 3 "b" sources).
    goodlength = np.lib.format.open_memmap('{}/group/goodlen.npy'.format(
        joint_folder_path), mode='w+', dtype=np.bool, shape=(len(agrouplengths),))
    di = max(1, len(agrouplengths) // 20)
    for i in range(0, len(agrouplengths), di):
        goodlength[i:i+di] = np.logical_not(grouplengthexceeded[i:i+di])
    acounters = np.lib.format.open_memmap('{}/group/acount.npy'.format(joint_folder_path),
                                          mode='w+', dtype=int, shape=(groupmax,))
    acounters[:] = 0
    bcounters = np.lib.format.open_memmap('{}/group/bcount.npy'.format(joint_folder_path),
                                          mode='w+', dtype=int, shape=(groupmax,))
    bcounters[:] = 0
    # open_memmap requires tuples of ints and np.amax doesn't play nicely with
    # memmapped arrays, making (memmap(result)) variables, so we force the
    # number to a simple int here.
    amaxlen, bmaxlen = 0, 0
    for i in range(0, len(agrouplengths), di):
        if np.sum(goodlength[i:i+di]) > 0:
            amaxlen = max(amaxlen, int(np.amax(agrouplengths[i:i+di][goodlength[i:i+di]])))
            bmaxlen = max(bmaxlen, int(np.amax(bgrouplengths[i:i+di][goodlength[i:i+di]])))
    alist = np.lib.format.open_memmap('{}/group/alist.npy'.format(joint_folder_path), mode='w+',
                                      dtype=int, shape=(amaxlen, groupmax), fortran_order=True)
    alist[:, :] = -1
    blist = np.lib.format.open_memmap('{}/group/blist.npy'.format(joint_folder_path), mode='w+',
                                      dtype=int, shape=(bmaxlen, groupmax), fortran_order=True)
    blist[:, :] = -1
    # Remember that we started groups from one, so convert to zero-indexing.
    # Loop over each source in turn, skipping any which belong to an island
    # too large to run, updating alist or blist with the corresponding island
    # number the source belongs to.
    for i in range(0, len(agroup)):
        if goodlength[agroup[i]-1]:
            alist[acounters[agroup[i]-1], agroup[i]-1] = i
            acounters[agroup[i]-1] += 1
    for i in range(0, len(bgroup)):
        if goodlength[bgroup[i]-1]:
            blist[bcounters[bgroup[i]-1], bgroup[i]-1] = i
            bcounters[bgroup[i]-1] += 1

    # Now, slightly convolutedly, we simply want to remove any sources from the
    # list with islands too large to run.
    newsecondlen = 0
    for i in range(0, len(agrouplengths), di):
        # Require the manual conversion to integer due to memmap issues again.
        newsecondlen += int(np.sum(goodlength[i:i+di]))

    di = max(1, newsecondlen // 20)
    # Essentially do a = a[q], or a = a[:, q], but via memmap.
    for cat_kind, list_array, maxlen, len_array in zip(
            ['a', 'b'], [alist, blist], [amaxlen, bmaxlen], [agrouplengths, bgrouplengths]):
        new_list_array = np.lib.format.open_memmap(
            '{}/group/{}list2.npy'.format(joint_folder_path, cat_kind), mode='w+', dtype=int,
            shape=(maxlen, newsecondlen), fortran_order=True)
        new_grouplengths = np.lib.format.open_memmap(
            '{}/group/{}grplen2.npy'.format(joint_folder_path, cat_kind), mode='w+', dtype=int,
            shape=(newsecondlen,))
        tick = 0
        for i in range(0, len(len_array), di):
            new_list_array[:, tick:tick+np.sum(goodlength[i:i+di])] = list_array[:, i:i+di][
                :, goodlength[i:i+di]]
            new_grouplengths[tick:tick+np.sum(goodlength[i:i+di])] = len_array[i:i+di][
                goodlength[i:i+di]]
            tick += np.sum(goodlength[i:i+di])

        os.system('mv {}/group/{}list2.npy {}/group/{}list.npy'.format(
            joint_folder_path, cat_kind, joint_folder_path, cat_kind))
        os.system('mv {}/group/{}grplen2.npy {}/group/{}grplen.npy'.format(
            joint_folder_path, cat_kind, joint_folder_path, cat_kind))

    # Tidy up temporary memmap arrays.
    for name in ['agroup', 'bgroup', 'grplenexceed', 'goodlen', 'acount', 'bcount']:
        os.remove('{}/group/{}.npy'.format(joint_folder_path, name))

    alist = np.load('{}/group/alist.npy'.format(joint_folder_path), mmap_mode='r+')
    blist = np.load('{}/group/blist.npy'.format(joint_folder_path), mmap_mode='r+')
    agrouplengths = np.load('{}/group/agrplen.npy'.format(joint_folder_path), mmap_mode='r+')
    bgrouplengths = np.load('{}/group/bgrplen.npy'.format(joint_folder_path), mmap_mode='r+')

    return alist, blist, agrouplengths, bgrouplengths


def _initial_group_numbering(aindices, bindices, aoverlap, boverlap, joint_folder_path):
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
    joint_folder_path : string
        Location of top-level folder containing the "group" folder in which
        index and overlap arrays are stored.

    Returns
    -------
    agroup : numpy.ndarray
        Array detailing the group number of each catalogue "a" source.
    bgroup : numpy.ndarray
        Array detailing the group number of each catalogue "b" source.
    '''
    agroup = np.lib.format.open_memmap('{}/group/agroup.npy'.format(joint_folder_path), mode='w+',
                                       dtype=int, shape=(len(aoverlap),))
    bgroup = np.lib.format.open_memmap('{}/group/bgroup.npy'.format(joint_folder_path), mode='w+',
                                       dtype=int, shape=(len(boverlap),))

    agroup[:] = 0
    bgroup[:] = 0

    group_num = 0
    # First search for catalogue "a" sources that are either lonely, with no
    # catalogue "b" object near them, or those with only one corresponding
    # object in catalogue "b", then iteratively group from a -> b -> a -> ...
    # any remaining objects.
    for i in range(0, len(agroup)):
        if aoverlap[i] == 0:
            group_num += 1
            agroup[i] = group_num
        elif aoverlap[i] == 1 and boverlap[aindices[0, i]] == 1:
            group_num += 1
            agroup[i] = group_num
            bgroup[aindices[0, i]] = group_num

    for i in range(0, len(bgroup)):
        if boverlap[i] == 0:
            group_num += 1
            bgroup[i] = group_num

    for i in range(0, len(agroup)):
        if agroup[i] == 0:
            group_num += 1
            _a_to_b(i, group_num, aoverlap[i], aindices, bindices,
                    aoverlap, boverlap, agroup, bgroup)

    return agroup, bgroup


def _a_to_b(ind, grp, N, aindices, bindices, aoverlap, boverlap, agroup, bgroup):
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
    N : integer
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
    for q in aindices[:N, ind]:
        if bgroup[q] != grp:
            _b_to_a(q, grp, boverlap[q], aindices, bindices, aoverlap, boverlap, agroup, bgroup)
    return


def _b_to_a(ind, grp, N, aindices, bindices, aoverlap, boverlap, agroup, bgroup):
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
    N : integer
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
    for f in bindices[:N, ind]:
        if agroup[f] != grp:
            _a_to_b(f, grp, aoverlap[f], aindices, bindices, aoverlap, boverlap, agroup, bgroup)
    return
