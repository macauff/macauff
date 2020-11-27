# Licensed under a 3-clause BSD style license - see LICENSE
'''
This module provides the framework for the creation of the photometric likelihoods
used in the cross-matching of the two catalogues.
'''

import sys
import numpy as np

from .misc_functions_fortran import misc_functions_fortran as mff

__all__ = ['compute_photometric_likelihoods']


def compute_photometric_likelihoods(joint_folder_path, a_cat_folder_path, b_cat_folder_path,
                                    afilts, bfilts, mem_chunk_num, cf_points, cf_areas,
                                    include_phot_like, use_phot_priors):
    '''

    '''

    print("Creating c(m, m) and f(m)...")

    print("Distributing sources into sky slices...")
    sys.stdout.flush()

    a_sky_inds = distribute_sky_indices(joint_folder_path, a_cat_folder_path, 'a', mem_chunk_num,
                                        cf_points)
    b_sky_inds = distribute_sky_indices(joint_folder_path, b_cat_folder_path, 'b', mem_chunk_num,
                                        cf_points)

    print("Making bins...")
    sys.stdout.flush()

    abinlengths, abinsarray, longabinlen = create_magnitude_bins(
        cf_points, afilts, mem_chunk_num, joint_folder_path, a_cat_folder_path, 'a', a_sky_inds)
    bbinlengths, bbinsarray, longbbinlen = create_magnitude_bins(
        cf_points, bfilts, mem_chunk_num, joint_folder_path, b_cat_folder_path, 'b', b_sky_inds)

    print("Calculating PDFs...")
    sys.stdout.flush()

    c_priors = np.lib.format.open_memmap(
        '{}/phot_like/c_priors.npy'.format(joint_folder_path), mode='w+', dtype=float,
        shape=(len(bfilts), len(afilts), len(cf_points)), fortran_order=True)
    fa_priors = np.lib.format.open_memmap(
        '{}/phot_like/fa_priors.npy'.format(joint_folder_path), mode='w+', dtype=float,
        shape=(len(bfilts), len(afilts), len(cf_points)), fortran_order=True)
    fb_priors = np.lib.format.open_memmap(
        '{}/phot_like/fb_priors.npy'.format(joint_folder_path), mode='w+', dtype=float,
        shape=(len(bfilts), len(afilts), len(cf_points)), fortran_order=True)
    c_array = np.lib.format.open_memmap(
        '{}/phot_like/c_array.npy'.format(joint_folder_path), mode='w+', dtype=float,
        shape=(longbbinlen-1, longabinlen-1, len(bfilts), len(afilts),
               len(cf_points)), fortran_order=True)
    fa_array = np.lib.format.open_memmap(
        '{}/phot_like/fa_array.npy'.format(joint_folder_path), mode='w+', dtype=float,
        shape=(longabinlen-1, len(bfilts), len(afilts), len(cf_points)), fortran_order=True)
    fb_array = np.lib.format.open_memmap(
        '{}/phot_like/fb_array.npy'.format(joint_folder_path), mode='w+', dtype=float,
        shape=(longbbinlen-1, len(bfilts), len(afilts), len(cf_points)), fortran_order=True)

    # Within each loop, since we've already assigned all sources to have a closest
    # c/f sky position pointing to be assigned to, we simply slice within the _load
    # functions on ID, and for the initial mem_chunk_num stepping load in the range
    # [m_, m_+mem_chunk_num).
    for m_ in range(0, len(cf_points), mem_chunk_num):
        a_small_phot, a_sky_ind_small = _load_multiple_sky_slice(
            joint_folder_path, 'a', m_, m_+mem_chunk_num, a_cat_folder_path, a_sky_inds)
        b_small_phot, b_sky_ind_small = _load_multiple_sky_slice(
            joint_folder_path, 'b', m_, m_+mem_chunk_num, b_cat_folder_path, b_sky_inds)
        for m in range(m_, min(len(cf_points), m_+mem_chunk_num)):
            area = cf_areas[m]
            a_sky_cut = _load_single_sky_slice(
                joint_folder_path, 'a', m, a_cat_folder_path, a_sky_ind_small)
            a_phot_cut = a_small_phot[a_sky_cut]
            b_sky_cut = _load_single_sky_slice(
                joint_folder_path, 'b', m, b_cat_folder_path, b_sky_ind_small)
            b_phot_cut = b_small_phot[b_sky_cut]
            for i in range(0, len(afilts)):
                if not include_phot_like and not use_phot_priors:
                    a_num_phot_cut = np.sum(~np.isnan(a_phot_cut[:, i]))
                    Na = a_num_phot_cut / area
                for j in range(0, len(bfilts)):
                    if not include_phot_like and not use_phot_priors:
                        b_num_phot_cut = np.sum(~np.isnan(b_phot_cut[:, j]))
                        Nb = b_num_phot_cut / area
                        # Without using photometric-based priors, all we can
                        # do is set the prior on one catalogue to 0.5 -- that
                        # is, equal chance of match or non-match; for this we
                        # use the less dense of the two catalogues as our
                        # "one-sided" match. Then, accordingly, we update the
                        # "field" source density of the more dense catalogue
                        # with its corresponding density, based on the input
                        # density and the counterpart density calculated.
                        c_prior = min(Na, Nb) / 2
                        fa_prior = Na - c_prior
                        fb_prior = Nb - c_prior
                        # To fake no photometric likelihoods, simply set all
                        # values to one, to cancel in the ratio later.
                        c_like, fa_like, fb_like = 1, 1, 1
                    elif not include_phot_like:
                        raise NotImplementedError("Only one-sided, asymmetric photometric "
                                                  "priors can currently be used.")
                    else:
                        raise NotImplementedError("Photometric likelihoods not currently "
                                                  "implemented. Please set include_phot_like "
                                                  "to False.")

                    c_priors[j, i, m] = c_prior
                    fa_priors[j, i, m] = fa_prior
                    fb_priors[j, i, m] = fb_prior
                    c_array[:bbinlengths[j, m]-1,
                            :abinlengths[i, m]-1, j, i, m] = c_like
                    fa_array[:abinlengths[i, m]-1, j, i, m] = fa_like
                    fb_array[:bbinlengths[j, m]-1, j, i, m] = fb_like

    # *binsarray is passed back from create_magnitude_bins as a memmapped array,
    # but *binlengths is just a numpy array, so quickly save these before returning.
    np.save('{}/phot_like/abinlengths.npy'.format(joint_folder_path), abinlengths)
    np.save('{}/phot_like/bbinlengths.npy'.format(joint_folder_path), bbinlengths)

    return


def distribute_sky_indices(joint_folder_path, cat_folder, name, mem_chunk_num, cf_points):
    n_sources = len(np.load('{}/con_cat_astro.npy'.format(cat_folder), mmap_mode='r'))
    sky_inds = np.lib.format.open_memmap('{}/phot_like/{}_sky_inds.npy'.format(
                                         joint_folder_path, name), mode='w+', dtype=int,
                                         shape=(n_sources,), fortran_order=True)
    for cnum in range(0, mem_chunk_num):
        lowind = np.floor(n_sources*cnum/mem_chunk_num).astype(int)
        highind = np.floor(n_sources*(cnum+1)/mem_chunk_num).astype(int)
        a = np.load('{}/con_cat_astro.npy'.format(cat_folder), mmap_mode='r')[lowind:highind]
        sky_inds[lowind:highind] = mff.find_nearest_point(a[:, 0], a[:, 1],
                                                          cf_points[:, 0], cf_points[:, 1])

    return sky_inds


def create_magnitude_bins(cf_points, filts, mem_chunk_num, joint_folder_path,
                          cat_folder_path, cat_type, sky_inds):
    binlengths = np.empty((len(filts), len(cf_points)), int)

    for m_ in range(0, len(cf_points), mem_chunk_num):
        a_phot_, sky_inds_ = _load_multiple_sky_slice(
            joint_folder_path, cat_type, m_, m_+mem_chunk_num, cat_folder_path, sky_inds)
        for m in range(m_, min(len(cf_points), m_+mem_chunk_num)):
            sky_cut = _load_single_sky_slice(
                joint_folder_path, cat_type, m, cat_folder_path, sky_inds_)
            for i in range(0, len(filts)):
                a = a_phot_[sky_cut, i]
                if np.sum(~np.isnan(a)) > 0:
                    f = make_bins(a[~np.isnan(a)])
                else:
                    f = np.array([0])
                del a
                binlengths[i, m] = len(f)

    longbinlen = np.amax(binlengths)
    binsarray = np.lib.format.open_memmap(
        '{}/phot_like/{}binsarray.npy'.format(joint_folder_path, cat_type), mode='w+', dtype=float,
        shape=(longbinlen, len(filts), len(cf_points)), fortran_order=True)
    binsarray[:, :, :] = -1
    for m_ in range(0, len(cf_points), mem_chunk_num):
        a_phot_, sky_inds_ = _load_multiple_sky_slice(
            joint_folder_path, cat_type, m_, m_+mem_chunk_num, cat_folder_path, sky_inds)
        for m in range(m_, min(len(cf_points), m_+mem_chunk_num)):
            sky_cut = _load_single_sky_slice(
                joint_folder_path, cat_type, m, cat_folder_path, sky_inds_)
            for i in range(0, len(filts)):
                a = a_phot_[sky_cut, i]
                if np.sum(~np.isnan(a)) > 0:
                    f = make_bins(a[~np.isnan(a)])
                else:
                    f = np.array([0])
                del a
                binsarray[:binlengths[i, m], i, m] = f

    return binlengths, binsarray, longbinlen


def make_bins(input_mags):
    '''
    Calculate bins for a catalogue's magnitude distribution, ensuring all stars
    are in histogram bins of sufficient number statistics.

    Parameters
    ----------
    input_mags : numpy.ndarray
        Array of magnitudes of given filter from the specific catalogue, to be
        placed in a histogram.

    Returns
    -------
    output_bins : numpy.ndarray
        Bins for the given catalogue-filter combination that produce robust
        numbers of sources within each magnitude interval.
    '''
    minamag = np.amin(input_mags)
    maxamag = np.amax(input_mags)
    da = 0.1
    maxa = da*np.ceil(maxamag/da)
    mina = da*np.floor(minamag/da)
    na = int((maxa - mina)/da + 1)
    output_bins = np.linspace(mina, maxa, na)

    hist, output_bins = np.histogram(input_mags, bins=output_bins)
    smalllist = []
    # Minimum number statistics in each 1-D bin.
    minnum = 250

    for i in range(0, len(output_bins)-1):
        if hist[i] < minnum:
            smalllist.extend([i])
    smalllist = np.array(smalllist)
    dellist = []
    if len(smalllist) > 0:
        for i in smalllist:
            if i not in dellist:
                flag = 0
                for j in range(i+1, len(output_bins)-1):
                    if np.sum(hist[i:j+1]) > minnum:
                        dellist.extend([k for k in range(i+1, j+1)])
                        flag = 1
                        break
                if flag == 0:
                    dellist.extend([k for k in range(i+1, len(output_bins)-1)])
    output_bins = np.delete(output_bins, dellist)

    return output_bins


def _load_multiple_sky_slice(joint_folder_path, cat_name, ind1, ind2, cat_folder_path, sky_inds):
    sky_cut = np.lib.format.open_memmap('{}/{}_temporary_sky_slice_combined.npy'.format(
        joint_folder_path, cat_name), mode='w+', dtype=np.bool, shape=(len(sky_inds),))

    di = max(1, len(sky_inds) // 20)

    for i in range(0, len(sky_inds), di):
        sky_cut[i:i+di] = (sky_inds[i:i+di] >= ind1) & (sky_inds[i:i+di] < ind2)

    photo_cutout = np.load('{}/con_cat_photo.npy'.format(cat_folder_path), mmap_mode='r')[sky_cut]
    sky_ind_cutout = np.load('{}/phot_like/{}_sky_inds.npy'.format(joint_folder_path, cat_name),
                             mmap_mode='r')[sky_cut]

    return photo_cutout, sky_ind_cutout


def _load_single_sky_slice(joint_folder_path, cat_name, ind, cat_folder_path, sky_inds):
    sky_cut = np.lib.format.open_memmap('{}/{}_small_sky_slice.npy'.format(
        joint_folder_path, cat_name), mode='w+', dtype=np.bool, shape=(len(sky_inds),))

    di = max(1, len(sky_inds) // 20)

    for i in range(0, len(sky_inds), di):
        sky_cut[i:i+di] = sky_inds[i:i+di] == ind

    return sky_cut
