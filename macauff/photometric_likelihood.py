# Licensed under a 3-clause BSD style license - see LICENSE
'''
This module provides the framework for the creation of the photometric likelihoods
used in the cross-matching of the two catalogues.
'''

import sys
import numpy as np


__all__ = ['compute_photometric_likelihoods']


def compute_photometric_likelihoods(joint_folder_path, a_cat_folder_path, b_cat_folder_path,
                                    afilts, bfilts, mem_chunk_num, ax1slices, ax2slices,
                                    include_phot_like, use_phot_priors):
    '''

    '''

    print("Creating c(m, m) and f(m)...")
    print("Making bins...")
    sys.stdout.flush()

    abinlengths, abinsarray, longabinlen = create_magnitude_bins(
        ax1slices, ax2slices, afilts, mem_chunk_num, joint_folder_path, a_cat_folder_path, 'a')
    bbinlengths, bbinsarray, longbbinlen = create_magnitude_bins(
        ax1slices, ax2slices, bfilts, mem_chunk_num, joint_folder_path, b_cat_folder_path, 'b')

    print("Calculating PDFs...")
    sys.stdout.flush()

    c_priors = np.lib.format.open_memmap(
        '{}/phot_like/c_priors.npy'.format(joint_folder_path), mode='w+', dtype=float,
        shape=(len(bfilts), len(afilts), len(ax2slices)-1, len(ax1slices)-1), fortran_order=True)
    fa_priors = np.lib.format.open_memmap(
        '{}/phot_like/fa_priors.npy'.format(joint_folder_path), mode='w+', dtype=float,
        shape=(len(bfilts), len(afilts), len(ax2slices)-1, len(ax1slices)-1), fortran_order=True)
    fb_priors = np.lib.format.open_memmap(
        '{}/phot_like/fb_priors.npy'.format(joint_folder_path), mode='w+', dtype=float,
        shape=(len(bfilts), len(afilts), len(ax2slices)-1, len(ax1slices)-1), fortran_order=True)
    c_array = np.lib.format.open_memmap(
        '{}/phot_like/c_array.npy'.format(joint_folder_path), mode='w+', dtype=float,
        shape=(longbbinlen-1, longabinlen-1, len(bfilts), len(afilts),
               len(ax2slices)-1, len(ax1slices)-1), fortran_order=True)
    fa_array = np.lib.format.open_memmap(
        '{}/phot_like/fa_array.npy'.format(joint_folder_path), mode='w+', dtype=float,
        shape=(longabinlen-1, len(bfilts), len(afilts), len(ax2slices)-1, len(ax1slices)-1),
        fortran_order=True)
    fb_array = np.lib.format.open_memmap(
        '{}/phot_like/fb_array.npy'.format(joint_folder_path), mode='w+', dtype=float,
        shape=(longbbinlen-1, len(bfilts), len(afilts), len(ax2slices)-1, len(ax1slices)-1),
        fortran_order=True)

    a_mm = np.load('{}/con_cat_astro.npy'.format(a_cat_folder_path), mmap_mode='r')
    b_mm = np.load('{}/con_cat_astro.npy'.format(b_cat_folder_path), mmap_mode='r')

    for m_ in range(0, len(ax1slices)-1, mem_chunk_num):
        ax1_1 = ax1slices[m_]
        ax1_2 = ax1slices[min(len(ax1slices)-1, m_+mem_chunk_num)]
        a_small, a_small_phot = _load_lon_slice(a_mm, joint_folder_path, 'a', ax1_1, ax1_2,
                                                a_cat_folder_path)
        b_small, b_small_phot = _load_lon_slice(b_mm, joint_folder_path, 'b', ax1_1, ax1_2,
                                                b_cat_folder_path)
        for m in range(m_, min(len(ax1slices)-1, m_+mem_chunk_num)):
            ax1_1 = ax1slices[m]
            ax1_2 = ax1slices[m+1]
            for n in range(0, len(ax2slices)-1):
                ax2_1 = ax2slices[n]
                ax2_2 = ax2slices[n+1]
                area = (ax2_2 - ax2_1) * (ax1_2 - ax1_1)
                a_sky_cut = _load_lon_lat_slice(a_small, joint_folder_path, 'a', ax1_1,
                                                ax1_2, ax2_1, ax2_2, a_cat_folder_path)
                a_phot_cut = a_small_phot[a_sky_cut]
                b_sky_cut = _load_lon_lat_slice(b_small, joint_folder_path, 'b', ax1_1,
                                                ax1_2, ax2_1, ax2_2, b_cat_folder_path)
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

                        c_priors[j, i, n, m] = c_prior
                        fa_priors[j, i, n, m] = fa_prior
                        fb_priors[j, i, n, m] = fb_prior
                        c_array[:bbinlengths[j, n, m]-1,
                                :abinlengths[i, n, m]-1, j, i, n, m] = c_like
                        fa_array[:abinlengths[i, n, m]-1, j, i, n, m] = fa_like
                        fb_array[:bbinlengths[j, n, m]-1, j, i, n, m] = fb_like

    # *binsarray is passed back from create_magnitude_bins as a memmapped array,
    # but *binlengths is just a numpy array, so quickly save these before returning.
    np.save('{}/phot_like/abinlengths.npy'.format(joint_folder_path), abinlengths)
    np.save('{}/phot_like/bbinlengths.npy'.format(joint_folder_path), bbinlengths)

    return


def create_magnitude_bins(ax1slices, ax2slices, filts, mem_chunk_num, joint_folder_path,
                          cat_folder_path, cat_type):
    binlengths = np.empty((len(filts), len(ax2slices)-1, len(ax1slices)-1), int)

    for m_ in range(0, len(ax1slices)-1, mem_chunk_num):
        ax1_1 = ax1slices[m_]
        ax1_2 = ax1slices[min(len(ax1slices)-1, m_+mem_chunk_num)]
        a_ = np.load('{}/con_cat_astro.npy'.format(cat_folder_path), mmap_mode='r')
        a_, a_phot_ = _load_lon_slice(a_, joint_folder_path, cat_type, ax1_1, ax1_2,
                                      cat_folder_path)
        for m in range(m_, min(len(ax1slices)-1, m_+mem_chunk_num)):
            ax1_1 = ax1slices[m]
            ax1_2 = ax1slices[m+1]
            for n in range(0, len(ax2slices)-1):
                ax2_1 = ax2slices[n]
                ax2_2 = ax2slices[n+1]
                sky_cut = _load_lon_lat_slice(a_, joint_folder_path, cat_type, ax1_1, ax1_2,
                                              ax2_1, ax2_2, cat_folder_path)
                for i in range(0, len(filts)):
                    a = a_phot_[sky_cut, i]
                    if np.sum(~np.isnan(a)) > 0:
                        f = make_bins(a[~np.isnan(a)])
                    else:
                        f = np.array([0])
                    del a
                    binlengths[i, n, m] = len(f)
    del a_
    longbinlen = np.amax(binlengths)
    binsarray = np.lib.format.open_memmap(
        '{}/phot_like/{}binsarray.npy'.format(joint_folder_path, cat_type), mode='w+', dtype=float,
        shape=(longbinlen, len(filts), len(ax2slices)-1, len(ax1slices)-1), fortran_order=True)
    binsarray[:, :, :, :] = -1
    for m_ in range(0, len(ax1slices)-1, mem_chunk_num):
        ax1_1 = ax1slices[m_]
        ax1_2 = ax1slices[min(len(ax1slices)-1, m_+mem_chunk_num)]
        a_ = np.load('{}/con_cat_astro.npy'.format(cat_folder_path), mmap_mode='r')
        a_, a_phot_ = _load_lon_slice(a_, joint_folder_path, cat_type, ax1_1, ax1_2,
                                      cat_folder_path)
        for m in range(m_, min(len(ax1slices)-1, m_+mem_chunk_num)):
            ax1_1 = ax1slices[m]
            ax1_2 = ax1slices[m+1]
            for n in range(0, len(ax2slices)-1):
                ax2_1 = ax2slices[n]
                ax2_2 = ax2slices[n+1]
                sky_cut = _load_lon_lat_slice(a_, joint_folder_path, cat_type, ax1_1, ax1_2,
                                              ax2_1, ax2_2, cat_folder_path)
                for i in range(0, len(filts)):
                    a = a_phot_[sky_cut, i]
                    if np.sum(~np.isnan(a)) > 0:
                        f = make_bins(a[~np.isnan(a)])
                    else:
                        f = np.array([0])
                    del a
                    binsarray[:binlengths[i, n, m], i, n, m] = f

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


def _load_lon_slice(a, joint_folder_path, cat_name, lon1, lon2, cat_folder_path):
    sky_cut_1 = np.lib.format.open_memmap('{}/{}_temporary_sky_slice_1.npy'.format(
        joint_folder_path, cat_name), mode='w+', dtype=np.bool, shape=(len(a),))
    sky_cut_2 = np.lib.format.open_memmap('{}/{}_temporary_sky_slice_2.npy'.format(
        joint_folder_path, cat_name), mode='w+', dtype=np.bool, shape=(len(a),))
    sky_cut = np.lib.format.open_memmap('{}/{}_temporary_sky_slice_combined.npy'.format(
        joint_folder_path, cat_name), mode='w+', dtype=np.bool, shape=(len(a),))

    di = max(1, len(a) // 20)

    for i in range(0, len(a), di):
        sky_cut_1[i:i+di] = (a[i:i+di, 0] >= lon1)
    for i in range(0, len(a), di):
        sky_cut_2[i:i+di] = (a[i:i+di, 0] <= lon2)

    for i in range(0, len(a), di):
        sky_cut[i:i+di] = (sky_cut_1[i:i+di] & sky_cut_2[i:i+di])

    a_cutout = np.load('{}/con_cat_astro.npy'.format(cat_folder_path), mmap_mode='r')[sky_cut]
    a_phot_cutout = np.load('{}/con_cat_photo.npy'.format(cat_folder_path), mmap_mode='r')[sky_cut]

    return a_cutout, a_phot_cutout


def _load_lon_lat_slice(a, joint_folder_path, cat_name, lon1, lon2, lat1, lat2, cat_folder_path):
    sky_cut_1 = np.lib.format.open_memmap('{}/{}_temporary_sky_slice_1.npy'.format(
        joint_folder_path, cat_name), mode='w+', dtype=np.bool, shape=(len(a),))
    sky_cut_2 = np.lib.format.open_memmap('{}/{}_temporary_sky_slice_2.npy'.format(
        joint_folder_path, cat_name), mode='w+', dtype=np.bool, shape=(len(a),))
    sky_cut_3 = np.lib.format.open_memmap('{}/{}_temporary_sky_slice_3.npy'.format(
        joint_folder_path, cat_name), mode='w+', dtype=np.bool, shape=(len(a),))
    sky_cut_4 = np.lib.format.open_memmap('{}/{}_temporary_sky_slice_4.npy'.format(
        joint_folder_path, cat_name), mode='w+', dtype=np.bool, shape=(len(a),))
    sky_cut = np.lib.format.open_memmap('{}/{}_temporary_sky_slice_combined.npy'.format(
        joint_folder_path, cat_name), mode='w+', dtype=np.bool, shape=(len(a),))

    di = max(1, len(a) // 20)

    for i in range(0, len(a), di):
        sky_cut_1[i:i+di] = (a[i:i+di, 0] >= lon1)
    for i in range(0, len(a), di):
        sky_cut_2[i:i+di] = (a[i:i+di, 0] <= lon2)

    for i in range(0, len(a), di):
        sky_cut_3[i:i+di] = (a[i:i+di, 1] >= lat1)
    for i in range(0, len(a), di):
        sky_cut_4[i:i+di] = (a[i:i+di, 1] <= lat2)

    for i in range(0, len(a), di):
        sky_cut[i:i+di] = (sky_cut_1[i:i+di] & sky_cut_2[i:i+di] &
                           sky_cut_3[i:i+di] & sky_cut_4[i:i+di])

    return sky_cut
