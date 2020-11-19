# Licensed under a 3-clause BSD style license - see LICENSE
'''
This module provides the framework for the creation of the photometric likelihoods
used in the cross-matching of the two catalogues.
'''

import sys
import os
import numpy as np


__all__ = ['compute_photometric_likelihoods']


def compute_photometric_likelihoods():
    '''

    '''

    print("Creating c(m, m) and f(m)...")
    print("Making bins...")
    sys.stdout.flush()

    create_magnitude_bins(ax1slices, ax2slices, afilts, mem_chunk_num, joint_folder_path,
                          a_cat_folder_path, 'a')
    create_magnitude_bins(ax1slices, ax2slices, bfilts, mem_chunk_num, joint_folder_path,
                          b_cat_folder_path, 'b')

    return


def create_magnitude_bins(ax1slices, ax2slices, filts, mem_chunk_num, joint_folder_path,
                          cat_folder_path, cat_type):
    binlengths = np.empty((len(filts), len(ax2slices)-1, len(ax1slices)-1), int)

    for m_ in range(0, len(ax1slices)-1, mem_chunk_num):
        ax1_1 = ax1slices[m_]
        ax1_2 = ax1slices[min(len(ax1slices)-1, m_+mem_chunk_num)]
        # TODO: update this part
        a_ = np.load('{}/con_cat_astro.npy'.format(cat_folder_path), mmap_mode='r')[:, 0]
        cutspacea = (a_ >= ax1_1) & (a_ < ax1_2)
        a_ = np.load('{}/con_cat_astro.npy'.format(cat_folder_path), mmap_mode='r')[cutspacea]
        del cutspacea
        for m in range(m_, min(len(ax1slices)-1, m_+mem_chunk_num)):
            ax1_1 = ax1slices[m]
            ax1_2 = ax1slices[m+1]
            for n in range(0, len(ax2slices)-1):
                ax2_1 = ax2slices[n]
                ax2_2 = ax2slices[n+1]
                cutspacea = (a_[:, 0] >= ax1_1) & (a_[:, 0] < ax1_2) & (a_[:, 1] >= ax2_1) & \
                            (a_[:, 1] < ax2_2)
                for i in range(0, len(filts)):
                    a = a_[cutspacea, i]
                    # TODO: update this check with nans
                    cuta = (a >= -900)
                    if np.sum(cuta) > 0:
                        f = make_bins(a[a >= -900])
                    else:
                        f = np.array([])
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
        a_ = np.load('{}/con_cat_astro.npy'.format(cat_folder_path), mmap_mode='r')[:, 0]
        cutspacea = (a_ >= ax1_1) & (a_ < ax1_2)
        a_ = np.load('{}/con_cat_astro.npy'.format(cat_folder_path), mmap_mode='r')[cutspacea]
        del cutspacea
        for m in range(m_, min(len(ax1slices)-1, m_+mem_chunk_num)):
            ax1_1 = ax1slices[m]
            ax1_2 = ax1slices[m+1]
            for n in range(0, len(ax2slices)-1):
                ax2_1 = ax2slices[n]
                ax2_2 = ax2slices[n+1]
                cutspacea = (a_[:, 0] >= ax1_1) & (a_[:, 0] < ax1_2) & (a_[:, 1] >= ax2_1) & \
                            (a_[:, 1] < ax2_2)
                for i in range(0, len(filts)):
                    a = a_[cutspacea, i]
                    cuta = (a >= -900)
                    if np.sum(cuta) > 0:
                        f = make_bins(a[a >= -900])
                    else:
                        f = np.array([])
                    del a
                    binsarray[:binlengths[i, n, m], i, n, m] = f

    return binlengths, binsarray


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
    output_bins[0] = da/10*np.floor(minamag/(da/10))-(da/20)
    output_bins[-1] = da/10*np.ceil(maxamag/(da/10))+(da/20)

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
                for j in range(i+1, len(output_bins)):
                    if np.sum(hist[i:j+1]) > minnum:
                        dellist.extend([k for k in range(i+1, j+1)])
                        flag = 1
                        break
                if flag == 0:
                    dellist.extend([k for k in range(i, len(output_bins)-1)])
    output_bins = np.delete(output_bins, dellist)

    return output_bins
