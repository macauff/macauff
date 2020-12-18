# Licensed under a 3-clause BSD style license - see LICENSE
'''
Tests for the "counterpart_pairing" module.
'''

import os
from numpy.testing import assert_allclose
import numpy as np
import pytest

from ..counterpart_pairing import source_pairing, _individual_island_probability
from ..counterpart_pairing_fortran import counterpart_pairing_fortran as cpf


def test_calculate_contamination_probabilities():
    rho = np.linspace(0, 100, 10000)
    drho = np.diff(rho)

    sigs = np.array([0.1, 0.2, 0.3, 0.4])
    seed = 96473
    rng = np.random.default_rng(seed)
    G = np.empty((len(rho)-1, len(sigs)), float)
    for i in range(len(sigs)):
        G[:, i] = np.exp(-2 * np.pi**2 * (rho[:-1]+drho/2)**2 * sigs[i]**2)
    for sep in rng.uniform(0, 0.5, 10):
        Gcc, Gcn, Gnc, Gnn = cpf.contam_match_prob(
            G[:, 0], G[:, 1], G[:, 2], G[:, 3], rho[:-1]+drho/2, drho, sep)
        for prob, sig in zip([Gcc, Gcn, Gnc, Gnn], sigs):
            assert_allclose(prob, 1/(2*np.pi*sig**2) * np.exp(-0.5 * sep**2 / sig**2),
                            rtol=1e-3, atol=1e-4)


class TestCounterpartPairing:
    def setup_class(self):
        seed = 8888
        rng = np.random.default_rng(seed)
        # Test will have three overlap islands: 2 a sources and 1 b source, and
        # a single lonely a source. The two islands will have two counterparts
        # and a single islands with three "field" sources.
        self.a_astro = np.empty((7, 3), float)
        self.b_astro = np.empty((4, 3), float)
        self.a_sig, self.b_sig = 0.1, 0.08

        self.a_astro[:3, :2] = np.array([[0, 0], [0.1, 0.1], [0.1, 0]])
        self.a_astro[3:6, :2] = self.a_astro[:3, :2] + rng.choice([-1, 1], size=(3, 2)) * \
            rng.uniform(2.1*self.a_sig/3600, 3*self.a_sig/3600, size=(3, 2))
        self.a_astro[6, :2] = np.array([0.1, 0.1])
        # Force the second source in the island with the counterpart a/b pair
        # in "a" to be 2-3 sigma away, while the counterpart is <=1 sigma distant.
        self.b_astro[:3, :2] = self.a_astro[:3, :2] + \
            rng.uniform(-1*self.b_sig/3600, self.b_sig/3600, size=(3, 2))
        # Swap the first and second indexes around
        self.b_astro[:2, 0] = self.b_astro[[1, 0], 0]
        self.b_astro[:2, 1] = self.b_astro[[1, 0], 1]
        self.b_astro[-1, :2] = np.array([0.05, 0.05])
        # Swap the last two indexes as well
        self.b_astro[-2:, 0] = self.b_astro[[-1, -2], 0]
        self.b_astro[-2:, 1] = self.b_astro[[-1, -2], 1]

        self.a_astro[:, 2] = self.a_sig
        self.b_astro[:, 2] = self.b_sig
        # Currently we don't care about the photometry, setting both
        # include_phot_like and use_phot_priors to False, so just fake:
        self.a_photo = np.ones((7, 3), float)
        self.b_photo = np.ones((4, 4), float)

        self.alist = np.array([[0, 3], [1, 4], [2, 5], [6, -1]]).T
        self.alist_ = self.alist
        # This blist doesn't have index 2, so it's unmatched by default.
        self.blist = np.array([[1], [0], [3], [-1]]).T
        self.blist_ = self.blist
        self.agrplen = np.array([2, 2, 2, 1])
        self.bgrplen = np.array([1, 1, 1, 0])

        self.rho = np.linspace(0, 100, 10000)
        self.drho = np.diff(self.rho)
        self.n_fracs = 3

        self.abinlengths = 2*np.ones((3, 1), int)
        self.bbinlengths = 2*np.ones((4, 1), int)
        # Having defaulted all photometry to a magnitude of 1, set bins:
        self.abinsarray = np.asfortranarray(np.broadcast_to(np.array([[[0]], [[2]]]), [2, 3, 1]))
        self.bbinsarray = np.asfortranarray(np.broadcast_to(np.array([[[0]], [[2]]]), [2, 4, 1]))
        self.c_array = np.ones((1, 1, 4, 3, 1), float, order='F')
        self.fa_array = np.ones((1, 4, 3, 1), float, order='F')
        self.fb_array = np.ones((1, 4, 3, 1), float, order='F')
        self.c_priors = np.asfortranarray((3/0.1**2 * 0.5) * np.ones((4, 3, 1), float))
        self.fa_priors = np.asfortranarray((7/0.1**2 - 3/0.1**2 * 0.5) * np.ones((4, 3, 1), float))
        self.fb_priors = np.asfortranarray((3/0.1**2 * 0.5) * np.ones((4, 3, 1), float))

        self.amagref = np.zeros((self.a_astro.shape[0]), int)
        self.bmagref = np.zeros((self.b_astro.shape[0]), int)

        self.amodelrefinds = np.zeros((3, 7), int, order='F')
        self.bmodelrefinds = np.zeros((3, 4), int, order='F')

        self.afrac_grids = np.zeros((self.n_fracs, 1, 3, 1), float, order='F')
        self.aflux_grids = np.zeros((1, 3, 1), float, order='F')
        self.bfrac_grids = np.zeros((self.n_fracs, 1, 4, 1), float, order='F')
        self.bflux_grids = np.zeros((1, 4, 1), float, order='F')
        self.afourier_grids = np.ones((len(self.rho)-1, 1, 3, 1), float, order='F')
        self.bfourier_grids = np.ones((len(self.rho)-1, 1, 4, 1), float, order='F')
        self.a_sky_inds = np.zeros((7), int)
        self.b_sky_inds = np.zeros((4), int)

        # We have to save various files for the source_pairing function to
        # pick up again.
        self.joint_folder_path = 'test_path'
        self.a_cat_folder_path = ' gaia_folder'
        self.b_cat_folder_path = 'wise_folder'
        self.a_auf_folder_path = 'gaia_auf_folder'
        self.b_auf_folder_path = 'wise_auf_folder'
        os.system('rm -r {}'.format(self.joint_folder_path))
        for f in ['pairing', 'phot_like', 'group']:
            os.makedirs('{}/{}'.format(self.joint_folder_path, f), exist_ok=True)
        for f in [self.a_cat_folder_path, self.b_cat_folder_path,
                  self.a_auf_folder_path, self.b_auf_folder_path]:
            os.makedirs(f, exist_ok=True)
        np.save('{}/con_cat_astro.npy'.format(self.a_cat_folder_path), self.a_astro)
        np.save('{}/con_cat_astro.npy'.format(self.b_cat_folder_path), self.b_astro)
        np.save('{}/con_cat_photo.npy'.format(self.a_cat_folder_path), self.a_photo)
        np.save('{}/con_cat_photo.npy'.format(self.b_cat_folder_path), self.b_photo)
        np.save('{}/magref.npy'.format(self.a_cat_folder_path), self.amagref)
        np.save('{}/magref.npy'.format(self.b_cat_folder_path), self.bmagref)

        np.save('{}/group/alist.npy'.format(self.joint_folder_path), self.alist)
        np.save('{}/group/blist.npy'.format(self.joint_folder_path), self.blist)
        np.save('{}/group/agrplen.npy'.format(self.joint_folder_path), self.agrplen)
        np.save('{}/group/bgrplen.npy'.format(self.joint_folder_path), self.bgrplen)

        np.save('{}/phot_like/abinsarray.npy'.format(self.joint_folder_path), self.abinsarray)
        np.save('{}/phot_like/bbinsarray.npy'.format(self.joint_folder_path), self.bbinsarray)
        np.save('{}/phot_like/abinlengths.npy'.format(self.joint_folder_path), self.abinlengths)
        np.save('{}/phot_like/bbinlengths.npy'.format(self.joint_folder_path), self.bbinlengths)

        np.save('{}/phot_like/c_priors.npy'.format(self.joint_folder_path), self.c_priors)
        np.save('{}/phot_like/c_array.npy'.format(self.joint_folder_path), self.c_array)
        np.save('{}/phot_like/fa_priors.npy'.format(self.joint_folder_path), self.fa_priors)
        np.save('{}/phot_like/fa_array.npy'.format(self.joint_folder_path), self.fa_array)
        np.save('{}/phot_like/fb_priors.npy'.format(self.joint_folder_path), self.fb_priors)
        np.save('{}/phot_like/fb_array.npy'.format(self.joint_folder_path), self.fb_array)

        np.save('{}/phot_like/a_sky_inds.npy'.format(self.joint_folder_path), self.a_sky_inds)
        np.save('{}/phot_like/b_sky_inds.npy'.format(self.joint_folder_path), self.b_sky_inds)

        np.save('{}/modelrefinds.npy'.format(self.a_auf_folder_path), self.amodelrefinds)
        np.save('{}/modelrefinds.npy'.format(self.b_auf_folder_path), self.bmodelrefinds)

        np.save('{}/arraylengths.npy'.format(self.a_auf_folder_path), np.ones((3, 1), int))
        np.save('{}/arraylengths.npy'.format(self.b_auf_folder_path), np.ones((4, 1), int))

        # We should have already made fourier_grid for each catalogue.
        np.save('{}/fourier_grid.npy'.format(self.a_auf_folder_path), self.afourier_grids)
        np.save('{}/fourier_grid.npy'.format(self.b_auf_folder_path), self.bfourier_grids)

        self.a_auf_pointings = np.array([[0.0, 0.0]])
        self.b_auf_pointings = np.array([[0.0, 0.0]])
        # However, source_pairing will create the flux and frac grids, this being
        # the first time we should have encountered them.
        self.a_filt_names = np.array(['BP', 'G', 'RP'])
        for i in range(0, len(self.a_filt_names)):
            os.makedirs('{}/0.0/0.0/{}'.format(self.a_auf_folder_path, self.a_filt_names[i]),
                        exist_ok=True)
            np.save('{}/0.0/0.0/{}/frac.npy'.format(self.a_auf_folder_path, self.a_filt_names[i]),
                    np.zeros((self.n_fracs, 1), float))
            np.save('{}/0.0/0.0/{}/flux.npy'.format(self.a_auf_folder_path, self.a_filt_names[i]),
                    np.zeros((1), float))
        self.b_filt_names = np.array(['W1', 'W2', 'W3', 'W4'])
        for i in range(0, len(self.b_filt_names)):
            os.makedirs('{}/0.0/0.0/{}'.format(self.b_auf_folder_path, self.b_filt_names[i]),
                        exist_ok=True)
            np.save('{}/0.0/0.0/{}/frac.npy'.format(self.b_auf_folder_path, self.b_filt_names[i]),
                    np.zeros((self.n_fracs, 1), float))
            np.save('{}/0.0/0.0/{}/flux.npy'.format(self.b_auf_folder_path, self.b_filt_names[i]),
                    np.zeros((1), float))

    def _calculate_prob_integral(self):
        self.o = np.sqrt(self.a_sig**2 + self.b_sig**2) / 3600
        self.sep = np.sqrt(((self.a_astro[0, 0] -
                             self.b_astro[1, 0])*np.cos(np.radians(self.b_astro[1, 1])))**2 +
                           (self.a_astro[0, 1] - self.b_astro[1, 1])**2)
        self.G = 1/(2 * np.pi * self.o**2) * np.exp(-0.5 * self.sep**2 / self.o**2)
        self.sep_wrong = np.sqrt(((self.a_astro[3, 0] -
                                   self.b_astro[1, 0])*np.cos(np.radians(self.a_astro[3, 1])))**2 +
                                 (self.a_astro[3, 1] - self.b_astro[1, 1])**2)
        self.G_wrong = 1/(2 * np.pi * self.o**2) * np.exp(-0.5 * self.sep_wrong**2 / self.o**2)
        self.Nc = self.c_priors[0, 0, 0]
        self.Nfa, self.Nfb = self.fa_priors[0, 0, 0], self.fb_priors[0, 0, 0]

    def test_individual_island_probability(self):
        i = 0
        wrapper = [
            i, self.a_astro, self.a_photo, self.b_astro, self.b_photo, self.alist, self.alist_,
            self.blist, self.blist_, self.agrplen, self.bgrplen, self.c_array, self.fa_array,
            self.fb_array, self.c_priors, self.fa_priors, self.fb_priors, self.amagref,
            self.bmagref, self.amodelrefinds, self.bmodelrefinds, self.abinsarray,
            self.abinlengths, self.bbinsarray, self.bbinlengths, self.afrac_grids,
            self.aflux_grids, self.bfrac_grids, self.bflux_grids, self.afourier_grids,
            self.bfourier_grids, self.a_sky_inds, self.b_sky_inds, self.rho, self.drho,
            self.n_fracs]
        results = _individual_island_probability(wrapper)
        (acrpts, bcrpts, acrptscontp, bcrptscontp, etacrpts, xicrpts, acrptflux, bcrptflux,
         afield, bfield, prob, integral) = results

        assert np.all(acrpts == np.array([0]))
        assert np.all(bcrpts == np.array([1]))
        assert np.all(acrptscontp == np.array([0]))
        assert np.all(bcrptscontp == np.array([0]))
        assert np.all(etacrpts == np.array([0]))

        self._calculate_prob_integral()
        assert_allclose(xicrpts, np.array([np.log10(self.G / self.fa_priors[0, 0, 0])]), rtol=1e-6)
        assert np.all(acrptflux == np.array([0]))
        assert np.all(bcrptflux == np.array([0]))
        assert np.all(afield == np.array([3]))
        assert len(bfield) == 0

        _integral = self.Nc*self.G*self.Nfa + self.Nc*self.G_wrong*self.Nfa +\
            self.Nfa*self.Nfa*self.Nfb
        assert_allclose(integral, _integral, rtol=1e-5)
        _prob = self.Nc*self.G*self.Nfa
        assert_allclose(prob, _prob, rtol=1e-5)

    def test_source_pairing(self):
        mem_chunk_num, n_pool = 2, 2
        source_pairing(
            self.joint_folder_path, self.a_cat_folder_path, self.b_cat_folder_path,
            self.a_auf_folder_path, self.b_auf_folder_path, self.a_filt_names, self.b_filt_names,
            self.a_auf_pointings, self.b_auf_pointings, self.rho, self.drho, self.n_fracs,
            mem_chunk_num, n_pool)

        bflux = np.load('{}/pairing/bcontamflux.npy'.format(self.joint_folder_path))
        assert np.all(bflux == np.zeros((3), float))

        a_matches = np.load('{}/pairing/ac.npy'.format(self.joint_folder_path))
        assert np.all([q in a_matches for q in [0, 1, 2]])
        assert np.all([q not in a_matches for q in [3, 4, 5, 6]])

        a_field = np.load('{}/pairing/af.npy'.format(self.joint_folder_path))
        assert np.all([q in a_field for q in [3, 4, 5, 6]])
        assert np.all([q not in a_field for q in [0, 1, 2]])

        b_matches = np.load('{}/pairing/bc.npy'.format(self.joint_folder_path))
        assert np.all([q in b_matches for q in [0, 1, 3]])
        assert np.all([q not in b_matches for q in [2]])

        b_field = np.load('{}/pairing/bf.npy'.format(self.joint_folder_path))
        assert np.all([q in b_field for q in [2]])
        assert np.all([q not in b_field for q in [0, 1, 3]])

        prob_counterpart = np.load('{}/pairing/pc.npy'.format(self.joint_folder_path))
        self._calculate_prob_integral()
        _integral = self.Nc*self.G*self.Nfa + self.Nc*self.G_wrong*self.Nfa +\
            self.Nfa*self.Nfa*self.Nfb
        _prob = self.Nc*self.G*self.Nfa
        norm_prob = _prob/_integral
        q = np.where([a_matches == 0])[0][0]
        assert_allclose(prob_counterpart[q], norm_prob, rtol=1e-5)
        xicrpts = np.load('{}/pairing/xi.npy'.format(self.joint_folder_path))
        assert_allclose(xicrpts[q], np.array([np.log10(self.G / self.fa_priors[0, 0, 0])]),
                        rtol=1e-6)

        prob_a_field = np.load('{}/pairing/pfa.npy'.format(self.joint_folder_path))
        a_field = np.load('{}/pairing/af.npy'.format(self.joint_folder_path))
        q = np.where(a_field == 6)[0][0]
        assert prob_a_field[q] == 1

        prob_b_field = np.load('{}/pairing/pfb.npy'.format(self.joint_folder_path))
        b_field = np.load('{}/pairing/bf.npy'.format(self.joint_folder_path))
        q = np.where(b_field == 2)[0][0]
        assert prob_b_field[q] == 1

    def test_including_b_reject(self):
        # Remove the third group, pretending it's rejected in the group stage.
        alist = self.alist[:, [0, 1, 3]]
        blist = self.blist[:, [0, 1, 3]]
        agrplen = self.agrplen[[0, 1, 3]]
        bgrplen = self.bgrplen[[0, 1, 3]]

        a_reject = np.array([2, 5])
        b_reject = np.array([3])
        os.makedirs('{}/reject'.format(self.joint_folder_path), exist_ok=True)
        np.save('{}/reject/reject_a.npy'.format(self.joint_folder_path), a_reject)
        np.save('{}/reject/reject_b.npy'.format(self.joint_folder_path), b_reject)

        np.save('{}/group/alist.npy'.format(self.joint_folder_path), alist)
        np.save('{}/group/blist.npy'.format(self.joint_folder_path), blist)
        np.save('{}/group/agrplen.npy'.format(self.joint_folder_path), agrplen)
        np.save('{}/group/bgrplen.npy'.format(self.joint_folder_path), bgrplen)

        mem_chunk_num, n_pool = 2, 2
        source_pairing(
            self.joint_folder_path, self.a_cat_folder_path, self.b_cat_folder_path,
            self.a_auf_folder_path, self.b_auf_folder_path, self.a_filt_names,
            self.b_filt_names, self.a_auf_pointings, self.b_auf_pointings, self.rho, self.drho,
            self.n_fracs, mem_chunk_num, n_pool)

        bflux = np.load('{}/pairing/bcontamflux.npy'.format(self.joint_folder_path))
        assert np.all(bflux == np.zeros((2), float))

        a_matches = np.load('{}/pairing/ac.npy'.format(self.joint_folder_path))
        assert np.all([q in a_matches for q in [0, 1]])
        assert np.all([q not in a_matches for q in [2, 3, 4, 5, 6]])

        a_field = np.load('{}/pairing/af.npy'.format(self.joint_folder_path))
        assert np.all([q in a_field for q in [3, 4, 6]])
        assert np.all([q not in a_field for q in [0, 1, 2, 5]])

        b_matches = np.load('{}/pairing/bc.npy'.format(self.joint_folder_path))
        assert np.all([q in b_matches for q in [0, 1]])
        assert np.all([q not in b_matches for q in [2, 3]])

        b_field = np.load('{}/pairing/bf.npy'.format(self.joint_folder_path))
        assert np.all([q in b_field for q in [2]])
        assert np.all([q not in b_field for q in [0, 1, 3]])

        prob_counterpart = np.load('{}/pairing/pc.npy'.format(self.joint_folder_path))
        self._calculate_prob_integral()
        _integral = self.Nc*self.G*self.Nfa + self.Nc*self.G_wrong*self.Nfa +\
            self.Nfa*self.Nfa*self.Nfb
        _prob = self.Nc*self.G*self.Nfa
        norm_prob = _prob/_integral
        q = np.where([a_matches == 0])[0][0]
        assert_allclose(prob_counterpart[q], norm_prob, rtol=1e-5)
        xicrpts = np.load('{}/pairing/xi.npy'.format(self.joint_folder_path))
        assert_allclose(xicrpts[q], np.array([np.log10(self.G / self.fa_priors[0, 0, 0])]),
                        rtol=1e-6)

        prob_a_field = np.load('{}/pairing/pfa.npy'.format(self.joint_folder_path))
        a_field = np.load('{}/pairing/af.npy'.format(self.joint_folder_path))
        q = np.where(a_field == 6)[0][0]
        assert prob_a_field[q] == 1

        prob_b_field = np.load('{}/pairing/pfb.npy'.format(self.joint_folder_path))
        b_field = np.load('{}/pairing/bf.npy'.format(self.joint_folder_path))
        q = np.where(b_field == 2)[0][0]
        assert prob_b_field[q] == 1

    def test_small_length_warnings(self):
        # Here want to test that the number of recorded matches -- either
        # counterpart, field, or rejected -- is lower than the total length.
        # To achieve this we fake reject length arrays smaller than their
        # supposed lengths.
        alist = self.alist[:, [0, 1]]
        blist = self.blist[:, [0, 1]]
        agrplen = self.agrplen[[0, 1]]
        bgrplen = self.bgrplen[[0, 1]]

        # Force catalogue a to have the wrong length by simply leaving a group
        # out of alist.
        a_reject = np.array([6])
        np.save('{}/reject/reject_a.npy'.format(self.joint_folder_path), a_reject)
        os.system('rm {}/reject/reject_b.npy'.format(self.joint_folder_path))

        np.save('{}/group/alist.npy'.format(self.joint_folder_path), alist)
        np.save('{}/group/blist.npy'.format(self.joint_folder_path), blist)
        np.save('{}/group/agrplen.npy'.format(self.joint_folder_path), agrplen)
        np.save('{}/group/bgrplen.npy'.format(self.joint_folder_path), bgrplen)

        mem_chunk_num, n_pool = 2, 2
        with pytest.warns(UserWarning) as record:
            source_pairing(
                self.joint_folder_path, self.a_cat_folder_path, self.b_cat_folder_path,
                self.a_auf_folder_path, self.b_auf_folder_path, self.a_filt_names,
                self.b_filt_names, self.a_auf_pointings, self.b_auf_pointings, self.rho, self.drho,
                self.n_fracs, mem_chunk_num, n_pool)
        assert len(record) == 2
        assert '2 catalogue a sources not in either counterpart, f' in record[0].message.args[0]
        assert '1 catalogue b source not in either counterpart, f' in record[1].message.args[0]

        bflux = np.load('{}/pairing/bcontamflux.npy'.format(self.joint_folder_path))
        assert np.all(bflux == np.zeros((2), float))

        a_matches = np.load('{}/pairing/ac.npy'.format(self.joint_folder_path))
        assert np.all([q in a_matches for q in [0, 1]])
        assert np.all([q not in a_matches for q in [2, 3, 4, 5, 6]])

        a_field = np.load('{}/pairing/af.npy'.format(self.joint_folder_path))
        assert np.all([q in a_field for q in [3, 4]])
        assert np.all([q not in a_field for q in [0, 1, 2, 5, 6]])

        a_reject = np.load('{}/reject/reject_a.npy'.format(self.joint_folder_path))
        assert np.all([q in a_reject for q in [6]])
        assert np.all([q not in a_reject for q in [0, 1, 2, 3, 4, 5]])

        b_matches = np.load('{}/pairing/bc.npy'.format(self.joint_folder_path))
        assert np.all([q in b_matches for q in [0, 1]])
        assert np.all([q not in b_matches for q in [2, 3]])

        b_field = np.load('{}/pairing/bf.npy'.format(self.joint_folder_path))
        assert np.all([q in b_field for q in [2, 3]])
        assert np.all([q not in b_field for q in [0, 1]])

    def test_large_length_warnings(self):
        # Here want to test that the number of recorded matches -- either
        # counterpart, field, or rejected -- is higher than the total length.
        # To achieve this we fake reject length arrays larger than their
        # supposed lengths.
        alist = self.alist[:, [0, 1]]
        blist = self.blist[:, [0, 1]]
        agrplen = self.agrplen[[0, 1]]
        bgrplen = self.bgrplen[[0, 1]]

        a_reject = np.array([2, 3, 4, 5, 6])
        b_reject = np.array([1, 3])
        np.save('{}/reject/reject_a.npy'.format(self.joint_folder_path), a_reject)
        np.save('{}/reject/reject_b.npy'.format(self.joint_folder_path), b_reject)

        np.save('{}/group/alist.npy'.format(self.joint_folder_path), alist)
        np.save('{}/group/blist.npy'.format(self.joint_folder_path), blist)
        np.save('{}/group/agrplen.npy'.format(self.joint_folder_path), agrplen)
        np.save('{}/group/bgrplen.npy'.format(self.joint_folder_path), bgrplen)

        mem_chunk_num, n_pool = 2, 2
        with pytest.warns(UserWarning) as record:
            source_pairing(
                self.joint_folder_path, self.a_cat_folder_path, self.b_cat_folder_path,
                self.a_auf_folder_path, self.b_auf_folder_path, self.a_filt_names,
                self.b_filt_names, self.a_auf_pointings, self.b_auf_pointings, self.rho, self.drho,
                self.n_fracs, mem_chunk_num, n_pool)
        assert len(record) == 2
        assert '2 additional catalogue a indices recorded' in record[0].message.args[0]
        assert '1 additional catalogue b index recorded' in record[1].message.args[0]

        bflux = np.load('{}/pairing/bcontamflux.npy'.format(self.joint_folder_path))
        assert np.all(bflux == np.zeros((2), float))

        a_matches = np.load('{}/pairing/ac.npy'.format(self.joint_folder_path))
        assert np.all([q in a_matches for q in [0, 1]])
        assert np.all([q not in a_matches for q in [2, 3, 4, 5, 6]])

        a_field = np.load('{}/pairing/af.npy'.format(self.joint_folder_path))
        assert np.all([q in a_field for q in [3, 4]])
        assert np.all([q not in a_field for q in [0, 1, 2, 5, 6]])

        a_reject = np.load('{}/reject/reject_a.npy'.format(self.joint_folder_path))
        assert np.all([q in a_reject for q in [2, 3, 4, 5, 6]])
        assert np.all([q not in a_reject for q in [0, 1]])

        b_matches = np.load('{}/pairing/bc.npy'.format(self.joint_folder_path))
        assert np.all([q in b_matches for q in [0, 1]])
        assert np.all([q not in b_matches for q in [2, 3]])

        b_field = np.load('{}/pairing/bf.npy'.format(self.joint_folder_path))
        assert np.all([q in b_field for q in [2]])
        assert np.all([q not in b_field for q in [0, 1, 3]])
