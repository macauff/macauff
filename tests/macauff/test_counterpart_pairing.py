# Licensed under a 3-clause BSD style license - see LICENSE
'''
Tests for the "counterpart_pairing" module.
'''

import itertools
import math
import os

import numpy as np
import pytest
from numpy.testing import assert_allclose
from test_utils import mock_filename

# pylint: disable=no-name-in-module,import-error
from macauff.counterpart_pairing import source_pairing
from macauff.counterpart_pairing_fortran import counterpart_pairing_fortran as cpf
from macauff.macauff import Macauff
from macauff.matching import CrossMatch

# pylint: enable=no-name-in-module,import-error
# pylint: disable=duplicate-code


def test_calculate_contamination_probabilities():
    rho = np.linspace(0, 100, 10000)
    drho = np.diff(rho)

    sigs = np.array([0.1, 0.2, 0.3, 0.4])
    seed = 96473
    rng = np.random.default_rng(seed)
    g = np.empty((len(rho)-1, len(sigs)), float)
    for i, sig in enumerate(sigs):
        g[:, i] = np.exp(-2 * np.pi**2 * (rho[:-1]+drho/2)**2 * sig**2)
    for sep in rng.uniform(0, 0.5, 10):
        gcc, gcn, gnc, gnn = cpf.contam_match_prob(
            g[:, 0], g[:, 1], g[:, 2], g[:, 3], rho[:-1]+drho/2, drho, sep)
        for prob, sig in zip([gcc, gcn, gnc, gnn], sigs):
            assert_allclose(prob, 1/(2*np.pi*sig**2) * np.exp(-0.5 * sep**2 / sig**2),
                            rtol=1e-3, atol=1e-4)


class TestCounterpartPairing:  # pylint: disable=too-many-instance-attributes
    def setup_class(self):  # pylint: disable=too-many-statements
        seed = 8888
        rng = np.random.default_rng(seed)
        # Test will have three overlap islands: two with 2 a sources and 1 b
        # source, with 1 unmatched a source, and one with 2 a + 1 b all unmatched
        # sources. Also two islands each with just one "field" object.
        self.a_cat = np.empty((7, 8), float)
        self.b_cat = np.empty((4, 9), float)
        self.a_sig, self.b_sig = 0.1, 0.08

        self.a_cat[:3, :2] = np.array([[0, 0], [0.1, 0.1], [0.1, 0]])
        self.a_cat[3:6, :2] = self.a_cat[:3, :2] + rng.choice([-1, 1], size=(3, 2)) * \
            rng.uniform(2.1*self.a_sig/3600, 3*self.a_sig/3600, size=(3, 2))
        self.a_cat[6, :2] = np.array([0.1, 0.1])
        # Force the second source in the island with the counterpart a/b pair
        # in "a" to be 2-3 sigma away, while the counterpart is <=1 sigma distant.
        self.b_cat[:3, :2] = self.a_cat[:3, :2] + \
            rng.uniform(-1*self.b_sig/3600, self.b_sig/3600, size=(3, 2))
        # Swap the first and second indexes around
        self.b_cat[:2, 0] = self.b_cat[[1, 0], 0]
        self.b_cat[:2, 1] = self.b_cat[[1, 0], 1]
        self.b_cat[-1, :2] = np.array([0.05, 0.05])
        # Swap the last two indexes as well
        self.b_cat[-2:, 0] = self.b_cat[[-1, -2], 0]
        self.b_cat[-2:, 1] = self.b_cat[[-1, -2], 1]
        # Force no match between the third island by adding distance between
        # a[2] and b[3]. Unphysical but effective.
        self.b_cat[3, 1] += 7*self.b_sig/3600

        self.a_cat[:, 2] = self.a_sig
        self.b_cat[:, 2] = self.b_sig
        # Currently we don't care about the photometry, setting both
        # include_phot_like and use_phot_priors to False, so just fake:
        self.a_cat[:, 3:6] = np.ones((7, 3), float)
        self.b_cat[:, 3:7] = np.ones((4, 4), float)

        self.alist = np.array([[0, 3], [1, 4], [2, 5], [6, -1], [-1, -1]]).T
        self.blist = np.array([[1], [0], [3], [-1], [2]]).T
        self.agrplen = np.array([2, 2, 2, 1, 0])
        self.bgrplen = np.array([1, 1, 1, 0, 1])

        self.rho = np.linspace(0, 100, 10000)
        self.drho = np.diff(self.rho)
        self.n_fracs = 2

        self.abinlengths = 2*np.ones((3, 1), int)
        self.bbinlengths = 2*np.ones((4, 1), int)
        # Having defaulted all photometry to a magnitude of 1, set bins:
        self.abinsarray = np.asfortranarray(np.broadcast_to(np.array([[[0]], [[2]]]), [2, 3, 1]))
        self.bbinsarray = np.asfortranarray(np.broadcast_to(np.array([[[0]], [[2]]]), [2, 4, 1]))
        self.c_array = np.ones((1, 1, 4, 3, 1), float, order='F')
        self.fa_array = np.ones((1, 4, 3, 1), float, order='F')
        self.fb_array = np.ones((1, 4, 3, 1), float, order='F')
        self.c_priors = np.asfortranarray((3/0.001**2 * 0.5) * np.ones((4, 3, 1), float))
        self.fa_priors = np.asfortranarray((7/0.001**2 - 3/0.001**2 * 0.5) *
                                           np.ones((4, 3, 1), float))
        self.fb_priors = np.asfortranarray((3/0.001**2 * 0.5) * np.ones((4, 3, 1), float))

        # best_mag_index and chunk_overlap respectively.
        self.a_cat[:, 7] = np.zeros((self.a_cat.shape[0]), int)
        self.b_cat[:, 8] = np.zeros((self.b_cat.shape[0]), int)
        self.a_cat[:, 6] = np.ones((self.a_cat.shape[0]), bool)
        self.b_cat[:, 7] = np.ones((self.b_cat.shape[0]), bool)

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
        self.joint_folder_path = 'test_path_9'
        self.a_cat_folder_path = 'gaia_folder_9'
        self.b_cat_folder_path = 'wise_folder_9'
        self.a_auf_folder_path = 'gaia_auf_folder_9'
        self.b_auf_folder_path = 'wise_auf_folder_9'

        os.system(f'rm -r {self.joint_folder_path}')
        os.makedirs(f'{self.joint_folder_path}', exist_ok=True)
        for f in [self.a_cat_folder_path, self.b_cat_folder_path,
                  self.a_auf_folder_path, self.b_auf_folder_path]:
            os.makedirs(f, exist_ok=True)
        with open(f'{self.a_cat_folder_path}/gaia.csv', "w", encoding='utf-8') as f:
            np.savetxt(f, self.a_cat, delimiter=",")
        with open(f'{self.b_cat_folder_path}/wise.csv', "w", encoding='utf-8') as f:
            np.savetxt(f, self.b_cat, delimiter=",")

        # We should have already made fourier_grid, frac_grid, and flux_grid
        # for each catalogue.
        self.a_perturb_auf_outputs = {}
        self.b_perturb_auf_outputs = {}
        self.a_perturb_auf_outputs['fourier_grid'] = self.afourier_grids
        self.b_perturb_auf_outputs['fourier_grid'] = self.bfourier_grids
        self.a_perturb_auf_outputs['frac_grid'] = self.afrac_grids
        self.b_perturb_auf_outputs['frac_grid'] = self.bfrac_grids
        self.a_perturb_auf_outputs['flux_grid'] = self.aflux_grids
        self.b_perturb_auf_outputs['flux_grid'] = self.bflux_grids

        self.large_len = max(len(self.a_cat), len(self.b_cat))

        with open(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.yaml'),
                  encoding='utf-8') as cm_p:
            self.cm_p_text = cm_p.read()
        with open(os.path.join(os.path.dirname(__file__), 'data/cat_a_params.yaml'),
                  encoding='utf-8') as ca_p:
            self.ca_p_text = ca_p.read()
        with open(os.path.join(os.path.dirname(__file__), 'data/cat_b_params.yaml'),
                  encoding='utf-8') as cb_p:
            self.cb_p_text = cb_p.read()

    def make_class(self):
        class A():  # pylint: disable=too-few-public-methods
            def __init__(self):
                pass

        a = A()
        a.delta_mag_cuts = np.array([2.5, 5])
        a.a_modelrefinds = self.amodelrefinds
        a.b_modelrefinds = self.bmodelrefinds
        a.a_perturb_auf_outputs = self.a_perturb_auf_outputs
        a.b_perturb_auf_outputs = self.b_perturb_auf_outputs
        a.joint_folder_path = self.joint_folder_path
        a.a_cat_folder_path = self.a_cat_folder_path
        a.b_cat_folder_path = self.b_cat_folder_path
        a.rho = self.rho
        a.drho = self.drho
        a.rank = 0
        a.chunk_id = 1
        a.abinsarray = self.abinsarray
        a.abinlengths = self.abinlengths
        a.bbinsarray = self.bbinsarray
        a.bbinlengths = self.bbinlengths
        a.a_sky_inds = self.a_sky_inds
        a.b_sky_inds = self.b_sky_inds
        a.c_priors = self.c_priors
        a.c_array = self.c_array
        a.fa_priors = self.fa_priors
        a.fa_array = self.fa_array
        a.fb_priors = self.fb_priors
        a.fb_array = self.fb_array
        a.alist = self.alist
        a.blist = self.blist
        a.agrplen = self.agrplen
        a.bgrplen = self.bgrplen
        a.lenrejecta = 0
        a.lenrejectb = 0
        a.a_astro = self.a_cat[:, :3]
        a.a_photo = self.a_cat[:, 3:6]
        a.b_astro = self.b_cat[:, :3]
        a.b_photo = self.b_cat[:, 3:7]
        a.a_magref = self.a_cat[:, 7]
        a.b_magref = self.b_cat[:, 8]

        return a

    def _calculate_prob_integral(self):
        self.o = np.sqrt(self.a_sig**2 + self.b_sig**2) / 3600
        self.sep = np.sqrt(((self.a_cat[0, 0] -
                             self.b_cat[1, 0])*np.cos(np.radians(self.b_cat[1, 1])))**2 +
                           (self.a_cat[0, 1] - self.b_cat[1, 1])**2)
        self.g = 1/(2 * np.pi * self.o**2) * np.exp(-0.5 * self.sep**2 / self.o**2)
        self.sep_wrong = np.sqrt(((self.a_cat[3, 0] -
                                   self.b_cat[1, 0])*np.cos(np.radians(self.a_cat[3, 1])))**2 +
                                 (self.a_cat[3, 1] - self.b_cat[1, 1])**2)
        self.g_wrong = 1/(2 * np.pi * self.o**2) * np.exp(-0.5 * self.sep_wrong**2 / self.o**2)
        self.nc = self.c_priors[0, 0, 0]
        self.nfa, self.nfb = self.fa_priors[0, 0, 0], self.fb_priors[0, 0, 0]

    def test_individual_island_probability(self):
        os.system(f'rm -r {self.joint_folder_path}')
        os.makedirs(f'{self.joint_folder_path}', exist_ok=True)
        i = 0
        wrapper = [
            self.a_cat[:, :3], self.a_cat[:, 3:6], self.b_cat[:, :3], self.b_cat[:, 3:7], self.c_array,
            self.fa_array, self.fb_array, self.c_priors, self.fa_priors, self.fb_priors, self.abinsarray,
            self.bbinsarray, self.abinlengths, self.bbinlengths, self.afrac_grids,
            self.aflux_grids, self.afourier_grids, self.bfrac_grids, self.bflux_grids,
            self.bfourier_grids, self.rho, self.drho, self.n_fracs, self.large_len,
            self.alist[:self.agrplen[i], i]+1, self.blist[:self.bgrplen[i], i]+1,
            self.a_cat[self.alist[:self.agrplen[i], i], 7]+1,
            self.a_sky_inds[self.alist[:self.agrplen[i], i]]+1,
            self.b_cat[self.blist[:self.bgrplen[i], i], 8]+1,
            self.b_sky_inds[self.blist[:self.bgrplen[i], i]]+1,
            self.amodelrefinds[:, self.alist[:self.agrplen[i], i]]+1,
            self.bmodelrefinds[:, self.blist[:self.bgrplen[i], i]]+1]
        results = cpf.find_single_island_prob(*wrapper)

        (acrpts, bcrpts, acrptscontp, bcrptscontp, etacrpts, xicrpts, acrptflux, bcrptflux,
         _, afield, bfield, _, _, _, _, _, _, _, _, prob, integral) = results

        assert np.all(acrpts == np.array([0]))
        assert np.all(bcrpts == np.array([1]))
        assert np.all(acrptscontp == np.array([0]))
        assert np.all(bcrptscontp == np.array([0]))
        assert np.all(etacrpts == np.array([0]))

        self._calculate_prob_integral()
        assert_allclose(xicrpts, np.array([np.log10(self.g / self.fa_priors[0, 0, 0])]), rtol=1e-6)
        assert np.all(acrptflux == np.array([0]))
        assert np.all(bcrptflux == np.array([0]))
        # 2 a vs 1 b with a match means only one rejected a source, but a
        # length 2 array; similarly, bfield will be empty but length one.
        assert np.all(afield == np.array([3, self.large_len+1]))
        assert len(bfield) == 1
        assert np.all(bfield == np.array([self.large_len+1]))

        _integral = self.nc*self.g*self.nfa + self.nc*self.g_wrong*self.nfa +\
            self.nfa*self.nfa*self.nfb
        assert_allclose(integral, _integral, rtol=1e-5)
        _prob = self.nc*self.g*self.nfa
        assert_allclose(prob, _prob, rtol=1e-5)

    def test_individual_island_zero_probabilities(self):
        os.system(f'rm -r {self.joint_folder_path}')
        os.makedirs(f'{self.joint_folder_path}', exist_ok=True)
        # Fake the extra fire extinguisher likelihood/prior used in the main code.
        fa_array = np.zeros_like(self.fa_array) + 1e-10
        fb_array = np.zeros_like(self.fb_array) + 1e-10
        fa_priors = np.zeros_like(self.fa_priors) + 1e-10
        fb_priors = np.zeros_like(self.fb_priors) + 1e-10
        i = 0
        wrapper = [
            self.a_cat[:, :3], self.a_cat[:, 3:6], self.b_cat[:, :3], self.b_cat[:, 3:7], self.c_array,
            fa_array, fb_array, self.c_priors, fa_priors, fb_priors, self.abinsarray,
            self.bbinsarray, self.abinlengths, self.bbinlengths, self.afrac_grids,
            self.aflux_grids, self.afourier_grids, self.bfrac_grids, self.bflux_grids,
            self.bfourier_grids, self.rho, self.drho, self.n_fracs, self.large_len,
            self.alist[:self.agrplen[i], i]+1, self.blist[:self.bgrplen[i], i]+1,
            self.a_cat[self.alist[:self.agrplen[i], i], 7]+1,
            self.a_sky_inds[self.alist[:self.agrplen[i], i]]+1,
            self.b_cat[self.blist[:self.bgrplen[i], i], 8]+1,
            self.b_sky_inds[self.blist[:self.bgrplen[i], i]]+1,
            self.amodelrefinds[:, self.alist[:self.agrplen[i], i]]+1,
            self.bmodelrefinds[:, self.blist[:self.bgrplen[i], i]]+1]
        results = cpf.find_single_island_prob(*wrapper)
        (acrpts, bcrpts, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
         prob, integral) = results

        assert np.all(acrpts == np.array([0]))
        assert np.all(bcrpts == np.array([1]))

        c_array = np.zeros_like(self.c_array) + 1e-10
        c_priors = np.zeros_like(self.c_priors) + 1e-10
        wrapper = [
            self.a_cat[:, :3], self.a_cat[:, 3:6], self.b_cat[:, :3], self.b_cat[:, 3:7], c_array,
            self.fa_array, self.fb_array, c_priors, self.fa_priors, self.fb_priors, self.abinsarray,
            self.bbinsarray, self.abinlengths, self.bbinlengths, self.afrac_grids,
            self.aflux_grids, self.afourier_grids, self.bfrac_grids, self.bflux_grids,
            self.bfourier_grids, self.rho, self.drho, self.n_fracs, self.large_len,
            self.alist[:self.agrplen[i], i]+1, self.blist[:self.bgrplen[i], i]+1,
            self.a_cat[self.alist[:self.agrplen[i], i], 7]+1,
            self.a_sky_inds[self.alist[:self.agrplen[i], i]]+1,
            self.b_cat[self.blist[:self.bgrplen[i], i], 8]+1,
            self.b_sky_inds[self.blist[:self.bgrplen[i], i]]+1,
            self.amodelrefinds[:, self.alist[:self.agrplen[i], i]]+1,
            self.bmodelrefinds[:, self.blist[:self.bgrplen[i], i]]+1]
        results = cpf.find_single_island_prob(*wrapper)
        (acrpts, bcrpts, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
         prob, integral) = results

        assert len(acrpts) == 1
        assert len(bcrpts) == 1
        assert np.all(acrpts == np.array([self.large_len+1]))
        assert np.all(bcrpts == np.array([self.large_len+1]))
        assert_allclose(prob/integral, 1)

    def test_source_pairing(self):
        # pylint: disable=no-member
        fake_cm = self.make_class()
        source_pairing(fake_cm)

        bflux = fake_cm.bcontamflux
        assert np.all(bflux == np.zeros((2), float))

        aflux = fake_cm.afieldflux
        assert np.all(aflux == np.zeros((5), float))
        bflux = fake_cm.bfieldflux
        assert np.all(bflux == np.zeros((2), float))

        a_matches = fake_cm.ac
        assert np.all([q in a_matches for q in [0, 1]])
        assert np.all([q not in a_matches for q in [2, 3, 4, 5, 6]])

        a_field = fake_cm.af
        assert np.all([q in a_field for q in [2, 3, 4, 5, 6]])
        assert np.all([q not in a_field for q in [0, 1]])

        b_matches = fake_cm.bc
        assert np.all([q in b_matches for q in [0, 1]])
        assert np.all([q not in b_matches for q in [2, 3]])

        b_field = fake_cm.bf
        assert np.all([q in b_field for q in [2, 3]])
        assert np.all([q not in b_field for q in [0, 1]])

        prob_counterpart = fake_cm.pc
        self._calculate_prob_integral()
        _integral = self.nc*self.g*self.nfa + self.nc*self.g_wrong*self.nfa +\
            self.nfa*self.nfa*self.nfb
        _prob = self.nc*self.g*self.nfa
        norm_prob = _prob/_integral
        q = np.where(a_matches == 0)[0][0]
        assert_allclose(prob_counterpart[q], norm_prob, rtol=1e-5)
        xicrpts = fake_cm.xi
        assert_allclose(xicrpts[q], np.array([np.log10(self.g / self.fa_priors[0, 0, 0])]),
                        rtol=1e-6)
        csep = fake_cm.crptseps
        assert_allclose(csep[q], self.sep * 3600, rtol=1e-6)
        assert len(csep) == len(a_matches)

        prob_a_field = fake_cm.pfa
        a_field = fake_cm.af
        q = np.where(a_field == 6)[0][0]
        assert prob_a_field[q] == 1

        prob_b_field = fake_cm.pfb
        b_field = fake_cm.bf
        q = np.where(b_field == 2)[0][0]
        assert prob_b_field[q] == 1

        afs = fake_cm.afieldseps
        afeta = fake_cm.afieldeta
        afxi = fake_cm.afieldxi
        q = np.where(a_field == 2)[0][0]
        fake_field_sep = np.sqrt(((self.a_cat[2, 0] -
                                   self.b_cat[3, 0])*np.cos(np.radians(self.b_cat[3, 1])))**2 +
                                 (self.a_cat[2, 1] - self.b_cat[3, 1])**2)
        assert_allclose(afs[q], fake_field_sep * 3600, rtol=1e-6)

        fake_field_g = 1/(2 * np.pi * self.o**2) * np.exp(-0.5 * fake_field_sep**2 / self.o**2)
        # c_priors and fb_priors are the same, so they cancel in the division.
        # Being in log space we can be relatively forgiving in our assertion limits.
        assert_allclose(afxi[q], np.log10(fake_field_g / self.nfa), rtol=0.01, atol=0.01)
        # Ignoring photometry here, so this should be equal probability.
        assert_allclose(afeta[q], np.log10(1.0))

    def test_including_b_reject(self):
        # pylint: disable=no-member
        os.system(f'rm -r {self.joint_folder_path}')
        os.makedirs(f'{self.joint_folder_path}', exist_ok=True)
        # Remove the third group, pretending it's rejected in the group stage.
        alist = self.alist[:, [0, 1, 3, 4]]
        blist = self.blist[:, [0, 1, 3, 4]]
        agrplen = self.agrplen[[0, 1, 3, 4]]
        bgrplen = self.bgrplen[[0, 1, 3, 4]]

        a_reject = np.array([2, 5])
        b_reject = np.array([3])

        fake_cm = self.make_class()
        fake_cm.alist = alist
        fake_cm.blist = blist
        fake_cm.agrplen = agrplen
        fake_cm.bgrplen = bgrplen
        fake_cm.lenrejecta = len(a_reject)
        fake_cm.lenrejectb = len(b_reject)
        source_pairing(fake_cm)

        bflux = fake_cm.bcontamflux
        assert np.all(bflux == np.zeros((2), float))

        a_matches = fake_cm.ac
        assert np.all([q in a_matches for q in [0, 1]])
        assert np.all([q not in a_matches for q in [2, 3, 4, 5, 6]])

        a_field = fake_cm.af
        assert np.all([q in a_field for q in [3, 4, 6]])
        assert np.all([q not in a_field for q in [0, 1, 2, 5]])

        b_matches = fake_cm.bc
        assert np.all([q in b_matches for q in [0, 1]])
        assert np.all([q not in b_matches for q in [2, 3]])

        b_field = fake_cm.bf
        assert np.all([q in b_field for q in [2]])
        assert np.all([q not in b_field for q in [0, 1, 3]])

        prob_counterpart = fake_cm.pc
        self._calculate_prob_integral()
        _integral = self.nc*self.g*self.nfa + self.nc*self.g_wrong*self.nfa + \
            self.nfa*self.nfa*self.nfb
        _prob = self.nc*self.g*self.nfa
        norm_prob = _prob/_integral
        q = np.where(a_matches == 0)[0][0]
        assert_allclose(prob_counterpart[q], norm_prob, rtol=1e-5)
        xicrpts = fake_cm.xi
        assert_allclose(xicrpts[q], np.array([np.log10(self.g / self.fa_priors[0, 0, 0])]),
                        rtol=1e-6)

        prob_a_field = fake_cm.pfa
        a_field = fake_cm.af
        q = np.where(a_field == 6)[0][0]
        assert prob_a_field[q] == 1

        prob_b_field = fake_cm.pfb
        b_field = fake_cm.bf
        q = np.where(b_field == 2)[0][0]
        assert prob_b_field[q] == 1

    def test_small_length_warnings(self):
        # pylint: disable=no-member
        os.system(f'rm -r {self.joint_folder_path}')
        os.makedirs(f'{self.joint_folder_path}', exist_ok=True)
        # Here want to test that the number of recorded matches -- either
        # counterpart, field, or rejected -- is lower than the total length.
        # To achieve this we fake reject length arrays smaller than their
        # supposed lengths.
        alist = self.alist[:, [0, 1, 4]]
        blist = self.blist[:, [0, 1, 4]]
        agrplen = self.agrplen[[0, 1, 4]]
        bgrplen = self.bgrplen[[0, 1, 4]]

        # Force catalogue a to have the wrong length by simply leaving a group
        # out of alist.
        a_reject = np.array([6])

        fake_cm = self.make_class()
        fake_cm.alist = alist
        fake_cm.blist = blist
        fake_cm.agrplen = agrplen
        fake_cm.bgrplen = bgrplen
        fake_cm.lenrejecta = len(a_reject)
        fake_cm.lenrejectb = 0
        with pytest.warns(UserWarning) as record:
            source_pairing(fake_cm)
        assert len(record) == 2
        assert '2 catalogue a sources not in either counterpart, f' in record[0].message.args[0]
        assert '1 catalogue b source not in either counterpart, f' in record[1].message.args[0]

        bflux = fake_cm.bcontamflux
        assert np.all(bflux == np.zeros((2), float))

        a_matches = fake_cm.ac
        assert np.all([q in a_matches for q in [0, 1]])
        assert np.all([q not in a_matches for q in [2, 3, 4, 5, 6]])

        a_field = fake_cm.af
        assert np.all([q in a_field for q in [3, 4]])
        assert np.all([q not in a_field for q in [0, 1, 2, 5, 6]])

        b_matches = fake_cm.bc
        assert np.all([q in b_matches for q in [0, 1]])
        assert np.all([q not in b_matches for q in [2, 3]])

        b_field = fake_cm.bf
        assert np.all([q in b_field for q in [2]])
        assert np.all([q not in b_field for q in [0, 1, 3]])

    def test_large_length_warnings(self):
        # pylint: disable=no-member
        os.system(f'rm -r {self.joint_folder_path}')
        os.makedirs(f'{self.joint_folder_path}', exist_ok=True)
        # Here want to test that the number of recorded matches -- either
        # counterpart, field, or rejected -- is higher than the total length.
        # To achieve this we fake reject length arrays larger than their
        # supposed lengths.
        alist = self.alist[:, [0, 1, 4]]
        blist = self.blist[:, [0, 1, 4]]
        agrplen = self.agrplen[[0, 1, 4]]
        bgrplen = self.bgrplen[[0, 1, 4]]

        a_reject = np.array([2, 3, 4, 5, 6])
        b_reject = np.array([1, 3])

        fake_cm = self.make_class()
        fake_cm.alist = alist
        fake_cm.blist = blist
        fake_cm.agrplen = agrplen
        fake_cm.bgrplen = bgrplen
        fake_cm.lenrejecta = len(a_reject)
        fake_cm.lenrejectb = len(b_reject)
        with pytest.warns(UserWarning) as record:
            source_pairing(fake_cm)
        assert len(record) == 2
        assert '2 additional catalogue a indices recorded' in record[0].message.args[0]
        assert '1 additional catalogue b index recorded' in record[1].message.args[0]

        bflux = fake_cm.bcontamflux
        assert np.all(bflux == np.zeros((2), float))

        a_matches = fake_cm.ac
        assert np.all([q in a_matches for q in [0, 1]])
        assert np.all([q not in a_matches for q in [2, 3, 4, 5, 6]])

        a_field = fake_cm.af
        assert np.all([q in a_field for q in [3, 4]])
        assert np.all([q not in a_field for q in [0, 1, 2, 5, 6]])

        b_matches = fake_cm.bc
        assert np.all([q in b_matches for q in [0, 1]])
        assert np.all([q not in b_matches for q in [2, 3]])

        b_field = fake_cm.bf
        assert np.all([q in b_field for q in [2]])
        assert np.all([q not in b_field for q in [0, 1, 3]])

    @pytest.mark.parametrize("with_and_without_photometry", [True, False])
    def test_pair_sources(self, with_and_without_photometry):  # pylint: disable=too-many-statements
        # pylint: disable=no-member
        os.system(f'rm -r {self.joint_folder_path}/*')
        # Same run as test_source_pairing, but called from CrossMatch rather than
        # directly this time.
        cm_p_ = self.cm_p_text.replace('  - [131, 134, 4, -1, 1, 3]', '  - [131, 131, 1, 0, 0, 1]')
        ca_p_ = self.ca_p_text.replace('  - [131, 134, 4, -1, 1, 3]', '  - [0, 0, 1, 0, 0, 1]')
        cb_p_ = self.cb_p_text.replace('  - [131, 134, 4, -1, 1, 4]', '  - [0, 0, 1, 0, 0, 1]')
        self.cm = CrossMatch(mock_filename(cm_p_.encode("utf-8")), mock_filename(ca_p_.encode("utf-8")),
                             mock_filename(cb_p_.encode("utf-8")))
        self.cm._load_metadata_config(9)
        self.cm._initialise_chunk()
        self.cm.delta_mag_cuts = np.array([2.5, 5])
        self.cm.a_modelrefinds = self.amodelrefinds
        self.cm.b_modelrefinds = self.bmodelrefinds
        self.cm.chunk_id = 1
        self.cm.a_perturb_auf_outputs = self.a_perturb_auf_outputs
        self.cm.b_perturb_auf_outputs = self.b_perturb_auf_outputs

        self.cm.count_pair_func = source_pairing

        self.cm.abinsarray = self.abinsarray
        self.cm.abinlengths = self.abinlengths
        self.cm.bbinsarray = self.bbinsarray
        self.cm.bbinlengths = self.bbinlengths
        self.cm.a_sky_inds = self.a_sky_inds
        self.cm.b_sky_inds = self.b_sky_inds
        self.cm.c_priors = self.c_priors
        self.cm.c_array = self.c_array
        self.cm.fa_priors = self.fa_priors
        self.cm.fa_array = self.fa_array
        self.cm.fb_priors = self.fb_priors
        self.cm.fb_array = self.fb_array
        self.cm.alist = self.alist
        self.cm.blist = self.blist
        self.cm.agrplen = self.agrplen
        self.cm.bgrplen = self.bgrplen
        self.cm.lenrejecta = 0
        self.cm.lenrejectb = 0

        if with_and_without_photometry:
            self.cm.include_phot_like = True
            self.cm.with_and_without_photometry = True
            self.cm.c_array = self.cm.c_array * 1.5

        mcff = Macauff(self.cm)
        mcff.pair_sources()

        bflux = self.cm.bcontamflux
        assert np.all(bflux == np.zeros((2), float))

        a_matches = self.cm.ac
        assert np.all([q in a_matches for q in [0, 1]])
        assert np.all([q not in a_matches for q in [2, 3, 4, 5, 6]])

        a_field = self.cm.af
        assert np.all([q in a_field for q in [2, 3, 4, 5, 6]])
        assert np.all([q not in a_field for q in [0, 1]])

        b_matches = self.cm.bc
        assert np.all([q in b_matches for q in [0, 1]])
        assert np.all([q not in b_matches for q in [2, 3]])

        b_field = self.cm.bf
        assert np.all([q in b_field for q in [2, 3]])
        assert np.all([q not in b_field for q in [0, 1]])

        if with_and_without_photometry:
            bflux = self.cm.bcontamflux_without_photometry
            assert np.all(bflux == np.zeros((2), float))

            a_matches = self.cm.ac_without_photometry
            assert np.all([q in a_matches for q in [0, 1]])
            assert np.all([q not in a_matches for q in [2, 3, 4, 5, 6]])

            a_field = self.cm.af_without_photometry
            assert np.all([q in a_field for q in [2, 3, 4, 5, 6]])
            assert np.all([q not in a_field for q in [0, 1]])

            b_matches = self.cm.bc_without_photometry
            assert np.all([q in b_matches for q in [0, 1]])
            assert np.all([q not in b_matches for q in [2, 3]])

            b_field = self.cm.bf_without_photometry
            assert np.all([q in b_field for q in [2, 3]])
            assert np.all([q not in b_field for q in [0, 1]])
        else:
            assert np.all([not hasattr(self.cm, f'{x}_without_photometry') for x in
                           ['bcontamflux', 'ac', 'af', 'bc', 'bf']])

        prob_counterpart = self.cm.pc
        self._calculate_prob_integral()
        if with_and_without_photometry:
            _integral = self.nc*self.g*self.nfa * 1.5 + self.nc*self.g_wrong*self.nfa * 1.5 + \
                self.nfa*self.nfa*self.nfb
            _prob = self.nc*self.g*self.nfa * 1.5
        else:
            _integral = self.nc*self.g*self.nfa + self.nc*self.g_wrong*self.nfa + \
                self.nfa*self.nfa*self.nfb
            _prob = self.nc*self.g*self.nfa
        norm_prob = _prob/_integral
        q = np.where(a_matches == 0)[0][0]
        assert_allclose(prob_counterpart[q], norm_prob, rtol=1e-5)
        xicrpts = self.cm.xi
        assert_allclose(xicrpts[q], np.array([np.log10(self.g / self.fa_priors[0, 0, 0])]),
                        rtol=1e-6)
        etacrpts = self.cm.eta
        fake_eta = np.log10(1.5) if with_and_without_photometry else np.log10(1.0)
        assert_allclose(etacrpts[q], np.array([fake_eta]), rtol=1e-6)

        if with_and_without_photometry:
            prob_counterpart = self.cm.pc_without_photometry
            self._calculate_prob_integral()
            _integral = self.nc*self.g*self.nfa + self.nc*self.g_wrong*self.nfa + \
                self.nfa*self.nfa*self.nfb
            _prob = self.nc*self.g*self.nfa
            norm_prob = _prob/_integral
            q = np.where(a_matches == 0)[0][0]
            assert_allclose(prob_counterpart[q], norm_prob, rtol=1e-5)
            xicrpts = self.cm.xi_without_photometry
            assert_allclose(xicrpts[q], np.array([np.log10(self.g / self.fa_priors[0, 0, 0])]),
                            rtol=1e-6)
            etacrpts = self.cm.eta_without_photometry
            fake_eta = np.log10(1.0)
            assert_allclose(etacrpts[q], np.array([fake_eta]), rtol=1e-6)
        else:
            assert np.all([not hasattr(self.cm, f'{x}_without_photometry') for x in
                           ['pc', 'xi']])

        prob_a_field = self.cm.pfa
        a_field = self.cm.af
        q = np.where(a_field == 6)[0][0]
        assert prob_a_field[q] == 1

        prob_b_field = self.cm.pfb
        b_field = self.cm.bf
        q = np.where(b_field == 2)[0][0]
        assert prob_b_field[q] == 1

        afs = self.cm.afieldseps
        afeta = self.cm.afieldeta
        afxi = self.cm.afieldxi
        q = np.where(a_field == 2)[0][0]
        fake_field_sep = np.sqrt(((self.a_cat[2, 0] -
                                   self.b_cat[3, 0])*np.cos(np.radians(self.b_cat[3, 1])))**2 +
                                 (self.a_cat[2, 1] - self.b_cat[3, 1])**2)
        assert_allclose(afs[q], fake_field_sep * 3600, rtol=1e-6)

        fake_field_g = 1/(2 * np.pi * self.o**2) * np.exp(-0.5 * fake_field_sep**2 / self.o**2)
        # c_priors and fb_priors are the same, so they cancel in the division.
        # Being in log space we can be relatively forgiving in our assertion limits.
        assert_allclose(afxi[q], np.log10(fake_field_g / self.nfa), rtol=0.01, atol=0.01)
        # Ignoring photometry here, so this should be equal probability.
        assert_allclose(afeta[q], np.log10(1.5) if with_and_without_photometry else np.log10(1.0))

        if with_and_without_photometry:
            prob_a_field = self.cm.pfa_without_photometry
            a_field = self.cm.af_without_photometry
            q = np.where(a_field == 6)[0][0]
            assert prob_a_field[q] == 1

            prob_b_field = self.cm.pfb_without_photometry
            b_field = self.cm.bf_without_photometry
            q = np.where(b_field == 2)[0][0]
            assert prob_b_field[q] == 1

            afs = self.cm.afieldseps_without_photometry
            afeta = self.cm.afieldeta_without_photometry
            afxi = self.cm.afieldxi_without_photometry
            q = np.where(a_field == 2)[0][0]
            fake_field_sep = np.sqrt(((self.a_cat[2, 0] -
                                       self.b_cat[3, 0])*np.cos(np.radians(self.b_cat[3, 1])))**2 +
                                     (self.a_cat[2, 1] - self.b_cat[3, 1])**2)
            assert_allclose(afs[q], fake_field_sep * 3600, rtol=1e-6)

            fake_field_g = 1/(2 * np.pi * self.o**2) * np.exp(-0.5 * fake_field_sep**2 / self.o**2)
            # c_priors and fb_priors are the same, so they cancel in the division.
            # Being in log space we can be relatively forgiving in our assertion limits.
            assert_allclose(afxi[q], np.log10(fake_field_g / self.nfa), rtol=0.01, atol=0.01)
            # Ignoring photometry here, so this should be equal probability.
            assert_allclose(afeta[q], np.log10(1.0))
        else:
            assert np.all([not hasattr(self.cm, f'{x}_without_photometry') for x in
                           ['pfa', 'af', 'pfb', 'bf', 'afieldseps', 'afieldeta', 'afieldxi']])


def test_f90_comb():
    for n in [2, 3, 4, 5]:
        for k in range(2, n+1, 1):
            n_combs = math.factorial(n) / math.factorial(k) / math.factorial(n - k)
            combs = cpf.calc_combs(n, n_combs, k).T
            new_combs = combs[np.lexsort([combs[:, i] for i in range(k)])]

            iter_combs = np.array(list(itertools.combinations(np.arange(1, n+1, 1), k)))
            new_iter_combs = iter_combs[np.lexsort([iter_combs[:, i] for i in range(k)])]
            assert np.all(new_combs == new_iter_combs)


def test_f90_perm_comb():
    for n in [2, 3, 4, 5]:
        for k in range(2, n+1, 1):
            n_combs = math.factorial(n) / math.factorial(k) / math.factorial(n - k)
            n_perms_per_comb = math.factorial(k)
            perms = cpf.calc_permcombs(n, k, n_perms_per_comb, n_combs).T
            new_perms = perms[np.lexsort([perms[:, i] for i in range(k)])]

            # Test against itertools, with n-pick-k: combinations then
            # permutations.
            iter_perms = np.array(list(itertools.permutations(np.arange(1, n+1, 1), r=k)))
            new_iter_perms = iter_perms[np.lexsort([iter_perms[:, i] for i in range(k)])]
            assert np.all(new_perms == new_iter_perms)


def test_factorial():
    for k in range(21):
        assert math.factorial(k) == cpf.factorial(k, k-1)
        assert math.factorial(k) == cpf.factorial(k, k)

    for k in range(21):
        assert cpf.factorial(k, 1) == k

    for k in range(21):
        for l in range(1, k+1):
            assert cpf.factorial(k, l) == math.factorial(k) / math.factorial(k - l)
