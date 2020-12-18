# Licensed under a 3-clause BSD style license - see LICENSE
'''
Tests for the "counterpart_pairing" module.
'''

from numpy.testing import assert_allclose
import numpy as np

from ..counterpart_pairing import source_pairing, _individual_island_probability
from ..counterpart_pairing_fortran import counterpart_pairing_fortran as cpf
from ..misc_functions_fortran import misc_functions_fortran as mff


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
        self.b_astro = np.empty((3, 3), float)
        self.a_sig, self.b_sig = 0.1, 0.08

        self.a_astro[:3, :2] = np.array([[0, 0], [0.1, 0.1], [0.1, 0]])
        self.a_astro[3:6, :2] = self.a_astro[:3, :2] + rng.choice([-1, 1], size=(3, 2)) * \
            rng.uniform(2.1*self.a_sig/3600, 3*self.a_sig/3600, size=(3, 2))
        self.a_astro[6, :2] = np.array([0.1, 0.1])
        # Force the second source in the island with the counterpart a/b pair
        # in "a" to be 2-3 sigma away, while the counterpart is <=1 sigma distant.
        self.b_astro[:, :2] = self.a_astro[:3, :2] + \
            rng.uniform(-1*self.b_sig/3600, self.b_sig/3600, size=(3, 2))
        # Swap the first and second indexes around
        self.b_astro[:2, 0] = self.b_astro[[1, 0], 0]
        self.b_astro[:2, 1] = self.b_astro[[1, 0], 1]

        self.a_astro[:, 2] = self.a_sig
        self.b_astro[:, 2] = self.b_sig
        # Currently we don't care about the photometry, setting both
        # include_phot_like and use_phot_priors to False, so just fake:
        self.a_photo = np.ones((7, 3), float)
        self.b_photo = np.ones((3, 4), float)

        self.alist = np.array([[0, 3], [1, 4], [2, 5], [6, -1]]).T
        self.alist_ = self.alist
        self.blist = np.array([[1], [0], [2]]).T
        self.blist_ = self.blist
        self.agrplen = np.array([2, 2, 2, 1])
        self.bgrplen = np.array([1, 1, 1])

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
        self.bmodelrefinds = np.zeros((3, 3), int, order='F')

        self.afrac_grids = np.zeros((self.n_fracs, 1, 3, 1), float, order='F')
        self.aflux_grids = np.zeros((1, 3, 1), float, order='F')
        self.bfrac_grids = np.zeros((self.n_fracs, 1, 4, 1), float, order='F')
        self.bflux_grids = np.zeros((1, 4, 1), float, order='F')
        self.afourier_grids = np.ones((len(self.rho)-1, 1, 3, 1), float, order='F')
        self.bfourier_grids = np.ones((len(self.rho)-1, 1, 4, 1), float, order='F')
        self.a_sky_inds = np.zeros((7), int)
        self.b_sky_inds = np.zeros((3), int)

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
        o = np.sqrt(self.a_sig**2 + self.b_sig**2) / 3600
        sep = np.sqrt(((self.a_astro[0, 0] -
                        self.b_astro[1, 0])*np.cos(np.radians(self.b_astro[1, 1])))**2 +
                      (self.a_astro[0, 1] - self.b_astro[1, 1])**2)
        G = 1/(2 * np.pi * o**2) * np.exp(-0.5 * sep**2 / o**2)
        assert_allclose(xicrpts, np.array([np.log10(G / self.fa_priors[0, 0, 0])]), rtol=1e-6)
        assert np.all(acrptflux == np.array([0]))
        assert np.all(bcrptflux == np.array([0]))
        assert np.all(afield == np.array([3]))
        assert len(bfield) == 0
        sep_wrong = np.sqrt(((self.a_astro[3, 0] -
                              self.b_astro[1, 0])*np.cos(np.radians(self.a_astro[3, 1])))**2 +
                            (self.a_astro[3, 1] - self.b_astro[1, 1])**2)
        G_wrong = 1/(2 * np.pi * o**2) * np.exp(-0.5 * sep_wrong**2 / o**2)
        Nc, Nfa, Nfb = self.c_priors[0, 0, 0], self.fa_priors[0, 0, 0], self.fb_priors[0, 0, 0]
        _integral = Nc*G*Nfa + Nc*G_wrong*Nfa + Nfa*Nfa*Nfb
        assert_allclose(integral, _integral, rtol=1e-5)
        _prob = Nc*G*Nfa
        assert_allclose(prob, _prob, rtol=1e-5)
