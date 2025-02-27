# Licensed under a 3-clause BSD style license - see LICENSE
'''
Tests for the "photometric_likelihood" module.
'''

import os

import numpy as np
from numpy.testing import assert_allclose
from test_matching import _replace_line

# pylint: disable=import-error,no-name-in-module
from macauff.macauff import Macauff
from macauff.matching import CrossMatch
from macauff.photometric_likelihood import compute_photometric_likelihoods, make_bins
from macauff.photometric_likelihood_fortran import photometric_likelihood_fortran as plf

# pylint: enable=import-error,no-name-in-module


class TestOneSidedPhotometricLikelihood:
    # pylint: disable=no-member
    def setup_class(self):
        self.cf_points = np.array([[a, b] for a in [131.5, 132.5, 133.5] for b in [-0.5, 0.5]])
        self.cf_areas = np.ones((6), float)
        self.joint_folder_path = 'test_path'
        self.a_cat_folder_path = 'gaia_folder'
        self.b_cat_folder_path = 'wise_folder'

        self.area = (134-131)*(1--1)

        self.afilts, self.bfilts = np.array(['G', 'BP', 'RP']), np.array(['W1', 'W2', 'W3', 'W4'])

        self.include_phot_like, self.use_phot_priors = False, False

        os.makedirs(self.a_cat_folder_path, exist_ok=True)
        os.makedirs(self.b_cat_folder_path, exist_ok=True)

        # Create a random selection of sources, and then NaN out 25% of each of
        # the filters, at random.
        seed = 98765
        rng = np.random.default_rng(seed)

        self.na, self.nb = 200000, 78000
        for n, filts, name in zip([self.na, self.nb], [self.afilts, self.bfilts], ['a', 'b']):
            a = np.empty((n, 3), float)
            a[:, 0] = rng.uniform(131, 134, n)
            a[:, 1] = rng.uniform(-1, 1, n)
            a[:, 2] = 0.1

            setattr(self, f'{name}_astro', a)

            a = rng.uniform(10.01, 14.99, (n, len(filts)))
            for i in range(len(filts)):
                q = rng.choice(n, size=n // 4, replace=False)
                a[q, i] = np.nan

            setattr(self, f'{name}_photo', a)

        old_line = 'cf_region_points = 131 134 4 -1 1 3'
        new_line = 'cf_region_points = 131.5 133.5 3 -0.5 0.5 2\n'
        with open(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                      idx, new_line, out_file=os.path.join(os.path.dirname(__file__),
                      'data/crossmatch_params_.txt'))
        old_line = 'cross_match_extent = 131 138 -3 3'
        new_line = 'cross_match_extent = 131 134 -3 3\n'
        with open(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params_.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params_.txt'),
                      idx, new_line)

    def make_class(self):
        class A():  # pylint: disable=too-few-public-methods
            def __init__(self):
                pass

        a = A()
        a.a_cat_folder_path = self.a_cat_folder_path
        a.b_cat_folder_path = self.b_cat_folder_path
        a.a_filt_names = self.afilts
        a.b_filt_names = self.bfilts
        a.cf_region_points = self.cf_points
        a.cf_areas = self.cf_areas
        a.include_phot_like = self.include_phot_like
        a.use_phot_priors = self.include_phot_like
        a.rank = 0
        a.chunk_id = 1
        a.a_astro = self.a_astro
        a.a_photo = self.a_photo
        a.b_astro = self.b_astro
        a.b_photo = self.b_photo

        a.auf_cdf_a = None
        a.auf_cdf_b = None

        return a

    def test_compute_photometric_likelihoods(self):
        na, nb, area = self.na, self.nb, self.area
        fake_cm = self.make_class()
        compute_photometric_likelihoods(fake_cm)

        for a, shape, value in zip([fake_cm.c_priors, fake_cm.fa_priors, fake_cm.fb_priors],
                                   [(4, 3, 6), (4, 3, 6), (4, 3, 6)],
                                   [0.75*nb/area/2, 0.75/area*(na-nb/2), 0.75/area*(nb-nb/2)]):
            assert np.all(a.shape == shape)
            # Allow 3% tolerance for counting statistics in the distribution above
            # caused by the rng.choice removal of objects in each filter.
            assert_allclose(a, value, rtol=0.03)

        abinlen = fake_cm.abinlengths
        bbinlen = fake_cm.bbinlengths

        assert np.all(abinlen == 51*np.ones((3, 6), int))
        assert np.all(bbinlen == 26*np.ones((4, 6), int))

        for a, shape in zip(
            [fake_cm.c_array, fake_cm.fa_array, fake_cm.fb_array], [(25, 50, 4, 3, 6), (50, 4, 3, 6),
                                                                    (25, 4, 3, 6)]):
            assert np.all(a.shape == shape)
            assert_allclose(a, np.ones(shape, float), atol=1e-30)

        abins = fake_cm.abinsarray
        bbins = fake_cm.bbinsarray
        for which_cat, filts, _bins in zip(['a', 'b'], [self.afilts, self.bfilts], [abins, bbins]):
            a = getattr(fake_cm, f'{which_cat}_astro')  # np.load(f'{folder}/con_cat_astro.npy')
            b = getattr(fake_cm, f'{which_cat}_photo')  # np.load(f'{folder}/con_cat_photo.npy')
            for i, (ax1, ax2) in enumerate(self.cf_points):
                q = ((a[:, 0] >= ax1-0.5) & (a[:, 0] <= ax1+0.5) &
                     (a[:, 1] >= ax2-0.5) & (a[:, 1] <= ax2+0.5))
                for k in range(len(filts)):
                    hist, _ = np.histogram(b[q & ~np.isnan(b[:, k]), k], bins=_bins[:, k, i])
                    assert np.all(hist >= 250)

    def test_empty_filter(self):
        # This test simply removes all "good" flags from a single pointing-filter
        # combination, to check the robustness of empty bin derivations.
        a = np.copy(self.b_photo)
        ax1_1, ax1_2 = 132, 133
        ax2_1, ax2_2 = -1, 0
        q = ((self.b_astro[:, 0] >= ax1_1) & (self.b_astro[:, 0] <= ax1_2) &
             (self.b_astro[:, 1] >= ax2_1) & (self.b_astro[:, 1] <= ax2_2))
        a[q, 2] = np.nan

        fake_cm = self.make_class()
        fake_cm.b_photo = a
        compute_photometric_likelihoods(fake_cm)

        abinlen = fake_cm.abinlengths
        assert np.all(abinlen == 51*np.ones((3, 6), int))

        bbinlen = fake_cm.bbinlengths
        b_check = 26*np.ones((4, 6), int)
        b_check[2, 2] = 1
        assert np.all(bbinlen == b_check)

        c_array = fake_cm.c_array
        c_check = np.ones((25, 50, 4, 3, 6), float)
        c_check[:, :, 2, :, 2] = 0
        assert_allclose(c_array, c_check, atol=1e-30)

    def test_small_number_bins(self):
        # This test checks if there are <N sources in a single pointing-filter
        # combination, then everything is simply lumped into one big bin.
        a = np.copy(self.b_photo)
        ax1_1, ax1_2 = 132, 133
        ax2_1, ax2_2 = -1, 0
        q = ((self.b_astro[:, 0] >= ax1_1) & (self.b_astro[:, 0] <= ax1_2) &
             (self.b_astro[:, 1] >= ax2_1) & (self.b_astro[:, 1] <= ax2_2))
        seed = 9999
        rng = np.random.default_rng(seed)
        a[q, 2] = np.append(rng.uniform(10, 15, (100,)), (np.sum(q)-100)*[np.nan])

        fake_cm = self.make_class()
        fake_cm.b_photo = a
        compute_photometric_likelihoods(fake_cm)

        abinlen = fake_cm.abinlengths
        assert np.all(abinlen == 51*np.ones((3, 6), int))

        bbinlen = fake_cm.bbinlengths
        b_check = 26*np.ones((4, 6), int)
        b_check[2, 2] = 2
        assert np.all(bbinlen == b_check)

        c_array = fake_cm.c_array
        c_check = np.ones((25, 50, 4, 3, 6), float)
        c_check[1:, :, 2, :, 2] = 0
        assert_allclose(c_array, c_check, atol=1e-30)

    def test_calculate_phot_like_input(self):
        # Here we also have to dump a random "magref" file to placate the
        # checks on CrossMatch.
        for folder, name in zip([self.a_cat_folder_path, self.b_cat_folder_path], ['a', 'b']):
            np.save(f'{folder}/con_cat_astro.npy', getattr(self, f'{name}_astro'))
            np.save(f'{folder}/con_cat_photo.npy', getattr(self, f'{name}_photo'))
            np.save(f'{folder}/magref.npy', np.zeros((len(getattr(self, f'{name}_astro')))))

        self.cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'))
        self.cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                               'data/crossmatch_params_.txt'),
                                  os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                                  os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        self.cm.chunk_id = 1
        self.cm.phot_like_func = compute_photometric_likelihoods
        self.cm.cf_areas = self.cf_areas
        self.cm.auf_cdf_a = None
        self.cm.auf_cdf_b = None
        mcff = Macauff(self.cm)
        mcff.calculate_phot_like()

        abinlen = self.cm.abinlengths
        bbinlen = self.cm.bbinlengths

        assert np.all(abinlen == 51*np.ones((3, 6), int))
        assert np.all(bbinlen == 26*np.ones((4, 6), int))

        for a, shape in zip(
            [self.cm.c_array, self.cm.fa_array,
             self.cm.fb_array], [(25, 50, 4, 3, 6), (50, 4, 3, 6), (25, 4, 3, 6)]):
            assert np.all(a.shape == shape)
            assert_allclose(a, np.ones(shape, float), atol=1e-30)

        abins = self.cm.abinsarray
        bbins = self.cm.bbinsarray
        for folder, filts, _bins in zip([self.a_cat_folder_path, self.b_cat_folder_path],
                                        [self.afilts, self.bfilts], [abins, bbins]):
            a = np.load(f'{folder}/con_cat_astro.npy')
            b = np.load(f'{folder}/con_cat_photo.npy')
            for i, (ax1, ax2) in enumerate(self.cf_points):
                q = ((a[:, 0] >= ax1-0.5) & (a[:, 0] <= ax1+0.5) &
                     (a[:, 1] >= ax2-0.5) & (a[:, 1] <= ax2+0.5))
                for k in range(len(filts)):
                    hist, _ = np.histogram(b[q & ~np.isnan(b[:, k]), k],
                                           bins=_bins[:, k, i])
                    assert np.all(hist >= 250)


def test_get_field_dists_fortran():
    rng = np.random.default_rng(11119999)

    ainds = np.arange(0, 10).reshape(1, 10, order='F')
    asize = np.array([1] * 10)
    aflags = np.ones(10, bool)
    bflags = np.ones(15, bool)
    amag = np.array([13] * 10)
    bmag = np.array([10] * 15)
    lowmag, uppmag = -999, 999

    fracs = np.array([0.5, 0.7, 0.9])

    auf_cdf = rng.uniform(0, 1, size=(1, 10)).reshape(1, 10, order='F')

    mask = plf.get_field_dists(auf_cdf, ainds, asize, fracs, aflags, amag, bflags, bmag,
                               lowmag, uppmag, lowmag, uppmag)
    test_mask = np.ones((10, len(fracs)), bool)
    for i in range(len(fracs)):
        test_mask[auf_cdf[0, :] <= fracs[i], i] = 0
    assert np.all(mask == test_mask)
    assert np.sum(mask) > 0


def test_brightest_mag_fortran():
    rng = np.random.default_rng(11119998)

    ainds = np.arange(0, 10).reshape(1, 10, order='F')
    asize = np.array([1] * 10)
    aerr_circ = np.pi * np.array([0.05/3600] * 10)**2
    aflags = np.ones(10, bool)
    bflags = np.ones(15, bool)
    amag = np.array([10] * 10)
    bmag = np.array([12] * 15)
    abin = np.array([9, 11])

    frac = 0.63
    auf_cdf = rng.uniform(0, 1, size=(1, 10)).reshape(1, 10, order='F')

    mask, area = plf.brightest_mag(auf_cdf, amag, bmag, ainds, asize, frac, aerr_circ, aflags, bflags, abin)
    test_mask = np.zeros((15, 1), bool)
    test_mask[:10][auf_cdf[0, :] <= frac, 0] = 1
    assert np.all(mask == test_mask)
    assert np.sum(mask) > 0
    assert_allclose(area, np.sum(aerr_circ[auf_cdf[0, :] <= frac]) / np.sum(auf_cdf[0, :] <= frac))


def test_make_bins():
    # Test a few combinations of input magnitude arrays to ensure make_bins
    # returns expected bins.

    dm = 0.1/251
    m = np.arange(10+1e-10, 15, dm)
    # Fudge edge magnitudes away from 'too close to bin edge' checks.
    m[m < 10.01] = 10.01
    m[m > 14.99] = 14.99
    bins = make_bins(m)
    assert_allclose(bins, np.arange(10, 15+1e-10, 0.1))
    h, _ = np.histogram(m, bins=bins)
    assert np.all(h == 251)

    dm = 0.1/2500
    m = np.arange(10+1e-10, 15, dm)
    m[m < 10.01] = 10.01
    m[m > 14.99] = 14.99
    bins = make_bins(m)
    assert_allclose(bins, np.arange(10, 15+1e-10, 0.1))
    h, _ = np.histogram(m, bins=bins)
    assert np.all(h == 2500)

    dm = 0.1/249
    m = np.arange(10+1e-10, 15, dm)
    m[m < 10.01] = 10.01
    m[m > 14.99] = 14.99
    bins = make_bins(m)
    assert_allclose(bins, np.arange(10, 15+1e-10, 0.2))
    h, _ = np.histogram(m, bins=bins)
    assert np.all(h == 2*249)

    dm = 0.1/124
    m = np.arange(10+1e-10, 15, dm)
    m[-1] = 15-1e-9
    # Don't fudge data away from bin edge this time.
    bins = make_bins(m)
    fake_bins = np.arange(10, 15.11, 0.3)
    fake_bins[0] = 10-1e-4
    fake_bins[-1] = 15+1e-4
    assert_allclose(bins, fake_bins)
    h, _ = np.histogram(m, bins=bins)
    # With this bin size not quite being exactly able to fit into
    # 10 -> 15 exactly, the final bin is only 0.2 wide, and hence
    # not a "triple" sized bin like the others.
    assert np.all(h[:-1] == 3*124) & (h[-1] == 2*124)


class TestFullPhotometricLikelihood:  # pylint: disable=too-many-instance-attributes
    def setup_class(self):  # pylint: disable=too-many-statements
        self.cf_points = np.array([[131.5, -0.5]])
        self.cf_areas = 0.25 * np.ones((1), float)
        self.joint_folder_path = 'test_path'
        self.a_cat_folder_path = 'gaia_folder'
        self.b_cat_folder_path = 'wise_folder'

        self.area = 0.25

        self.afilts, self.bfilts = np.array(['G']), np.array(['G'])

        self.include_phot_like, self.use_phot_priors = True, True

        os.makedirs(self.a_cat_folder_path, exist_ok=True)
        os.makedirs(self.b_cat_folder_path, exist_ok=True)

        # Create a random selection of sources, and then NaN out 25% of each of
        # the filters, at random.
        seed = 98763
        rng = np.random.default_rng(seed)

        asig, bsig = 0.1, 0.15
        self.ntot = 45000

        aa = np.empty((self.ntot, 3), float)
        aa[:, 0] = rng.uniform(131.25, 131.75, self.ntot)
        aa[:, 1] = rng.uniform(-0.75, -0.25, self.ntot)
        aa[:, 2] = asig

        ba = np.empty((self.ntot, 3), float)
        d = rng.rayleigh(scale=np.sqrt(asig**2 + bsig**2)/3600, size=self.ntot)
        t = rng.uniform(0, 2*np.pi, size=self.ntot)
        ba[:, 0] = aa[:, 0] + d * np.cos(t)
        ba[:, 1] = aa[:, 1] + d * np.sin(t)
        ba[:, 2] = bsig

        a_phot_sig, b_phot_sig = 0.05, 0.05

        ap = rng.uniform(10, 15, (self.ntot, len(self.afilts)))
        bp = ap + rng.normal(loc=0, scale=b_phot_sig, size=(self.ntot, len(self.bfilts)))
        ap = ap + rng.normal(loc=0, scale=a_phot_sig, size=(self.ntot, len(self.afilts)))

        a_cut, b_cut = ap[:, 0] > 11, bp[:, 0] < 14.5
        self.nc = np.sum(a_cut & b_cut)

        aa[~a_cut, 0] = rng.uniform(131.25, 131.75, np.sum(~a_cut))
        aa[~a_cut, 1] = rng.uniform(-0.75, -0.25, np.sum(~a_cut))
        ba[~b_cut, 0] = rng.uniform(131.25, 131.75, np.sum(~b_cut))
        ba[~b_cut, 1] = rng.uniform(-0.75, -0.25, np.sum(~b_cut))

        self.a_astro = aa
        self.b_astro = ba
        self.a_photo = ap
        self.b_photo = bp

        # Have to pre-create the various overlap arrays, and integral lengths:
        self.asize = np.ones(self.ntot, int)
        self.asize[~a_cut] = 0
        self.asize[~b_cut] = 0
        self.ainds = -1*np.ones((1, self.ntot), int, order='F')
        self.ainds[0, :] = np.arange(0, self.ntot)
        self.ainds[0, ~a_cut] = -1
        self.ainds[0, ~b_cut] = -1
        self.bsize = np.ones(self.ntot, int)
        self.bsize[~a_cut] = 0
        self.bsize[~b_cut] = 0
        self.binds = -1*np.ones((1, self.ntot), int, order='F')
        self.binds[0, :] = np.arange(0, self.ntot)
        self.binds[0, ~a_cut] = -1
        self.binds[0, ~b_cut] = -1

        # Integrate 2-D Gaussian to N*sigma radius gives probability Y of
        # 1 - exp(-0.5 N^2 sigma^2 / sigma^2) = 1 - exp(-0.5 N^2).
        # Rearranging for N gives N = sqrt(-2 ln(1 - Y))
        self.y_f, self.y_b = 0.99, 0.63
        n_b = np.sqrt(-2 * np.log(1 - self.y_b))
        n_f = np.sqrt(-2 * np.log(1 - self.y_f))
        self.ab_area = np.pi * (n_b * np.sqrt(asig**2 + bsig**2) * np.ones(self.ntot, float) / 3600)**2
        self.af_area = np.pi * (n_f * np.sqrt(asig**2 + bsig**2) * np.ones(self.ntot, float) / 3600)**2
        self.bb_area = np.pi * (n_b * np.sqrt(asig**2 + bsig**2) * np.ones(self.ntot, float) / 3600)**2
        self.bf_area = np.pi * (n_f * np.sqrt(asig**2 + bsig**2) * np.ones(self.ntot, float) / 3600)**2

        self.auf_cdf_a = (1 - np.exp(-0.5 * d**2 / ((asig**2 + bsig**2) / 3600**2))).reshape(1, -1, order='F')
        # Even though we've moved the objects, distance d preserves the object
        # separation, so we have to give both "a" and "b" sources CDFs of 1
        # when resetting the bonds. This way we have no false-match overlap,
        # but that's a <1% error. This is also true above, where we have to
        # remove the *size and *inds bonds for bright and faint magnitudes.
        self.auf_cdf_a[0, ~a_cut] = 1.0
        self.auf_cdf_a[0, ~b_cut] = 1.0
        self.auf_cdf_b = (1 - np.exp(-0.5 * d**2 / ((asig**2 + bsig**2) / 3600**2))).reshape(1, -1, order='F')
        self.auf_cdf_b[0, ~a_cut] = 1.0
        self.auf_cdf_b[0, ~b_cut] = 1.0

    def make_class(self):
        class A():  # pylint: disable=too-few-public-methods
            def __init__(self):
                pass

        a = A()
        a.a_cat_folder_path = self.a_cat_folder_path
        a.b_cat_folder_path = self.b_cat_folder_path
        a.a_filt_names = self.afilts
        a.b_filt_names = self.bfilts
        a.cf_region_points = self.cf_points
        a.cf_areas = self.cf_areas
        a.include_phot_like = self.include_phot_like
        a.use_phot_priors = self.include_phot_like
        a.int_fracs = np.array([self.y_b, self.y_f])
        a.rank = 0
        a.chunk_id = 1
        a.ab_area = self.ab_area
        a.af_area = self.af_area
        a.bb_area = self.bb_area
        a.bf_area = self.bf_area
        a.ainds = self.ainds
        a.asize = self.asize
        a.binds = self.binds
        a.bsize = self.bsize
        a.a_astro = self.a_astro
        a.a_photo = self.a_photo
        a.b_astro = self.b_astro
        a.b_photo = self.b_photo

        a.auf_cdf_a = self.auf_cdf_a
        a.auf_cdf_b = self.auf_cdf_b

        return a

    def test_compute_phot_like(self):
        fake_cm = self.make_class()
        compute_photometric_likelihoods(fake_cm)

        # pylint: disable=no-member
        c_p = fake_cm.c_priors
        c_l = fake_cm.c_array
        fa_p = fake_cm.fa_priors
        fa_l = fake_cm.fa_array
        fb_p = fake_cm.fb_priors
        fb_l = fake_cm.fb_array
        # pylint: enable=no-member

        fake_c_p = self.nc / self.area
        fake_fa_p = (self.ntot - self.nc) / self.area
        fake_fb_p = (self.ntot - self.nc) / self.area
        assert_allclose(c_p, fake_c_p, rtol=0.05)
        assert_allclose(fa_p, fake_fa_p, rtol=0.02)
        assert_allclose(fb_p, fake_fb_p, rtol=0.02)

        abins = make_bins(self.a_photo[~np.isnan(self.a_photo[:, 0]), 0])
        bbins = make_bins(self.b_photo[~np.isnan(self.b_photo[:, 0]), 0])
        a = self.a_photo[~np.isnan(self.a_photo[:, 0]), 0]
        b = self.b_photo[~np.isnan(self.b_photo[:, 0]), 0]
        q = (a <= 11) | (b >= 14.5)

        fake_fa_l = np.zeros((len(abins) - 1, 1, 1, 1), float)
        h, _ = np.histogram(a[q], bins=abins, density=True)
        fake_fa_l[:, 0, 0, 0] = h
        fake_abins = fake_cm.abinsarray[:, 0, 0]  # pylint: disable=no-member
        assert np.all(abins == fake_abins)
        assert np.all(fake_fa_l.shape == fa_l.shape)
        assert_allclose(fa_l, fake_fa_l, atol=0.02)

        fake_fb_l = np.zeros((len(bbins) - 1, 1, 1, 1), float)
        h, _ = np.histogram(b[q], bins=bbins, density=True)
        fake_fb_l[:, 0, 0, 0] = h
        fake_bbins = fake_cm.bbinsarray[:, 0, 0]  # pylint: disable=no-member
        assert np.all(bbins == fake_bbins)
        assert np.all(fake_fb_l.shape == fb_l.shape)
        assert_allclose(fb_l, fake_fb_l, atol=0.02)

        h, _, _ = np.histogram2d(a[~q], b[~q], bins=[abins, bbins])
        fake_c_l = np.zeros((len(bbins)-1, len(abins) - 1, 1, 1, 1), float, order='F')
        pa, _ = np.histogram(a, bins=abins, density=True)
        integral = 0
        for i in range(len(abins)-1):
            if np.sum(h[i]) > 0:
                fake_c_l[:, i, 0, 0, 0] = h[i] / np.sum(h[i]) / np.diff(bbins) * pa[i]
            else:
                fake_c_l[:, i, 0, 0, 0] = 0
            integral += np.sum(fake_c_l[:, i, 0, 0, 0] * np.diff(bbins) * (abins[i+1] - abins[i]))
        fake_c_l /= integral
        assert np.all(fake_c_l.shape == c_l.shape)
        assert_allclose(c_l, fake_c_l, rtol=0.1, atol=0.05)

    def test_compute_phot_like_use_priors_only(self):
        fake_cm = self.make_class()
        fake_cm.include_phot_like = False
        compute_photometric_likelihoods(fake_cm)

        # pylint: disable=no-member
        c_p = fake_cm.c_priors
        c_l = fake_cm.c_array
        fa_p = fake_cm.fa_priors
        fa_l = fake_cm.fa_array
        fb_p = fake_cm.fb_priors
        fb_l = fake_cm.fb_array
        # pylint: enable=no-member

        fake_c_p = self.nc / self.area
        fake_fa_p = (self.ntot - self.nc) / self.area
        fake_fb_p = (self.ntot - self.nc) / self.area
        assert_allclose(c_p, fake_c_p, rtol=0.05)
        assert_allclose(fa_p, fake_fa_p, rtol=0.02)
        assert_allclose(fb_p, fake_fb_p, rtol=0.02)

        abins = make_bins(self.a_photo[~np.isnan(self.a_photo[:, 0]), 0])
        bbins = make_bins(self.b_photo[~np.isnan(self.b_photo[:, 0]), 0])

        fake_fa_l = np.ones((len(abins) - 1, 1, 1, 1), float)
        fake_abins = fake_cm.abinsarray[:, 0, 0]  # pylint: disable=no-member
        assert np.all(abins == fake_abins)
        assert np.all(fake_fa_l.shape == fa_l.shape)
        assert_allclose(fa_l, fake_fa_l)

        fake_fb_l = np.ones((len(bbins) - 1, 1, 1, 1), float)
        fake_bbins = fake_cm.bbinsarray[:, 0, 0]  # pylint: disable=no-member
        assert np.all(bbins == fake_bbins)
        assert np.all(fake_fb_l.shape == fb_l.shape)
        assert_allclose(fb_l, fake_fb_l)

        fake_c_l = np.ones((len(bbins)-1, len(abins) - 1, 1, 1, 1), float, order='F')
        assert np.all(fake_c_l.shape == c_l.shape)
        assert_allclose(c_l, fake_c_l)
