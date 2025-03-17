# Licensed under a 3-clause BSD style license - see LICENSE
'''
Tests for the "perturbation_auf" module.
'''

# pylint: disable=too-many-lines,duplicate-code

import math
import os

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import j0, j1  # pylint: disable=no-name-in-module
from scipy.stats import skewnorm
from test_matching import _replace_line

# pylint: disable=import-error,no-name-in-module
from macauff.macauff import Macauff
from macauff.matching import CrossMatch
from macauff.misc_functions_fortran import misc_functions_fortran as mff
from macauff.perturbation_auf import (
    _calculate_magnitude_offsets,
    download_trilegal_simulation,
    make_perturb_aufs,
    make_tri_counts,
)
from macauff.perturbation_auf_fortran import perturbation_auf_fortran as paf

# pylint: enable=import-error,no-name-in-module


class TestCreatePerturbAUF:
    def setup_class(self):
        os.makedirs('gaia_auf_folder', exist_ok=True)
        os.makedirs('wise_auf_folder', exist_ok=True)
        os.makedirs('gaia_folder', exist_ok=True)
        os.makedirs('wise_folder', exist_ok=True)
        os.makedirs('test_path', exist_ok=True)
        self.cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'))
        self.cm._initialise_chunk(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                                  os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                                  os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        self.cm.a_auf_region_points = np.array([[0, 0], [50, 50]], dtype=float)
        self.cm.b_auf_region_points = np.array([[0, 0], [50, 50]], dtype=float)
        self.cm.perturb_auf_func = make_perturb_aufs

    def test_no_perturb_outputs(self):
        # pylint: disable=no-member
        # Randomly generate two catalogues (x3 files) between coordinates
        # 0, 0 and 50, 50.
        rng = np.random.default_rng()
        for nf, size in zip([3, 4], [25, 54]):
            cat = np.zeros((size, 3), float)
            rand_inds = rng.permutation(cat.shape[0])[:size // 2 - 1]
            cat[rand_inds, 0] = 50
            cat[rand_inds, 1] = 50
            cat += rng.uniform(-0.1, 0.1, cat.shape)
            setattr(self.cm, f"{'a' if size == 25 else 'b'}_astro", cat)

            cat = rng.uniform(10, 20, (size, nf))
            setattr(self.cm, f"{'a' if size == 25 else 'b'}_photo", cat)

            cat = rng.choice(nf, size=(size,))
            setattr(self.cm, f"{'a' if size == 25 else 'b'}_magref", cat)

        self.cm.include_perturb_auf = False
        self.cm.chunk_id = 1
        mcff = Macauff(self.cm)
        mcff.create_perturb_auf()
        p_a_o = self.cm.b_perturb_auf_outputs
        lenr = len(self.cm.r)
        lenrho = len(self.cm.rho)
        for coord in ['0.0', '50.0']:
            for filt in ['W1', 'W2', 'W3', 'W4']:
                perturb_auf_combo = f'{coord}-{coord}-{filt}'
                for filename, shape in zip(['frac', 'flux', 'offset', 'cumulative', 'fourier',
                                            'Narray', 'magarray'],
                                           [(1, 1), (1,), (lenr-1, 1), (lenr-1, 1), (lenrho-1, 1),
                                            (1, 1), (1, 1)]):
                    file = p_a_o[perturb_auf_combo][filename]
                    assert np.all(file.shape == shape)
                assert np.all(p_a_o[perturb_auf_combo]['frac'] == 0)
                assert np.all(p_a_o[perturb_auf_combo]['cumulative'] == 1)
                assert np.all(p_a_o[perturb_auf_combo]['fourier'] == 1)
                assert np.all(p_a_o[perturb_auf_combo]['magarray'] == 1)
                file = p_a_o[perturb_auf_combo]['offset']
                assert np.all(file[1:] == 0)
                assert file[0] == 1/(2 * np.pi * (self.cm.r[0] + self.cm.dr[0]/2) * self.cm.dr[0])

        file = self.cm.a_modelrefinds
        assert np.all(file[0, :] == 0)
        assert np.all(file[1, :] == self.cm.a_magref)

        # Select AUF pointing index based on a 0 vs 50 cut in longitude.
        cat = self.cm.a_astro
        inds = np.ones(file.shape[1], int)
        inds[np.where(cat[:, 0] < 1)[0]] = 0
        assert np.all(file[2, :] == inds)


# pylint: disable=too-many-locals,too-many-statements
def test_perturb_aufs():
    # Poisson distribution with mean 0.08 gives 92.3% zero, 7.4% one, and 0.3% two draws.
    mean = 0.08
    prob_0_draw = mean**0 * np.exp(-mean) / math.factorial(0)
    prob_1_draw = mean**1 * np.exp(-mean) / math.factorial(1)
    prob_2_draw = mean**2 * np.exp(-mean) / math.factorial(2)

    n = np.array([1.0])
    m = np.array([0.0])
    psf_r = 1.185*6.1
    sig = 6.1 / (2 * np.sqrt(2 * np.log(2)))
    r = np.linspace(0, psf_r, 1500)
    dr = np.diff(r)
    rho = np.linspace(0, 100, 10000)
    drho = np.diff(rho)
    j0s = mff.calc_j0(r[:-1]+dr/2, rho[:-1]+drho/2)

    model_count = 1.0
    num_trials = 100000
    mag_cut = np.array([5.0])
    model_mags = np.array([0])
    model_mags_interval = np.array([1.0e-8])

    log10y = np.log10(mean / model_mags_interval / np.pi / (psf_r / 3600)**2)

    track_sp_hist = np.zeros(len(r)-1, float)
    track_pa_hist = np.zeros(len(r)-1, float)
    # Have to keep track of the uncertainty in the counts, to compare fairly
    # with expected Poissonian counting statistics scatter.
    track_pa_hist_err_sq = np.zeros(len(r)-1, float)

    track_pa_fourier = np.zeros(len(rho)-1, float)

    seed_size = paf.get_random_seed_size()
    rng = np.random.default_rng(seed=123124)

    # Limit the size of each simulation, but run many to aggregate
    # better counting statistics.
    num = 50
    _ddp, _l, agt = np.array([[1]]), np.array([1]), 'fw'
    for _ in range(num):
        seed = rng.choice(1000000, size=(seed_size, 1))
        offsets, fracs, fluxs = paf.scatter_perturbers(np.array([mean]), m, psf_r, 5, mag_cut, sig,
                                                       _ddp, _l, model_mags_interval, agt,
                                                       num_trials, seed[:, 0])
        hist, _ = np.histogram(offsets, bins=r)
        assert_allclose(fracs[0], 1-prob_0_draw, rtol=0.05)
        assert_allclose(np.mean(fluxs), prob_0_draw*0+prob_1_draw*1+prob_2_draw*2, rtol=0.05)

        frac, flux, fourieroffset, offsets, _ = paf.perturb_aufs(
            n, m, r[:-1]+dr/2, dr, r, j0s,
            model_mags+model_mags_interval/2, model_mags_interval, log10y, model_count,
            np.array([1]), mag_cut, psf_r, sig, num_trials, seed, _ddp, _l, agt)

        assert_allclose(frac, 1-prob_0_draw, rtol=0.05)
        assert_allclose(flux, prob_0_draw*0+prob_1_draw*1+prob_2_draw*2, rtol=0.05)

        track_sp_hist += hist / np.sum(hist) / (np.pi * (r[1:]**2 - r[:-1]**2)) / num

        track_pa_hist += offsets[:, 0] / num
        track_pa_hist_err_sq += (np.sqrt(
            offsets[:, 0] * np.sum(hist) * (np.pi * (r[1:]**2 - r[:-1]**2))) / (
            np.sum(hist) * (np.pi * (r[1:]**2 - r[:-1]**2))) / num)**2

        track_pa_fourier += fourieroffset[:, 0] / num

    # Here we assume that the direct and wrapper calls had the same seed for
    # each call, and identical results, and hence should basically agree perfectly.
    assert_allclose(track_pa_hist, track_sp_hist, atol=2e-6)

    offsets_fake = np.zeros_like(r[:-1])
    offsets_fake[0] += prob_0_draw / (np.pi * (r[1]**2 - r[0]**2))
    # Given that we force equal-brightness "binaries", we expect a maximum
    # perturbation of half the total radius.
    q = np.where(r[1:] > psf_r/2)[0]
    offsets_fake[:q[0]] += prob_1_draw / np.pi / (psf_r/2)**2
    final_bin_frac = (psf_r/2 - r[q[0]]) / (r[q[0]+1] - r[q[0]])
    offsets_fake[q[0]] += final_bin_frac * prob_1_draw / np.pi / (psf_r/2)**2

    track_pa_hist_err_sq[track_pa_hist_err_sq == 0] = 1e-8
    for i in range(len(r)-1):
        assert_allclose(track_pa_hist[i], offsets_fake[i], rtol=0.05,
                        atol=1e-5 + 4 * np.sqrt(track_pa_hist_err_sq[i]))

    # Take the average fourier representation of the offsets and compare. The
    # first expression is the integral of the constant 1/(pi (R/2)^2) probability
    # for the randomly placed, single-draw perturbations.
    fake_fourier = (1-prob_0_draw) * 2 / (np.pi * psf_r * (rho[:-1]+drho/2)) * j1(
        np.pi * psf_r * (rho[:-1]+drho/2))
    # The second necessary half of the fourier representation should be a delta
    # function, which becomes the unity function, but since we do numerical
    # integrations, is in practice just the fraction of zero-draw "perturbations"
    # in an annulus of the first radial bin, Fourier-Bessel transformed. This
    # gives f(r) = P / (2 pi r dr), which then cancels everything except J0.
    fake_fourier += prob_0_draw * j0(2 * np.pi * (r[0]+dr[0]/2) * (rho[:-1]+drho/2))

    # Quickly verify fourier_transform by comparing the theoretical
    # real-space distribution, fourier transformed, to the theoretical
    # fourier transformation as well.
    another_fake_fourier = paf.fourier_transform(offsets_fake, r[:-1]+dr/2, dr, j0s)

    assert_allclose(fake_fourier, another_fake_fourier, rtol=5e-3)
    assert_allclose(fake_fourier, track_pa_fourier, rtol=5e-3)


def test_histogram():
    rng = np.random.default_rng(11111)
    x = rng.uniform(0, 1, 10000)
    bins = np.linspace(0, 1, 15)

    _, counts_f = paf.histogram1d_dp(x, bins[0], bins[-1], len(bins)-1,
                                     np.array([True] * (len(x))), np.ones_like(x))
    counts_p, _ = np.histogram(x, bins=bins)
    assert np.all(counts_f == counts_p)


@pytest.mark.parametrize("position", ["inside", "corner", "edge", "random"])
# pylint: disable-next=too-many-branches
def test_circle_area(position):
    rng = np.random.default_rng(346675614)
    r = 0.1

    x_edges = np.array([0, 1])
    y_edges = np.array([0, 1])

    hull_x = x_edges[[0, 0, 1, 1, 0]]
    hull_y = y_edges[[0, 1, 1, 0, 0]]

    if position == "inside":
        # If circle is inside rectangle, get full area:
        done = 0
        while done < 100:
            seed = rng.choice(100000, size=(paf.get_random_seed_size(), 1))
            [x, y] = rng.uniform(0, 1, size=2)
            if (x - r >= x_edges[0] and x + r <= x_edges[1] and
                    y - r >= y_edges[0] and y + r <= y_edges[1]):
                calc_area = paf.get_circle_area_overlap([x], [y], r, hull_x, hull_y, seed)
                assert_allclose(calc_area, np.pi * r**2)
                done += 1

    if position == "corner":
        # Now, if the circle is exactly on the corners of the rectangle
        # we should have a quarter the area:
        for x, y in zip([0, 0, 1, 1], [0, 1, 0, 1]):
            seed = rng.choice(100000, size=(paf.get_random_seed_size(), 1))
            calc_area = paf.get_circle_area_overlap([x], [y], r, hull_x, hull_y, seed)
            # We have a random process in this calculation, will result in small
            # variations in the area.
            assert_allclose(calc_area, np.pi * r**2 / 4, rtol=0.02, atol=5e-5)

    if position == "edge":
        # In the middle of an edge we should have half the circle area:
        for _ in range(100):
            for x, y in zip([0, 0.5, 1, 0.5], [0.5, 0, 0.5, 1]):
                if x == 0.5:
                    x = rng.uniform(0 + r + 1e-4, 1 - r - 1e-4, size=1)
                if y == 0.5:
                    y = rng.uniform(0 + r + 1e-4, 1 - r - 1e-4, size=1)
                seed = rng.choice(100000, size=(paf.get_random_seed_size(), 1))
                calc_area = paf.get_circle_area_overlap([x], [y], r, hull_x, hull_y, seed)
                assert_allclose(calc_area, np.pi * r**2 / 2)

        # Otherwise, we have a more random amount of missing circle:
        xp = np.linspace(*x_edges, 800)
        yp = np.linspace(*y_edges, 800)
        dx, dy = xp[1] - xp[0], yp[1] - yp[0]
        for _ in range(100):
            for x0, y0 in zip([0, 0.5, 1, 0.5], [0.5, 0, 0.5, 1]):
                if x0 == 0.5:
                    x = rng.uniform(0 + r + 1e-4, 1 - r - 1e-4, size=1)
                    if y0 == 0:
                        y = rng.uniform(y0 + 1e-4, y0 + r - 1e-4, size=1)
                    else:
                        y = rng.uniform(y0 - r + 1e-4, y0 - 1e-4, size=1)
                if y0 == 0.5:
                    y = rng.uniform(0 + r + 1e-4, 1 - r - 1e-4, size=1)
                    if x0 == 0:
                        x = rng.uniform(x0 + 1e-4, x0 + r - 1e-4, size=1)
                    else:
                        x = rng.uniform(x0 - r + 1e-4, x0 - 1e-4, size=1)
                seed = rng.choice(100000, size=(paf.get_random_seed_size(), 1))
                calc_area = paf.get_circle_area_overlap([x], [y], r, hull_x, hull_y, seed)
                if x0 == 0.5:
                    if y0 == 0:
                        h = r - (y - y0)
                    else:
                        h = r - (y0 - y)
                if y0 == 0.5:
                    if x0 == 0:
                        h = r - (x - x0)
                    else:
                        h = r - (x0 - x)
                # pylint: disable-next=possibly-used-before-assignment
                chord_area = r**2 * np.arccos(1 - h / r) - (r - h) * np.sqrt(r**2 - (r - h)**2)
                remaining_area = np.pi * r**2 - chord_area
                assert_allclose(calc_area, remaining_area)

    # pylint: disable-next=too-many-nested-blocks
    if position == "random":
        # Verify a few randomly placed circles too:
        done = 0
        xp = np.linspace(*x_edges, 400)
        yp = np.linspace(*y_edges, 400)
        dx, dy = xp[1] - xp[0], yp[1] - yp[0]
        while done < 100:
            seed = rng.choice(100000, size=(paf.get_random_seed_size(), 1))
            [x, y] = rng.uniform(0, 1, size=2)
            if np.any([x - r < x_edges[0], x + r > x_edges[1],
                       y - r < y_edges[0], y + r > y_edges[1]]):
                calc_area = paf.get_circle_area_overlap([x], [y], r, hull_x, hull_y, seed)
                manual_area = 0
                for x_p in xp:
                    for y_p in yp:
                        if np.sqrt((x_p - x)**2 + (y_p - y)**2) <= r:
                            manual_area += dx*dy
                assert_allclose(calc_area, manual_area, rtol=0.05)
            done += 1

        # Verify that we don't mind if coordinates are negative:
        x, y = -0.1, 0.08
        x_edges = np.array([-0.15, 0.15])
        y_edges = np.array([-0.15, 0.15])
        hull_x = x_edges[[0, 0, 1, 1, 0]]
        hull_y = y_edges[[0, 1, 1, 0, 0]]
        seed = rng.choice(100000, size=(paf.get_random_seed_size(), 1))
        calc_area = paf.get_circle_area_overlap([x], [y], r, hull_x, hull_y, seed)
        xp = np.linspace(*x_edges, 200)
        yp = np.linspace(*y_edges, 200)
        dx, dy = xp[1] - xp[0], yp[1] - yp[0]
        manual_area = 0
        for x_p in xp:
            for y_p in yp:
                if np.sqrt((x_p - x)**2 + (y_p - y)**2) <= r:
                    manual_area += dx*dy
        assert_allclose(calc_area, manual_area, rtol=0.05)


def test_psf_perturb():
    l_cut = np.load(os.path.join(os.path.dirname(__file__), 'data/l_cut.npy'))
    dd_params = np.load(os.path.join(os.path.dirname(__file__), 'data/dd_params.npy'))

    psf_r = 1.185 * 6.1
    psf_sig = 6.1 / (2 * np.sqrt(2 * np.log(2)))

    rng = np.random.default_rng(34242)
    # These magnitudes are such that the fluxes lie in the four algorithm
    # regimes we care about: , ~100%, >~54%, >15%, and <15%.
    for flux in [0.999, 0.75, 0.26, 0.12]:
        for _ in range(50):
            r, t = np.sqrt(rng.uniform(0, psf_r)), rng.uniform(0, 2*np.pi)
            x, y = r * np.cos(t), r * np.sin(t)
            fluxes = flux + rng.uniform(-0.001, 0.001)
            xav, yav = paf.psf_perturb(x, y, r, fluxes, dd_params, flux, psf_sig, l_cut)

            if flux >= l_cut[2]:
                manual_x = x * fluxes / (1 + fluxes)
                manual_y = y * fluxes / (1 + fluxes)
            elif flux >= l_cut[0]:
                # sigma, mu, alpha, T, rc
                params = np.zeros((5), float)
                for i in range(dd_params.shape[1]):
                    params += dd_params[:, i, 1 if flux >= l_cut[1] else 0] * flux**i

                if np.abs(x) <= params[4] * psf_sig:
                    manual_x = x * fluxes / (1 + fluxes)
                else:

                    manual_x = (np.sign(x) * params[3] * fluxes *
                                skewnorm.pdf(np.abs(x) / psf_sig, params[2],
                                             loc=params[1], scale=params[0])) * psf_sig

                if np.abs(y) <= params[4] * psf_sig:
                    manual_y = y * fluxes / (1 + fluxes)
                else:

                    manual_y = (np.sign(y) * params[3] * fluxes *
                                skewnorm.pdf(np.abs(y) / psf_sig, params[2],
                                             loc=params[1], scale=params[0])) * psf_sig
            else:
                exps = np.exp(-0.25 * (x**2 + y**2) / psf_sig**2)
                manual_x = fluxes * x * exps
                manual_y = fluxes * y * exps

            assert_allclose(manual_x, xav, rtol=0.005)
            assert_allclose(manual_y, yav, rtol=0.005)


def test_fit_skew():
    rng = np.random.default_rng(34587345)
    for _ in range(100):
        params = rng.uniform(0, 4, 4)
        x = rng.uniform(0, 2)
        l = rng.uniform(0.15, 1)
        y1 = paf.fit_skew(params, x, l)
        y2 = params[3] * l * skewnorm.pdf(x, params[2], loc=params[1], scale=params[0])
        assert_allclose(y1, y2, atol=1e-15)


def test_calc_mag_offsets():
    # First, test the results where relative flux of secondary vs noise is
    # the dominant contributor.
    mag_array = np.array([13.99])
    model_mag_mids = np.array([14])
    model_mags_interval = np.array([0.1])
    log10y = np.array([0])
    r = 1.185 * 6.1
    n_norm = 1
    b = 0.05
    snr = 19.91
    count_array = np.array([1])
    dm = _calculate_magnitude_offsets(count_array, mag_array, b, snr, model_mag_mids, log10y,
                                      model_mags_interval, r, n_norm)
    assert_allclose(dm, 6.5, atol=0.0003)

    rng = np.random.default_rng(28937482734)
    # Second, verify the outputs of no perturber 1% of the time. For this we need
    # to fake slightly more involved data, though.
    # B / snr = 1 gives dm_max_snr = 0
    snr = 0.05
    for n in [0.5, 0.15]:
        model_mag_mids = np.arange(14, 20, 0.1)
        model_mags_interval = 0.1 * np.ones_like(model_mag_mids)
        log10y = np.zeros_like(model_mag_mids)
        count_array = np.ones(1, float) / (0.1 * np.pi * (r/3600)**2) * n

        density = 10**log10y * model_mags_interval * np.pi * (r/3600)**2 * count_array[0] / n_norm
        dm = _calculate_magnitude_offsets(count_array, mag_array, b, snr, model_mag_mids, log10y,
                                          model_mags_interval, r, n_norm)
        q = model_mag_mids <= mag_array[0] + dm[0]
        draws = rng.poisson(lam=density[q], size=(100000, np.sum(q)))
        frac = np.sum(np.sum(draws, axis=1) == 0) / draws.shape[0]
        assert frac < 0.01


class GalCountValues():  # pylint: disable=too-few-public-methods
    def __init__(self):
        # Update these values if this is changed in CrossMatch
        self.cmau = np.empty((5, 2, 4), float)
        self.cmau[0, :, :] = [[-24.286513, 1.141760, 2.655846, np.nan],
                              [-23.192520, 1.778718, 1.668292, np.nan]]
        self.cmau[1, :, :] = [[0.001487, 2.918841, 0.000510, np.nan],
                              [0.000560, 7.691261, 0.003330, -0.065565]]
        self.cmau[2, :, :] = [[-1.257761, 0.021362, np.nan, np.nan],
                              [-0.309077, -0.067411, np.nan, np.nan]]
        self.cmau[3, :, :] = [[-0.302018, 0.034203, np.nan, np.nan],
                              [-0.713062, 0.233366, np.nan, np.nan]]
        self.cmau[4, :, :] = [[1.233627, -0.322347, np.nan, np.nan],
                              [1.068926, -0.385984, np.nan, np.nan]]
        self.alpha0 = [[2.079, 3.524, 1.917, 1.992, 2.536], [2.461, 2.358, 2.568, 2.268, 2.402]]
        self.alpha1 = [[2.265, 3.862, 1.921, 1.685, 2.480], [2.410, 2.340, 2.200, 2.540, 2.464]]
        self.alphaweight = [[3.47e+09, 3.31e+06, 2.13e+09, 1.64e+10, 1.01e+09],
                            [3.84e+09, 1.57e+06, 3.91e+08, 4.66e+10, 3.03e+07]]


gal_values = GalCountValues()


class TestMakePerturbAUFs():
    def setup_class(self):
        self.auf_folder = 'auf_folder'
        self.cat_folder = 'cat_folder'
        os.system(f'rm -r {self.auf_folder}')
        os.system(f'rm -r {self.cat_folder}')
        os.makedirs(self.auf_folder)
        os.makedirs(self.cat_folder)

        self.filters = np.array(['W1'])
        self.tri_filt_names = np.copy(self.filters)
        self.auf_points = np.array([[0.0, 0.0]])
        self.ax_lims = np.array([0, 1, 0, 1])

        self.psf_fwhms = np.array([6.1])
        self.r = np.linspace(0, 1.185 * self.psf_fwhms[0], 2500)
        self.dr = np.diff(self.r)
        self.rho = np.linspace(0, 100, 5000)
        self.drho = np.diff(self.rho)
        self.include_perturb_auf = True
        self.num_trials = 50000
        self.j0s = mff.calc_j0(self.rho[:-1]+self.drho/2, self.r[:-1]+self.dr/2)

        self.delta_mag_cuts = np.array([10])

    def make_class(self):
        class A():  # pylint: disable=too-few-public-methods,too-many-instance-attributes
            def __init__(self):
                pass

        a = A()
        a.b_cat_folder_path = self.cat_folder
        a.b_auf_folder_path = self.auf_folder
        a.b_filt_names = self.filters
        a.b_auf_region_points = self.auf_points
        a.r = self.r
        a.dr = self.dr
        a.rho = self.rho
        a.drho = self.drho
        a.j0s = self.j0s
        a.include_perturb_auf = self.include_perturb_auf
        a.rank = None
        a.chunk_id = None

        return a

    @pytest.mark.remote_data
    def test_create_single_low_numbers(self):
        density_radius = np.sqrt(1 / np.pi / np.exp(8.7))

        d_mag = 0.1

        # Fake up a TRILEGAL simulation data file.
        text = ('#area = 140.0 sq deg\n#Av at infinity = 1\n'
                'Gc logAge [M/H] m_ini   logL   logTe logg  m-M0   Av    '
                'm2/m1 mbol   J      H      Ks     IRAC_3.6 IRAC_4.5 IRAC_5.8 IRAC_8.0 MIPS_24 '
                'MIPS_70 MIPS_160 W1     W2     W3     W4       Mact\n')
        # This won't be enough objects, so we'll trigger the error message later.
        for _ in range(3):
            text = text + (
                '1   6.65 -0.39  0.02415 -2.701 3.397  4.057 14.00  8.354 0.00 25.523 25.839 '
                '24.409 23.524 22.583 22.387 22.292 22.015 21.144 19.380 20.878 15.001 22.391 '
                '21.637 21.342  0.024\n1   6.65 -0.39  0.02415 -2.701 3.397  4.057 14.00 8.354 '
                '0.00 25.523 25.839 24.409 23.524 22.583 22.387 22.292 22.015 21.144 19.380 20.878 '
                '15.002 22.391 21.637 21.342  0.024\n 1   6.65 -0.39  0.02415 -2.701 3.397  4.057 '
                '14.00 8.354 0.00 25.523 25.839 24.409 23.524 22.583 22.387 22.292 22.015 21.144 '
                '19.380 20.878 15.003 22.391 21.637 21.342  0.024\n 1   6.65 -0.39  0.02415 -2.701 '
                '3.397  4.057 14.00  8.354 0.00 25.523 25.839 24.409 23.524 22.583 22.387 22.292 '
                '22.015 21.144 19.380 20.878 15.004 22.391 21.637 21.342  0.024\n\n 1   6.65 -0.39 '
                ' 0.02415 -2.701 3.397  4.057 14.00  8.354 0.00 25.523 25.839 24.409 23.524 22.583 '
                '22.387 22.292 22.015 21.144 19.380 20.878 100.99 22.391 21.637 21.342  0.024\n')

        os.makedirs(f'{self.auf_folder}/{self.auf_points[0][0]}/{self.auf_points[0][1]}', exist_ok=True)
        with open(f'{self.auf_folder}/{self.auf_points[0][0]}/{self.auf_points[0][1]}/'
                  'trilegal_auf_simulation_faint.dat', "w", encoding='utf-8') as f:
            f.write(text)
        f.close()

        snr_mag_params = np.array([[[0.0109, 46.08, 0.119, 130, 0]]])

        self.fake_cm = self.make_class()
        self.fake_cm.b_tri_set_name = 'WISE'
        self.fake_cm.b_tri_filt_num = 11
        self.fake_cm.b_tri_filt_names = self.tri_filt_names
        self.fake_cm.b_tri_maglim_faint = 32
        self.fake_cm.b_tri_num_faint = 1000000
        self.fake_cm.b_auf_region_frame = 'galactic'
        self.fake_cm.b_psf_fwhms = self.psf_fwhms
        self.fake_cm.num_trials = self.num_trials
        self.fake_cm.d_mag = d_mag
        self.fake_cm.delta_mag_cuts = self.delta_mag_cuts
        self.fake_cm.b_fit_gal_flag = False
        self.fake_cm.b_dens_dist = density_radius
        self.fake_cm.b_run_fw_auf = True
        self.fake_cm.b_run_psf_auf = False
        self.fake_cm.b_snr_mag_params = snr_mag_params
        self.fake_cm.b_gal_al_avs = [0]
        self.fake_cm.b_download_tri = False
        self.fake_cm.b_astro = np.array([[0.3, 0.3, 0.1]] * 101)
        self.fake_cm.b_astro[0, [0, 1]] = [0.3, 0.31]
        self.fake_cm.b_astro[1, [0, 1]] = [0.31, 0.3]
        self.fake_cm.b_astro[2, [0, 1]] = [0.31, 0.31]
        self.fake_cm.b_photo = np.array([np.concatenate(([14.99], [100]*100))]).T
        self.fake_cm.b_magref = np.array([0] * 101)
        self.fake_cm.b_dens_hist_tri_list = [None] * len(self.filters)
        self.fake_cm.b_tri_model_mags_list = [None] * len(self.filters)
        self.fake_cm.b_tri_model_mag_mids_list = [None] * len(self.filters)
        self.fake_cm.b_tri_model_mags_interval_list = [None] * len(self.filters)
        self.fake_cm.b_tri_n_bright_sources_star_list = [None] * len(self.filters)

        with pytest.raises(ValueError, match="The number of simulated objects in this sky patch "):
            make_perturb_aufs(self.fake_cm, 'b')

    @pytest.mark.remote_data
    def test_psf_algorithm(self):  # pylint: disable=too-many-locals
        # Number of sources per PSF circle, on average, solved backwards to ensure
        # that local density ends up exactly in the middle of a count_array bin.
        # This should be approximately 0.15 sources per PSF circle.
        psf_mean = np.exp(9.38) * np.pi * (1.185 * self.psf_fwhms[0] / 3600)**2

        density_radius = np.sqrt(1 / np.pi / np.exp(9.38))

        new_auf_points = np.vstack((self.auf_points, np.array([[10, 10]])))

        d_mag = 0.1

        # These magnitudes need to create a 0.25 magnitude bin such that they
        # lie in the three algorithm regimes we care about: >~54%, >15%, and <15%.
        for mag in [15.15, 16.1, 19.1]:
            # Fake up a TRILEGAL simulation data file.
            text = ('#area = 140.0 sq deg\n#Av at infinity = 1\n'
                    'Gc logAge [M/H] m_ini   logL   logTe logg  m-M0   Av    '
                    'm2/m1 mbol   J      H      Ks     IRAC_3.6 IRAC_4.5 IRAC_5.8 IRAC_8.0 MIPS_24 '
                    'MIPS_70 MIPS_160 W1     W2     W3     W4       Mact\n')
            for _ in range(35):
                text = text + (
                    '1   6.65 -0.39  0.02415 -2.701 3.397  4.057 14.00  8.354 0.00 25.523 25.839 '
                    f'24.409 23.524 22.583 22.387 22.292 22.015 21.144 19.380 20.878 {mag} 22.391 '
                    '21.637 21.342  0.024\n1   6.65 -0.39  0.02415 -2.701 3.397  4.057 14.00 8.354 '
                    '0.00 25.523 25.839 24.409 23.524 22.583 22.387 22.292 22.015 21.144 19.380 '
                    f'20.878 {mag+0.001} 22.391 21.637 21.342  0.024\n 1   6.65 -0.39  0.02415 -2.701 '
                    '3.397  4.057 14.00 8.354 0.00 25.523 25.839 24.409 23.524 22.583 22.387 '
                    f'22.292 22.015 21.144 19.380 20.878 {mag+0.002} 22.391 21.637 21.342  0.024\n 1   '
                    '6.65 -0.39  0.02415 -2.701 3.397  4.057 14.00  8.354 0.00 25.523 25.839 '
                    f'24.409 23.524 22.583 22.387 22.292 22.015 21.144 19.380 20.878 {mag+0.003} 22.391 '
                    '21.637 21.342  0.024\n\n 1   6.65 -0.39  0.02415 -2.701 3.397  4.057 14.00  '
                    '8.354 0.00 25.523 25.839 24.409 23.524 22.583 22.387 22.292 22.015 21.144 '
                    '19.380 20.878 100.99 22.391 21.637 21.342  0.024\n')
            for new_auf_point in new_auf_points:
                os.makedirs(f'{self.auf_folder}/{new_auf_point[0]}/{new_auf_point[1]}', exist_ok=True)
                with open(f'{self.auf_folder}/{new_auf_point[0]}/{new_auf_point[1]}/'
                          'trilegal_auf_simulation_faint.dat', "w", encoding='utf-8') as f:
                    f.write(text)

            prob_0_draw = psf_mean**0 * np.exp(-psf_mean) / math.factorial(0)
            prob_1_draw = psf_mean**1 * np.exp(-psf_mean) / math.factorial(1)
            prob_2_draw = psf_mean**2 * np.exp(-psf_mean) / math.factorial(2)

            ax1, ax2 = self.auf_points[0]

            keep_frac = np.zeros((len(self.delta_mag_cuts), 1), float)
            if mag > 17:
                keep_flux = np.zeros((1,), float)

            photo_array = np.array([np.concatenate(([14.99], [100]*100, [10], [10], [10], [10]))]).T
            # Catalogue bins for the source:
            a_photo = photo_array[0, :]
            dmag = 0.25
            mag_min = dmag * np.floor(np.amin(a_photo[0])/dmag)
            mag_max = dmag * np.ceil(np.amax(a_photo[0])/dmag)
            mag_bins = np.arange(mag_min, mag_max+1e-10, dmag)
            mag_bin = 0.5 * (mag_bins[1:]+mag_bins[:-1])
            # Model magnitude bins:
            tri_mags = np.array([mag, mag+0.001, mag+0.002, mag+0.003])
            minmag = d_mag * np.floor(np.amin(tri_mags)/d_mag)
            maxmag = d_mag * np.ceil(np.amax(tri_mags)/d_mag)
            mod_bins = np.arange(minmag, maxmag+1e-10, d_mag)
            mod_bin = mod_bins[:-1] + np.diff(mod_bins)/2
            mag_offset = mod_bin - mag_bin
            rel_flux = 10**(-1/2.5 * mag_offset)
            dfluxes1 = 10**(-(mag_offset-d_mag/2)/2) - 10**(-mag_offset/2.5)
            dfluxes2 = 10**(-mag_offset/2.5) - 10**(-(mag_offset+d_mag/2)/2.5)

            snr_mag_params = np.array([[[0.0109, 46.08, 0.119, 130, 0]]])
            l_cut = np.load(os.path.join(os.path.dirname(__file__), 'data/l_cut.npy'))
            dd_params = np.load(os.path.join(os.path.dirname(__file__), 'data/dd_params.npy'))
            run_fw = mag >= 19
            self.fake_cm = self.make_class()
            self.fake_cm.b_auf_region_points = new_auf_points
            self.fake_cm.b_tri_set_name = 'WISE'
            self.fake_cm.b_tri_filt_num = 11
            self.fake_cm.b_tri_filt_names = self.tri_filt_names
            self.fake_cm.b_tri_maglim_faint = 32
            self.fake_cm.b_tri_num_faint = 1000000
            self.fake_cm.b_auf_region_frame = 'galactic'
            self.fake_cm.b_psf_fwhms = self.psf_fwhms
            self.fake_cm.num_trials = self.num_trials
            self.fake_cm.d_mag = d_mag
            self.fake_cm.delta_mag_cuts = self.delta_mag_cuts
            self.fake_cm.b_fit_gal_flag = False
            self.fake_cm.b_dens_dist = density_radius
            self.fake_cm.b_run_fw_auf = run_fw
            self.fake_cm.b_run_psf_auf = True
            self.fake_cm.b_snr_mag_params = snr_mag_params
            self.fake_cm.b_dd_params = dd_params
            self.fake_cm.b_l_cut = l_cut
            self.fake_cm.b_gal_al_avs = [0]
            self.fake_cm.b_download_tri = False
            self.fake_cm.b_dens_hist_tri_list = [None] * len(self.filters)
            self.fake_cm.b_tri_model_mags_list = [None] * len(self.filters)
            self.fake_cm.b_tri_model_mag_mids_list = [None] * len(self.filters)
            self.fake_cm.b_tri_model_mags_interval_list = [None] * len(self.filters)
            self.fake_cm.b_tri_n_bright_sources_star_list = [None] * len(self.filters)
            # Have to fudge extra sources to keep our 15th mag source in the local
            # density cutout.
            # Add extra sources to force a 0.1-0.9 square convex hull cutout.
            self.fake_cm.b_astro = np.concatenate(
                ([0.3, 0.3, 0.1] * 101, [0.1, 0.1, 0.1], [0.1, 0.9, 0.1],
                 [0.9, 0.1, 0.1], [0.9, 0.9, 0.1])).reshape(-1, 3)
            self.fake_cm.b_photo = photo_array
            self.fake_cm.b_magref = np.array([0] * 105)
            _, p_a_o = make_perturb_aufs(self.fake_cm, 'b')

            perturb_auf_combo = f'{ax1}-{ax2}-{self.filters[0]}'
            for name, size in zip(
                    ['frac', 'flux', 'offset', 'cumulative', 'fourier', 'Narray', 'magarray'],
                    [(len(self.delta_mag_cuts), 3), (3,), (len(self.r)-1, 3),
                     (len(self.r)-1, 3), (len(self.rho)-1, 3), (3,), (3,)]):
                var = p_a_o[perturb_auf_combo][name]
                assert np.all(var.shape == size)

            keep_frac = var = p_a_o[perturb_auf_combo]['frac']
            if mag > 17:
                keep_flux = var = p_a_o[perturb_auf_combo]['flux']

            # Have more relaxed conditions on assertion than in test_perturb_aufs
            # above, as we can't arbitrarily force the magnitude bin widths to be
            # very small, and hence have a blur on relative fluxes allowed.
            assert_allclose(keep_frac[0, 0], 1-prob_0_draw, rtol=0.1)

            if mag > 17:
                psf_r = 1.185 * self.psf_fwhms[0]
                psf_sig = self.psf_fwhms[0] / (2 * np.sqrt(2 * np.log(2)))

                rel_flux = rel_flux + 0.5 * (dfluxes1 + dfluxes2)
                _y = np.linspace(-psf_r, psf_r, 10000)
                offsets, sum_q = 0, 0
                for x in np.linspace(-psf_r, psf_r, 10000):
                    y = _y[np.sqrt(x**2 + _y**2) <= psf_r]
                    exps = np.exp(-0.25 * (x**2 + y**2) / psf_sig**2)
                    offsets += np.sum(exps)
                    sum_q += len(y)
                offset = offsets / sum_q
                df = rel_flux * offset
                assert_allclose(keep_flux[0], (prob_0_draw*0 + prob_1_draw*df +
                                prob_2_draw*2*df), rtol=0.1, atol=0.005)

    @pytest.mark.remote_data
    @pytest.mark.parametrize("precompute_tri_hists", [True, False])
    def test_compute_local_density(self, precompute_tri_hists):
        # pylint: disable=no-member
        # Number of sources per PSF circle, on average, solved backwards to ensure
        # that local density ends up exactly in the middle of a count_array bin.
        # This should be approximately 0.076 sources per PSF circle.
        psf_mean = np.exp(8.7) * np.pi * (1.185 * self.psf_fwhms[0] / 3600)**2
        # We want to calculate the local density on the fly. We
        # therefore have to choose our "density radius" to set the appropriate
        # local density for our single source.
        density_radius = np.sqrt(1 / np.pi / np.exp(8.7))

        new_auf_points = np.vstack((self.auf_points, np.array([[10, 10]])))

        # Have to fudge extra sources to keep our 15th mag source in the local
        # density cutout.
        # Force the 0.1-0.9 square with extra objects for the convex hull to pick up.
        x = np.concatenate(([0.3, 0.3, 0.1] * 101, [0.1, 0.1, 0.1], [0.1, 0.9, 0.1],
                            [0.9, 0.1, 0.1], [0.9, 0.9, 0.1])).reshape(-1, 3)
        np.save(f'{self.cat_folder}/con_cat_astro.npy', x)
        np.save(f'{self.cat_folder}/con_cat_photo.npy',
                np.array([np.concatenate(([14.99], [100]*100, [10], [10], [10], [10]))]).T)
        np.save(f'{self.cat_folder}/magref.npy', np.array([0] * 105))

        # Fake up a TRILEGAL simulation data file.
        text = ('#area = 140.0 sq deg\n#Av at infinity = 1\n'
                'Gc logAge [M/H] m_ini   logL   logTe logg  m-M0   Av    '
                'm2/m1 mbol   J      H      Ks     IRAC_3.6 IRAC_4.5 IRAC_5.8 IRAC_8.0 MIPS_24 '
                'MIPS_70 MIPS_160 W1     W2     W3     W4       Mact\n')
        for _ in range(35):
            text = text + (
                '1   6.65 -0.39  0.02415 -2.701 3.397  4.057 14.00  8.354 0.00 25.523 25.839 '
                '24.409 23.524 22.583 22.387 22.292 22.015 21.144 19.380 20.878 15.001 22.391 '
                '21.637 21.342  0.024\n1   6.65 -0.39  0.02415 -2.701 3.397  4.057 14.00 8.354 '
                '0.00 25.523 25.839 24.409 23.524 22.583 22.387 22.292 22.015 21.144 19.380 20.878 '
                '15.002 22.391 21.637 21.342  0.024\n 1   6.65 -0.39  0.02415 -2.701 3.397  4.057 '
                '14.00 8.354 0.00 25.523 25.839 24.409 23.524 22.583 22.387 22.292 22.015 21.144 '
                '19.380 20.878 15.003 22.391 21.637 21.342  0.024\n 1   6.65 -0.39  0.02415 -2.701 '
                '3.397  4.057 14.00  8.354 0.00 25.523 25.839 24.409 23.524 22.583 22.387 22.292 '
                '22.015 21.144 19.380 20.878 15.004 22.391 21.637 21.342  0.024\n\n 1   6.65 -0.39 '
                ' 0.02415 -2.701 3.397  4.057 14.00  8.354 0.00 25.523 25.839 24.409 23.524 22.583 '
                '22.387 22.292 22.015 21.144 19.380 20.878 100.99 22.391 21.637 21.342  0.024\n')
        for new_auf_point in new_auf_points:
            os.makedirs(f'{self.auf_folder}/{new_auf_point[0]}/{new_auf_point[1]}', exist_ok=True)
            with open(f'{self.auf_folder}/{new_auf_point[0]}/{new_auf_point[1]}/'
                      'trilegal_auf_simulation_faint.dat', "w", encoding='utf-8') as f:
                f.write(text)

        prob_0_draw = psf_mean**0 * np.exp(-psf_mean) / math.factorial(0)
        prob_1_draw = psf_mean**1 * np.exp(-psf_mean) / math.factorial(1)
        prob_2_draw = psf_mean**2 * np.exp(-psf_mean) / math.factorial(2)

        ax1, ax2 = self.auf_points[0]

        d_mag = 0.1
        # Catalogue bins for the source, only keeping the single
        # source currently under consideration, and ignoring the two extra objects
        # used to force the local density to be calculated properly.
        a_photo = np.load(f'{self.cat_folder}/con_cat_photo.npy')[0, :]
        dmag = 0.25
        mag_min = dmag * np.floor(np.amin(a_photo[0])/dmag)
        mag_max = dmag * np.ceil(np.amax(a_photo[0])/dmag)
        mag_bins = np.arange(mag_min, mag_max+1e-10, dmag)
        mag_bin = 0.5 * (mag_bins[1:]+mag_bins[:-1])
        # Model magnitude bins:
        tri_mags = np.array([15.001, 15.002, 15.003, 15.004])
        minmag = d_mag * np.floor(np.amin(tri_mags)/d_mag)
        maxmag = d_mag * np.ceil(np.amax(tri_mags)/d_mag)
        mod_bins = np.arange(minmag, maxmag+1e-10, d_mag)
        mod_bin = mod_bins[:-1] + np.diff(mod_bins)/2
        mag_offset = mod_bin - mag_bin
        rel_flux = 10**(-1/2.5 * mag_offset)

        ol, nl = 'include_perturb_auf = no', 'include_perturb_auf = yes\n'
        with open(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        idx = np.where([ol in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                      idx, nl, out_file=os.path.join(os.path.dirname(__file__),
                      'data/crossmatch_params_.txt'))

        ol, nl = 'filt_names = G_BP G G_RP', 'filt_names = G\n'
        with open(os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        idx = np.where([ol in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                      idx, nl, out_file=os.path.join(os.path.dirname(__file__),
                      'data/cat_a_params_.txt'))
        for ol, nl in zip(['psf_fwhms = 0.12 0.12 0.12', 'cat_folder_path = gaia_folder',
                           'auf_folder_path = gaia_auf_folder', 'tri_filt_names = G_BP G G_RP',
                           'gal_al_avs = '],
                          ['psf_fwhms = 0.12\n', 'cat_folder_path = cat_folder\n',
                           'auf_folder_path = auf_folder\n', 'tri_filt_names = W1\n',
                           'gal_al_avs = 0\n']):
            with open(os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                      encoding='utf-8') as file:
                f = file.readlines()
            idx = np.where([ol in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_a_params_.txt'),
                          idx, nl)

        ol, nl = 'filt_names = W1 W2 W3 W4', 'filt_names = W1\n'
        with open(os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        idx = np.where([ol in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'),
                      idx, nl, out_file=os.path.join(os.path.dirname(__file__),
                      'data/cat_b_params_.txt'))
        for ol, nl in zip(['psf_fwhms = 6.08 6.84 7.36 11.99', 'cat_folder_path = wise_folder',
                           'auf_folder_path = wise_auf_folder', 'tri_filt_names = W1 W2 W3 W4',
                           'gal_al_avs = '],
                          ['psf_fwhms = 6.08\n', 'cat_folder_path = cat_folder\n',
                           'auf_folder_path = auf_folder\n', 'tri_filt_names = W1\n',
                           'gal_al_avs = 0\n']):
            with open(os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'),
                      encoding='utf-8') as file:
                f = file.readlines()
            idx = np.where([ol in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_b_params_.txt'),
                          idx, nl)

        os.makedirs('a_snr_mag', exist_ok=True)
        os.makedirs('b_snr_mag', exist_ok=True)
        np.save('a_snr_mag/snr_mag_params.npy', np.array([[[0.0109, 46.08, 0.119, 130, 0]]]))
        np.save('b_snr_mag/snr_mag_params.npy', np.array([[[0.0109, 46.08, 0.119, 130, 0]]]))

        # Fake this the easy way both times, then below correct the parameters
        # for the precompute_hist version of the test.
        with open(os.path.join(os.path.dirname(__file__), 'data/cat_a_params_.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        old_line = 'auf_folder_path = auf_folder'
        new_line = ('auf_folder_path = auf_folder\ndens_hist_tri_location = None\n'
                    'tri_model_mags_location = None\ntri_model_mag_mids_location = None\n'
                    'tri_model_mags_interval_location = None\ntri_dens_uncert_location = None\n'
                    'tri_n_bright_sources_star_location = None\n')
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_a_params_.txt'), idx, new_line)
        with open(os.path.join(os.path.dirname(__file__), 'data/cat_b_params_.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        old_line = 'auf_folder_path = auf_folder'
        new_line = ('auf_folder_path = auf_folder\ndens_hist_tri_location = None\n'
                    'tri_model_mags_location = None\ntri_model_mag_mids_location = None\n'
                    'tri_model_mags_interval_location = None\ntri_dens_uncert_location = None\n'
                    'tri_n_bright_sources_star_location = None\n')
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_b_params_.txt'), idx, new_line)

        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'))
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                          'data/crossmatch_params_.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params_.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params_.txt'))

        if precompute_tri_hists:
            for flag in ['a_', 'b_']:
                _a_photo = np.load(f'{self.cat_folder}/con_cat_photo.npy')
                hist, bins = np.histogram(_a_photo[~np.isnan(_a_photo)], bins='auto')
                dens_mag = (bins[:-1]+np.diff(bins)/2)[np.argmax(hist)] - 0.5
                dens, tri_mags, tri_mags_mids, dtri_mags, _, num_bright_obj = make_tri_counts(
                    f'{self.auf_folder}/{new_auf_points[0][0]}/{new_auf_points[0][1]}/',
                    'trilegal_auf_simulation', getattr(cm, f'{flag}tri_filt_names')[0], cm.d_mag,
                    np.amin(a_photo), dens_mag)
                setattr(cm, f'{flag}dens_hist_tri_list', [dens])
                setattr(cm, f'{flag}tri_model_mags_list', [tri_mags])
                setattr(cm, f'{flag}tri_model_mag_mids_list', [tri_mags_mids])
                setattr(cm, f'{flag}tri_model_mags_interval_list', [dtri_mags])
                setattr(cm, f'{flag}tri_n_bright_sources_star_list', [num_bright_obj])

            cm.a_auf_folder_path = None
            cm.b_auf_folder_path = None
            cm.a_tri_set_name = None
            cm.b_tri_set_name = None
            cm.a_tri_maglim_faint = None
            cm.b_tri_maglim_faint = None
            cm.a_tri_num_faint = None
            cm.b_tri_num_faint = None
            cm.a_download_tri = None
            cm.b_download_tri = None
            cm.a_tri_filt_num = None
            cm.b_tri_filt_num = None
            cm.a_tri_filt_names = [None] * len(cm.a_tri_filt_names)
            cm.b_tri_maglim_faint = [None] * len(cm.b_tri_filt_names)

        cm.a_auf_region_points = new_auf_points
        cm.b_auf_region_points = new_auf_points
        cm.cross_match_extent = self.ax_lims
        cm.a_dens_dist = density_radius
        cm.b_dens_dist = density_radius
        cm.r = self.r
        cm.dr = self.dr
        cm.rho = self.rho
        cm.drho = self.drho
        cm.j0s = self.j0s
        cm.num_trials = self.num_trials
        cm.a_fit_gal_flag = False
        cm.b_fit_gal_flag = False

        cm.a_run_fw = True
        cm.a_run_psf = False
        cm.b_run_fw = True
        cm.b_run_psf = False

        cm.chunk_id = 1

        cm.perturb_auf_func = make_perturb_aufs

        mcff = Macauff(cm)
        mcff.create_perturb_auf()

        perturb_auf_combo = f'{ax1}-{ax2}-{self.filters[0]}'
        fracs = cm.b_perturb_auf_outputs[perturb_auf_combo]['frac']
        fluxs = cm.b_perturb_auf_outputs[perturb_auf_combo]['flux']
        fourier = cm.b_perturb_auf_outputs[perturb_auf_combo]['fourier']

        assert_allclose(fracs[0, 0], 1-prob_0_draw, rtol=0.1)
        assert_allclose(fluxs[0], (prob_0_draw*0 + prob_1_draw*rel_flux +
                        prob_2_draw*2*rel_flux), rtol=0.1)

        psf_r = 1.185 * self.psf_fwhms[0]
        small_r = psf_r * rel_flux / (1 + rel_flux)

        fake_fourier = (1-prob_0_draw) / (np.pi * small_r * (cm.rho[:-1]+cm.drho/2)) * j1(
            2 * np.pi * small_r * (cm.rho[:-1]+cm.drho/2))
        fake_fourier += prob_0_draw * j0(2 * np.pi * (cm.r[0]+cm.dr[0]/2) *
                                         (cm.rho[:-1]+cm.drho/2))

        assert_allclose(fake_fourier, fourier[:, 0], rtol=0.05)

    @pytest.mark.remote_data
    def test_with_galaxy_counts(self):
        # Number of sources per PSF circle, on average, solved backwards to ensure
        # that local density ends up exactly in the middle of a count_array bin.
        # This should be approximately 0.076 sources per PSF circle.
        psf_mean = np.exp(8.7) * np.pi * (1.185 * self.psf_fwhms[0] / 3600)**2

        density_radius = np.sqrt(1 / np.pi / np.exp(8.7))

        new_auf_points = np.vstack((self.auf_points, np.array([[10, 10]])))

        # Force the 0.1-0.9 square with extra objects for the convex hull to pick up.
        x = np.concatenate(([0.3, 0.3, 0.1] * 101, [0.1, 0.1, 0.1], [0.1, 0.9, 0.1],
                            [0.9, 0.1, 0.1], [0.9, 0.9, 0.1])).reshape(-1, 3)
        np.save(f'{self.cat_folder}/con_cat_astro.npy', x)
        rng = np.random.default_rng(seed=83458923)
        main_mags = rng.uniform(24.95, 25.05, size=100)
        np.save(f'{self.cat_folder}/con_cat_photo.npy',
                np.array([np.concatenate(([14.99], main_mags, [10], [10], [10], [10]))]).T)
        np.save(f'{self.cat_folder}/magref.npy', np.array([0] * 105))

        # Fake up a TRILEGAL simulation data file.
        text = ('#area = 140.0 sq deg\n#Av at infinity = 1\n'
                'Gc logAge [M/H] m_ini   logL   logTe logg  m-M0   Av    '
                'm2/m1 mbol   J      H      Ks     IRAC_3.6 IRAC_4.5 IRAC_5.8 IRAC_8.0 MIPS_24 '
                'MIPS_70 MIPS_160 W1     W2     W3     W4       Mact\n')
        for _ in range(35):
            text = text + (
                '1   6.65 -0.39  0.02415 -2.701 3.397  4.057 14.00  8.354 0.00 25.523 25.839 '
                '24.409 23.524 22.583 22.387 22.292 22.015 21.144 19.380 20.878 15.001 22.391 '
                '21.637 21.342  0.024\n1   6.65 -0.39  0.02415 -2.701 3.397  4.057 14.00 8.354 '
                '0.00 25.523 25.839 24.409 23.524 22.583 22.387 22.292 22.015 21.144 19.380 20.878 '
                '15.002 22.391 21.637 21.342  0.024\n 1   6.65 -0.39  0.02415 -2.701 3.397  4.057 '
                '14.00 8.354 0.00 25.523 25.839 24.409 23.524 22.583 22.387 22.292 22.015 21.144 '
                '19.380 20.878 15.003 22.391 21.637 21.342  0.024\n 1   6.65 -0.39  0.02415 -2.701 '
                '3.397  4.057 14.00  8.354 0.00 25.523 25.839 24.409 23.524 22.583 22.387 22.292 '
                '22.015 21.144 19.380 20.878 15.004 22.391 21.637 21.342  0.024\n\n 1   6.65 -0.39 '
                ' 0.02415 -2.701 3.397  4.057 14.00  8.354 0.00 25.523 25.839 24.409 23.524 22.583 '
                '22.387 22.292 22.015 21.144 19.380 20.878 25.99 22.391 21.637 21.342  0.024\n')
        for new_auf_point in new_auf_points:
            os.makedirs(f'{self.auf_folder}/{new_auf_point[0]}/{new_auf_point[1]}', exist_ok=True)
            with open(f'{self.auf_folder}/{new_auf_point[0]}/{new_auf_point[1]}/'
                      'trilegal_auf_simulation_faint.dat', "w", encoding='utf-8') as f:
                f.write(text)

        prob_0_draw = psf_mean**0 * np.exp(-psf_mean) / math.factorial(0)
        prob_1_draw = psf_mean**1 * np.exp(-psf_mean) / math.factorial(1)
        prob_2_draw = psf_mean**2 * np.exp(-psf_mean) / math.factorial(2)

        ax1, ax2 = self.auf_points[0]

        # Catalogue bins for the source:
        a_photo = np.load(f'{self.cat_folder}/con_cat_photo.npy')[0, :]
        dmag = 0.25
        mag_min = dmag * np.floor(np.amin(a_photo[0])/dmag)
        mag_max = dmag * np.ceil(np.amax(a_photo[0])/dmag)
        mag_bins = np.arange(mag_min, mag_max+1e-10, dmag)
        mag_bin = 0.5 * (mag_bins[1:]+mag_bins[:-1])
        # Model magnitude bins:
        d_mag = 0.1
        tri_mags = np.array([15.001, 15.002, 15.003, 15.004])
        minmag = d_mag * np.floor(np.amin(tri_mags)/d_mag)
        maxmag = d_mag * np.ceil(np.amax(tri_mags)/d_mag)
        mod_bins = np.arange(minmag, maxmag+1e-10, d_mag)
        mod_bin = mod_bins[:-1] + np.diff(mod_bins)/2
        mag_offset = mod_bin - mag_bin
        rel_flux = 10**(-1/2.5 * mag_offset)

        ol, nl = 'include_perturb_auf = no', 'include_perturb_auf = yes\n'
        with open(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        idx = np.where([ol in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                      idx, nl, out_file=os.path.join(os.path.dirname(__file__),
                      'data/crossmatch_params_.txt'))

        ol, nl = 'filt_names = G_BP G G_RP', 'filt_names = G\n'
        with open(os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        idx = np.where([ol in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                      idx, nl, out_file=os.path.join(os.path.dirname(__file__),
                      'data/cat_a_params_.txt'))
        for ol, nl in zip(['psf_fwhms = 0.12 0.12 0.12', 'cat_folder_path = gaia_folder',
                           'auf_folder_path = gaia_auf_folder', 'tri_filt_names = G_BP G G_RP',
                           'gal_al_avs = '],
                          ['psf_fwhms = 0.12\n', 'cat_folder_path = cat_folder\n',
                           'auf_folder_path = auf_folder\n', 'tri_filt_names = W1\n',
                           'gal_al_avs = 0\n']):
            with open(os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                      encoding='utf-8') as file:
                f = file.readlines()
            idx = np.where([ol in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_a_params_.txt'),
                          idx, nl)

        ol, nl = 'filt_names = W1 W2 W3 W4', 'filt_names = W1\n'
        with open(os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        idx = np.where([ol in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'),
                      idx, nl, out_file=os.path.join(os.path.dirname(__file__),
                      'data/cat_b_params_.txt'))
        for ol, nl in zip(['psf_fwhms = 6.08 6.84 7.36 11.99', 'cat_folder_path = wise_folder',
                           'auf_folder_path = wise_auf_folder', 'tri_filt_names = W1 W2 W3 W4',
                           'gal_al_avs = '],
                          ['psf_fwhms = 6.08\n', 'cat_folder_path = cat_folder\n',
                           'auf_folder_path = auf_folder\n', 'tri_filt_names = W1\n',
                           'gal_al_avs = 0\n']):
            with open(os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'),
                      encoding='utf-8') as file:
                f = file.readlines()
            idx = np.where([ol in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_b_params_.txt'),
                          idx, nl)

        os.makedirs('a_snr_mag', exist_ok=True)
        os.makedirs('b_snr_mag', exist_ok=True)
        np.save('a_snr_mag/snr_mag_params.npy', np.array([[[0.0109, 46.08, 0.119, 130, 0]]]))
        np.save('b_snr_mag/snr_mag_params.npy', np.array([[[0.0109, 46.08, 0.119, 130, 0]]]))

        with open(os.path.join(os.path.dirname(__file__), 'data/cat_a_params_.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        old_line = 'auf_folder_path = auf_folder'
        new_line = ('auf_folder_path = auf_folder\ndens_hist_tri_location = None\n'
                    'tri_model_mags_location = None\ntri_model_mag_mids_location = None\n'
                    'tri_model_mags_interval_location = None\ntri_dens_uncert_location = None\n'
                    'tri_n_bright_sources_star_location = None\n')
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_a_params_.txt'), idx, new_line)
        with open(os.path.join(os.path.dirname(__file__), 'data/cat_b_params_.txt'),
                  encoding='utf-8') as file:
            f = file.readlines()
        old_line = 'auf_folder_path = auf_folder'
        new_line = ('auf_folder_path = auf_folder\ndens_hist_tri_location = None\n'
                    'tri_model_mags_location = None\ntri_model_mag_mids_location = None\n'
                    'tri_model_mags_interval_location = None\ntri_dens_uncert_location = None\n'
                    'tri_n_bright_sources_star_location = None\n')
        idx = np.where([old_line in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_b_params_.txt'), idx, new_line)

        cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data'))
        cm._initialise_chunk(os.path.join(os.path.dirname(__file__),
                                          'data/crossmatch_params_.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params_.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params_.txt'))

        cm.a_auf_region_points = new_auf_points
        cm.b_auf_region_points = new_auf_points
        cm.cross_match_extent = self.ax_lims
        cm.r = self.r
        cm.dr = self.dr
        cm.rho = self.rho
        cm.drho = self.drho
        cm.j0s = self.j0s
        cm.cross_match_extent = self.ax_lims
        cm.num_trials = self.num_trials
        cm.a_fit_gal_flag = True
        cm.b_fit_gal_flag = True

        cm.a_gal_wavs = np.array([0.641])
        cm.a_gal_zmax = np.array([0.001])
        cm.a_gal_nzs = np.array([2])
        cm.a_gal_aboffsets = np.array([0.105])
        cm.a_gal_filternames = np.array(['gaiadr2-G'])
        cm.a_gal_al_avs = np.array([0])
        cm.a_saturation_magnitudes = np.array([5])

        cm.b_gal_wavs = np.array([3.4])
        cm.b_gal_zmax = np.array([1])
        cm.b_gal_nzs = np.array([10])
        cm.b_gal_aboffsets = np.array([2.699])
        cm.b_gal_filternames = ['wise2010-W1']
        cm.b_gal_al_avs = np.array([0])
        cm.b_saturation_magnitudes = np.array([5])

        cm.a_run_fw = True
        cm.a_run_psf = False
        cm.b_run_fw = True
        cm.b_run_psf = False

        cm.chunk_id = 1

        cm.a_dens_dist = density_radius
        cm.b_dens_dist = density_radius

        cm.perturb_auf_func = make_perturb_aufs

        mcff = Macauff(cm)
        mcff.create_perturb_auf()

        perturb_auf_combo = f'{ax1}-{ax2}-{self.filters[0]}'
        fracs = cm.b_perturb_auf_outputs[perturb_auf_combo]['frac']
        fluxs = cm.b_perturb_auf_outputs[perturb_auf_combo]['flux']
        fourier = cm.b_perturb_auf_outputs[perturb_auf_combo]['fourier']

        assert_allclose(fracs[0, 0], 1-prob_0_draw, rtol=0.1)
        assert_allclose(fluxs[0], (prob_0_draw*0 + prob_1_draw*rel_flux +
                        prob_2_draw*2*rel_flux), rtol=0.1)

        psf_r = 1.185 * self.psf_fwhms[0]
        small_r = psf_r * rel_flux / (1 + rel_flux)

        fake_fourier = (1-prob_0_draw) / (np.pi * small_r * (cm.rho[:-1]+cm.drho/2)) * j1(
            2 * np.pi * small_r * (cm.rho[:-1]+cm.drho/2))
        fake_fourier += prob_0_draw * j0(2 * np.pi * (cm.r[0]+cm.dr[0]/2) *
                                         (cm.rho[:-1]+cm.drho/2))

        assert_allclose(fake_fourier, fourier[:, 0], rtol=0.05)


@pytest.mark.parametrize("run_type", ['faint', 'bright', 'both', 'neither'])
def test_make_tri_counts(run_type):  # pylint: disable=too-many-branches
    # Faint from 10-20, bright from 5-15
    rng = np.random.default_rng(seed=464564399234)
    n_b, n_f = 100000, 90000
    if run_type != 'bright':
        script = '#area = 1 sq deg\n#Av at infinity = 1\n#W1 Av\n'
        mags = rng.uniform(10, 20, size=n_f)
        for mag in mags:
            script += f'{mag} 1\n'
        with open('trilegal_auf_simulation_faint.dat', 'w', encoding='utf-8') as out:
            out.writelines(script)
    if run_type != 'faint':
        script = '#area = 1 sq deg\n#Av at infinity = 1\n#W1 Av\n'
        mags = rng.uniform(5, 15, size=n_b)
        fake_mags = np.arange(5, 15.01, 0.1)
        h, _ = np.histogram(mags, fake_mags)
        if run_type == 'both':
            # Fake it that the final bin has the most sources to avoid
            # make_tri_count's physical-counts-driven completeness limit
            # cull.
            big_bin_value = np.argmax(h)
            big_number = np.amax(h)
            final_number = h[-1]
            big_bin_values = np.where((mags >= fake_mags[big_bin_value]) &
                                      (mags <= fake_mags[big_bin_value+1]))[0]
            for i in big_bin_values[:(big_number-final_number)+2]:
                mags[i] = rng.uniform(fake_mags[-2], fake_mags[-1])

        for mag in mags:
            script += f'{mag} 1\n'
        with open('trilegal_auf_simulation_bright.dat', 'w', encoding='utf-8') as out:
            out.writelines(script)
    if run_type != "neither":
        dens, tri_mags, tri_mags_mids, _, uncert, n = make_tri_counts(
            '.', 'trilegal_auf_simulation', 'W1', 0.1, 5 if run_type != "faint" else 10, 20,
            use_bright=run_type != "faint", use_faint=run_type != "bright")
        assert n
        if run_type == "both":
            assert_allclose(tri_mags[0], 5, atol=0.1)
            assert_allclose(tri_mags[-1], 20, atol=0.1)
        if run_type == "faint":
            assert_allclose(tri_mags[0], 10, atol=0.1)
            assert_allclose(tri_mags[-1], 20, atol=0.1)
        if run_type == "bright":
            assert_allclose(tri_mags[0], 5, atol=0.1)
            assert_allclose(tri_mags[-1], 15, atol=0.1)
        for i, tri_mag_mid in enumerate(tri_mags_mids):
            if tri_mag_mid < 10:
                # 10-20 vs 5-15 mags with 0.1 mag bins for N/100. This is
                # not len(tri_mags) since each gets its density separately
                # from where its 10 mags of dynamic range get placed in the
                # larger bin set!
                expect_dens = n_b/100 / 0.1 / 1
            elif tri_mag_mid > 15:
                expect_dens = n_f/100 / 0.1 / 1
            else:
                if run_type == 'faint':
                    expect_dens = n_f/100 / 0.1 / 1
                elif run_type == 'bright':
                    expect_dens = n_b/100 / 0.1 / 1
                else:
                    d_u_f = np.sqrt(n_f/100) / 0.1 / 1
                    d_u_b = np.sqrt(n_b/100) / 0.1 / 1
                    w_f, w_b = 1 / d_u_f**2, 1 / d_u_b**2
                    d_f, d_b = n_f/100 / 0.1 / 1, n_b/100 / 0.1 / 1
                    expect_dens = (d_b * w_b + d_f * w_f) / (w_b + w_f)
            assert_allclose(expect_dens, dens[i], atol=3*uncert[i], rtol=0.01)
    else:
        with pytest.raises(ValueError, match="use_bright and use_faint cannot both be "):
            dens, tri_mags, tri_mags_mids, _, uncert, n = make_tri_counts(
                '.', 'trilegal_auf_simulation', 'W1', 0.1, 10, 20, use_bright=False,
                use_faint=False)

    if run_type == "both":
        with pytest.raises(ValueError, match="If one of al_av or av_grid is provided "):
            dens, tri_mags, tri_mags_mids, _, uncert, n = make_tri_counts(
                '.', 'trilegal_auf_simulation', 'W1', 0.1, 5, 20,
                use_bright=False, use_faint=True, al_av=0.9)

        dens, tri_mags, tri_mags_mids, _, uncert, n = make_tri_counts(
            '.', 'trilegal_auf_simulation', 'W1', 0.1, 5, 20, use_bright=True, use_faint=True,
            al_av=0.9, av_grid=np.array([2, 2, 2, 2, 2]))
        assert n
        assert_allclose(tri_mags[0], 5 + 0.9, atol=0.1)
        assert_allclose(tri_mags[-1], 19 + 0.9, atol=0.1)
        for i, tri_mag_mid in enumerate(tri_mags_mids):
            if tri_mag_mid < 10 + 0.9:
                expect_dens = n_b/100 / 0.1 / 1
            elif tri_mag_mid > 15 + 0.9:
                expect_dens = n_f/100 / 0.1 / 1
            else:
                d_u_f = np.sqrt(n_f/100) / 0.1 / 1
                d_u_b = np.sqrt(n_b/100) / 0.1 / 1
                w_f, w_b = 1 / d_u_f**2, 1 / d_u_b**2
                d_f, d_b = n_f/100 / 0.1 / 1, n_b/100 / 0.1 / 1
                expect_dens = (d_b * w_b + d_f * w_f) / (w_b + w_f)
            assert_allclose(expect_dens, dens[i], atol=3*uncert[i], rtol=0.01)
    if run_type == "faint":
        ol = '#Av at infinity = 1'
        nl = '#Av at infinity = 0.05\n'
        with open('trilegal_auf_simulation_faint.dat', encoding='utf-8') as file:
            f = file.readlines()
        idx = np.where([ol in line for line in f])[0][0]
        _replace_line('trilegal_auf_simulation_faint.dat', idx, nl)
        with pytest.raises(ValueError, match="tri_av_inf_faint cannot be smaller than 0.1 while"):
            dens, tri_mags, tri_mags_mids, _, uncert, n = make_tri_counts(
                '.', 'trilegal_auf_simulation', 'W1', 0.1, 10, 20,
                use_bright=False, use_faint=True, al_av=0.9, av_grid=np.array([2, 2, 2, 2]))
    if run_type == "bright":
        ol = '#Av at infinity = 1'
        nl = '#Av at infinity = 0.05\n'
        with open('trilegal_auf_simulation_bright.dat', encoding='utf-8') as file:
            f = file.readlines()
        idx = np.where([ol in line for line in f])[0][0]
        _replace_line('trilegal_auf_simulation_bright.dat', idx, nl)
        with pytest.raises(ValueError, match="tri_av_inf_bright cannot be smaller than 0.1 while"):
            dens, tri_mags, tri_mags_mids, _, uncert, n = make_tri_counts(
                '.', 'trilegal_auf_simulation', 'W1', 0.1, 5, 20,
                use_bright=True, use_faint=False, al_av=0.9, av_grid=np.array([2, 2, 2, 2]))


@pytest.mark.remote_data
def test_trilegal_download():
    tri_folder = '.'
    download_trilegal_simulation(tri_folder, 'gaiaDR2', 15, 6, 1, 'galactic', 32, 0.01, total_objs=10000)
    tri_name = 'trilegal_auf_simulation'
    with open(f'{tri_folder}/{tri_name}.dat', "r", encoding='utf-8') as f:
        line = f.readline()
    bits = line.split(' ')
    tri_area = float(bits[2])
    tri = np.genfromtxt(f'{tri_folder}/{tri_name}.dat', delimiter=None, names=True,
                        comments='#', skip_header=2)
    assert np.all(tri[:]['G'] <= 32)
    assert tri_area <= 10
