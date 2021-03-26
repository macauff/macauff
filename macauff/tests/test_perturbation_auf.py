# Licensed under a 3-clause BSD style license - see LICENSE
'''
Tests for the "perturbation_auf" module.
'''

import pytest
import os
import numpy as np
from numpy.testing import assert_allclose
from scipy.special import j0, j1

from ..matching import CrossMatch
from ..misc_functions_fortran import misc_functions_fortran as mff
from ..perturbation_auf import (make_perturb_aufs, get_circle_overlap_area,
                                download_trilegal_simulation)
from ..perturbation_auf_fortran import perturbation_auf_fortran as paf

from .test_matching import _replace_line


class TestCreatePerturbAUF:
    def setup_class(self):
        os.makedirs('gaia_auf_folder', exist_ok=True)
        os.makedirs('wise_auf_folder', exist_ok=True)
        self.cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        self.cm.a_auf_region_points = np.array([[0, 0], [50, 50]], dtype=float)
        self.cm.b_auf_region_points = np.array([[0, 0], [50, 50]], dtype=float)
        self.cm.mem_chunk_num = 4
        self.files_per_auf_sim = 7

    def test_no_perturb_outputs(self):
        # Randomly generate two catalogues (x3 files) between coordinates
        # 0, 0 and 50, 50.
        rng = np.random.default_rng()
        for path, Nf, size in zip([self.cm.a_cat_folder_path, self.cm.b_cat_folder_path], [3, 4],
                                  [25, 54]):
            cat = np.zeros((size, 3), float)
            rand_inds = rng.permutation(cat.shape[0])[:size // 2 - 1]
            cat[rand_inds, 0] = 50
            cat[rand_inds, 1] = 50
            cat += rng.uniform(-0.1, 0.1, cat.shape)
            np.save('{}/con_cat_astro.npy'.format(path), cat)

            cat = rng.uniform(10, 20, (size, Nf))
            np.save('{}/con_cat_photo.npy'.format(path), cat)

            cat = rng.choice(Nf, size=(size,))
            np.save('{}/magref.npy'.format(path), cat)

        self.cm.include_perturb_auf = False
        self.cm.run_auf = True
        self.cm.create_perturb_auf(self.files_per_auf_sim)
        lenr = len(self.cm.r)
        lenrho = len(self.cm.rho)
        for coord in ['0.0', '50.0']:
            for filt in ['W1', 'W2', 'W3', 'W4']:
                path = '{}/{}/{}/{}'.format(self.cm.b_auf_folder_path, coord, coord, filt)
                for filename, shape in zip(['frac', 'flux', 'offset', 'cumulative', 'fourier',
                                            'N', 'mag'],
                                           [(1, 1), (1,), (lenr-1, 1), (lenr-1, 1), (lenrho-1, 1),
                                            (1, 1), (1, 1)]):
                    assert os.path.isfile('{}/{}.npy'.format(path, filename))
                    file = np.load('{}/{}.npy'.format(path, filename))
                    assert np.all(file.shape == shape)
                assert np.all(np.load('{}/frac.npy'.format(path)) == 0)
                assert np.all(np.load('{}/cumulative.npy'.format(path)) == 1)
                assert np.all(np.load('{}/fourier.npy'.format(path)) == 1)
                assert np.all(np.load('{}/mag.npy'.format(path)) == 1)
                file = np.load('{}/offset.npy'.format(path))
                assert np.all(file[1:] == 0)
                assert file[0] == 1/(2 * np.pi * (self.cm.r[0] + self.cm.dr[0]/2) * self.cm.dr[0])

        file = np.load('{}/modelrefinds.npy'.format(self.cm.a_auf_folder_path))
        assert np.all(file[0, :] == 0)
        assert np.all(file[1, :] == np.load('{}/magref.npy'.format(self.cm.a_cat_folder_path)))

        # Select AUF pointing index based on a 0 vs 50 cut in longitude.
        cat = np.load('{}/con_cat_astro.npy'.format(self.cm.a_cat_folder_path))
        inds = np.ones(file.shape[1], int)
        inds[np.where(cat[:, 0] < 1)[0]] = 0
        assert np.all(file[2, :] == inds)

    def test_run_auf_file_number(self):
        # Reset any saved files from the above tests
        os.system("rm -rf {}/*".format(self.cm.a_auf_folder_path))
        os.system("rm -rf {}/*".format(self.cm.b_auf_folder_path))
        self.cm.run_auf = False
        with pytest.warns(UserWarning, match='Incorrect number of files in catalogue "a"'):
            self.cm.create_perturb_auf(self.files_per_auf_sim)

        # Now create fake files to simulate catalogue "a" having the right files.
        # For 2 AUF pointings this comes to 5 + 2*N_filt*files_per_auf_sim files.
        os.system("rm -rf {}/*".format(self.cm.a_auf_folder_path))
        for i in range(5 + 2 * 3 * self.files_per_auf_sim):
            np.save('{}/random_file_{}.npy'.format(self.cm.a_auf_folder_path, i), np.zeros(1))

        # This should still return the same warning, just for catalogue "b" now.
        with pytest.warns(UserWarning) as record:
            self.cm.create_perturb_auf(self.files_per_auf_sim)
        assert len(record) == 1
        assert 'Incorrect number of files in catalogue "b"' in record[0].message.args[0]

    @pytest.mark.filterwarnings("ignore:Incorrect number of files in")
    def test_load_auf_print(self, capsys):
        # Reset any saved files from the above tests
        os.system("rm -rf {}/*".format(self.cm.a_auf_folder_path))
        os.system("rm -rf {}/*".format(self.cm.b_auf_folder_path))

        # Generate new dummy data for catalogue "b"'s AUF folder.
        for i in range(5 + 2 * 4 * self.files_per_auf_sim):
            np.save('{}/random_file_{}.npy'.format(self.cm.b_auf_folder_path, i), np.zeros(1))
        capsys.readouterr()
        # This test will create catalogue "a" files because of the wrong
        # number of files (zero) in the folder.
        self.cm.create_perturb_auf(self.files_per_auf_sim)
        output = capsys.readouterr().out
        assert 'Loading empirical perturbation AUFs for catalogue "a"' not in output
        assert 'Loading empirical perturbation AUFs for catalogue "b"' in output

        os.system("rm -rf {}/*".format(self.cm.a_auf_folder_path))
        os.system("rm -rf {}/*".format(self.cm.b_auf_folder_path))
        # Generate new dummy data for each catalogue's AUF folder.
        for path, fn in zip([self.cm.a_auf_folder_path, self.cm.b_auf_folder_path], [3, 4]):
            for i in range(5 + 2 * fn * self.files_per_auf_sim):
                np.save('{}/random_file_{}.npy'.format(path, i), np.zeros(1))
        capsys.readouterr()
        self.cm.create_perturb_auf(self.files_per_auf_sim)
        output = capsys.readouterr().out
        assert 'Loading empirical perturbation AUFs for catalogue "a"' in output
        assert 'Loading empirical perturbation AUFs for catalogue "b"' in output


def test_perturb_aufs():
    # Poisson distribution with mean 0.08 gives 92.3% zero, 7.4% one, and 0.3% two draws.
    mean = 0.08
    prob_0_draw = mean**0 * np.exp(-mean) / np.math.factorial(0)
    prob_1_draw = mean**1 * np.exp(-mean) / np.math.factorial(1)
    prob_2_draw = mean**2 * np.exp(-mean) / np.math.factorial(2)

    N = np.array([1.0])
    m = np.array([0.0])
    R = 1.185*6.1
    r = np.linspace(0, R, 1500)
    dr = np.diff(r)
    rho = np.linspace(0, 100, 10000)
    drho = np.diff(rho)
    j0s = mff.calc_j0(r[:-1]+dr/2, rho[:-1]+drho/2)

    model_count = 1.0
    num_trials = 100000
    mag_cut = np.array([5.0])
    model_mags = np.array([0])
    model_mags_interval = np.array([1.0e-8])

    log10y = np.log10(mean / model_mags_interval / np.pi / (R / 3600)**2)

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
    num = 100
    for _ in range(num):
        seed = rng.choice(1000000, size=seed_size)
        offsets, fracs, fluxs = paf.scatter_perturbers(np.array([mean]), m, R, 5, mag_cut,
                                                       model_mags_interval, num_trials, seed)
        hist, _ = np.histogram(offsets, bins=r)
        assert_allclose(fracs[0], 1-prob_0_draw, rtol=0.05)
        assert_allclose(np.mean(fluxs), prob_0_draw*0+prob_1_draw*1+prob_2_draw*2, rtol=0.05)

        Frac, Flux, fourieroffset, offsets, cumulative = paf.perturb_aufs(
            N, m, r[:-1]+dr/2, dr, r, j0s,
            model_mags+model_mags_interval/2, model_mags_interval, log10y, model_count,
            np.array([1]), mag_cut, R, num_trials, seed)

        assert_allclose(Frac, 1-prob_0_draw, rtol=0.05)
        assert_allclose(Flux, prob_0_draw*0+prob_1_draw*1+prob_2_draw*2, rtol=0.05)

        track_sp_hist += hist / np.sum(hist) / (np.pi * (r[1:]**2 - r[:-1]**2)) / num

        track_pa_hist += offsets[:, 0] / num
        track_pa_hist_err_sq += (np.sqrt(
            offsets[:, 0] * np.sum(hist) * (np.pi * (r[1:]**2 - r[:-1]**2))) / (
            np.sum(hist) * (np.pi * (r[1:]**2 - r[:-1]**2))) / num)**2

        track_pa_fourier += fourieroffset[:, 0] / num

    # Here we assume that the direct and wrapper calls had the same seed for
    # each call, and identical results, and hence should basically agree perfectly.
    assert_allclose(track_pa_hist, track_sp_hist, atol=1e-6)

    offsets_fake = np.zeros_like(r[:-1])
    offsets_fake[0] += prob_0_draw / (np.pi * (r[1]**2 - r[0]**2))
    # Given that we force equal-brightness "binaries", we expect a maximum
    # perturbation of half the total radius.
    q = np.where(r[1:] > R/2)[0]
    offsets_fake[:q[0]] += prob_1_draw / np.pi / (R/2)**2
    final_bin_frac = (R/2 - r[q[0]]) / (r[q[0]+1] - r[q[0]])
    offsets_fake[q[0]] += final_bin_frac * prob_1_draw / np.pi / (R/2)**2

    track_pa_hist_err_sq[track_pa_hist_err_sq == 0] = 1e-8
    for i in range(len(r)-1):
        assert_allclose(track_pa_hist[i], offsets_fake[i], rtol=0.05,
                        atol=1e-5 + 4 * np.sqrt(track_pa_hist_err_sq[i]))

    # Take the average fourier representation of the offsets and compare. The
    # first expression is the integral of the constant 1/(pi (R/2)^2) probability
    # for the randomly placed, single-draw perturbations.
    fake_fourier = (1-prob_0_draw) * 2 / (np.pi * R * (rho[:-1]+drho/2)) * j1(
        np.pi * R * (rho[:-1]+drho/2))
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

    counts_f = paf.histogram(x, bins)
    counts_p, _ = np.histogram(x, bins=bins)
    assert np.all(counts_f == counts_p)


def test_circle_area():
    rng = np.random.default_rng(123897123)
    R = 0.1

    x_edges = [0, 1]
    y_edges = [0, 1]

    # If circle is inside rectangle, get full area:
    done = 0
    while done < 10:
        [x, y] = rng.uniform(0, 1, size=2)
        if (x - R >= x_edges[0] and x + R <= x_edges[1] and
                y - R >= y_edges[0] and y + R <= y_edges[1]):
            calc_area = get_circle_overlap_area(R, x_edges, y_edges, [x, y])
            assert_allclose(calc_area, np.pi * R**2)
            done += 1

    # Now, if the circle is exactly on the corners of the rectangle
    # we should have a quarter the area:
    for x, y in zip([0, 0, 1, 1], [0, 1, 0, 1]):
        calc_area = get_circle_overlap_area(R, x_edges, y_edges, [x, y])
        assert_allclose(calc_area, np.pi * R**2 / 4)

    # In the middle of an edge we should have half the circle area:
    for x, y in zip([0, 0.5, 1, 0.5], [0.5, 0, 0.5, 1]):
        calc_area = get_circle_overlap_area(R, x_edges, y_edges, [x, y])
        assert_allclose(calc_area, np.pi * R**2 / 2)

    # Verify a few randomly placed circles too:
    done = 0
    xp = np.linspace(*x_edges, 100)
    yp = np.linspace(*y_edges, 100)
    dx, dy = xp[1] - xp[0], yp[1] - yp[0]
    while done < 20:
        [x, y] = rng.uniform(0, 1, size=2)
        if np.any([x - R < x_edges[0], x + R > x_edges[1],
                   y - R < y_edges[0], y + R > y_edges[1]]):
            calc_area = get_circle_overlap_area(R, x_edges, y_edges, [x, y])
            manual_area = 0
            for i in range(len(xp)):
                for j in range(len(yp)):
                    if np.sqrt((xp[i] - x)**2 + (yp[j] - y)**2) <= R:
                        manual_area += dx*dy
            assert_allclose(calc_area, manual_area, rtol=0.05)
        done += 1


class TestMakePerturbAUFs():
    def setup_class(self):
        self.auf_folder = 'auf_folder'
        self.cat_folder = 'cat_folder'
        os.system('rm -r {}'.format(self.auf_folder))
        os.system('rm -r {}'.format(self.cat_folder))
        os.makedirs(self.auf_folder)
        os.makedirs(self.cat_folder)

        self.filters = np.array(['W1'])
        self.tri_filt_names = np.copy(self.filters)
        self.auf_points = np.array([[0.0, 0.0]])
        self.ax_lims = np.array([0, 1, 0, 1])

        self.psf_fwhms = np.array([6.1])
        self.r = np.linspace(0, 1.185 * self.psf_fwhms[0], 2500)
        self.dr = np.diff(self.r)
        self.rho = np.linspace(0, 100, 10000)
        self.drho = np.diff(self.rho)
        self.which_cat = 'b'
        self.include_perturb_auf = True
        self.num_trials = 100000

        self.mem_chunk_num = 1
        self.delta_mag_cuts = np.array([2.5, 5])

        self.args = [self.auf_folder, self.cat_folder, self.filters, self.auf_points,
                     self.r, self.dr, self.rho, self.drho, self.which_cat,
                     self.include_perturb_auf, self.mem_chunk_num]

        self.files_per_auf_sim = 7

    def test_raise_value_errors(self):
        with pytest.raises(ValueError, match='tri_set_name must be given if include_perturb_auf ' +
                           'and tri_download_flag are both True'):
            make_perturb_aufs(*self.args, tri_download_flag=True)
        with pytest.raises(ValueError, match='tri_filt_num must be given if include_perturb_auf ' +
                           'and tri_download_flag are both True'):
            make_perturb_aufs(*self.args, tri_download_flag=True, tri_set_name='WISE')
        with pytest.raises(ValueError, match='auf_region_frame must be given if ' +
                           'include_perturb_auf and tri_download_flag are both True'):
            make_perturb_aufs(*self.args, tri_download_flag=True, tri_set_name='WISE',
                              tri_filt_num=1)

        with pytest.raises(ValueError, match='tri_filt_names must be given if ' +
                           'include_perturb_auf is True'):
            make_perturb_aufs(*self.args)
        with pytest.raises(ValueError, match='delta_mag_cuts must be given if ' +
                           'include_perturb_auf is True'):
            make_perturb_aufs(*self.args, tri_filt_names=1)
        with pytest.raises(ValueError, match='psf_fwhms must be given if ' +
                           'include_perturb_auf is True'):
            make_perturb_aufs(*self.args, tri_filt_names=1, delta_mag_cuts=1)
        with pytest.raises(ValueError, match='num_trials must be given if ' +
                           'include_perturb_auf is True'):
            make_perturb_aufs(*self.args, tri_filt_names=1, delta_mag_cuts=1, psf_fwhms=1)
        with pytest.raises(ValueError, match='j0s must be given if ' +
                           'include_perturb_auf is True'):
            make_perturb_aufs(*self.args, tri_filt_names=1, delta_mag_cuts=1, psf_fwhms=1,
                              num_trials=1)
        with pytest.raises(ValueError, match='density_mags must be given if ' +
                           'include_perturb_auf is True'):
            make_perturb_aufs(*self.args, tri_filt_names=1, delta_mag_cuts=1, psf_fwhms=1,
                              num_trials=1, j0s=1)
        with pytest.raises(ValueError, match='dm_max must be given if ' +
                           'include_perturb_auf is True'):
            make_perturb_aufs(*self.args, tri_filt_names=1, delta_mag_cuts=1, psf_fwhms=1,
                              num_trials=1, j0s=1, density_mags=1)
        with pytest.raises(ValueError, match='d_mag must be given if ' +
                           'include_perturb_auf is True'):
            make_perturb_aufs(*self.args, tri_filt_names=1, delta_mag_cuts=1, psf_fwhms=1,
                              num_trials=1, j0s=1, density_mags=1, dm_max=1)
        with pytest.raises(ValueError, match='compute_local_density must be given if ' +
                           'include_perturb_auf is True'):
            make_perturb_aufs(*self.args, tri_filt_names=1, delta_mag_cuts=1, psf_fwhms=1,
                              num_trials=1, j0s=1, density_mags=1, dm_max=1, d_mag=1)
        with pytest.raises(ValueError, match='density_radius must be given if ' +
                           'include_perturb_auf and compute_local_density are both True'):
            make_perturb_aufs(*self.args, tri_filt_names=1, delta_mag_cuts=1,
                              compute_local_density=True, psf_fwhms=1, num_trials=1, j0s=1,
                              density_mags=1, dm_max=1, d_mag=1)

    def test_without_compute_local_density(self):
        # Number of sources per PSF circle, on average, solved backwards to ensure
        # that local density ends up exactly in the middle of a count_array bin.
        # This should be approximately 0.076 sources per PSF circle.
        psf_mean = np.exp(8.7) * np.pi * (1.185 * self.psf_fwhms[0] / 3600)**2
        # Local density is the controllable variable to ensure that we get
        # the expected sources per PSF circle, with most variables cancelling
        # mean divided by circle area sets the density needed.
        local_dens = psf_mean / (np.pi * (1.185 * self.psf_fwhms[0] / 3600)**2)
        np.save('{}/local_N.npy'.format(self.auf_folder), np.array([[local_dens]]))

        np.save('{}/con_cat_astro.npy'.format(self.cat_folder), np.array([[0.3, 0.3, 0.1]]))
        np.save('{}/con_cat_photo.npy'.format(self.cat_folder), np.array([[14.99]]))
        np.save('{}/magref.npy'.format(self.cat_folder), np.array([0]))

        num_trials = 100000
        j0s = mff.calc_j0(self.rho[:-1]+self.drho/2, self.r[:-1]+self.dr/2)
        cutoff_mags = np.array([20])
        dm_max = np.array([10])
        d_mag = 0.1

        # Fake up a TRILEGAL simulation data file. Need to paste the same source
        # four times to pass a check for more than three sources in a histogram.
        text = ('#area = 4.0 sq deg\n#Gc logAge [M/H] m_ini   logL   logTe logg  m-M0   Av    ' +
                'm2/m1 mbol   J      H      Ks     IRAC_3.6 IRAC_4.5 IRAC_5.8 IRAC_8.0 MIPS_24 ' +
                'MIPS_70 MIPS_160 W1     W2     W3     W4       Mact\n 1   6.65 -0.39  0.02415 ' +
                '-2.701 3.397  4.057 14.00  8.354 0.00 25.523 25.839 24.409 23.524 22.583 ' +
                '22.387 22.292 22.015 21.144 19.380 20.878 15.001 22.391 21.637 21.342  0.024\n ' +
                '1   6.65 -0.39  0.02415 -2.701 3.397  4.057 14.00  8.354 0.00 25.523 25.839 ' +
                '24.409 23.524 22.583 22.387 22.292 22.015 21.144 19.380 20.878 15.002 22.391 ' +
                '21.637 21.342  0.024\n 1   6.65 -0.39  0.02415 -2.701 3.397  4.057 14.00  ' +
                '8.354 0.00 25.523 25.839 24.409 23.524 22.583 22.387 22.292 22.015 21.144 ' +
                '19.380 20.878 15.003 22.391 21.637 21.342  0.024\n 1   6.65 -0.39  0.02415 ' +
                '-2.701 3.397  4.057 14.00  8.354 0.00 25.523 25.839 24.409 23.524 22.583 ' +
                '22.387 22.292 22.015 21.144 19.380 20.878 15.004 22.391 21.637 21.342  0.024')
        os.makedirs('{}/{}/{}'.format(
            self.auf_folder, self.auf_points[0][0], self.auf_points[0][1]), exist_ok=True)
        with open('{}/{}/{}/trilegal_auf_simulation.dat'.format(
                  self.auf_folder, self.auf_points[0][0], self.auf_points[0][1]), "w") as f:
            f.write(text)

        prob_0_draw = psf_mean**0 * np.exp(-psf_mean) / np.math.factorial(0)
        prob_1_draw = psf_mean**1 * np.exp(-psf_mean) / np.math.factorial(1)
        prob_2_draw = psf_mean**2 * np.exp(-psf_mean) / np.math.factorial(2)

        ax1, ax2 = self.auf_points[0]

        keep_frac = np.zeros((len(self.delta_mag_cuts), 1), float)
        keep_flux = np.zeros((1,), float)
        track_fourier = np.zeros(len(self.rho)-1, float)

        # Catalogue bins for the source:
        a_photo = np.load('{}/con_cat_photo.npy'.format(self.cat_folder))
        dmag = 0.25
        mag_min = dmag * np.floor(np.amin(a_photo)/dmag)
        mag_max = dmag * np.ceil(np.amax(a_photo)/dmag)
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

        N = 25
        for i in range(N):
            make_perturb_aufs(*self.args, psf_fwhms=self.psf_fwhms, num_trials=num_trials, j0s=j0s,
                              density_mags=cutoff_mags, dm_max=dm_max, d_mag=d_mag,
                              delta_mag_cuts=self.delta_mag_cuts, compute_local_density=False,
                              tri_filt_names=self.tri_filt_names)

            if i == 0:
                for name, size in zip(
                        ['frac', 'flux', 'offset', 'cumulative', 'fourier', 'N', 'mag'],
                        [(len(self.delta_mag_cuts), 1), (1,), (len(self.r)-1, 1),
                         (len(self.r)-1, 1), (len(self.rho)-1, 1), (1,), (1,)]):
                    var = np.load('{}/{}/{}/{}/{}.npy'.format(
                                  self.auf_folder, ax1, ax2, self.filters[0], name))
                    assert np.all(var.shape == size)

            fracs = np.load('{}/{}/{}/{}/frac.npy'.format(
                self.auf_folder, ax1, ax2, self.filters[0]))
            fluxs = np.load('{}/{}/{}/{}/flux.npy'.format(
                self.auf_folder, ax1, ax2, self.filters[0]))
            fourier = np.load('{}/{}/{}/{}/fourier.npy'.format(
                self.auf_folder, ax1, ax2, self.filters[0]))

            keep_frac += fracs / N
            keep_flux += fluxs / N
            track_fourier += fourier[:, 0] / N

        # Have more relaxed conditions on assertion than in test_perturb_aufs
        # above, as we can't arbitrarily force the magnitude bin widths to be
        # very small, and hence have a blur on relative fluxes allowed.
        assert_allclose(keep_frac[0, 0], 1-prob_0_draw, rtol=0.1)
        assert_allclose(keep_flux[0], (prob_0_draw*0 + prob_1_draw*rel_flux +
                        prob_2_draw*2*rel_flux), rtol=0.1)

        R = 1.185 * self.psf_fwhms[0]
        small_R = R * rel_flux / (1 + rel_flux)

        fake_fourier = (1-prob_0_draw) / (np.pi * small_R * (self.rho[:-1]+self.drho/2)) * j1(
            2 * np.pi * small_R * (self.rho[:-1]+self.drho/2))
        fake_fourier += prob_0_draw * j0(2 * np.pi * (self.r[0]+self.dr[0]/2) *
                                         (self.rho[:-1]+self.drho/2))

        assert_allclose(fake_fourier, track_fourier, rtol=0.05)

    def test_with_compute_local_density(self):
        # Number of sources per PSF circle, on average, solved backwards to ensure
        # that local density ends up exactly in the middle of a count_array bin.
        # This should be approximately 0.076 sources per PSF circle.
        psf_mean = np.exp(8.7) * np.pi * (1.185 * self.psf_fwhms[0] / 3600)**2
        # This time we want to calculate the local density on the fly, but still
        # get the same value we did in the without compute local density test. We
        # therefore have to choose our "density radius" to set the appropriate
        # local density for our single source.
        density_mags = np.array([20])
        density_radius = np.sqrt(1 / np.pi / np.exp(8.7))

        new_auf_points = np.vstack((self.auf_points, np.array([[10, 10]])))

        # Have to fudge extra sources to keep our 15th mag source in the local
        # density cutout.
        np.save('{}/con_cat_astro.npy'.format(self.cat_folder),
                np.array([[0.3, 0.3, 0.1], [0.1, 0.1, 0.1], [0.9, 0.9, 0.1]]))
        np.save('{}/con_cat_photo.npy'.format(self.cat_folder), np.array([[14.99], [10], [10]]))
        np.save('{}/magref.npy'.format(self.cat_folder), np.array([0, 0, 0]))

        # Fake up a TRILEGAL simulation data file. Need to paste the same source
        # four times to pass a check for more than three sources in a histogram.
        text = ('#area = 4.0 sq deg\n#Gc logAge [M/H] m_ini   logL   logTe logg  m-M0   Av    ' +
                'm2/m1 mbol   J      H      Ks     IRAC_3.6 IRAC_4.5 IRAC_5.8 IRAC_8.0 MIPS_24 ' +
                'MIPS_70 MIPS_160 W1     W2     W3     W4       Mact\n 1   6.65 -0.39  0.02415 ' +
                '-2.701 3.397  4.057 14.00  8.354 0.00 25.523 25.839 24.409 23.524 22.583 ' +
                '22.387 22.292 22.015 21.144 19.380 20.878 15.001 22.391 21.637 21.342  0.024\n ' +
                '1   6.65 -0.39  0.02415 -2.701 3.397  4.057 14.00  8.354 0.00 25.523 25.839 ' +
                '24.409 23.524 22.583 22.387 22.292 22.015 21.144 19.380 20.878 15.002 22.391 ' +
                '21.637 21.342  0.024\n 1   6.65 -0.39  0.02415 -2.701 3.397  4.057 14.00  ' +
                '8.354 0.00 25.523 25.839 24.409 23.524 22.583 22.387 22.292 22.015 21.144 ' +
                '19.380 20.878 15.003 22.391 21.637 21.342  0.024\n 1   6.65 -0.39  0.02415 ' +
                '-2.701 3.397  4.057 14.00  8.354 0.00 25.523 25.839 24.409 23.524 22.583 ' +
                '22.387 22.292 22.015 21.144 19.380 20.878 15.004 22.391 21.637 21.342  0.024')
        for i in range(len(new_auf_points)):
            os.makedirs('{}/{}/{}'.format(
                self.auf_folder, new_auf_points[i][0], new_auf_points[i][1]), exist_ok=True)
            with open('{}/{}/{}/trilegal_auf_simulation.dat'.format(
                      self.auf_folder, new_auf_points[i][0], new_auf_points[i][1]), "w") as f:
                f.write(text)

        prob_0_draw = psf_mean**0 * np.exp(-psf_mean) / np.math.factorial(0)
        prob_1_draw = psf_mean**1 * np.exp(-psf_mean) / np.math.factorial(1)
        prob_2_draw = psf_mean**2 * np.exp(-psf_mean) / np.math.factorial(2)

        ax1, ax2 = self.auf_points[0]

        d_mag = 0.1
        # Catalogue bins for the source, only keeping the single
        # source currently under consideration, and ignoring the two extra objects
        # used to force the local density to be calculated properly.
        a_photo = np.load('{}/con_cat_photo.npy'.format(self.cat_folder))[0, :]
        dmag = 0.25
        mag_min = dmag * np.floor(np.amin(a_photo)/dmag)
        mag_max = dmag * np.ceil(np.amax(a_photo)/dmag)
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
        f = open(os.path.join(os.path.dirname(__file__),
                              'data/crossmatch_params.txt')).readlines()
        idx = np.where([ol in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                      idx, nl, out_file=os.path.join(os.path.dirname(__file__),
                      'data/crossmatch_params_.txt'))

        ol, nl = 'filt_names = G_BP G G_RP', 'filt_names = G\n'
        f = open(os.path.join(os.path.dirname(__file__),
                              'data/cat_a_params.txt')).readlines()
        idx = np.where([ol in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                      idx, nl, out_file=os.path.join(os.path.dirname(__file__),
                      'data/cat_a_params_.txt'))
        for ol, nl in zip(['psf_fwhms = 0.12 0.12 0.12', 'cat_folder_path = gaia_folder',
                           'auf_folder_path = gaia_auf_folder', 'tri_filt_names = G_BP G G_RP',
                           'dens_mags = 20 20 20'],
                          ['psf_fwhms = 0.12\n', 'cat_folder_path = cat_folder\n',
                           'auf_folder_path = auf_folder\n', 'tri_filt_names = W1\n',
                           'dens_mags = 20\n']):
            f = open(os.path.join(os.path.dirname(__file__),
                                  'data/cat_a_params.txt')).readlines()
            idx = np.where([ol in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_a_params_.txt'),
                          idx, nl)

        ol, nl = 'filt_names = W1 W2 W3 W4', 'filt_names = W1\n'
        f = open(os.path.join(os.path.dirname(__file__),
                              'data/cat_b_params.txt')).readlines()
        idx = np.where([ol in line for line in f])[0][0]
        _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'),
                      idx, nl, out_file=os.path.join(os.path.dirname(__file__),
                      'data/cat_b_params_.txt'))
        for ol, nl in zip(['psf_fwhms = 6.08 6.84 7.36 11.99', 'cat_folder_path = wise_folder',
                           'auf_folder_path = wise_auf_folder', 'tri_filt_names = W1 W2 W3 W4',
                           'dens_mags = 20 20 20 20'],
                          ['psf_fwhms = 6.08\n', 'cat_folder_path = cat_folder\n',
                           'auf_folder_path = auf_folder\n', 'tri_filt_names = W1\n',
                           'dens_mags = 20\n']):
            f = open(os.path.join(os.path.dirname(__file__),
                                  'data/cat_b_params.txt')).readlines()
            idx = np.where([ol in line for line in f])[0][0]
            _replace_line(os.path.join(os.path.dirname(__file__), 'data/cat_b_params_.txt'),
                          idx, nl)

        cm = CrossMatch(os.path.join(os.path.dirname(__file__),
                                     'data/crossmatch_params_.txt'),
                        os.path.join(os.path.dirname(__file__), 'data/cat_a_params_.txt'),
                        os.path.join(os.path.dirname(__file__), 'data/cat_b_params_.txt'))

        cm.a_auf_region_points = new_auf_points
        cm.b_auf_region_points = new_auf_points
        cm.cross_match_extent = self.ax_lims
        cm.a_dens_mags = density_mags
        cm.b_dens_mags = density_mags
        cm.a_dens_dist = density_radius
        cm.b_dens_dist = density_radius
        cm.compute_local_density = True
        cm.run_auf = True
        cm.run_group = True
        cm.run_cf = True
        cm.run_source = True
        cm.num_trials = self.num_trials

        cm.create_perturb_auf(self.files_per_auf_sim)

        fracs = np.load('{}/{}/{}/{}/frac.npy'.format(
            self.auf_folder, ax1, ax2, self.filters[0]))
        fluxs = np.load('{}/{}/{}/{}/flux.npy'.format(
            self.auf_folder, ax1, ax2, self.filters[0]))
        fourier = np.load('{}/{}/{}/{}/fourier.npy'.format(
            self.auf_folder, ax1, ax2, self.filters[0]))

        assert_allclose(fracs[0, 0], 1-prob_0_draw, rtol=0.1)
        assert_allclose(fluxs[0], (prob_0_draw*0 + prob_1_draw*rel_flux +
                        prob_2_draw*2*rel_flux), rtol=0.1)

        R = 1.185 * self.psf_fwhms[0]
        small_R = R * rel_flux / (1 + rel_flux)

        fake_fourier = (1-prob_0_draw) / (np.pi * small_R * (cm.rho[:-1]+cm.drho/2)) * j1(
            2 * np.pi * small_R * (cm.rho[:-1]+cm.drho/2))
        fake_fourier += prob_0_draw * j0(2 * np.pi * (cm.r[0]+cm.dr[0]/2) *
                                         (cm.rho[:-1]+cm.drho/2))

        assert_allclose(fake_fourier, fourier[:, 0], rtol=0.05)


@pytest.mark.remote_data
def test_trilegal_download():
    tri_folder = '.'
    download_trilegal_simulation(tri_folder, 'gaiaDR2', 180, 20, 1, 'galactic', total_objs=10000)
    tri_name = 'trilegal_auf_simulation'
    f = open('{}/{}.dat'.format(tri_folder, tri_name), "r")
    line = f.readline()
    f.close()
    bits = line.split(' ')
    tri_area = float(bits[2])
    tri = np.genfromtxt('{}/{}.dat'.format(tri_folder, tri_name), delimiter=None,
                        names=True, comments='#', skip_header=1)
    assert np.all(tri[:]['G'] <= 32)
    assert tri_area <= 10
