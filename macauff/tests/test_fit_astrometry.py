# Licensed under a 3-clause BSD style license - see LICENSE
'''
Tests for the "fit_astrometry" module.
'''


import numpy as np
from numpy.testing import assert_allclose
import os

from ..fit_astrometry import AstrometricCorrections


class TestAstroCorrection:
    def setup_class(self):
        self.rng = np.random.default_rng(seed=43578345)
        self.N = 5000
        choice = self.rng.choice(self.N, size=self.N, replace=False)
        self.true_ra = np.linspace(100, 110, self.N)[choice]
        self.true_dec = np.linspace(-3, 3, self.N)[choice]

        os.makedirs('store_data', exist_ok=True)

        os.makedirs('tri_folder', exist_ok=True)
        # Fake some TRILEGAL downloads with random data.
        text = ('#area = 4.0 sq deg\n#Av at infinity = 0\n' +
                'Gc logAge [M/H] m_ini   logL   logTe logg  m-M0   Av    ' +
                'm2/m1 mbol   J      H      Ks     IRAC_3.6 IRAC_4.5 IRAC_5.8 IRAC_8.0 MIPS_24 ' +
                'MIPS_70 MIPS_160 W1     W2     W3     W4       Mact\n')
        w1s = self.rng.uniform(14, 16, size=1000)
        for w1 in w1s:
            text = text + (
                '1   6.65 -0.39  0.02415 -2.701 3.397  4.057 14.00  8.354 0.00 25.523 25.839 ' +
                '24.409 23.524 22.583 22.387 22.292 22.015 21.144 19.380 20.878 '
                '{} 22.391 21.637 21.342  0.024\n '.format(w1))
        with open('tri_folder/trilegal_sim_105.0_0.0_bright.dat', "w") as f:
            f.write(text)
        with open('tri_folder/trilegal_sim_105.0_0.0_faint.dat', "w") as f:
            f.write(text)

    def fake_cata_cutout(self, l1, l2, b1, b2):
        astro_uncert = self.rng.uniform(0.001, 0.002, size=self.N)
        mag = self.rng.uniform(12, 12.1, size=self.N)
        mag_uncert = self.rng.uniform(0.01, 0.02, size=self.N)
        a = np.array([self.true_ra, self.true_dec, astro_uncert, mag, mag_uncert]).T
        np.save('store_data/a_cat{}{}{}{}.npy'.format(l1, l2, b1, b2), a)

    def fake_catb_cutout(self, l1, l2, b1, b2):
        mag = np.empty(self.N, float)
        # Fake some data to plot SNR(S) = S / sqrt(c S + b (aS)^2)
        mag[:50] = np.linspace(10, 18, 50)
        mag[50:100] = mag[:50] + self.rng.uniform(-0.0001, 0.0001, size=50)
        s = 10**(-1/2.5 * mag[:50])
        snr = s / np.sqrt(3.5e-16 * s + 8e-17 + (1.2e-2 * s)**2)
        mag_uncert = np.empty(self.N, float)
        mag_uncert[:50] = 2.5 * np.log10(1 + 1/snr)
        mag_uncert[50:100] = mag_uncert[:50] + self.rng.uniform(-0.001, 0.001, size=50)
        astro_uncert = np.empty(self.N, float)
        astro_uncert[:100] = 0.01
        # Divide the N-100 objects at the 0/21/44/70/100 interval, for a
        # 21/23/26/31 split.
        i_list = [100, int((self.N-100)*0.21 + 100),
                  int((self.N-100)*0.44 + 100), int((self.N-100)*0.7 + 100)]
        j_list = [i_list[1], i_list[2], i_list[3], self.N]
        for i, j, mag_mid, sig_mid in zip(i_list, j_list,
                                          [14.07, 14.17, 14.27, 14.37], [0.05, 0.075, 0.1, 0.12]):
            mag[i:j] = self.rng.uniform(mag_mid-0.05, mag_mid+0.05, size=j-i)
            snr_mag = mag_mid / np.sqrt(3.5e-16 * mag_mid + 8e-17 + (1.2e-2 * mag_mid)**2)
            dm_mag = 2.5 * np.log10(1 + 1/snr_mag)
            mag_uncert[i:j] = self.rng.uniform(dm_mag-0.005, dm_mag+0.005, size=j-i)
            astro_uncert[i:j] = self.rng.uniform(sig_mid, sig_mid+0.01, size=j-i)
        angle = self.rng.uniform(0, 2*np.pi, size=self.N)
        ra_angle, dec_angle = np.cos(angle), np.sin(angle)
        # Key is that objects are distributed over TWICE their quoted uncertainty!
        # Also remember that uncertainty needs to be in arcseconds but
        # offset in deg.
        dist = self.rng.rayleigh(scale=2*astro_uncert / 3600, size=self.N)
        rand_ra = self.true_ra + dist * ra_angle
        rand_dec = self.true_dec + dist * dec_angle
        b = np.array([rand_ra, rand_dec, astro_uncert, mag, mag_uncert]).T
        np.save('store_data/b_cat{}{}{}{}.npy'.format(l1, l2, b1, b2), b)

    def test_fit_astrometry(self):
        dd_params = np.load(os.path.join(os.path.dirname(__file__), 'data/dd_params.npy'))
        l_cut = np.load(os.path.join(os.path.dirname(__file__), 'data/l_cut.npy'))
        lmids, bmids = np.array([105], dtype=float), np.array([0], dtype=float)
        magarray = np.array([14.07, 14.17, 14.27, 14.37])
        magslice = np.array([0.05, 0.05, 0.05, 0.05])
        sigslice = np.array([0.1, 0.1, 0.1, 0.1])
        ac = AstrometricCorrections(
            psf_fwhm=6.1, numtrials=10000, nn_radius=30, dens_search_radius=900,
            save_folder='ac_save_folder', trifolder='tri_folder', triname='trilegal_sim',
            maglim_b=13, maglim_f=25, magnum=11, trifilterset='2mass_spitzer_wise',
            trifiltname='W1', gal_wav_micron=3.35, gal_ab_offset=2.699, gal_filtname='wise2010-W1',
            gal_alav=0.039, bright_mag=16, dm=0.1, dd_params=dd_params, l_cut=l_cut, lmids=lmids,
            bmids=bmids, lb_dimension=1, cutout_area=60, cutout_height=6, mag_array=magarray,
            mag_slice=magslice, sig_slice=sigslice, n_pool=1)

        a_cat_func = self.fake_cata_cutout
        b_cat_func = self.fake_catb_cutout
        a_cat_name = 'store_data/a_cat{}{}{}{}.npy'
        b_cat_name = 'store_data/b_cat{}{}{}{}.npy'
        ac(a_cat_func, b_cat_func, a_cat_name, b_cat_name, cat_recreate=True,
           snr_model_recreate=True, count_recreate=True, tri_download=False, dens_recreate=True,
           nn_recreate=True, auf_sim_recreate=True, auf_pdf_recreate=True,
           h_o_fit_recreate=True, fit_x2s_recreate=True, make_plots=True, make_summary_plot=True)

        assert os.path.isfile('ac_save_folder/pdf/auf_fits_105.0_0.0.pdf')
        assert os.path.isfile('ac_save_folder/pdf/counts_comparison.pdf')
        assert os.path.isfile('ac_save_folder/pdf/s_vs_snr.pdf')
        assert os.path.isfile('ac_save_folder/pdf/sig_fit_comparisons.pdf')
        assert os.path.isfile('ac_save_folder/pdf/sig_h_stats.pdf')

        marray = np.load('ac_save_folder/npy/m_sigs_array.npy')
        narray = np.load('ac_save_folder/npy/n_sigs_array.npy')
        assert_allclose([marray[0], narray[0]], [2, 0], rtol=0.1, atol=0.01)

        lmids = np.load('ac_save_folder/npy/lmids.npy')
        bmids = np.load('ac_save_folder/npy/bmids.npy')
        assert_allclose([lmids[0], bmids[0]], [105, 0], atol=0.001)

        abc_array = np.load('ac_save_folder/npy/snr_model.npy')
        assert_allclose(abc_array[0, 0], 1.2e-2, rtol=0.05, atol=0.001)
        assert_allclose(abc_array[0, 1], 8e-17, rtol=0.05, atol=5e-19)
