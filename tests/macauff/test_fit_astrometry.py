# Licensed under a 3-clause BSD style license - see LICENSE
'''
Tests for the "fit_astrometry" module.
'''


import os

import numpy as np
import pytest
from numpy.testing import assert_allclose

from macauff.fit_astrometry import AstrometricCorrections, SNRMagnitudeRelationship


class TestAstroCorrection:
    def setup_method(self):
        self.rng = np.random.default_rng(seed=43578345)
        self.n = 25000
        choice = self.rng.choice(self.n, size=self.n, replace=True)
        self.true_ra = np.linspace(100, 110, self.n)[choice]
        choice = self.rng.choice(self.n, size=self.n, replace=True)
        self.true_dec = np.linspace(-3, 3, self.n)[choice]

        os.makedirs('store_data', exist_ok=True)

        os.makedirs('tri_folder', exist_ok=True)
        # Fake some TRILEGAL downloads with random data.
        text = ('#area = 4.0 sq deg\n#Av at infinity = 1\n' +
                'Gc logAge [M/H] m_ini   logL   logTe logg  m-M0   Av    ' +
                'm2/m1 mbol   J      H      Ks     IRAC_3.6 IRAC_4.5 IRAC_5.8 IRAC_8.0 MIPS_24 ' +
                'MIPS_70 MIPS_160 W1     W2     W3     W4       Mact\n')
        w1s = self.rng.uniform(13.5, 15.5, size=1000)
        for w1 in w1s:
            text = text + (
                '1   6.65 -0.39  0.02415 -2.701 3.397  4.057 14.00  8.354 0.00 25.523 25.839 ' +
                '24.409 23.524 22.583 22.387 22.292 22.015 21.144 19.380 20.878 '
                f'{w1} 22.391 21.637 21.342  0.024\n ')
        with open('tri_folder/trilegal_sim_105.0_0.0_bright.dat', "w", encoding='utf-8') as f:
            f.write(text)
        with open('tri_folder/trilegal_sim_105.0_0.0_faint.dat', "w", encoding='utf-8') as f:
            f.write(text)

    def fake_cata_cutout(self, lmin, lmax, bmin, bmax, *cat_args):  # pylint: disable=unused-argument
        astro_uncert = self.rng.uniform(0.001, 0.002, size=self.n)
        mag = self.rng.uniform(12, 12.1, size=self.n)
        mag_uncert = self.rng.uniform(0.01, 0.02, size=self.n)
        a = np.array([self.true_ra, self.true_dec, astro_uncert, mag, mag_uncert]).T
        if self.npy_or_csv == 'npy':
            np.save(self.a_cat_name.format(*cat_args), a)
        else:
            np.savetxt(self.a_cat_name.format(*cat_args), a, delimiter=',')

    def fake_catb_cutout(self, lmin, lmax, bmin, bmax, *cat_args):  # pylint: disable=unused-argument
        mag = np.empty(self.n, float)
        # Fake some data to plot SNR(S) = S / sqrt(c S + b (aS)^2)
        mag[:50] = np.linspace(10, 18, 50)
        mag[50:100] = mag[:50] + self.rng.uniform(-0.0001, 0.0001, size=50)
        s = 10**(-1/2.5 * mag[:50])
        snr = s / np.sqrt(3.5e-16 * s + 8e-17 + (1.2e-2 * s)**2)
        mag_uncert = np.empty(self.n, float)
        mag_uncert[:50] = 2.5 * np.log10(1 + 1/snr)
        mag_uncert[50:100] = mag_uncert[:50] + self.rng.uniform(-0.001, 0.001, size=50)
        astro_uncert = np.empty(self.n, float)
        astro_uncert[:100] = 0.01
        # Divide the N-100 objects at the 0/16/33/52/75/100 interval, for a
        # 16/17/19/23/25 split.
        i_list = [100, int((self.n-100)*0.16 + 100),
                  int((self.n-100)*0.33 + 100), int((self.n-100)*0.52 + 100),
                  int((self.n-100)*0.75 + 100)]
        j_list = [i_list[1], i_list[2], i_list[3], i_list[4], self.n]
        for i, j, mag_mid, sig_mid in zip(i_list, j_list,
                                          [14.07, 14.17, 14.27, 14.37, 14.47],
                                          [0.01, 0.02, 0.06, 0.12, 0.4]):
            mag[i:j] = self.rng.uniform(mag_mid-0.05, mag_mid+0.05, size=j-i)
            snr_mag = mag_mid / np.sqrt(3.5e-16 * mag_mid + 8e-17 + (1.2e-2 * mag_mid)**2)
            dm_mag = 2.5 * np.log10(1 + 1/snr_mag)
            mag_uncert[i:j] = self.rng.uniform(dm_mag-0.005, dm_mag+0.005, size=j-i)
            astro_uncert[i:j] = self.rng.uniform(sig_mid, sig_mid+0.01, size=j-i)
        angle = self.rng.uniform(0, 2*np.pi, size=self.n)
        ra_angle, dec_angle = np.cos(angle), np.sin(angle)
        # Key is that objects are distributed over TWICE their quoted uncertainty!
        # Also remember that uncertainty needs to be in arcseconds but
        # offset in deg.
        dist = self.rng.rayleigh(scale=2*astro_uncert / 3600, size=self.n)
        rand_ra = self.true_ra + dist * ra_angle
        rand_dec = self.true_dec + dist * dec_angle
        b = np.array([rand_ra, rand_dec, astro_uncert, mag, mag_uncert]).T
        if self.npy_or_csv == 'npy':
            np.save(self.b_cat_name.format(*cat_args), b)
        else:
            np.savetxt(self.b_cat_name.format(*cat_args), b, delimiter=',')

    def test_fit_astrometry_load_errors(self):  # pylint: disable=too-many-statements
        dd_params = np.load(os.path.join(os.path.dirname(__file__), 'data/dd_params.npy'))
        l_cut = np.load(os.path.join(os.path.dirname(__file__), 'data/l_cut.npy'))
        ax1_mids, ax2_mids = np.array([105], dtype=float), np.array([0], dtype=float)
        magarray = np.array([14.07, 14.17, 14.27, 14.37, 14.47])
        magslice = np.array([0.05, 0.05, 0.05, 0.05, 0.05])
        sigslice = np.array([0.01, 0.01, 0.01, 0.01, 0.01])

        _kwargs = {
            'psf_fwhm': 6.1, 'numtrials': 10000, 'nn_radius': 30, 'dens_search_radius': 0.25,
            'save_folder': 'ac_save_folder', 'trifilepath': 'tri_folder/trilegal_sim.dat',
            'maglim_f': 25, 'magnum': 11, 'tri_num_faint': 1500000,
            'trifilterset': '2mass_spitzer_wise', 'trifiltname': 'W1', 'gal_wav_micron': 3.35,
            'gal_ab_offset': 2.699, 'gal_filtname': 'wise2010-W1', 'gal_alav': 0.039,
            'dm': 0.1, 'dd_params': dd_params, 'l_cut': l_cut, 'ax1_mids': ax1_mids,
            'ax2_mids': ax2_mids, 'cutout_area': 60, 'cutout_height': 6, 'mag_array': magarray,
            'mag_slice': magslice, 'sig_slice': sigslice, 'n_pool': 1,
            'pos_and_err_indices': [[0, 1, 2], [0, 1, 2]], 'mag_indices': [3],
            'mag_unc_indices': [4], 'mag_names': ['W1'], 'correct_astro_mag_indices_index': 0,
            'n_r': 5000, 'n_rho': 5000, 'max_rho': 100, 'saturation_magnitudes': [15],
            'mn_fit_type': 'quadratic'}

        with pytest.raises(ValueError, match='single_sided_auf must be True.'):
            AstrometricCorrections(
                **_kwargs, single_sided_auf=False, ax_dimension=1, npy_or_csv='npy',
                coord_or_chunk='coord', coord_system='equatorial', pregenerate_cutouts=True)
        for ax_dim in [3, 'A']:
            with pytest.raises(ValueError, match="ax_dimension must either be '1' or "):
                AstrometricCorrections(
                    **_kwargs, ax_dimension=ax_dim, npy_or_csv='npy',
                    coord_or_chunk='coord', coord_system='equatorial', pregenerate_cutouts=True)
        for n_or_c in ['x', 4, 'npys']:
            with pytest.raises(ValueError, match="npy_or_csv must either be 'npy' or"):
                AstrometricCorrections(
                    **_kwargs, ax_dimension=1, npy_or_csv=n_or_c,
                    coord_or_chunk='coord', coord_system='equatorial', pregenerate_cutouts=True)
        for c_or_c in ['x', 4, 'npys']:
            with pytest.raises(ValueError, match="coord_or_chunk must either be 'coord' or"):
                AstrometricCorrections(
                    **_kwargs, ax_dimension=1, npy_or_csv='csv',
                    coord_or_chunk=c_or_c, coord_system='equatorial', pregenerate_cutouts=True)
        with pytest.raises(ValueError, match="chunks must be provided"):
            AstrometricCorrections(
                **_kwargs, ax_dimension=1, npy_or_csv='csv',
                coord_or_chunk='chunk', coord_system='equatorial', pregenerate_cutouts=True)
        with pytest.raises(ValueError, match="ax_dimension must be 2, and ax1-ax2 pairings "):
            AstrometricCorrections(
                **_kwargs, ax_dimension=1, npy_or_csv='csv', pregenerate_cutouts=True,
                coord_or_chunk='chunk', coord_system='equatorial', chunks=[2017])
        with pytest.raises(ValueError, match="ax1_mids, ax2_mids, and chunks must all be the "):
            AstrometricCorrections(
                **_kwargs, ax_dimension=2, npy_or_csv='csv', pregenerate_cutouts=True,
                coord_or_chunk='chunk', coord_system='equatorial', chunks=[2017, 2018])
        for e_or_g in ['x', 4, 'galacticorial']:
            with pytest.raises(ValueError, match="coord_system must either be 'equatorial'"):
                AstrometricCorrections(
                    **_kwargs, ax_dimension=1, npy_or_csv='csv',
                    coord_or_chunk='coord', coord_system=e_or_g, pregenerate_cutouts=True)
        for pregen_cut in [2, 'x', 'true']:
            with pytest.raises(ValueError, match="pregenerate_cutouts should either be 'None', 'True' or "):
                AstrometricCorrections(
                    **_kwargs, ax_dimension=1, npy_or_csv='csv', pregenerate_cutouts=pregen_cut,
                    coord_or_chunk='coord', coord_system='equatorial')
        del _kwargs['cutout_height']
        with pytest.raises(ValueError, match="cutout_height must be given if pregenerate_cutouts"):
            AstrometricCorrections(
                **_kwargs, ax_dimension=1, npy_or_csv='csv', pregenerate_cutouts=False,
                coord_or_chunk='coord', coord_system='equatorial')
        del _kwargs['cutout_area']
        with pytest.raises(ValueError, match="cutout_area must be given if pregenerate_cutouts"):
            AstrometricCorrections(
                **_kwargs, ax_dimension=1, npy_or_csv='csv', pregenerate_cutouts=False,
                coord_or_chunk='coord', coord_system='equatorial')
        with pytest.raises(ValueError, match="use_photometric_uncertainties must either be True "):
            ac = AstrometricCorrections(
                **_kwargs, ax_dimension=1, npy_or_csv='csv', pregenerate_cutouts=False,
                coord_or_chunk='coord', coord_system='equatorial', cutout_area=60, cutout_height=6,
                use_photometric_uncertainties='yes')
        with pytest.raises(ValueError, match="return_nm must either be True "):
            ac = AstrometricCorrections(
                **_kwargs, ax_dimension=1, npy_or_csv='csv', pregenerate_cutouts=False,
                coord_or_chunk='coord', coord_system='equatorial', cutout_area=60, cutout_height=6,
                use_photometric_uncertainties=True, return_nm='f')
        del _kwargs['mn_fit_type']
        with pytest.raises(ValueError, match="mn_fit_type must either be 'quad"):
            ac = AstrometricCorrections(
                **_kwargs, ax_dimension=1, npy_or_csv='csv', pregenerate_cutouts=False,
                coord_or_chunk='coord', coord_system='equatorial', cutout_area=60, cutout_height=6,
                use_photometric_uncertainties=True, return_nm=False, mn_fit_type='something else')
        ac = AstrometricCorrections(
            **_kwargs, ax_dimension=1, npy_or_csv='csv', pregenerate_cutouts=False,
            coord_or_chunk='coord', coord_system='equatorial', cutout_area=60, cutout_height=6,
            mn_fit_type='quadratic')
        self.a_cat_name = 'store_data/a_cat{}{}.npy'
        self.b_cat_name = 'store_data/b_cat{}{}.npy'
        with pytest.raises(ValueError, match='a_cat_func must be given if pregenerate_cutouts '):
            ac(a_cat_name=self.a_cat_name, b_cat_name=self.b_cat_name, a_cat_func=None, b_cat_func=None,
               tri_download=False, make_plots=True, make_summary_plot=True)
        with pytest.raises(ValueError, match='b_cat_func must be given if pregenerate_cutouts '):
            ac(a_cat_name=self.a_cat_name, b_cat_name=self.b_cat_name, a_cat_func=self.fake_cata_cutout,
               b_cat_func=None, tri_download=False, make_plots=True, make_summary_plot=True)
        dd_params = np.load(os.path.join(os.path.dirname(__file__), 'data/dd_params.npy'))
        l_cut = np.load(os.path.join(os.path.dirname(__file__), 'data/l_cut.npy'))
        ax1_mids, ax2_mids = np.array([105], dtype=float), np.array([0], dtype=float)
        magarray = np.array([14.07, 14.17, 14.27, 14.37, 14.47])
        magslice = np.array([0.05, 0.05, 0.05, 0.05, 0.05])
        sigslice = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
        chunks = None
        ax_dimension = 1
        ac = AstrometricCorrections(
            psf_fwhm=6.1, numtrials=1000, nn_radius=30, dens_search_radius=0.25,
            save_folder='ac_save_folder', trifilepath='tri_folder/trilegal_sim_{}_{}.dat',
            maglim_f=25, magnum=11, tri_num_faint=1500000, trifilterset='2mass_spitzer_wise',
            trifiltname='W1', gal_wav_micron=3.35, gal_ab_offset=2.699, gal_filtname='wise2010-W1',
            gal_alav=0.039, dm=0.1, dd_params=dd_params, l_cut=l_cut, ax1_mids=ax1_mids,
            ax2_mids=ax2_mids, ax_dimension=ax_dimension, cutout_area=60, cutout_height=6,
            mag_array=magarray, mag_slice=magslice, sig_slice=sigslice, n_pool=1, npy_or_csv='npy',
            coord_or_chunk='coord', pos_and_err_indices=[[0, 1, 2], [0, 1, 2]], mag_indices=[3],
            mag_unc_indices=[4], mag_names=['W1'], correct_astro_mag_indices_index=0,
            coord_system='equatorial', chunks=chunks, pregenerate_cutouts=True, n_r=2000, n_rho=2000,
            max_rho=40, saturation_magnitudes=[15], mn_fit_type='quadratic')
        with pytest.raises(ValueError, match="a_cat and b_cat must either both be None or "):
            ac(a_cat=None, b_cat=np.array([0]), a_cat_name=None, b_cat_name=None, a_cat_func=None,
               b_cat_func=None, tri_download=False, make_plots=True, make_summary_plot=True)
        with pytest.raises(ValueError, match="a_cat_name and b_cat_name must either both be None or "):
            ac(a_cat=None, b_cat=None, a_cat_name=None, b_cat_name='text', a_cat_func=None,
               b_cat_func=None, tri_download=False, make_plots=True, make_summary_plot=True)
        with pytest.raises(ValueError, match="pregenerate_cutouts must be None if a_cat is not None"):
            ac(a_cat=np.array([0]), b_cat=np.array([0]), a_cat_name=None, b_cat_name=None, a_cat_func=None,
               b_cat_func=None, tri_download=False, make_plots=True, make_summary_plot=True)
        ac.pregenerate_cutouts = None
        with pytest.raises(ValueError, match="a_cat_func must be None if a_cat is not "):
            ac(a_cat=np.array([0]), b_cat=np.array([0]), a_cat_name=None, b_cat_name=None,
               a_cat_func=np.array([0]), b_cat_func=None, tri_download=False, make_plots=True,
               make_summary_plot=True)
        with pytest.raises(ValueError, match="b_cat_func must be None if b_cat is not "):
            ac(a_cat=np.array([0]), b_cat=np.array([0]), a_cat_name=None, b_cat_name=None,
               a_cat_func=None, b_cat_func=np.array([0]), tri_download=False, make_plots=True,
               make_summary_plot=True)
        with pytest.raises(ValueError, match="a_cat must not be None if pregenerate_cutouts is None."):
            ac(a_cat=None, b_cat=None, a_cat_name=None, b_cat_name=None,
               a_cat_func=None, b_cat_func=None, tri_download=False, make_plots=True,
               make_summary_plot=True)
        ac.pregenerate_cutouts = True
        with pytest.raises(ValueError, match="a_cat and a_cat_name must not both be None or both not be "):
            ac(a_cat=None, b_cat=None, a_cat_name=None, b_cat_name=None,
               a_cat_func=None, b_cat_func=None, tri_download=False, make_plots=True,
               make_summary_plot=True)

        self.npy_or_csv = 'npy'
        cat_args = (105.0, 0.0)
        if os.path.isfile(self.a_cat_name.format(*cat_args)):
            os.remove(self.a_cat_name.format(*cat_args))
        if os.path.isfile(self.b_cat_name.format(*cat_args)):
            os.remove(self.b_cat_name.format(*cat_args))
        a_cat_func = None
        b_cat_func = None
        ac.pregenerate_cutouts = True
        with pytest.raises(ValueError, match="If pregenerate_cutouts is 'True' all files must "
                           f"exist already, but {self.a_cat_name.format(*cat_args)} does not."):
            ac(a_cat_name=self.a_cat_name, b_cat_name=self.b_cat_name, a_cat_func=a_cat_func,
               b_cat_func=b_cat_func, tri_download=False, make_plots=True, make_summary_plot=True,
               seeing_ranges=[1, 2, 3])
        ax1_min, ax1_max, ax2_min, ax2_max = 100, 110, -3, 3
        self.fake_cata_cutout(ax1_min, ax1_max, ax2_min, ax2_max, *cat_args)
        with pytest.raises(ValueError, match="If pregenerate_cutouts is 'True' all files must "
                           f"exist already, but {self.b_cat_name.format(*cat_args)} does not."):
            ac(a_cat_name=self.a_cat_name, b_cat_name=self.b_cat_name, a_cat_func=a_cat_func,
               b_cat_func=b_cat_func, tri_download=False, make_plots=True, make_summary_plot=True,
               seeing_ranges=[1, 2, 3])
        with pytest.raises(ValueError, match="seeing_ranges must be provided if make_plots"):
            ac(a_cat=None, b_cat=None, a_cat_name=self.a_cat_name, b_cat_name=self.b_cat_name,
               a_cat_func=None, b_cat_func=None, tri_download=False, make_plots=True,
               make_summary_plot=True)
        ac.pregenerate_cutouts = None
        with pytest.raises(ValueError, match="seeing_ranges must be a list of length 1, 2"):
            ac(a_cat='a', b_cat='b', a_cat_name=None, b_cat_name=None,
               a_cat_func=None, b_cat_func=None, tri_download=False, make_plots=True,
               make_summary_plot=True, seeing_ranges=[1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match="seeing_ranges should be a list of floats"):
            ac(a_cat='a', b_cat='b', a_cat_name=None, b_cat_name=None,
               a_cat_func=None, b_cat_func=None, tri_download=False, make_plots=True,
               make_summary_plot=True, seeing_ranges=['a', 'b', 'c'])
        with pytest.raises(ValueError, match="single_or_repeat must either be 'single' "):
            ac(a_cat='a', b_cat='b', a_cat_name=None, b_cat_name=None,
               a_cat_func=None, b_cat_func=None, tri_download=False, make_plots=True,
               make_summary_plot=True, seeing_ranges=np.array([1, 2]), single_or_repeat='something else')
        ac.pregenerate_cutouts = False
        with pytest.raises(ValueError, match="single_or_repeat cannot be 'repeat' unless pregenerate_cutouts "
                           "is None and a_cat "):
            ac(a_cat=None, b_cat=None, a_cat_name=self.a_cat_name, b_cat_name=self.b_cat_name,
               a_cat_func=np.array([0]), b_cat_func=np.array([0]), tri_download=False, make_plots=True,
               make_summary_plot=True, seeing_ranges=np.array([1, 2]), single_or_repeat='repeat')
        ac.pregenerate_cutouts = None
        with pytest.raises(ValueError, match="repeat_unique_visits_list must be provided if "):
            ac(a_cat='a', b_cat='b', a_cat_name=None, b_cat_name=None,
               a_cat_func=None, b_cat_func=None, tri_download=False, make_plots=True,
               make_summary_plot=True, seeing_ranges=np.array([1, 2]), single_or_repeat='repeat')

        os.system('rm -r store_data')

    @pytest.mark.remote_data
    @pytest.mark.parametrize("npy_or_csv,coord_or_chunk,coord_system,pregenerate_cutouts,return_nm,in_memory",
                             [("csv", "chunk", "equatorial", True, False, False),
                              ("npy", "coord", "galactic", None, True, True),
                              ("npy", "chunk", "equatorial", False, False, False)])
    # pylint: disable-next=too-many-statements,too-many-branches
    def test_fit_astrometry(self, npy_or_csv, coord_or_chunk, coord_system, pregenerate_cutouts, return_nm,
                            in_memory):
        self.npy_or_csv = npy_or_csv
        dd_params = np.load(os.path.join(os.path.dirname(__file__), 'data/dd_params.npy'))
        l_cut = np.load(os.path.join(os.path.dirname(__file__), 'data/l_cut.npy'))
        # Flag telling us to test for the non-running of all sightlines,
        # but to leave pre-generated ones alone
        half_run_flag = (npy_or_csv == "npy" and coord_or_chunk == "chunk" and
                         coord_system == "equatorial" and pregenerate_cutouts is False)
        if half_run_flag:
            ax1_mids, ax2_mids = np.array([105, 120], dtype=float), np.array([0, 10], dtype=float)
        else:
            ax1_mids, ax2_mids = np.array([105], dtype=float), np.array([0], dtype=float)
        magarray = np.array([14.07, 14.17, 14.27, 14.37, 14.47])
        magslice = np.array([0.05, 0.05, 0.05, 0.05, 0.05])
        sigslice = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
        if coord_or_chunk == 'coord':
            chunks = None
            ax_dimension = 1
        else:
            if half_run_flag:
                chunks = [2017, 2018]
            else:
                chunks = [2017]
            ax_dimension = 2
        ac = AstrometricCorrections(
            psf_fwhm=6.1, numtrials=1000, nn_radius=30, dens_search_radius=1,
            save_folder='ac_save_folder', trifilepath='tri_folder/trilegal_sim_{}_{}.dat',
            maglim_f=25, magnum=11, tri_num_faint=1500000, trifilterset='2mass_spitzer_wise',
            trifiltname='W1', gal_wav_micron=3.35, gal_ab_offset=2.699, gal_filtname='wise2010-W1',
            gal_alav=0.039, dm=0.1, dd_params=dd_params, l_cut=l_cut, ax1_mids=ax1_mids,
            ax2_mids=ax2_mids, ax_dimension=ax_dimension, mag_array=magarray, mag_slice=magslice,
            sig_slice=sigslice, n_pool=1, npy_or_csv=npy_or_csv, coord_or_chunk=coord_or_chunk,
            pos_and_err_indices=[[0, 1, 2], [0, 1, 2]], mag_indices=[3], mag_unc_indices=[4],
            mag_names=['W1'], correct_astro_mag_indices_index=0, coord_system=coord_system, chunks=chunks,
            pregenerate_cutouts=pregenerate_cutouts,
            cutout_area=60 if pregenerate_cutouts is False else None,
            cutout_height=6 if pregenerate_cutouts is False else None, n_r=2000, n_rho=2000, max_rho=40,
            return_nm=return_nm, saturation_magnitudes=[5],
            mn_fit_type='quadratic' if pregenerate_cutouts is False else 'linear')

        if coord_or_chunk == 'coord':
            self.a_cat_name = 'store_data/a_cat{}{}' + ('.csv' if npy_or_csv == 'csv' else '.npy')
            self.b_cat_name = 'store_data/b_cat{}{}' + ('.csv' if npy_or_csv == 'csv' else '.npy')
        else:
            self.a_cat_name = 'store_data/a_cat{}' + ('.csv' if npy_or_csv == 'csv' else '.npy')
            self.b_cat_name = 'store_data/b_cat{}' + ('.csv' if npy_or_csv == 'csv' else '.npy')
        if pregenerate_cutouts:
            # Cutout area is 60 sq deg with a height of 6 deg for a 10x6 box around (105, 0).
            cat_args = (chunks[0],)
            ax1_min, ax1_max, ax2_min, ax2_max = 100, 110, -3, 3
            self.fake_cata_cutout(ax1_min, ax1_max, ax2_min, ax2_max, *cat_args)
            self.fake_catb_cutout(ax1_min, ax1_max, ax2_min, ax2_max, *cat_args)
            a_cat_func = None
            b_cat_func = None
        else:
            a_cat_func = self.fake_cata_cutout
            b_cat_func = self.fake_catb_cutout
        if os.path.isfile('ac_save_folder/npy/snr_mag_params.npy'):
            os.remove('ac_save_folder/npy/snr_mag_params.npy')
        if os.path.isfile('ac_save_folder/npy/m_sigs_array.npy'):
            os.remove('ac_save_folder/npy/m_sigs_array.npy')
        if os.path.isfile('ac_save_folder/npy/n_sigs_array.npy'):
            os.remove('ac_save_folder/npy/n_sigs_array.npy')

        if half_run_flag:
            np.save('ac_save_folder/npy/snr_mag_params.npy',
                    np.array([[[-1, -1, -1, -1, -1], [-1, -2, -3, -4, -5]]], dtype=float))
            np.save('ac_save_folder/npy/m_sigs_array.npy', np.array([-1, 12], dtype=float))
            np.save('ac_save_folder/npy/n_sigs_array.npy', np.array([-1, 15], dtype=float))
        if in_memory:
            cat_args = (105.0, 0.0)
            ax1_min, ax1_max, ax2_min, ax2_max = 100, 110, -3, 3
            self.fake_cata_cutout(ax1_min, ax1_max, ax2_min, ax2_max, *cat_args)
            self.fake_catb_cutout(ax1_min, ax1_max, ax2_min, ax2_max, *cat_args)
            a_cat = np.load(self.a_cat_name.format(*cat_args))
            b_cat = np.load(self.b_cat_name.format(*cat_args))
        if return_nm:
            if in_memory:
                marray, narray, abc_array = ac(
                    a_cat=a_cat, b_cat=b_cat, a_cat_name=None, b_cat_name=None, a_cat_func=None,
                    b_cat_func=None, tri_download=False, make_plots=True, make_summary_plot=True,
                    seeing_ranges=np.array([0.5, 1, 1.5]))
            else:
                marray, narray, abc_array = ac(
                    a_cat=None, b_cat=None, a_cat_name=self.a_cat_name, b_cat_name=self.b_cat_name,
                    a_cat_func=a_cat_func, b_cat_func=b_cat_func, tri_download=False, make_plots=True,
                    make_summary_plot=True, seeing_ranges=np.array([0.5, 1, 1.5]))
        else:
            if in_memory:
                ac(a_cat=a_cat, b_cat=b_cat, a_cat_name=None, b_cat_name=None, a_cat_func=None,
                    b_cat_func=None, tri_download=False, make_plots=True, make_summary_plot=True,
                    seeing_ranges=np.array([0.5, 1, 1.5]), single_or_repeat='repeat',
                   repeat_unique_visits_list=np.array([1] * len(b_cat)))
            else:
                ac(a_cat=None, b_cat=None, a_cat_name=self.a_cat_name, b_cat_name=self.b_cat_name,
                   a_cat_func=a_cat_func, b_cat_func=b_cat_func, tri_download=False, make_plots=True,
                   make_summary_plot=True, seeing_ranges=np.array([0.5, 1, 1.5]))

        if coord_or_chunk == 'coord':
            assert os.path.isfile('ac_save_folder/pdf/auf_fits_105.0_0.0.pdf')
            assert os.path.isfile('ac_save_folder/pdf/counts_comparison_105.0_0.0.pdf')
            assert os.path.isfile('ac_save_folder/pdf/s_vs_snr_105.0_0.0.pdf')
            assert os.path.isfile('ac_save_folder/pdf/histogram_mag_vs_sig_vs_snr_105.0_0.0.pdf')
        else:
            assert os.path.isfile('ac_save_folder/pdf/auf_fits_2017.pdf')
            assert os.path.isfile('ac_save_folder/pdf/counts_comparison_2017.pdf')
            assert os.path.isfile('ac_save_folder/pdf/s_vs_snr_2017.pdf')
            assert os.path.isfile('ac_save_folder/pdf/histogram_mag_vs_sig_vs_snr_2017.pdf')

        assert os.path.isfile('ac_save_folder/pdf/summary_mn_ind_cdfs.pdf')
        assert os.path.isfile('ac_save_folder/pdf/summary_mn_sky.pdf')
        assert os.path.isfile('ac_save_folder/pdf/summary_individual_sig_vs_sig.pdf')

        if not return_nm:
            marray = np.load('ac_save_folder/npy/m_sigs_array.npy')
            narray = np.load('ac_save_folder/npy/n_sigs_array.npy')
        # pylint: disable-next=possibly-used-before-assignment
        assert_allclose([marray[0], narray[0]], [2, 0], rtol=0.1, atol=0.01)

        if not return_nm:
            abc_array = np.load('ac_save_folder/npy/snr_mag_params.npy')
        # pylint: disable-next=possibly-used-before-assignment
        assert_allclose([abc_array[0, 0, 3], abc_array[0, 0, 4]], [105, 0], atol=0.001)

        assert_allclose(abc_array[0, 0, 0], 1.2e-2, rtol=0.05, atol=0.001)
        assert_allclose(abc_array[0, 0, 1], 8e-17, rtol=0.05, atol=5e-19)

        if pregenerate_cutouts is False:
            assert_allclose(ac.ax1_mins[0], 100, rtol=0.01)
            assert_allclose(ac.ax1_maxs[0], 110, rtol=0.01)
            assert_allclose(ac.ax2_mins[0], -3, rtol=0.01)
            assert_allclose(ac.ax2_maxs[0], 3, rtol=0.01)

        if half_run_flag:
            # For the pre-determined set of parameters we should have skipped
            # one of the sightlines and want to check if its parameters are
            # unchanged.
            assert_allclose([marray[1], narray[1]], [12, 15], atol=0.001)
            assert_allclose(abc_array[0, 1, 0], -1, atol=0.001)
            assert_allclose(abc_array[0, 1, 1], -2, atol=0.001)
            assert_allclose(abc_array[0, 1, 3], -4, atol=0.001)

        os.system('rm -r store_data')


class TestSNRMagRelation:
    def setup_method(self):
        self.rng = np.random.default_rng(seed=3478989767)
        self.n = 5000
        choice = self.rng.choice(self.n, size=self.n, replace=False)
        self.true_ra = np.linspace(100, 110, self.n)[choice]
        self.true_dec = np.linspace(-3, 3, self.n)[choice]

        os.makedirs('store_data', exist_ok=True)

    def test_snr_mag_relation_load_errors(self):
        ax1_mids, ax2_mids = np.array([105], dtype=float), np.array([0], dtype=float)
        _kwargs = {
            'save_folder': 'ac_save_folder', 'ax1_mids': ax1_mids, 'ax2_mids': ax2_mids,
            'pos_and_err_indices': [0, 1, 2], 'mag_indices': [3], 'mag_unc_indices': [4],
            'mag_names': ['W1']}

        for ax_dim in [3, 'A']:
            with pytest.raises(ValueError, match="ax_dimension must either be '1' or "):
                SNRMagnitudeRelationship(**_kwargs, ax_dimension=ax_dim, npy_or_csv='npy',
                                         coord_or_chunk='coord', coord_system='equatorial')
        for n_or_c in ['x', 4, 'npys']:
            with pytest.raises(ValueError, match="npy_or_csv must either be 'npy' or"):
                SNRMagnitudeRelationship(**_kwargs, ax_dimension=1, npy_or_csv=n_or_c,
                                         coord_or_chunk='coord', coord_system='equatorial')
        for c_or_c in ['x', 4, 'npys']:
            with pytest.raises(ValueError, match="coord_or_chunk must either be 'coord' or"):
                SNRMagnitudeRelationship(**_kwargs, ax_dimension=1, npy_or_csv='csv',
                                         coord_or_chunk=c_or_c, coord_system='equatorial')
        with pytest.raises(ValueError, match="chunks must be provided"):
            SNRMagnitudeRelationship(**_kwargs, ax_dimension=1, npy_or_csv='csv',
                                     coord_or_chunk='chunk', coord_system='equatorial')
        with pytest.raises(ValueError, match="ax_dimension must be 2, and ax1-ax2 pairings "):
            SNRMagnitudeRelationship(**_kwargs, ax_dimension=1, npy_or_csv='csv',
                                     coord_or_chunk='chunk', chunks=[2017],
                                     coord_system='equatorial')
        with pytest.raises(ValueError, match="ax1_mids, ax2_mids, and chunks must all be the "):
            SNRMagnitudeRelationship(**_kwargs, ax_dimension=2, npy_or_csv='csv',
                                     coord_or_chunk='chunk', chunks=[2017, 2018],
                                     coord_system='equatorial')
        for e_or_g in ['x', 4, 'galacticorial']:
            with pytest.raises(ValueError, match="coord_system must either be 'equatorial'"):
                SNRMagnitudeRelationship(**_kwargs, ax_dimension=1, npy_or_csv='csv',
                                         coord_or_chunk='coord', coord_system=e_or_g)
        with pytest.raises(ValueError, match="return_nm must either be True "):
            SNRMagnitudeRelationship(**_kwargs, ax_dimension=1, npy_or_csv='csv',
                                     coord_or_chunk='coord', coord_system='equatorial', return_nm='f')
        smr = SNRMagnitudeRelationship(**_kwargs, ax_dimension=1, npy_or_csv='csv',
                                       coord_or_chunk='coord', coord_system='equatorial', return_nm=False)
        with pytest.raises(ValueError, match="One of b_cat and b_cat_name "):
            smr(b_cat=None, b_cat_name=None)
        with pytest.raises(ValueError, match="Only one of b_cat and b_cat_name "):
            smr(b_cat=np.array([0]), b_cat_name='name')

        os.system('rm -r store_data')

    @pytest.mark.parametrize("npy_or_csv,coord_or_chunk,coord_system,return_nm,in_memory",
                             [("csv", "chunk", "equatorial", True, False),
                              ("npy", "chunk", "galactic", True, True),
                              ("npy", "coord", "galactic", False, False),
                              ("npy", "chunk", "equatorial", False, False)])
    # pylint: disable-next=too-many-statements,too-many-branches
    def test_snr_mag_relation_fit(self, npy_or_csv, coord_or_chunk, coord_system, return_nm, in_memory):
        self.npy_or_csv = npy_or_csv
        # Flag telling us to test for the non-running of all sightlines,
        # but to leave pre-generated ones alone
        half_run_flag = (npy_or_csv == "npy" and coord_or_chunk == "chunk" and
                         coord_system == "equatorial")
        if half_run_flag:
            ax1_mids, ax2_mids = np.array([105, 120], dtype=float), np.array([0, 10], dtype=float)
        else:
            ax1_mids, ax2_mids = np.array([105], dtype=float), np.array([0], dtype=float)

        if coord_or_chunk == 'coord':
            chunks = None
            ax_dimension = 1
        else:
            if half_run_flag:
                chunks = [2017, 2018]
            else:
                chunks = [2017]
            ax_dimension = 2
        smr = SNRMagnitudeRelationship(
            save_folder='ac_save_folder', ax1_mids=ax1_mids, ax2_mids=ax2_mids,
            ax_dimension=ax_dimension, npy_or_csv=npy_or_csv, coord_or_chunk=coord_or_chunk,
            pos_and_err_indices=[0, 1, 2], mag_indices=[3], mag_unc_indices=[4], mag_names=['W1'],
            coord_system=coord_system, chunks=chunks, return_nm=return_nm)

        if coord_or_chunk == 'coord':
            self.a_cat_name = 'store_data/a_cat{}{}' + ('.csv' if npy_or_csv == 'csv' else '.npy')
            self.b_cat_name = 'store_data/b_cat{}{}' + ('.csv' if npy_or_csv == 'csv' else '.npy')
        else:
            self.a_cat_name = 'store_data/a_cat{}' + ('.csv' if npy_or_csv == 'csv' else '.npy')
            self.b_cat_name = 'store_data/b_cat{}' + ('.csv' if npy_or_csv == 'csv' else '.npy')

        # Cutout area is 60 sq deg with a height of 6 deg for a 10x6 box around (105, 0).
        if coord_or_chunk == 'coord':
            cat_args = (ax1_mids[0], ax2_mids[0])
        else:
            cat_args = (chunks[0],)
        ax1_min, ax1_max, ax2_min, ax2_max = 100, 110, -3, 3
        t_a_c = TestAstroCorrection()
        t_a_c.npy_or_csv = self.npy_or_csv
        t_a_c.n = self.n
        t_a_c.rng = self.rng
        t_a_c.true_ra = self.true_ra
        t_a_c.true_dec = self.true_dec
        t_a_c.b_cat_name = self.b_cat_name
        t_a_c.fake_catb_cutout(ax1_min, ax1_max, ax2_min, ax2_max, *cat_args)
        if half_run_flag:
            t_a_c.fake_catb_cutout(ax1_min, ax1_max, ax2_min, ax2_max, *(chunks[1],))

        if os.path.isfile('ac_save_folder/npy/snr_mag_params.npy'):
            os.remove('ac_save_folder/npy/snr_mag_params.npy')
        if os.path.isfile('ac_save_folder/npy/m_sigs_array.npy'):
            os.remove('ac_save_folder/npy/m_sigs_array.npy')
        if os.path.isfile('ac_save_folder/npy/n_sigs_array.npy'):
            os.remove('ac_save_folder/npy/n_sigs_array.npy')

        if half_run_flag:
            np.save('ac_save_folder/npy/snr_mag_params.npy',
                    np.array([[[-1, -1, -1, -1, -1], [-1, -2, -3, -4, -5]]], dtype=float))
            np.save('ac_save_folder/npy/m_sigs_array.npy', np.array([-1, 12], dtype=float))
            np.save('ac_save_folder/npy/n_sigs_array.npy', np.array([-1, 15], dtype=float))
        if in_memory:
            cat_args = (chunks[0],)
            ax1_min, ax1_max, ax2_min, ax2_max = 100, 110, -3, 3
            t_a_c.fake_catb_cutout(ax1_min, ax1_max, ax2_min, ax2_max, *cat_args)
            b_cat = [np.load(self.b_cat_name.format(*cat_args))]
        if return_nm:
            if in_memory:
                abc_array = smr(b_cat=b_cat, make_plots=True)
            else:
                abc_array = smr(b_cat_name=self.b_cat_name, make_plots=True)
        else:
            if in_memory:
                smr(b_cat=b_cat, make_plots=True)
            else:
                smr(b_cat_name=self.b_cat_name, make_plots=True)

        if coord_or_chunk == 'coord':
            assert os.path.isfile('ac_save_folder/pdf/s_vs_snr_105.0_0.0.pdf')
        else:
            assert os.path.isfile('ac_save_folder/pdf/s_vs_snr_2017.pdf')

        if not return_nm:
            abc_array = np.load('ac_save_folder/npy/snr_mag_params.npy')
        # pylint: disable-next=possibly-used-before-assignment
        assert_allclose([abc_array[0, 0, 3], abc_array[0, 0, 4]], [105, 0], atol=0.001)

        assert_allclose(abc_array[0, 0, 0], 1.2e-2, rtol=0.05, atol=0.001)
        assert_allclose(abc_array[0, 0, 1], 8e-17, rtol=0.06, atol=5e-19)

        if half_run_flag:
            # For the pre-determined set of parameters we should have skipped
            # one of the sightlines and want to check if its parameters are
            # unchanged.
            assert_allclose(abc_array[0, 1, 0], -1, atol=0.001)
            assert_allclose(abc_array[0, 1, 1], -2, atol=0.001)
            assert_allclose(abc_array[0, 1, 3], -4, atol=0.001)

        os.system('rm -r store_data')
