# Licensed under a 3-clause BSD style license - see LICENSE
'''
Tests for the "galaxy_counts" module.
'''

import astropy.units as u
import numpy as np
from numpy.testing import assert_allclose
from test_perturbation_auf import GalCountValues

from macauff.galaxy_counts import (create_galaxy_counts,
                                   generate_speclite_filters)

gal_values = GalCountValues()


class TestCreateGalaxyCounts():
    def setup_class(self):
        self.mag_bins = np.arange(10, 15, 0.1)
        self.z_array = np.array([0, 0.01])

        self.ab_offset = 0

        # Avoid using astropy's functions to independently check working.
        # For LambdaCDM, H0 = 67.7 km/s/Mpc, Ol = 0.69, Om = 0.31, Ok = 0:
        # From e.g. https://ned.ipac.caltech.edu/level5/Hogg/Hogg_contents.html:
        # dV = dh * (1+z)**2 * da**2 / E(z) dO dz
        # da = dm / (1 + z); dm = dc for Omega_k = 0; dc = dh \int_0^z dz' / E(z')
        # dV = dh**3 * (\int_0^z dz' / E(z'))**2 / E(z) dOdz
        # E(z) = sqrt(Om (1+z)^3 + Ok (1 + z)^2 + Ol) = sqrt(Om (1+z)^3 + Ol)
        Om, Ol = 0.31, 0.69
        hubble_distance = 3000/0.677  # 3000/h Mpc
        __z = np.linspace(self.z_array[0], self.z_array[1], 1001)
        E__z = np.sqrt(Om * (1 + __z)**3 + Ol)
        int_ez = np.sum(1/E__z) * (__z[1] - __z[0])
        _dV_dOmega_dz_1 = 0  # at exactly z = 0 \int_0^z dz' / E(z') = 0
        sterad_per_sq_deg = (np.pi/180)**2
        _dV_dOmega_dz_2 = hubble_distance**3 * int_ez**2 / E__z[-1] * sterad_per_sq_deg
        self.dV_dOmega = 0.5 * (_dV_dOmega_dz_1 + _dV_dOmega_dz_2) * np.diff(self.z_array)

        self.fake_mags = np.linspace(-60, 50, 1101)
        # m ~= M + 5 log10(dl(z)) + 25; dl = (1 + z) * dm = (1 + z) * dh \int_0^z dz' / E(z')
        __z = np.linspace(self.z_array[0], 0.5 * np.sum(self.z_array), 1001)
        E__z = np.sqrt(Om * (1 + __z)**3 + Ol)
        int_ez = np.sum(1/E__z) * (__z[1] - __z[0])
        dl = (1 + 0.5 * np.sum(self.z_array)) * hubble_distance * int_ez
        self.fake_app_mags = self.fake_mags + 5 * np.log10(dl) + 25

    def _calculate_params(self, lwav, i):
        # Assume we're close enough to zero redshift to not need P and Q.
        _cmau = gal_values.cmau[:, i, :]
        fake_m = _cmau[0, 2] * np.exp(-lwav * _cmau[0, 1]) + _cmau[0, 0]
        if i == 0:
            fake_phi = _cmau[1, 2] * np.exp(-lwav * _cmau[1, 1]) + _cmau[1, 0]
        else:
            fake_phi = _cmau[1, 2] * np.exp(-0.5 * (lwav - _cmau[1, 3])**2 *
                                            _cmau[1, 1]) + _cmau[1, 0]
        fake_alpha = _cmau[2, 1] * lwav + _cmau[2, 0]

        return fake_m, fake_phi, fake_alpha

    def test_create_galaxy_counts(self):
        wav = 3.4  # microns
        filter_name = 'wise2010-W1'
        gal_dens = create_galaxy_counts(gal_values.cmau, self.mag_bins, self.z_array, wav,
                                        gal_values.alpha0, gal_values.alpha1,
                                        gal_values.alphaweight, self.ab_offset, filter_name, [0])
        tot_fake_sch = np.zeros_like(gal_dens)
        for i in [0, 1]:
            fake_m, fake_phi, fake_alpha = self._calculate_params(np.log10(wav), i)
            fake_schechter = (0.4 * np.log(10) * fake_phi *
                              (10**(-0.4 * (self.fake_mags - fake_m)))**(fake_alpha+1) *
                              np.exp(-10**(-0.4 * (self.fake_mags - fake_m))))
            tot_fake_sch += np.interp(self.mag_bins, self.fake_app_mags,
                                      fake_schechter) * self.dV_dOmega

        assert_allclose(tot_fake_sch, gal_dens, rtol=0.01, atol=1e-4)

    def test_create_filter_and_galaxy_counts(self):
        wav = 0.16  # microns
        f, n = 'filter', 'uuu'
        filter_name = '{}-{}'.format(f, n)

        generate_speclite_filters(f, [n], [np.array([0.159, 0.16, 0.161])], [np.array([0, 1, 0])],
                                  u.micron)

        gal_dens = create_galaxy_counts(gal_values.cmau, self.mag_bins, self.z_array, wav,
                                        gal_values.alpha0, gal_values.alpha1,
                                        gal_values.alphaweight, self.ab_offset, filter_name, [1.5])
        tot_fake_sch = np.zeros_like(gal_dens)
        for i in [0, 1]:
            fake_m, fake_phi, fake_alpha = self._calculate_params(np.log10(wav), i)
            fake_schechter = (0.4 * np.log(10) * fake_phi *
                              (10**(-0.4 * (self.fake_mags - fake_m)))**(fake_alpha+1) *
                              np.exp(-10**(-0.4 * (self.fake_mags - fake_m))))
            tot_fake_sch += np.interp(self.mag_bins, self.fake_app_mags + 1.5,
                                      fake_schechter) * self.dV_dOmega

        assert_allclose(tot_fake_sch, gal_dens, rtol=0.01, atol=1e-4)
