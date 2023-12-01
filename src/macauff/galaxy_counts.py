# Licensed under a 3-clause BSD style license - see LICENSE
'''
Provides the framework for simulating brightness distributions of galaxies
in any chosen filterset by parameterised double Schechter functions.
'''

import numpy as np
import skypy.galaxies as skygal
from astropy.cosmology import default_cosmology
from astropy.modeling.models import Exponential1D, Linear1D
from speclite.filters import FilterResponse, load_filters

__all__ = ['create_galaxy_counts', 'generate_speclite_filters']


def create_galaxy_counts(cmau_array, mag_bins, z_array, wav, alpha0, alpha1, weight,
                         ab_offset, filter_name, al_grid):
    r'''
    Create a simulated distribution of galaxy magnitudes for a particular
    bandpass by consideration of double Schechter functions (for blue and
    red galaxies) in a specified range of redshifts, following [1]_.

    Parameters
    ----------
    cmau_array : numpy.ndarray
        Array holding the c/m/a/u values that describe the parameterisation
        of the Schechter functions with wavelength, following Wilson (2022, RNAAS,
        6, 60) [1]_. Shape should be `(5, 2, 4)`, with 5 parameters for both
        blue and red galaxies.
    mag_bins : numpy.ndarray
        The apparent magnitudes at which to evaluate the on-sky differential
        galaxy density.
    z_array : numpy.ndarray
        Redshift bins to evaluate Schechter densities in the middle of.
    wav : float
        The wavelength, in microns, of the bandpass observations should be
        simulated in. Should likely be the effective wavelength.
    alpha0 : list of numpy.ndarray or numpy.ndarray
        List of arrays of parameters :math:`\alpha_{i, 0}` used to calculate
        Dirichlet-distributed SED coefficients. Should either be a two-element
        list of arrays of 5 elements, or an array of shape ``(2, 5)``, with
        coefficients for blue galaxies before red galaxies. See [2]_ and [3]_
        for more details.
    alpha1 : list of numpy.ndarray or numpy.ndarray
        :math:`\alpha_{i, 1}` used in the calculation of Dirichlet-distributed SED
        coefficients. Two-element list or ``(2, 5)`` shape array of blue
        then red galaxy coefficients.
    weight : list of numpy.ndarray or numpy.ndarray
        Corresponding weights for the derivation of Dirichlet `kcorrect`
        coefficients. Must match shape of ``alpha0`` and ``alpha1``.
    ab_offset : float
        Zeropoint offset for differential galaxy count observations in a
        non-AB magnitude system. Must be in the sense of m_desired = m_AB - offset.
    filter_name : str
        ``speclite`` compound filterset-filter name for the response curve
        of the particular observations. If observations are in a filter system
        not provided by ``speclite``, response curve can be generated using
        ``generate_speclite_filters``.
    al_grid : list or numpy.ndarray of floats
        The reddenings at infinity by which to extinct all galaxy magnitudes,
        for various different potential sub-sightlines within a field of view.

    Returns
    -------
    gal_dens : numpy.ndarray
        Simulated numbers of galaxies per square degree per magnitude in the
        specified observed bandpass.

    References
    ----------
    .. [1] Wilson T. J. (2022), RNAAS, 6, 60
    .. [2] Herbel J., Kacprzak T., Amara A., et al. (2017), JCAP, 8, 35
    .. [3] Blanton M. R., Roweis S. (2007), AJ, 133, 734

    '''
    cosmology = default_cosmology.get()
    gal_dens = np.zeros_like(mag_bins)
    log_wav = np.log10(wav)

    alpha0_blue, alpha0_red = alpha0
    alpha1_blue, alpha1_red = alpha1
    weight_blue, weight_red = weight

    # Currently just set up a very wide absolute magnitude bin range to ensure
    # we get the dynamic range right. Inefficient but should be reliable...
    abs_mag_bins = np.linspace(-60, 50, 1100)
    for i in range(len(z_array)-1):
        mini_z_array = z_array[[i, i+1]]
        z = 0.5 * np.sum(mini_z_array)

        phi_model1 = generate_phi(cmau_array, 0, log_wav, z, abs_mag_bins)
        phi_model2 = generate_phi(cmau_array, 1, log_wav, z, abs_mag_bins)

        # differential_comoving_volume is "per redshift per steradian" at each
        # redshift, so we take the average and "integrate" over z.
        dV_dOmega = np.sum(cosmology.differential_comoving_volume(
            mini_z_array).to_value('Mpc3 / deg2'))/2 * np.diff(mini_z_array)

        model_densities = [phi_model1 * dV_dOmega, phi_model2 * dV_dOmega]

        # Blanton & Roweis (2007) kcorrect templates, via skypy.
        w = skygal.spectrum.kcorrect.wavelength
        t = skygal.spectrum.kcorrect.templates
        # Generate redshifts and coefficients and k-corrections for each
        # realisation, and then take the median k-correction.
        for _alpha0, _alpha1, _weight, model_density in zip(
                [alpha0_blue, alpha0_red], [alpha1_blue, alpha1_red],
                [weight_blue, weight_red], model_densities):
            rng = np.random.default_rng()
            redshift = rng.uniform(z_array[i], z_array[i+1], 100)
            spectral_coefficients = skygal.spectrum.dirichlet_coefficients(
                redshift=redshift, alpha0=_alpha0, alpha1=_alpha1, weight=_weight)

            kcorr = np.empty_like(redshift)
            for j in range(len(redshift)):
                _z = redshift[j]
                f = load_filters(filter_name)[0]
                fs = f.create_shifted(_z)
                non_shift_ab_maggy, shift_ab_maggy = 0, 0
                for k in range(len(t)):
                    try:
                        non_shift_ab_maggy += spectral_coefficients[j, k] * f.get_ab_maggies(t[k],
                                                                                             w)
                    except ValueError:
                        _t, _w = fs.pad_spectrum(t[k], w, method='edge')
                        non_shift_ab_maggy += spectral_coefficients[j, k] * fs.get_ab_maggies(_t,
                                                                                              _w)
                    try:
                        shift_ab_maggy += spectral_coefficients[j, k] * fs.get_ab_maggies(t[k], w)
                    except ValueError:
                        _t, _w = fs.pad_spectrum(t[k], w, method='edge')
                        shift_ab_maggy += spectral_coefficients[j, k] * fs.get_ab_maggies(_t, _w)
                # Backwards to Hogg+ astro-ph/0210394, our "shifted" bandpass is the rest-frame
                # as opposed to the observer frame.
                kcorr[j] = -2.5 * np.log10(1/(1+_z) * shift_ab_maggy / non_shift_ab_maggy)
            # Take the average of the NxM AVs that went into al_grid.
            for al in al_grid:
                # e.g. Loveday+2015 for absolute -> apparent magnitude conversion
                gal_dens += np.interp(mag_bins, abs_mag_bins + cosmology.distmod(z).value +
                                      np.percentile(kcorr, 50) - ab_offset + al,
                                      model_density) / len(al_grid)

    return gal_dens


def generate_phi(cmau_array, cmau_ind, log_wav, z, abs_mag_bins):
    r'''
    Generate a Schechter (1976) [1]_ function from interpolated,
    wavelength-dependent parameters.

    Parameters
    ----------
    cmau_array : numpy.ndarray
        A shape ``(5, 2, 4)`` array, containing the c, m, a, and u parameters
        for  :math:`M^*_0`, :math:\phi^*_0`, :math:`\alpha`, P, and Q for "blue"
        and "red" galaxies respectively.
    cmau_ind : {0, 1}
        Value indicating whether we're generating galaxy densities for blue
        or red galaxies.
    log_wav : float
        The log-10 value of the wavelength of the particular observations, in
        microns.
    z : float
        The redshift at which to evaluate the Schechter luminosity function.
    abs_mag_bins : numpy.ndarray
        The absolute magnitudes at which to evaluate the Schechter function.

    Returns
    -------
    phi_model : numpy.ndarray
        The Schechter function differential galaxy density at ``z``.

    References
    ----------
    .. [1] Schechter P. (1976), ApJ, 203, 297

    '''
    M_star0 = function_evaluation_lookup(cmau_array, 0, cmau_ind, log_wav)
    phi_star0 = function_evaluation_lookup(cmau_array, 1, cmau_ind, log_wav)
    alpha = function_evaluation_lookup(cmau_array, 2, cmau_ind, log_wav)
    P = function_evaluation_lookup(cmau_array, 3, cmau_ind, log_wav)
    # All other parameters are a function of wavelength, but we want Q(P).
    Q = function_evaluation_lookup(cmau_array, 4, cmau_ind, P)
    # phi*(z) = phi* * 10**(0.4 P z) = phi* exp(0.4 * ln(10) P z)
    # and thus Exponential1D being of the form exp(x / tau),
    tau = 1 / (0.4 * np.log(10) * P)
    m_star = Linear1D(slope=-Q, intercept=M_star0)
    phi_star = Exponential1D(amplitude=phi_star0, tau=tau)
    L = 10**(-0.4 * (abs_mag_bins - m_star(z)))
    phi_model = 0.4 * np.log(10) * phi_star(z) * L**(alpha+1) * np.exp(-L)

    return phi_model


def function_evaluation_lookup(cmau, ind1, ind2, x):
    '''
    Generate the appropriate parameterisation derived by Wilson (2022) [1]_
    based on Schechter function parameter wavelength dependencies.

    Parameters
    ----------
    cmau : numpy.ndarray
        Array holding the c/m/a/u values that describe the parameterisation
        of the Schechter functions. Shape should be `(5, 2, 4)`, with 5
        parameters for both blue and red galaxies.
    ind1, ind2 : integer
        The indices into the specific parameter and galaxy type respectively to
        derive parameter value for.
    x : float
        The variable controlling the change in the particular Schechter function
        parameter.

    Returns
    -------
    float
        The evaluation of the parameterisation of this Schechter parameter
        at a given wavelength, with the particular functional form dictated by
        whether ``a`` or ``u`` have been fit for previously.

    References
    ----------
    .. [1] Wilson T. J. (2022), RNAAS, 6, 60

    '''
    c, m, a, u = cmau[ind1, ind2]
    if np.isnan(a) and np.isnan(u):
        return m * x + c
    elif np.isnan(u):
        return a * np.exp(-x * m) + c
    else:
        return a * np.exp(-0.5 * (x - u)**2 * m) + c


def generate_speclite_filters(group_name, filter_names, wavelength_list, response_list,
                              wavelength_unit):
    '''
    Convenience function to create a new set of ``speclite`` filters, if bandpasses
    other than those provided in the module already are required. Currently, this
    generates a temporary ``speclite.filters.FilterResponse`` object that can be
    loaded using the standard ``speclite`` syntax of ``group_name-band_name``.

    Parameters
    ----------
    group_name : string
        The overall name to be given to the set of filters being generated. For
        example, ``speclite`` currently provide ``sdss2010``.
    filter_names : list of string
        The individual names of each filter to be generated. For the above example
        of ``sdss2010``, ``filter_names`` would be ``[u, g, r, i, z]``.
    wavelength_list : list of numpy.ndarray
        The wavelengths of the filter response curves being loaded, corresponding
        to each entry in ``filter_names``.
    response_list : list of numpy.ndarray
        The response curve values of the filters, for each ``wavelength_list`` item.
        Each element in the list should also match an entry in ``filter_names``.
    wavelength_unit : ``astropy.units.Unit`` object
        The relevant ``astropy`` unit (e.g., ``u.micron``, ``u.angstrom``) for the
        units all ``wavelength_list`` entries are given in. All response curves
        currently have to be in common units.
    '''
    for filt_name, wavelength, response in zip(filter_names, wavelength_list, response_list):
        FilterResponse(wavelength=wavelength*wavelength_unit, response=response,
                       meta=dict(group_name=group_name, band_name=filt_name))
