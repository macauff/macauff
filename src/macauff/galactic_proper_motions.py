# Licensed under a 3-clause BSD style license - see LICENSE
'''
Provides the mathematical framework for simulating distributions of proper
motions based on the distance and Galactic coordinates of a set of Galactic
stars.
'''

import itertools

import numpy as np

__all__ = ['calculate_proper_motions']

from macauff.misc_functions import make_pool


def calculate_proper_motions(d, l, b, temp, n):
    '''
    Calculates the statistical distribution of proper motions of a sightline,
    from a set of theoretical sources within the particular sky area.

    Parameters
    ----------
    d : numpy.ndarray
        The set of distances to each theoretical source.
    l : numpy.ndarray
        The Galactic longitudes of each simulated source.
    b : numpy.ndarray
        The Galactic latitude of each source.
    temp : numpy.ndarray
        The simulated effective temperature of each object.
    n : integer
        The number of realisations of each object's Galactic velocities to draw
        from its dispersion relation.

    Returns
    -------
    pm : numpy.ndarray
        The Galactic and Equatorial proper motions of the ``len(d)`` simulated
        sources, for the three simulated Galactic components each with ``n``
        velocity dispersion realisations.
    type_fracs : numpy.ndarray
        The density-based weightings of each of the three Galactic components
        for each source in ``d``, broadcast to shape ``(len(d), 3, n)`` to match
        the shape of ``pm``.
    '''
    # Height of the Sun above the Galactic plane from Juric et al. (2008, ApJ, 673, 864).
    z_sol = 0.025
    # Solar Galactic radius from Mroz et al. (2019, ApJ, 870, 10).
    r_sol = 8.09
    # Scale lengths also from Juric et al. (2008).
    l_thin, h_thin = 2.6, 0.3
    l_thick, h_thick = 3.6, 0.9
    f_thick = 0.13
    f_h, q, n = 0.0051, 0.64, 2.77
    h_b = 0.8
    # Bulge-to-disc normalisation, from Jackson et al. (2002, MNRAS, 337, 749).
    zeta_b = 2

    # These coordinates do not account for any z_sol offset, and hence z = 0 is at b = 0.
    r, z = convert_dist_coord_to_cylindrical(d, l, b, r_sol)
    # This comes out with shape (len(d), 3); have to factor an additional 1/N in the weights
    # so that when we histogram an extra N times as many proper motions we smooth to the same
    # counts as the original data.
    fractions = fraction_density_component(r, z, r_sol, z_sol, l_thin, h_thin, l_thick, h_thick,
                                           f_thick, f_h, q, n, zeta_b, h_b).T / n
    # We then tile it to (len(d), len(types), n)
    type_fracs = np.tile(fractions[:, :, np.newaxis], (1, 1, n))

    # V_a values come from Robin et al. (2003), A&A, 409, 523; Pasetto et al. (2012), A&A, 547,
    # A70; & (220km/s Halo drift velocity from Reddy (2009), Proc. IAU Symp., 265)
    # 240 km/s Halo V_a from Golubov et al. (2013), A&A, 557, A92.
    drift_vels = np.array([10, 49, 240], dtype=float)
    # Assume our rotation curve was constructed from thin disc objects, so take the relative
    # asymmetric drift from that. Objects in Mroz et al. are Classical Cepheids, so that's fine.
    drift_vels -= drift_vels[0]

    # Derive Oort constants from Pecaut & Mamajek (2013, ApJS, 208, 9) and
    # Olling & Dehnen (2003, ApJ, 599, 275).
    # First, get (B - V)_0 from Teff based on P&M scalings:
    bv0 = np.empty_like(temp)
    bv0[temp < 10000] = 5.07836 * np.exp(-0.27083 * temp[temp < 10000] / 1000) - 0.40739
    bv0[temp >= 10000] = 0.69012 * np.exp(-0.08179 * temp[temp >= 10000] / 1000) - 0.35093
    # Next, get A/B from O&D based on intrinsic B-V colour:
    oort_a = 1.94553 * bv0 + 11.33138
    oort_b = -2.63360 * bv0 - 13.60611

    rng = np.random.default_rng()
    pm = np.empty((len(d), 3, n, 4))
    n_pool = 40
    counter = np.arange(0, len(d))
    iter_group = zip(counter, itertools.repeat(drift_vels), itertools.repeat(rng),
                     itertools.repeat([l, b, d]), itertools.repeat(n), itertools.repeat([r, z]),
                     itertools.repeat([r_sol, z_sol]), itertools.repeat([oort_a, oort_b]),
                     itertools.repeat([l_thin, l_thick]))
    with make_pool(n_pool) as pool:
        for stuff in pool.imap_unordered(calc_pm, iter_group,
                                         chunksize=max(1, len(d) // n_pool)):
            j, mu_a, mu_d, mu_l, mu_b = stuff
            pm[j, :, :, 0] = mu_a
            pm[j, :, :, 1] = mu_d
            pm[j, :, :, 2] = mu_l
            pm[j, :, :, 3] = mu_b

    pool.join()

    return pm, type_fracs


def convert_dist_coord_to_cylindrical(d, l, b, r_sol):
    '''
    Convert distance and Galactic lat/lon to Galactocentric Cartesian coordinates.

    Parameters
    ----------
    d : numpy.ndarray
        Distance to a set of theoretical objects.
    l : numpy.ndarray
        Galactic longitude for each source.
    b : numpy.ndarray
        Galactic latitude for all sources.
    r_sol : float
        Galactocentric Cylindrical radius of the Sun from the Galactic center
        in kpc.

    Returns
    -------
    r : numpy.ndarray
        The Galactocentric Cylindrical radius of each object from the Galactic
        center.
    z : numpy.ndarray
        The Galactocentric Cylindrical height of each object from the Galactic
        plane.
    '''
    # Following notation where X points towards the GC from the Sun, with a right-handed system,
    # but choice of handedness is irrelevant here since we only care about radial distance.
    x = d * np.cos(np.radians(l)) * np.cos(np.radians(b)) - r_sol
    y = d * np.sin(np.radians(l)) * np.cos(np.radians(b))
    z = d * np.sin(np.radians(b))
    r = np.sqrt(x**2 + y**2)
    return r, z


def fraction_density_component(r, z, r_sol, z_sol, l_thin, h_thin, l_thick, h_thick, f_thick,
                               f_halo, q, n, zeta_b, h_b):  # pylint: disable=unused-argument
    '''
    Calculates the relative densities of each of the Galactic components used
    in simulating potential proper motions of objects. At present, this is the
    thin and thick discs, and the (outer) Galactic halo.

    Parameters
    ----------
    r : numpy.ndarray
        Galactocentric Cylindrical radii for each source used to simulate
        proper motions for the particular sightline.
    z : numpy.ndarray
        Galactocentric Cylindrical heights above the Galactic plane for each
        source.
    r_sol : float
        The Sun's Galactocentric Cylindrical radius.
    z_sol : float
        Solar Galactocentric Cylindrical height above the plane.
    l_thin : float
        Scale length of the thin disc in the radial direction.
    h_thin : float
        Vertical scale length of the thin disc.
    l_thick : float
        Thick disc radial scale length.
    h_thick : float
        Thick disc vertical scale length.
    f_thick : float
        Relative normalisation of the thick disc to the thin disc density.
    f_halo : float
        Normalisation of the outer Galactic halo to the thin disc.
    q : float
        Oblateness of the outer Galactic halo in the vertical direction.
    n : float
        Scaling relation of the halo with Galactocentric (oblate) Spherical
        distance.
    zeta_b : float
        Relative normalisation of the Galactic bulge to the thin disc.
    h_b : float
        Spherical scale length of the Galactic bulge.

    Returns
    -------
    rel_dens : numpy.ndarray
        The relative densities of the implemented Galactic components.
    '''
    # First three densities from eq 22-24 and table 10 of Juric et al. (2008), plus
    # corrections and extension from eq 9-12 of Ivezic et al. (2008, ApJ, 684, 287), albeit with the
    # exception of the removal of rho_d(r_sol, 0) as a normalising factor in all functions
    f_thin = thin_disc_density(r, z, r_sol, z_sol, l_thin, h_thin)
    f_thick = thick_disc_density(r, z, r_sol, z_sol, l_thick, h_thick, f_thick)
    f_halo = halo_density(r, z, r_sol, f_halo, q, n)
    # To avoid a singularity within the solar circle, just set the halo component density
    # to its value at r_sol, allowing for the thin+thick disc to overcome it towards
    # the Galactic center. Test with e.g. Sandage (1987), AJ, 93, 610, Fig 2b.
    f_halo[r < r_sol] = f_halo[np.argmin(np.abs(r - r_sol))]
    # Currently not using f_bulge due to no parameterisation for its proper
    # motion at present.
    # f_bulge = bulge_density(r, z, zeta_b, h_b, q)
    norm = f_thin + f_thick + f_halo  # + f_bulge
    return np.array([f_thin, f_thick, f_halo]) / norm


def thin_disc_density(r, z, r_sol, z_sol, l_thin, h_thin):
    '''
    Calculates the relative density of the thin disc.

    Parameters
    ----------
    r : numpy.ndarray
        Galactocentric Cylindrical radii for each source used to simulate
        proper motions for the particular sightline.
    z : numpy.ndarray
        Galactocentric Cylindrical heights above the Galactic plane for each
        source.
    r_sol : float
        The Sun's Galactocentric Cylindrical radius.
    z_sol : float
        Solar Galactocentric Cylindrical height above the plane.
    l_thin : float
        Scale length of the thin disc in the radial direction.
    h_thin : float
        Vertical scale length of the thin disc.

    Returns
    -------
    disc_density : numpy.ndarray
        The density of the thin disc evaluated at ``r`` and ``z``.
    '''
    return disc_density(r, z, l_thin, h_thin, r_sol, z_sol)


def disc_density(r, z, l, h, r_sol, z_sol):
    '''
    Calculates the relative density of an exponential disc, either
    the thin or thick discs, depending on ``l`` and ``h``.

    Parameters
    ----------
    r : numpy.ndarray
        Galactocentric Cylindrical radii for each source used to simulate
        proper motions for the particular sightline.
    z : numpy.ndarray
        Galactocentric Cylindrical heights above the Galactic plane for each
        source.
    r_sol : float
        The Sun's Galactocentric Cylindrical radius.
    z_sol : float
        Solar Galactocentric Cylindrical height above the plane.
    l : float
        Scale length of the disc in the radial direction.
    h : float
        Vertical scale length of the disc.

    Returns
    -------
    disc_density : numpy.ndarray
        The density of the disc evaluated at ``r`` and ``z``.
    '''
    return np.exp(-(r - r_sol) / l - np.abs(z + z_sol)/h)


def thick_disc_density(r, z, r_sol, z_sol, l_thick, h_thick, f_thick):
    '''
    Calculates the relative density of the thick disc.

    Parameters
    ----------
    r : numpy.ndarray
        Galactocentric Cylindrical radii for each source used to simulate
        proper motions for the particular sightline.
    z : numpy.ndarray
        Galactocentric Cylindrical heights above the Galactic plane for each
        source.
    r_sol : float
        The Sun's Galactocentric Cylindrical radius.
    z_sol : float
        Solar Galactocentric Cylindrical height above the plane.
    l_thick : float
        Scale length of the thick disc in the radial direction.
    h_thick : float
        Vertical scale length of the thick disc.
    f_thick : float
        Relative scaling between the thin disc and thick disc.

    Returns
    -------
    disc_density : numpy.ndarray
        The density of the thick disc evaluated at ``r`` and ``z``.
    '''
    return f_thick * disc_density(r, z, l_thick, h_thick, r_sol, z_sol)


def halo_density(r, z, r_sol, f_h, q, n):
    '''
    Calculates the relative density of the halo.

    Parameters
    ----------
    r : numpy.ndarray
        Galactocentric Cylindrical radii for each source used to simulate
        proper motions for the particular sightline.
    z : numpy.ndarray
        Galactocentric Cylindrical heights above the Galactic plane for each
        source.
    r_sol : float
        The Sun's Galactocentric Cylindrical radius.
    f_h : float
        The normalising constant between thin disc and halo parameterisations.
    q : float
        Oblateness of the halo.
    n : float
        Power law scaling relation for halo density.

    Returns
    -------
    halo_density : numpy.ndarray
        The density of the halo evaluated at ``r`` and ``z``.
    '''
    return f_h * (r_sol / np.sqrt(r**2 + (z/q)**2))**n


def bulge_density(r, z, zeta_b, h_b):
    '''
    Calculates the relative density of the Galactic bulge. Currently
    not implemented.

    Parameters
    ----------
    r : numpy.ndarray
        Galactocentric Cylindrical radii for each source used to simulate
        proper motions for the particular sightline.
    z : numpy.ndarray
        Galactocentric Cylindrical heights above the Galactic plane for each
        source.
    zeta_b : float
        Normalising constant, setting the relative densities between the
        bulge and thin disc.
    h_b : float
        Scale length of the bulge density.

    Returns
    -------
    bulge_density : numpy.ndarray
        The density of the bulge evaluated at ``r`` and ``z``.
    '''
    # Bulge parameters from Jackson et al. (2002), equation 8.
    x = np.sqrt(r**2 + z**2)

    # We drop C entirely, assuming it to be the equivalent of rho_d(r_sol, 0) for Jackson et al.
    return zeta_b * np.exp(-x/h_b)


def calc_pm(iterable):  # pylint: disable=too-many-locals
    '''
    Calculate an individual simulated object's distribution of proper motions
    based on its different motions if it were a thin disc, thick disc, or outer
    halo object, combined with relative weightings for its assignment to those
    components of the Galaxy.

    Parameters
    ----------
    iterable : list
        The list of various variables passed through to ``calc_pm`` by
        ``multiprocessing``.

    Returns
    -------
    j : integer
        The index of the particular simulated source having proper motions
        derived for it.
    mu_a : numpy.ndarray
        The simulated proper motions for this object, ``n`` simulated velocities
        per ``drift_vels`` Galactic component, in right ascension.
    mu_d : numpy.ndarray
        The corresponding declination proper motions to ``mu_a``.
    _mu_l : numpy.ndarray
        Transposed Galactic longitude proper motions for the source's ``mu_a``
        and ``mu_d``.
    _mu_b : numpy.ndarray
        Transposed Galactic latitude proper motions for the source's ``mu_a``
        and ``mu_d``.
    '''
    j, drift_vels, rng, [l, b, d], n, [r, z], [r_sol, _], \
        [oort_a, oort_b], [l_thin, l_thick] = iterable
    mu_a, mu_d = np.empty((len(drift_vels), n), float), np.empty((len(drift_vels), n), float)
    _mu_l, _mu_b = np.empty((len(drift_vels), n), float), np.empty((len(drift_vels), n), float)

    sinl, cosl = np.sin(np.radians(l[j])), np.cos(np.radians(l[j]))
    sinb, cosb = np.sin(np.radians(b[j])), np.cos(np.radians(b[j]))

    d_ip = cosb * d[j]

    # Rotation+mirror matrix, to put R-phi-z dispersion along Vr-Vt-Vz plane
    rot = np.array(
        [[(r[j]**2 + d_ip**2 - r_sol**2) / (2 * r[j] * d_ip), r_sol / r[j] * sinl, 0],
         [r_sol / r[j] * sinl, -(r[j]**2 + d_ip**2 - r_sol**2) / (2 * r[j] * d_ip), 0],
         [0, 0, 1]])

    us, vs, ws = 0, 0, 0
    usol, vsol, wsol = 11.1, 12.2, 7.3  # km/s
    sinbeta = d[j] * sinl / r[j]
    cosbeta = (r_sol**2 + r[j]**2 - d[j]**2) / (2 * r_sol * r[j])

    theta_sol = 233.6  # km/s
    __b = 0.72
    a1, a2, a3 = 235.0, 0.89, 1.31
    x = (r[j] / r_sol) / a2
    theta_d_sq = a1**2 * __b * 1.97 * x**1.22 / (x**2 + 0.78**2)**1.43
    theta_h_sq = a1**2 * (1 - __b) * x**2 * (1 + a3**2) / (x**2 + a3**2)
    theta_r = np.sqrt(theta_d_sq + theta_h_sq)

    u1 = us * cosbeta + (vs + theta_r) * sinbeta - usol
    v1 = -us*sinbeta + (vs + theta_r) * cosbeta - vsol - theta_sol
    w1 = ws - wsol

    # Based on Mroz et al. (2019), our components of Heliocentric Cylindrical
    # coordinates are:
    vr = u1 * cosl + v1 * sinl
    vt = v1 * cosl - u1 * sinl
    vz = w1

    for i, drift_vel in enumerate(drift_vels):
        drift_vel_rtz = np.matmul(rot, np.array([[0], [-drift_vel], [0]]))
        mean = np.array([vr + drift_vel_rtz[0, 0], vt + drift_vel_rtz[1, 0], vz])
        if i == 0:
            cov = find_thin_disc_dispersion(r[j], z[j], l[j], b[j], d[j], oort_a[j], oort_b[j], l_thin,
                                            r_sol)
        if i == 1:
            cov = find_thick_disc_dispersion(r[j], l[j], b[j], d[j], l_thick, r_sol)
        if i == 2:
            cov = find_halo_dispersion(l[j], b[j], d[j])

        new_uvw = rng.multivariate_normal(mean, cov, n)  # pylint: disable=possibly-used-before-assignment
        v_d, v_l, v_z = new_uvw[:, 0], new_uvw[:, 1], new_uvw[:, 2]

        # 1 km/s/kpc = 0.2108 mas/year
        mu_lstar = v_l / d[j] * 0.2108
        mu_b = 0.2108/d[j] * (v_z * cosb - v_d * sinb)

        mu_a[i, :], mu_d[i, :] = galactic_to_equatorial(np.radians(l[j]), np.radians(b[j]),
                                                        mu_lstar, mu_b)

        _mu_l[i, :], _mu_b[i, :] = mu_lstar, mu_b

    return (j, mu_a, mu_d, _mu_l, _mu_b)


def find_thin_disc_dispersion(r, z, l, b, d, oort_a, oort_b, h, r_sol):
    '''
    Calculate the dispersion in the thin disc as a function of Galactic position.

    Parameters
    ----------
    r : float
        The Galactic Cylindrical radius of the object.
    z : float
        The Galactic Cylindrical height of the source.
    l : float
        Galactic longitude of the star.
    b : float
        Galactic latitude of the star.
    oort_a : float
        Oort constant.
    oort_b : float
        Oort constant.
    h : float
        Scale length of the thin disc.
    r_sol : float
        Galactic Cylindrical radius of the Sun.

    Returns
    -------
    cov_rtz : numpy.ndarray
        The covariance matrix dispersion vector, in Galactic Cylindrical
        coordinates.
    '''
    # Data from Pasetto et al. (2012), A&A, 547, A71, tables 4-9
    _r_sol = 8.5  # kpc; have to use the Pasetto et al. result of R0 ~ 8.4-8.6 kpc for consistency

    # Assume that sig_rr^2 goes as R^2 exp(-2 R / h) by a stable Toomre Parameter (Lewis & Freeman
    # 1989, AJ, 97, 139) where rotation curve is flat (Mroz et al.), where h is the scale length
    # of the disc. Pasetto et al. have a sig_rr^2(R0, 0) ~ 715 km^2/s^2
    sig_rr2_00 = 715.93  # km^2/s^2
    sig_rr2_r0 = sig_rr2_00 * (r / _r_sol)**2 * np.exp(-2 * (r - _r_sol) / h)
    # Now assume that the vertical gradient of the thin disc of Pasetto et al. goes for ~1kpc
    # with a gradient of roughly 1200km^2/s^2/kpc at R=R0, but scaling the same as the dispersion.
    sig_rr2_z_grad_0 = 1236.97  # km^2/s^2/kpc
    sig_rr2_z_grad = sig_rr2_z_grad_0  # * (r / r_sol)**2 * np.exp(-2 * (r - r_sol) / h)
    sig_rr2 = sig_rr2_r0 + min(1, np.abs(z)) * sig_rr2_z_grad

    # Assume that the phi variance sigma can be approximated from the relation with
    # the R sigma:
    sig_phiphi2 = (-oort_b / (oort_a - oort_b)) * sig_rr2

    # We assume the correlation along the R-phi covariance is zero, following Vallenari et al.
    sig_rphi2 = 0

    # Following the Pasetto et al. discussion, we conclude that the phi-z dispersion is
    # essentially zero, to their uncertainties, and force it to be so:
    sig_phiz2 = 0

    # Similar to sig_rr, we assume that sig_zz^2 = sig_zz^2(R0, 0) * exp(-(r - r_sol) / h)
    # where sig_zz^2(R0, 0) ~ 243 km^2/s^2
    sig_zz2_00 = 243.71  # km^2/s^2
    sig_zz2_r0 = sig_zz2_00 * np.exp(-(r - _r_sol) / h)
    # Now, again, extrapolate the vertical gradient seen in sig_zz(R, z) to 1 kpc above/below plane
    # with a gradient, again, that scales with the radial gradient. Assume ~300km^2/s^2/kpc R = R0.
    sig_zz2_z_grad_0 = 306.84  # km^2/s^2/kpc
    sig_zz2_z_grad = sig_zz2_z_grad_0  # * np.exp(-(r - r_sol) / h)
    sig_zz2 = sig_zz2_r0 + min(1, np.abs(z)) * sig_zz2_z_grad

    # Given Vallenari et al. (2006), A&A, 451, 125, following Cuddeford & Amendt (1991), we
    # assume that the vertical tilt goes as the gradient of sig^2_rz, which is given by
    # the difference between sig^2_rr(R, 0) and sig^2_zz(R, 0); we assume lambda = 0.6, for now.
    dsig2_rz_dz = 0.6 * (sig_rr2_r0 - sig_zz2_r0) / r
    sig_rz2 = z * dsig2_rz_dz
    # If correlation gets above one, force sig_rz2 = +- sqrt(sig_rr^2) * sqrt(sig_zz^2) at largest
    if np.abs(sig_rz2) > np.sqrt(sig_rr2) * np.sqrt(sig_zz2):
        sig_rz2 = 0.99 * np.sign(sig_rz2) * np.sqrt(sig_rr2) * np.sqrt(sig_zz2)

    cov = np.empty((3, 3), float)
    cov[0, 0] = sig_rr2
    cov[1, 1] = sig_phiphi2
    cov[2, 2] = sig_zz2
    cov[0, 1] = cov[1, 0] = sig_rphi2
    cov[0, 2] = cov[2, 0] = sig_rz2
    cov[1, 2] = cov[2, 1] = sig_phiz2

    d_ip = np.cos(np.radians(b)) * d
    # Rotation matrix, to put R-phi-z dispersion along Vr-Vt-Vz plane
    sinl = np.sin(np.radians(l))
    rot = np.array(
        [[(r**2 + d_ip**2 - r_sol**2) / (2 * r * d_ip), r_sol / r * sinl, 0],
         [r_sol / r * sinl, -(r**2 + d_ip**2 - r_sol**2) / (2 * r * d_ip), 0],
         [0, 0, 1]])

    cov_rtz = np.matmul(rot, np.matmul(cov, rot.T))

    return cov_rtz


def find_thick_disc_dispersion(r, l, b, d, h, r_sol):
    '''
    Calculate the dispersion in the thick disc as a function of Galactic position.

    Parameters
    ----------
    r : float
        The Galactic Cylindrical radius of the object.
    l : float
        Galactic longitude of the star.
    b : float
        Galactic latitude of the star.
    d : float
        THree-dimensional distance to the star.
    h : float
        Scale length of the thick disc.
    r_sol : float
        Galactic Cylindrical radius of the Sun.

    Returns
    -------
    cov_rtz : numpy.ndarray
        The covariance matrix dispersion vector, in Galactic Cylindrical
        coordinates.
    '''
    # Data from Pasetto et al. (2012), A&A, 547, A70, table 3-4
    # Assume sig_phiphi / sig_rr = const, so scale both the same. Currently assume there are no
    # cross-term products.

    _r_sol = 8.5  # kpc; use the Pasetto et al. thin disc rough R0, figuring it should be equal

    if r < _r_sol:
        sig_rr2 = 60.2**2 * (r / _r_sol)**2 * np.exp(-2 * (r - _r_sol) / h)
        sig_rphi2 = 0  # 37.6 km/s(!)
        sig_rz2 = 0  # 13.3 km/s(!)
        sig_phiphi2 = 44.7**2 * (r / _r_sol)**2 * np.exp(-2 * (r - _r_sol) / h)
        sig_phiz2 = 0  # 4.0 km/s(!)
        sig_zz2 = 37.2**2 * np.exp(-(r - _r_sol) / h)
    else:
        sig_rr2 = 55.8**2 * (r / _r_sol)**2 * np.exp(-2 * (r - _r_sol) / h)
        sig_rphi2 = 0  # 35.5 km/s(!)
        sig_rz2 = 0  # 9.6 km/s(!)
        sig_phiphi2 = 45.2**2 * (r / _r_sol)**2 * np.exp(-2 * (r - _r_sol) / h)
        sig_phiz2 = 0  # 3.8 km/s(!)
        sig_zz2 = 36.3**2 * np.exp(-(r - _r_sol) / h)

    cov = np.empty((3, 3), float)
    cov[0, 0] = sig_rr2
    cov[1, 1] = sig_phiphi2
    cov[2, 2] = sig_zz2
    cov[0, 1] = cov[1, 0] = sig_rphi2
    cov[0, 2] = cov[2, 0] = sig_rz2
    cov[1, 2] = cov[2, 1] = sig_phiz2

    d_ip = np.cos(np.radians(b)) * d
    # Rotation matrix, to put R-phi-z dispersion along Vr-Vt-Vz plane
    sinl = np.sin(np.radians(l))
    rot = np.array(
        [[(r**2 + d_ip**2 - r_sol**2) / (2 * r * d_ip), r_sol / r * sinl, 0],
         [r_sol / r * sinl, -(r**2 + d_ip**2 - r_sol**2) / (2 * r * d_ip), 0],
         [0, 0, 1]])

    cov_rtz = np.matmul(rot, np.matmul(cov, rot.T))

    return cov_rtz


def find_halo_dispersion(l, b, d):
    '''
    Calculate the dispersion in the halo as a function of Galactic position.

    Parameters
    ----------
    l : float
        Galactic longitude of the star.
    b : float
        Galactic latitude of the star.
    d : float
        The Galactic Spherical distance of the object from the Galactic center.

    Returns
    -------
    cov_rtz : numpy.ndarray
        The covariance matrix dispersion vector, in Galactic Cylindrical
        coordinates.
    '''
    # Data from King et al. (2015), ApJ, 813, 89, table 3; conversions from Appendix A
    # Heliocentric d, l, b (x, y, z) become Galactocentric R, phi, theta (X, Y, Z)
    r_vals = np.array([8.4, 10.1, 11.1, 12.0, 13.1, 14.4, 16.7, 22.4])
    sig_rr = np.array([155.3, 156.5, 107.4, 150.1, 105.0, 95.1, 56.8, 76.8])
    sig_phiphi = np.array([88.3, 86.2, 110.3, 85.9, 194.0, 172.6, 225.8, 195.8])
    sig_thetatheta = np.array([109.8, 98.2, 117.6, 38.5, 165.4, 205.3, 256.0, 159.1])
    cov_rphi = np.array([-271.8, 2442.2, -2923.5, 530.2, 5801.2, 1448.5, 494.8, 30.9])
    cov_rtheta = np.array([428.7, -123.2, -2806.8, -2088.1, 1896.4, 3053.6, 1243.5, 57.2])
    cov_phitheta = np.array([-80.4, -456.0, -6839.6, -4335.6, 5498.6, 2843.3, 831.7, 10559.0])

    r_sol = 8  # kpc; use the King et al. result for self-consistency
    z_sol = 0.0196  # kpc

    # Being unsure of why the r = 12kpc covariance matrix is not positive semi-definite, we
    # just remove the covariances for the time being. The covariances are large, but with large
    # uncertainties: cov_rphi is 0.25sigma away from zero, cov_rtheta 1.95 sigma from zero, and
    # cov_phitheta 1.6 sigma from zero, so this isn't a terrible assumption to make at this time.
    cov_rphi[3], cov_rtheta[3], cov_phitheta[3] = 0, 0, 0

    sinl, cosl = np.sin(np.radians(l)), np.cos(np.radians(l))
    sinb, cosb = np.sin(np.radians(b)), np.cos(np.radians(b))

    x = d * cosl * cosb
    y = d * sinl * cosb
    z = d * sinb
    x_ = x - r_sol
    y_ = y
    z_ = z + z_sol
    r = np.sqrt(x_**2 + y_**2 + z_**2)

    q = np.argmin(np.abs(r - r_vals))

    cov = np.empty((3, 3), float)
    cov[0, 0] = sig_rr[q]**2
    cov[1, 1] = sig_phiphi[q]**2
    cov[2, 2] = sig_thetatheta[q]**2
    cov[0, 1] = cov[1, 0] = cov_rphi[q]
    cov[0, 2] = cov[2, 0] = cov_rtheta[q]
    cov[1, 2] = cov[2, 1] = cov_phitheta[q]

    rho = r_sol**2 + d**2 - 2 * r_sol * d * cosb
    r = rho * cosb

    cosbeta = (r_sol**2 + rho**2 - d**2) / (2 * r_sol * rho)
    d_ip = d * cosb
    # Rotation matrix, to put R-phi-theta dispersion along Vr-Vt-Vz plane
    # First put R-phi-theta into R-phi-z, but remembering the opposing definition
    # of phi between the two coordinate frames.
    rot_sph_to_cyl = np.array([[cosbeta, 0, d/rho * sinb],
                               [0, -1, 0],
                               [-d/rho*sinb, 0, cosbeta]])
    # Second put R-phi-z into vd-vl-vz, with phi/vl being of the same sign, and
    # z/vz being aligned.
    sinl = np.sin(np.radians(l))
    rot_cyl_to_cyl = np.array(
        [[(r**2 + d_ip**2 - r_sol**2) / (2 * r * d_ip), r_sol / r * sinl, 0],
         [r_sol / r * sinl, -(r**2 + d_ip**2 - r_sol**2) / (2 * r * d_ip), 0],
         [0, 0, 1]])

    rot = np.matmul(rot_cyl_to_cyl, rot_sph_to_cyl)

    cov_rtz = np.matmul(rot, np.matmul(cov, rot.T))

    return cov_rtz


def galactic_to_equatorial(l, b, mu_l, mu_b):
    '''
    Transform Galactic coordinate proper motions into proper motions in
    equatorial coordinates.

    Parameters
    ----------
    l : float
        Galactic longitude of the source.
    b : float
        Galactic latitude of the object.
    mu_l : numpy.ndarray
        Galactic longitude component of the proper motion.
    mu_b : numpy.ndarray
        Galactic latitude component of the proper motion.

    Returns
    -------
    mu_a : numpy.ndarray
        Right ascension component of the equatorial proper motions.
    mu_d : numpy.ndarray
        Declination component of the equatorial proper motions.
    '''
    # Rotation frame maths from https://gea.esac.esa.int/archive/documentation/GDR2/
    # Data_processing/chap_cu3ast/sec_cu3ast_intro/ssec_cu3ast_intro_tansforms.html
    ag_p = np.array([[-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
                     [+0.4941094278755837, -0.4448296299600112, +0.7469822444972189],
                     [-0.8676661490190047, -0.1980763734312015, +0.4559837761750669]])
    ag = ag_p.T

    r_gal = np.array([[np.cos(l) * np.cos(b)], [np.sin(l) * np.cos(b)], [np.sin(b)]])
    r_icrs = np.matmul(ag, r_gal)

    a = np.arctan2(r_icrs[1], r_icrs[0])[0]
    d = np.arctan2(r_icrs[2], np.sqrt(r_icrs[0]**2 + r_icrs[1]**2))[0]

    p_gal = np.array([[-np.sin(l)], [np.cos(l)], [0]])
    p_icrs = np.array([[-np.sin(a)], [np.cos(a)], [0]])

    q_gal = np.array([[-np.cos(l) * np.sin(b)], [-np.sin(l) * np.sin(b)], [np.cos(b)]])
    q_icrs = np.array([[-np.cos(a) * np.sin(d)], [-np.sin(a) * np.sin(d)], [np.cos(d)]])

    mu_gal = p_gal * mu_l + q_gal * mu_b

    mu_icrs = np.matmul(ag, mu_gal)

    mu_a = np.matmul(np.transpose(p_icrs), mu_icrs)
    mu_d = np.matmul(np.transpose(q_icrs), mu_icrs)

    return mu_a, mu_d


def equatorial_to_galactic(a, d, mu_a, mu_d):
    '''
    Transform equatorial coordinate proper motions into proper motions in
    Galactic coordinates.

    Parameters
    ----------
    l : float
        Galactic longitude of the source.
    b : float
        Galactic latitude of the object.
    mu_a : numpy.ndarray
        Right ascension component of the equatorial proper motions.
    mu_d : numpy.ndarray
        Declination component of the equatorial proper motions.

    Returns
    -------
    mu_l : numpy.ndarray
        Galactic longitude component of the proper motion.
    mu_b : numpy.ndarray
        Galactic latitude component of the proper motion.
    '''
    a = np.radians(np.copy(a))
    d = np.radians(np.copy(d))
    # Rotation frame maths from https://gea.esac.esa.int/archive/documentation/GDR2/
    # Data_processing/chap_cu3ast/sec_cu3ast_intro/ssec_cu3ast_intro_tansforms.html
    ag_p = np.array([[-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
                     [+0.4941094278755837, -0.4448296299600112, +0.7469822444972189],
                     [-0.8676661490190047, -0.1980763734312015, +0.4559837761750669]])

    r_icrs = np.array([[np.cos(a) * np.cos(d)], [np.sin(a) * np.cos(d)], [np.sin(d)]])
    r_gal = np.matmul(ag_p, r_icrs)

    l = np.arctan2(r_gal[1], r_gal[0])[0]
    b = np.arctan2(r_gal[2], np.sqrt(r_gal[0]**2 + r_gal[1]**2))[0]

    p_gal = np.array([[-np.sin(l)], [np.cos(l)], [0]])
    p_icrs = np.array([[-np.sin(a)], [np.cos(a)], [0]])

    q_gal = np.array([[-np.cos(l) * np.sin(b)], [-np.sin(l) * np.sin(b)], [np.cos(b)]])
    q_icrs = np.array([[-np.cos(a) * np.sin(d)], [-np.sin(a) * np.sin(d)], [np.cos(d)]])

    mu_icrs = p_icrs * mu_a + q_icrs * mu_d

    mu_gal = np.matmul(ag_p, mu_icrs)

    mu_l = np.matmul(np.transpose(p_gal), mu_gal)
    mu_b = np.matmul(np.transpose(q_gal), mu_gal)

    return mu_l, mu_b


def weighted_quantile(values, quantiles, sample_weight=None,
                      values_sorted=False, old_style=False):
    """
    Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    Adapted from https://stackoverflow.com/questions/21844024/weighted-
        percentile-using-numpy/29677616#29677616
    """
    values = np.array(values).flatten()
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight).flatten()
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'
    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]
    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be consistent with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)
