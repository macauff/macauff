# Licensed under a 3-clause BSD style license - see LICENSE
'''
Provides the framework for calculating theoretical perturbations due to faint, unresolved
objects blended with brighter sources, under the assumption that sources are fit with PSF
photometry while in a sky background-dominated regime such that noise is constant across
the detector.
'''

import itertools
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, gridspec
from matplotlib.colors import Normalize
from scipy.optimize import basinhopping, minimize
from scipy.special import erf  # pylint: disable=no-name-in-module

from macauff.misc_functions import make_pool

# Assume that usetex = False only applies for tests where no TeX is installed
# at all, instead of users having half-installed TeX, dvipng et al. somewhere.
usetex = not not shutil.which("tex")  # pylint: disable=unneeded-not
if usetex:
    plt.rcParams.update({"text.usetex": True, "text.latex.preamble": r"\usepackage{amsmath}"})


__all__ = ['FitPSFPerturbations']


class FitPSFPerturbations:
    """
    Class that derives a parameterisation of the effect of a hidden, blended
    contaminant within a brighter source in a photometric image, based on
    fitting the composite object with a single Gaussian PSF in the limit that
    sky background dominates and noise is constant across the image.
    """
    def __init__(self, psf_fwhm, d_di, d_li, n_pool, data_save_folder, plot_save_folder=None):
        """
        Initialise FitPSFPerturbations with necessary parameters.

        Parameters
        ----------
        psf_fwhm : float
            The full-width at half-maximum of the PSF to simulate fitting
            for perturbations.
        d_di : float
            Separation between perturber offsets, setting the precision of the
            simulated perturbations.
        d_li : float
            Separation between relative perturber fluxes, affecting the
            precision of the derived offset relations.
        n_pool : integer
            Number of parallel threads to use for ``multiprocessing`` calls.
        data_save_folder : string
            Relative or absolute path into which to save the data created by
            the fitting process.
        plot_save_folder : string
            Relative or absolute path, used to save plots generated as part
            of the PSF algorithm AUF perturbation parameters.
        """
        self.psf_fwhm = psf_fwhm
        self.psf_sig = self.psf_fwhm / (2 * np.sqrt(2 * np.log(2)))
        # Set the maximum offset to be the Rayleigh criterion, assuming
        # we would resolve the blended object outside of this all of the time.
        self.di = np.arange(0, 1.185*self.psf_fwhm, d_di)
        # Handle l=1 separately
        self.li = np.arange(0.15, 1-d_li+1e-10, d_li)
        self.n_pool = n_pool

        self.data_save_folder = data_save_folder
        self.plot_save_folder = plot_save_folder

        if not os.path.exists(self.data_save_folder):
            os.makedirs(self.data_save_folder)
        if self.plot_save_folder is not None and not os.path.exists(self.plot_save_folder):
            os.makedirs(self.plot_save_folder)

    def __call__(self, run_initial_ld, run_skew_fit, run_polynomial_fit, make_fit_plots,
                 draw_sim_num=500000):
        """
        Call function for FitPSFPerturbations.

        Parameters
        ----------
        run_initial_ld : boolean
            Flag for whether to run the initial fitting of perturbation at
            various relative flux-distance combinations if data already exist.
        run_skew_fit : boolean
            Toggle for whether to always run the fitting of each relative flux's
            perturbation as a function of perturber position with a skew normal.
        run_polynomial_fit : boolean
            Flag indicating whether to re-run fitting of skew normal parameters
            as a function of relative perturber flux even if already run
            previously.
        make_fit_plots : boolean
            Controls whether summary plot is created for parameterisation run.
        draw_sim_num : integer, optional
            Number of realisations to draw when plotting summary figure for
            goodness-of-fit distributions.
        """
        if self.plot_save_folder is None and not make_fit_plots:
            raise ValueError("plot_save_folder cannot be None if plots are "
                             "to be made. Please specify a folder for plots "
                             "to be saved into.")
        self.run_initial_ld = run_initial_ld
        self.run_skew_fit = run_skew_fit
        self.run_polynomial_fit = run_polynomial_fit
        self.make_fit_plots = make_fit_plots
        self.draw_sim_num = draw_sim_num

        if self.run_initial_ld or not os.path.exists(f'{self.data_save_folder}/dd_ld.npy'):
            self.make_initial_ld_perturbations()
        if self.run_skew_fit or not os.path.exists(f'{self.data_save_folder}/dd_skew_pars.npy'):
            self.fit_skew_distributions()
        if self.run_polynomial_fit or not np.all(
                [os.path.exists(f'{self.data_save_folder}/{q}.npy') for q in
                 ['ns', 'dd_ic', 'dd_params', 'dd_params_full', 'l_cut']]):
            self.fit_polynomial_parameterisations()

        if self.make_fit_plots:
            self.plot_fits()

    def make_initial_ld_perturbations(self):
        """
        Derive full perturbations due to additional blended object within PSF
        for a matrix of relative flux-positional offset combinations.
        """
        # Save d and delta-d, derived "properly", for each l/d combo
        dd = np.empty((len(self.li), len(self.di), 2), float)

        ij = itertools.product(np.arange(len(self.li)), np.arange(len(self.di)))
        iter_array = zip(ij, itertools.repeat([self.li, self.di, self.psf_sig]))
        with make_pool(self.n_pool) as pool:
            for return_values in pool.imap_unordered(
                    self.min_parallel_dd_fit, iter_array,
                    chunksize=int(len(self.li) * len(self.di)/self.n_pool)):
                (i, j), res_xy = return_values
                res = res_xy.x
                dd[i, j, 0] = np.sqrt(res[0]**2 + res[1]**2)
                dd[i, j, 1] = self.di[j]

        pool.join()
        np.save(f'{self.data_save_folder}/dd_ld.npy', dd)

    def min_parallel_dd_fit(self, iterable):
        """
        Wrapper for minimisation routine used in fitting for positional
        offset caused by fitting two blended sources with a single PSF model.

        Parameters
        ----------
        iterable : list
            List of inputs from ``multiprocessing``, including indices into
            perturber position and flux arrays, the perturber parameter
            arrays, and the width of the Gaussian used in modelling the PSF.

        Returns
        -------
        i : integer
            The index into the relative flux array for this particular call.
        j : integer
            The perturber positional offset array index.
        res : ~`scipy.optimize.OptimizeResult`
            The `scipy` optimisation output.
        """
        (i, j), (l, di, c) = iterable
        l, d = l[i], di[j]
        res = minimize(self.min_dd_fit_xy, x0=np.array([l, d * l / (1 + l), 0]),
                       args=(1, [d], [1e-4], [l], c),
                       jac=True, method='newton-cg', hess=self.hess_dd_fit_xy,
                       options={'xtol': 1e-15})

        return (i, j), res

    def min_dd_fit_xy(self, p, l, xis, yis, lis, sig):
        """
        Minimisation function for fitting one PSF model to two or more simulated
        sources, including the first-order derivatives with respect to position
        offset and flux brightening of the single PSF.

        Parameters
        ----------
        p : list
            List of the current values of perturbation offset and
            flux brightening caused by all perturbers hidden within the
            central object's PSF.
        l : float
            Flux of the central object.
        xis : numpy.ndarray
            x-axis positions of all perturbers being fit with a single PSF.
        yis : numpy.ndarray
            Positions of perturbers in the opposing orthogonal axis.
        lis : numpy.ndarray
            Flux of perturbers relative to the central source flux ``l``.
        sig : float
            The Gaussian sigma of the PSFs being fit.

        Returns
        -------
        numpy.ndarray
            Negative log-likelihood and its derivative with respect to position
            and flux offsets.
        """
        dx, dy, dl = p
        di_dd = np.array([xis - dx, yis - dy])
        dd = np.array([dx, dy])
        fx = ((l+dl) * np.sum(lis * self.psi(di_dd, sig)) +
              l * (l+dl) * self.psi(dd, sig) - 0.5*(l+dl)**2)
        dfx = np.array([(l+dl) * np.sum(lis * (xis - dx)/2/sig**2 * self.psi(di_dd, sig)) -
                        l * (l+dl) * dx/2/sig**2 * self.psi(dd, sig),
                        (l+dl) * np.sum(lis * (yis - dy)/2/sig**2 * self.psi(di_dd, sig)) -
                        l * (l+dl) * dy/2/sig**2 * self.psi(dd, sig),
                        -(l + dl) + np.sum(lis * self.psi(di_dd, sig)) + l * self.psi(dd, sig)])

        return -1*fx, -1*dfx

    def hess_dd_fit_xy(self, p, l, xis, yis, lis, sig):
        """
        Second-order derivatives of the fit of one PSF model to two or more
        simulated sources, following the minimisation routine ``min_dd_fit_xy``.

        Parameters
        ----------
        p : list
            List of the current values of perturbation offset and
            flux brightening caused by all perturbers hidden within the
            central object's PSF.
        l : float
            Flux of the central object.
        xis : numpy.ndarray
            x-axis positions of all perturbers being fit with a single PSF.
        yis : numpy.ndarray
            Positions of perturbers in the opposing orthogonal axis.
        lis : numpy.ndarray
            Flux of perturbers relative to the central source flux ``l``.
        sig : float
            The Gaussian sigma of the PSFs being fit.

        Returns
        -------
        numpy.ndarray
            Negative of the Hessian of the log-likelihood fit between one PSF
            model and multiple injected blended sources.
        """
        dx, dy, dl = p
        di_dd = np.array([xis - dx, yis - dy])
        dd = np.array([dx, dy])
        dx2 = ((l+dl) * np.sum(lis * self.psi(di_dd, sig) * ((xis - dx)**2/2/sig**2 - 1)) / 2 /
               sig**2 + l * (l+dl) * self.psi(dd, sig) * (dx**2 / 2 / sig**2 - 1) / 2 / sig**2)
        dxdy = ((l+dl) * np.sum(lis * self.psi(di_dd, sig) * (xis - dx) * (yis - dy)/4/sig**4) +
                l * (l+dl) * self.psi(dd, sig) * dx*dy / 4 / sig**4)
        dy2 = ((l+dl) * np.sum(lis * self.psi(di_dd, sig) * ((yis - dy)**2/2/sig**2 - 1)) / 2 /
               sig**2 + l * (l+dl) * self.psi(dd, sig) * (dy**2 / 2 / sig**2 - 1) / 2 / sig**2)
        dxdl = np.sum(lis*(xis - dx)/2/sig**2 *
                      self.psi(di_dd, sig)) - l * dx/2/sig**2 * self.psi(dd, sig)
        dydl = np.sum(lis*(yis - dy)/2/sig**2 *
                      self.psi(di_dd, sig)) - l * dy/2/sig**2 * self.psi(dd, sig)
        dl2 = -1

        return -1*np.array([[dx2, dxdy, dxdl], [dxdy, dy2, dydl], [dxdl, dydl, dl2]])

    def psi(self, x, sig):
        r"""
        Calculate Psi, the convolution of two PSFs :math:`\phi` (equation 2,
        Plewa & Sari, 2018, MNRAS, 476, 4372).

        Parameters
        ----------
        x : numpy.ndarray
            Separation at which to evaluate the phi convolution. The first
            axis should contain the cartesian x- and y-axis positions, with
            each unique object filling the subsequent axes.
        sig : float
            Gaussian sigma of the PSFs being convolved.

        Returns
        -------
        numpy.ndarray
            The convolution of two PSFs :math:`\phi` evaluated at `x`.
        """
        return np.exp(-0.25 * np.sum(x**2, axis=0) / sig**2)

    def fit_skew_distributions(self):
        """
        Function for fitting sets of position offsets as a function of perturber
        position with skew-normal distributions, accounting for a break at small
        offsets where relationship becomes linear, for each value of perturber
        relative flux.
        """
        dd = np.load(f'{self.data_save_folder}/dd_ld.npy')
        dd_skew_pars = np.empty((len(self.li), 5), float)

        # Fit each l distribution for its parameters as a function of d.
        for i, li in enumerate(self.li):
            _x, _y = dd[i, 1:, 1] / self.psf_sig, dd[i, 1:, 0] / self.psf_sig
            slices = np.where(np.abs((_y - _x * li/(1 + li))/_y) > 0.01)[0]
            if len(slices) == 0:
                dd_skew_pars[i, :-1] = 0
                dd_skew_pars[i, -1] = _x[-1] + 0.1
                print(r'All derived perturbations within 1% of linear fit for '
                      rf'relative flux of {li[i]}. No skew-normal fit needed.')
                continue
            cutr = _x[slices[0]]

            x = _x[slices[0]:]
            y = _y[slices[0]:]

            w = np.ones_like(x)
            w[:10] = 100

            n_pools = self.n_pool
            n_overloop = 2
            niters = 150
            counter = np.arange(0, n_pools*n_overloop)
            xy_step = 0.1
            temp = 0.01
            x0 = None
            method = 'L-BFGS-B'
            min_kwarg = {'method': method, 'args': (x, y, w, self.li[i]), 'jac': True}
            iter_rep = itertools.repeat([min_kwarg, niters, x0, xy_step, temp])
            iter_group = zip(counter, iter_rep)
            res = None
            min_val = None
            with make_pool(n_pools) as pool:
                for return_res in pool.imap_unordered(self.dd_fitting_wrapper, iter_group,
                                                      chunksize=n_overloop):
                    if min_val is None or return_res.fun < min_val:
                        res = return_res
                        min_val = return_res.fun
            pool.join()

            dd_skew_pars[i, :-1] = res.x
            dd_skew_pars[i, -1] = cutr

        # With degeneracy between T/sig/alpha, we need to ensure values are of
        # the correct sign. T and sig should always be positive, so if negative
        # their sign needs correcting. However, if sig is negative then alpha will
        # also be of the wrong sign, to correct for a * (x - u) / c within the
        # skew-normal CDF. Thus, if either T or sig is negative, correct all
        # three values.
        q = (dd_skew_pars[:, 3] < 0) | (dd_skew_pars[:, 0] < 0)
        dd_skew_pars[q, 0] *= -1
        dd_skew_pars[q, 2] *= -1
        dd_skew_pars[q, 3] *= -1

        np.save(f'{self.data_save_folder}/dd_skew_pars.npy', dd_skew_pars)

    def dd_fitting_wrapper(self, iterable):
        """
        Wrapper for fitting the parameters for the skew-normal distribution
        to a position offset-perturber position relationship of a particular
        flux value of perturber, via basinhopping.

        Parameters
        ----------
        iterable : list
            List of values passed through ``multiprocessing``, including the
            index into the relative flux array, the keywords passed through
            to `basinhopping` (containing the method used, the arguments to
            pass to `basinhopping`, and things like whether to compute the
            Jacobian), the number of basin hop iterations, starting argument
            values, and basin hop step size and "temperature".

        Returns
        -------
        res : ~`scipy.optimize.OptimizeResult`
            The `scipy` optimisation output containing the best-fit
            skew-normal parameters from all basins.
        """
        def sum_one_skew(p, x, y, w, l):
            """
            Calculate the weighted sum-of-square-residuals between the
            skew-normal and input data values, as well as its derivative
            with respect to the parameters of the skew-normal.

            Parameters
            ----------
            p : list
                Input values of Gaussian sigma, mean, skewness, and amplitude
                for this particular skew-normal.
            x : numpy.ndarray
                Perturber distance from central source.
            y : numpy.ndarray
                Perturbation offsets, as calculated from full model residual
                fitting.
            w : numpy.ndarray
                Weights for each data point.
            l : numpy.ndarray
                The relative flux of the perturber.

            Returns
            -------
            numpy.ndarray
                Weighted sum of data-model fits and derivatives of the
                goodness-of-fit parameter.
            """
            f, df = self.fit_one_skew(p, x, l)
            return np.sum((y - f)**2 * w), np.array([np.sum(-2 * w * (y - f) * q) for q in df])

        rng = np.random.default_rng()
        _, (min_kwarg, niters, x0, s, t) = iterable
        if x0 is None:
            # sigma, mu, alpha, T
            x0 = [rng.uniform(0, 3), rng.uniform(0, 4), rng.uniform(-1, 1), rng.uniform(0, 1)]
        res = basinhopping(sum_one_skew, x0, minimizer_kwargs=min_kwarg, niter=niters, T=t,
                           stepsize=s)

        return res

    def fit_one_skew(self, p, x, l):
        """
        Function used in the fitting of a skew-normal distribution, calculating
        the value of the skew-normal and its derivative with respect to its
        parameters at the given value.

        Parameters
        ----------
        p : list
            List of skew-normal parameters, the Gaussian sigma, mean, skewness
            and amplitude respectively.
        x : numpy.ndarray
            Input values at which to evaluate the skew-normal distribution.
        l : float
            Relative flux of perturber for which perturbation-position relations
            are being fit with the skew-normal distribution.

        Returns
        -------
        f : numpy.ndarray
            The skew-normal distribution evaluated at all ``x``.
        df : numpy.ndarray
            The derivative of `f` with respect to its Gaussian sigma, mean,
            skewness, and amplitude respectively.
        """
        def calc_psi_pdf(x):
            r"""
            Probability density function :math:`\psi` of the standard normal
            distribution, used in calculating the skew-normal distribution.

            Parameters
            ----------
            x : numpy.ndarray
                Values at which to evaluate the standard normal distribution.

            Returns
            -------
            numpy.ndarray
                :math:`\psi` at every `x` value.
            """
            return 1/np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2)

        def calc_psi_cdf(x):
            r"""
            Cumulative distribution function, :math:`\psi` integrated from
            :math:`-\infty` to `x`, of the standard normal distribution, used
            in calculating the skew-normal distribution.

            Parameters
            ----------
            x : numpy.ndarray
                Values at which to evaluate the CDF of the standard normal
                distribution.

            Returns
            -------
            numpy.ndarray
                :math:`\psi` integrated from :math:`-\infty` to `x`, at every
                `x`.
            """
            return 0.5 * (1 + erf(x / np.sqrt(2)))

        c, u, a, t = p

        x_ = (x - u) / c

        psi_pdf = calc_psi_pdf(x_)
        psi_pdf_skew = calc_psi_pdf(a * x_)
        psi_cdf_skew = calc_psi_cdf(a * x_)
        dx_dc = -(x - u) / c**2
        dx_du = -1 / c
        f = 2 * t / c * l * psi_pdf * psi_cdf_skew
        dfdc = (2 * t / c * l * psi_pdf * (dx_dc * (-x_ * psi_cdf_skew +
                                                    psi_pdf_skew * a) - psi_cdf_skew/c))
        dfdu = 2 * t / c * l * psi_pdf * dx_du * (-psi_cdf_skew * x_ + a * psi_pdf_skew)
        dfda = 2 * t / c * l * psi_pdf * psi_pdf_skew * x_
        dfdt = 2 / c * l * psi_pdf * psi_cdf_skew

        df = np.array([dfdc, dfdu, dfda, dfdt])

        return f, df

    def fit_polynomial_parameterisations(self):
        """
        Function controlling the derivation of power law polynomial fits to
        skew-normal distribution parameters as a function of the relative flux
        of the perturber.
        """
        dd = np.load(f'{self.data_save_folder}/dd_ld.npy')
        dd_skew_pars = np.load(f'{self.data_save_folder}/dd_skew_pars.npy')

        l_cut = [0.15, self.li[self.li < 0.75][np.argmin(dd_skew_pars[self.li < 0.75, 0])],
                 self.li[np.argmin(dd_skew_pars[:, 2])]]

        ns = np.arange(4, 25)
        dd_params = np.empty((len(ns), dd_skew_pars.shape[1], np.amax(ns), 2), float)

        for k, q in enumerate([self.li <= l_cut[1], (self.li > l_cut[1]) & (self.li < l_cut[2])]):
            ij = np.arange(len(ns))
            iter_array = zip(ij, itertools.repeat([ns, dd_skew_pars[q, :], self.li[q],
                                                   self.di[-1]/self.psf_sig]))
            with make_pool(self.n_pool) as pool:
                for results in pool.imap_unordered(self.min_parallel_dd_param_fit, iter_array,
                                                   chunksize=max(1, int(len(ns)/self.n_pool))):
                    j, n, resses = results
                    for i, res in enumerate(resses):
                        dd_params[j, i, :n, k] = res

            pool.join()

        # Keep track of the total goodness-of-fit values across all di-li
        # combinations, as well as the goodness-of-fits just for individual
        # lis, across di.
        dd_ic = np.zeros((len(ns), 2), float)
        dd_x2s = np.empty((len(ns), len(self.li), 2), float)

        for j, n in enumerate(ns):
            dd_x2 = [0, 0]
            for i, li in enumerate(self.li):
                x = dd[i, :, 1] / self.psf_sig
                y = dd[i, :, 0] / self.psf_sig
                ddparams = self.return_ddparams(li, l_cut, dd_params, n, j)
                q = (~np.isnan(x)) & (~np.isnan(y))
                x2_w = np.sum((y[q] - self.dd_combined_fit(ddparams, x[q], li,
                                                           l_cut[2]))**2 / y[q]**2)
                x2_nw = np.sum((y[q] - self.dd_combined_fit(ddparams, x[q], li, l_cut[2]))**2)
                dd_x2[0] += x2_w
                dd_x2[1] += x2_nw
                dd_x2s[j, i, :] = [x2_w, x2_nw]

            dd_ic[j, :] = dd_x2

        np.save(f'{self.data_save_folder}/ns.npy', ns)
        np.save(f'{self.data_save_folder}/dd_ic.npy', dd_ic)
        np.save(f'{self.data_save_folder}/dd_x2s.npy', dd_x2s)
        np.save(f'{self.data_save_folder}/dd_params_full.npy', dd_params)
        np.save(f'{self.data_save_folder}/l_cut.npy', l_cut)

        # Use the relative sum-of-square-residuals as the metric for deciding
        # which polynomial order is the best, but with an AIC complexity factor.
        n_ind = np.unravel_index(np.argmin(dd_ic[:, 0] + 2*ns), dd_ic[:, 1].shape)[0]
        dd_params_final = dd_params[n_ind, :, :ns[n_ind]]
        np.save(f'{self.data_save_folder}/dd_params.npy', dd_params_final)

    def min_parallel_dd_param_fit(self, iterable):
        """
        Wrapper function for calculating the best-fit polynomial parameters
        for describing skew-normal distribution parameters as a function of
        relative perturber flux.

        Parameters
        ----------
        iterable : list
            The parameters passed through ``multiprocessing``, including
            index of polynomial order, the list of polynomial orders being
            fit across all parallel threads, the set of skew-normal
            distribution parameters at all fluxes, the fluxes evaluated for
            perturbation offsets, and the maximum perturber position.

        Returns
        -------
        j : integer
            The index into the polynomial order array.
        n : integer
            The polynomial order being fit.
        resses : list
            A list containing the best-fit polynomial weights for each
            parameters in the skew-normal distributuon (Gaussian sigma, mean,
            skewness, amplitude) and linear-regime radius cutoff.
        """
        def sum_poly(p, x, y):
            """
            Computes the sum-of-squares goodness-of-fit of a polynomial to
            some data, and returns it and its first-order derivative with
            respect to the polynomial parameters.

            Parameters
            ----------
            p : list
                The polynomial weights.
            x : numpy.ndarray
                Values at which to evaluate the polynomial
            y : numpy.ndarray
                The data points, each of which corresponds to an element in
                ``x``.

            Returns
            -------
            numpy.ndarray
                The sum-of-squares between the model and data `y`, as well as
                the derivative of the sum-of-squares with respect to each
                parameter in the list `p`.
            """
            f, df = self.fit_poly(p, x)
            return np.sum((y - f)**2), np.array([np.sum(-2 * (y - f) * q) for q in df])

        j, (ns, dd_skew_pars, li, dcut) = iterable
        n = ns[j]
        resses = []
        for i in range(dd_skew_pars.shape[1]):
            # Filter for any "non-fit" fluxes, which are indicated by having
            # a linear-regime cutoff radius larger than the maximum extent
            # of dd probed.
            q = dd_skew_pars[:, -1] < dcut
            resses.append(minimize(sum_poly, x0=[0.5]*n,
                                   args=(li[q], dd_skew_pars[q, i]),
                                   jac=True, method='L-BFGS-B', options={'ftol': 1e-12}).x)

        return j, n, resses

    def fit_poly(self, p, x):
        r"""
        Wrapper function for evaluating a polynomial with weights `p` at
        all `x`, :math:`y_j = \sum_{i=0}^N p_i x_j^i`, and its derivative with
        respect to each parameter `p`.

        Parameters
        ----------
        p : list or numpy.ndarray
            The weights of the polynomial.
        x : numpy.ndarray
            Values at which to calculate the polynomial.

        Returns
        -------
        y : numpy.ndarray
            The evaluation of the polynomial with parameters `p` at
            each `x`.
        dy : list
            The derivative of `y` with respect to each `p_i` parameter.
        """
        x = np.atleast_1d(x)
        y = np.empty((len(p), len(x)), float)
        # y = sum_i=0^N p_i x**i
        y[0, :] = p[0]
        for i in np.arange(1, len(p)):
            y[i, :] = y[i-1, :] * x / p[i-1] * p[i]

        y = np.sum(y, axis=0)

        dy = [np.ones_like(x)]
        for _ in np.arange(1, len(p)):
            dy.append(dy[-1] * x)

        return y, dy

    def return_ddparams(self, li, l_cut, dd_params, n, n_ind):
        """
        Convenience function for deriving skew-normal distribution parameters
        as a function of perturber flux from fit polynomial distributions.

        Parameters
        ----------
        li : float
            The flux of the perturber relative to the flux of the central source.
        l_cut : list or numpy.ndarray
            List of key cut-off relative fluxes, at which parameterisations of
            perturber effects change.
        dd_params : numpy.ndarray
            Array of polynomial weights for each skew-normal distribution
            parameter, for multiple polynomial orders, for parameterisations
            above and below ``l_cut[1]``.
        n : integer
            Number of polynomial terms being used to calculate the skew-normal
            distribution values.
        n_ind : integer
            Index of the chosen number of polynomial terms in ``dd_params``.

        Returns
        -------
        dd_skew_params : numpy.ndarray
            Values of the skew-normal sigma, mean, skewness, and amplitude,
            evaluated at `li`, from a polynomial of order `N+1`.
        """
        dd_skew_params = []
        p = 0 if li <= l_cut[1] else 1
        for q in range(dd_params.shape[1]):
            dd_skew_params.append(self.fit_poly(dd_params[n_ind, q, :n, p], li)[0])
        dd_skew_params = np.array(dd_skew_params)
        return dd_skew_params

    def dd_combined_fit(self, p, x, l, l_cut):
        """
        Wrapper function for calculating the combined description of the
        perturber offsets as a linear relation up to a particular offset
        and a skew-normal distribution beyond that.

        Parameters
        ----------
        p : list
            The values of the skew-normal distribution (sigma, mean, skewness,
            amplitude), as well as the cutoff radius inside which to follow
            a linear relation.
        x : numpy.ndarray
            Values at which to evaluate the offset due to a blended perturber.
        l : float
            Relative flux of the perturber as compared with the central object.
        l_cut : float
            Cutoff relative flux, above which no skew-normal distribution is
            used at all.

        Returns
        -------
        y : numpy.ndarray
            The composite perturbation function, linear within ``p[-1]`` and
            a skew-normal distribution outside.
        """
        if l >= l_cut:
            return x * l / (l + 1)
        y = np.empty_like(x)
        q = np.where(x <= p[-1])
        y[q] = x[q] * l / (l + 1)
        q = np.where(x > p[-1])
        y[q] = self.fit_one_skew(p[:-1], x[q], l)[0]
        return y

    def plot_fits(self):  # pylint: disable=too-many-statements
        """
        Visualisation function, plotting the various parameterisations fit in
        FitPSFPerturbations, enabling quality checks to be carried out.
        """
        dd = np.load(f'{self.data_save_folder}/dd_ld.npy')
        dd_skew_pars = np.load(f'{self.data_save_folder}/dd_skew_pars.npy')
        ns = np.load(f'{self.data_save_folder}/ns.npy')
        dd_ic = np.load(f'{self.data_save_folder}/dd_ic.npy')
        dd_x2s = np.load(f'{self.data_save_folder}/dd_x2s.npy')
        l_cut = np.load(f'{self.data_save_folder}/l_cut.npy')
        dd_params_full = np.load(f'{self.data_save_folder}/dd_params_full.npy')
        n_ind = np.unravel_index(np.argmin(dd_ic[:, 0] + 2*ns), dd_ic[:, 1].shape)[0]

        plt.figure('figure', figsize=(40, 16))
        gs = gridspec.GridSpec(2, 4)
        norm = Normalize(vmin=self.li[0], vmax=self.li[len(self.li)-1])
        ax = plt.subplot(gs[0, 0])
        # Work backwards from maximum li to minimum in 0.05 li steps.
        for i in range(len(self.li)-1, -1, -int(np.ceil(0.05/(self.li[1] - self.li[0])))):
            x = dd[i, :, 1] / self.psf_sig
            y = dd[i, :, 0] / self.psf_sig
            q = (~np.isnan(x)) & (~np.isnan(y))
            x, y = x[q], y[q]
            ax.plot(x, y, ls='-', c=cm.viridis(norm(self.li[i])))  # pylint: disable=no-member

            ddparams = self.return_ddparams(self.li[i], l_cut, dd_params_full,
                                            ns[n_ind], n_ind)
            ax.plot(x, self.dd_combined_fit(ddparams, x, self.li[i], l_cut[2]), ls='--',
                    c=cm.viridis(norm(self.li[i])))  # pylint: disable=no-member

        if usetex:
            ax.set_xlabel(r'$x / \sigma_\mathrm{PSF}$')
            ax.set_ylabel(r'$\Delta x / \sigma_\mathrm{PSF}$')
        else:
            ax.set_xlabel(r'x / sigma_PSF')
            ax.set_ylabel(r'Delta x / sigma_PSF')

        ax = plt.subplot(gs[0, 1])
        if usetex:
            label_list = [r'$\sigma$', r'$\mu$', r'$\alpha$', r'$T$', r'$r_\mathrm{c}$']
        else:
            label_list = [r'sigma', r'mu', r'alpha', r'T', r'r_c']
        for i, (label, c, c2) in enumerate(zip(label_list, ['k', 'r', 'b', 'g', 'purple'],
                                           ['gray', 'orange', 'aquamarine', 'olive', 'violet'])):
            ax.plot(self.li, dd_skew_pars[:, i], ls='-', c=c, label=label)
            q = self.li <= l_cut[1]
            ax.plot(self.li[q], self.fit_poly(dd_params_full[n_ind, i, :ns[n_ind], 0],
                                              self.li[q])[0], ls='--', c=c2, lw=4)
            q = (self.li > l_cut[1]) & (self.li < l_cut[2])
            ax.plot(self.li[q], self.fit_poly(dd_params_full[n_ind, i, :ns[n_ind], 1],
                                              self.li[q])[0], ls='-.', c=c2, lw=4)

        ax.legend(fontsize=14, ncol=2)
        ax.set_xlabel('Relative perturber flux')
        ax.set_ylabel('Perturbation terms')

        ax = plt.subplot(gs[0, 2])
        ax1 = ax.twinx()
        ax.plot(ns, np.log10(dd_ic[:, 1]), ls='-', c='k')
        ax.axvline(ns[n_ind], ls='-', c='k')
        ax1.plot(ns, np.log10(dd_ic[:, 0]), ls='-', c='r')

        ax.set_xlabel('Order of polynomial fit')
        if usetex:
            ax.set_ylabel(r'$\mathrm{log}_{10}(\sum_i (y_i - f(x_i))^2)$')
            ax1.set_ylabel(r'$\mathrm{log}_{10}(\sum_i \frac{(y_i - f(x_i))^2}{y_i^2})$', c='r')
        else:
            ax.set_ylabel(r'log10(sum_i (y_i - f(x_i))**2)')
            ax1.set_ylabel(r'log10(sum_i (y_i - f(x_i))**2 / y_i**2)', c='r')

        ax = plt.subplot(gs[1, 0])
        ax1 = ax.twinx()
        ax.plot(self.li, np.log10(dd_x2s[n_ind, :, 1]), ls='-', c='k')
        ax.plot(self.li, np.log10(dd_x2s[n_ind, :, 0]), ls='-', c='r')

        ax.set_xlabel('Relative perturber flux')
        if usetex:
            ax.set_ylabel(r'$\mathrm{log}_{10}(\sum_i (y_i - f(x_i))^2)$')
            ax1.set_ylabel(r'$\mathrm{log}_{10}(\sum_i \frac{(y_i - f(x_i))^2}{y_i^2})$', c='r')
        else:
            ax.set_ylabel(r'log10(sum_i (y_i - f(x_i))**2)')
            ax1.set_ylabel(r'log10(sum_i (y_i - f(x_i))**2 / y_i**2)', c='r')

        ax = plt.subplot(gs[1, 1])
        max_perc_diff = np.empty_like(self.li)
        for i, li in enumerate(self.li):
            x = dd[i, 1:, 1] / self.psf_sig
            y = dd[i, 1:, 0] / self.psf_sig
            q = (~np.isnan(x)) & (~np.isnan(y))
            x, y = x[q], y[q]
            ddparams = self.return_ddparams(li, l_cut, dd_params_full, ns[n_ind], n_ind)
            max_perc_diff[i] = np.amax(np.abs((self.dd_combined_fit(ddparams, x, li,
                                               l_cut[2]) - y) / (y + 1e-5)))
        ax.plot(self.li, max_perc_diff, ls='-', c='k')

        ax.set_xlabel('Relative perturber flux')
        if usetex:
            ax.set_ylabel(r'Max $\lvert\frac{\Delta x_\mathrm{f} / \sigma_\mathrm{PSF} - '
                          r'\Delta x_\mathrm{t} / \sigma_\mathrm{PSF}}{\Delta x_\mathrm{t} / '
                          r'\sigma_\mathrm{PSF}}\lvert$')
        else:
            ax.set_ylabel(r'Max abs((Delta x_f / sigma_PSF - Delta x_t / sigma_PSF) / '
                          r'(Delta x_t / sigma_PSF))')

        ax = plt.subplot(gs[1, 2])
        max_abs_diff = np.empty_like(self.li)
        for i, li in enumerate(self.li):
            x = dd[i, 1:, 1] / self.psf_sig
            y = dd[i, 1:, 0] / self.psf_sig
            q = (~np.isnan(x)) & (~np.isnan(y))
            x, y = x[q], y[q]
            ddparams = self.return_ddparams(li, l_cut, dd_params_full, ns[n_ind], n_ind)
            max_abs_diff[i] = np.amax(np.abs(self.dd_combined_fit(ddparams, x, li, l_cut[2]) - y))
        ax.plot(self.li, max_abs_diff, ls='-', c='k')

        ax.set_xlabel('Relative perturber flux')
        if usetex:
            ax.set_ylabel(r'Max $\lvert\Delta x_\mathrm{f} / \sigma_\mathrm{PSF} - '
                          r'\Delta x_\mathrm{t} / \sigma_\mathrm{PSF}\lvert$')
        else:
            ax.set_ylabel(r'Max abs(Delta x_f / sigma_PSF - Delta x_t / sigma_PSF)')

        diff = np.empty((self.draw_sim_num, 6), float)
        ij = np.arange(diff.shape[0])
        iter_array = zip(ij, itertools.repeat([self.psf_fwhm, self.psf_sig, l_cut,
                                               dd_params_full, ns[n_ind], n_ind]))
        with make_pool(self.n_pool) as pool:
            for results in pool.imap_unordered(self.loop_ind_fit, iter_array,
                                               chunksize=int(diff.shape[0]/self.n_pool)):
                i, dx, _li, x, ddparams = results
                dx_fit = self.dd_combined_fit(ddparams, np.array([x]), _li, l_cut[2])
                diff[i, 0] = np.amax(np.abs((dx_fit - dx)/(dx + 1e-3)))
                diff[i, 1] = np.amax(np.abs(dx_fit - dx))
                diff[i, 2] = _li
                diff[i, 3] = x
                diff[i, 4] = dx
                diff[i, 5] = dx_fit[0]

        pool.join()

        ax = plt.subplot(gs[0, 3])
        hist, bins = np.histogram(diff[:, 0], bins='auto')
        ax.plot(bins, np.append(hist, 0), 'k-')
        if usetex:
            ax.set_xlabel(r'$\lvert\frac{\Delta x_\mathrm{f} / \sigma_\mathrm{PSF} - '
                          r'\Delta x_\mathrm{t} / \sigma_\mathrm{PSF}}{\Delta x_\mathrm{t} / '
                          r'\sigma_\mathrm{PSF}}\lvert$')
        else:
            ax.set_xlabel(r'Max abs((Delta x_f / sigma_PSF - Delta x_t / sigma_PSF) / '
                          r'(Delta x_t / sigma_PSF))')
        ax.set_ylabel('N')

        ax = plt.subplot(gs[1, 3])
        hist, bins = np.histogram(diff[:, 1], bins='auto')
        ax.plot(bins, np.append(hist, 0), 'k-')
        if usetex:
            ax.set_xlabel(r'$\lvert\Delta x_\mathrm{f} / \sigma_\mathrm{PSF} - '
                          r'\Delta x_\mathrm{t} / \sigma_\mathrm{PSF}\lvert$')
        else:
            ax.set_xlabel(r'Max abs(Delta x_f / sigma_PSF - Delta x_t / sigma_PSF)')
        ax.set_ylabel('N')

        plt.tight_layout()
        plt.savefig(f'{self.plot_save_folder}/dd_params_visualisation.pdf')

    def loop_ind_fit(self, iterable):
        """
        Wrapper function for fitting perturbation offset and flux brightening
        of one PSF model fit to multiple unresolved objects, for arbitrary
        relative flux and position.

        Parameters
        ----------
        iterable : list
            List passed through ``multiprocessing``, containing the index of
            the fit, as well as PSF FWHM and Gaussian sigma, maximum relative
            flux at which skew-normal distributions still describe
            perturbation offsets, the array of skew-normal parameters as a
            function of relative flux, the polynomial order being fit, and
            an index into the polynomial order array.

        Returns
        -------
        i : integer
            The index of the fit, looping through the number of simulations
            to draw.
        dx : float
            The derived perturbation offset, normalised to the PSF sigma.
        li : float
            The randomly drawn relative flux of the perturbing object.
        x : float
            Position of the perturbing object relative to the central source.
        ddparams : numpy.ndarray
            The calculated skew-normal distribution parameters evaluated at
            the simulated `li` for the chosen polynomial order.
        """
        rng = np.random.default_rng()
        i, (psf_fwhm, psf_sig, t_cut, dd_params, n, n_ind) = iterable
        li = rng.uniform(0.15, 1)
        x_ = 2 * psf_fwhm
        while x_ > 1.185*psf_fwhm:
            x_ = rng.rayleigh(scale=psf_sig)
        res = minimize(self.min_dd_fit_xy, x0=np.array([li, x_ * li / (1 + li), 0]),
                       args=(1, [x_], [1e-4], [li], psf_sig),
                       jac=True, method='newton-cg', hess=self.hess_dd_fit_xy,
                       options={'xtol': 1e-15})
        dx = np.sqrt(res.x[0]**2 + res.x[1]**2) / psf_sig
        x = x_ / psf_sig

        ddparams = self.return_ddparams(li, t_cut, dd_params, n, n_ind)

        return i, dx, li, x, ddparams
