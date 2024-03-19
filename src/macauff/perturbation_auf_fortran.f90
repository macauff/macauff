! Licensed under a 3-clause BSD style license - see LICENSE

module perturbation_auf_fortran
! This module provides the Fortran code for the handling of the creation of perturbation
! component of the Astrometric Uncertainty Function.

implicit none

integer, parameter :: dp = kind(0.0d0)  ! double precision
real(dp), parameter :: pi = 4.0_dp*atan(1.0_dp)

contains

subroutine get_density(a_ax1, a_ax2, b_ax1, b_ax2, maxdist, counts)
    ! Calculate the number of sources in a given catalogue within a specified radius of each source.
    integer, parameter :: dp = kind(0.0d0)  ! double precision
    ! Sky coordinates for catalogues a and b.
    real(dp), intent(in) :: a_ax1(:), a_ax2(:), b_ax1(:), b_ax2(:)
    ! Separation to consider the number of objects within.
    real(dp), intent(in) :: maxdist
    ! Number of objects within maxdist of each catalogue a source.
    integer, intent(out) :: counts(size(a_ax1))
    ! Loop counters.
    integer :: i, j
    ! Sky separations.
    real(dp) :: dist, dx, d_ax1, d_ax2

    counts = 0
!$OMP PARALLEL DO DEFAULT(NONE) PRIVATE(i, j, dist, dx, d_ax1, d_ax2) SHARED(a_ax1, a_ax2, b_ax1, b_ax2, counts, maxdist)
    do j = 1, size(a_ax1)
        do i = 1, size(b_ax1)
            ! Difference in latitude is always just the absolute difference
            d_ax2 = abs(a_ax2(j) - b_ax2(i))
            if (d_ax2 <= maxdist) then
                ! Need reduction of Haversine formula for longitude difference, remembering to convert to degrees:
                d_ax1 = 2.0_dp * asin(abs(cos(a_ax2(j) / 180.0_dp * pi) * &
                                      sin((a_ax1(j) - b_ax1(i))/2.0_dp / 180.0_dp * pi))) * 180.0_dp / pi
                if (d_ax1 <= maxdist) then
                    call haversine(a_ax1(j), b_ax1(i), a_ax2(j), b_ax2(i), dist)
                    if (dist <= maxdist) then
                        counts(j) = counts(j) + 1
                    end if
                end if
            end if
        end do
    end do
!$OMP END PARALLEL DO

end subroutine get_density

subroutine get_circle_area_overlap(cat_ax1, cat_ax2, density_radius, hull_ax1, hull_ax2, seed, circ_overlap_area)
    ! Calculates the amount of circle overlap with a rectangle of particular coordinates. Adapts
    ! code provided by B. Retter, from Retter, Hatchell & Naylor (2019, MNRAS, 487, 887).
    integer, parameter :: dp = kind(0.0d0)  ! double precision
    ! Coordinates in orthogonal sky axes, and radius of, circles to calculate the overlap with sky position of.
    real(dp), intent(in) :: cat_ax1(:), cat_ax2(:), density_radius
    ! Sky coordinates of the points defining the sky cutout region. Note that this should be cyclical, such
    ! that hull_ax1(1) == hull_ax1(size(hull_ax1)), hull_ax2(1) == hull_ax2(size(hull_ax2)).
    real(dp), intent(in) :: hull_ax1(:), hull_ax2(:)
    ! RNG seed.
    integer, intent(in) :: seed(:, :)
    ! Relative amount of circle inside rectangle for each unique point.
    real(dp), intent(out) :: circ_overlap_area(size(cat_ax1))

    ! Loop counters.
    integer :: i, j, k
    ! Flags for forks in the logic of calculating circle area.
    logical :: circle_too_near_edge, hull_point_inside_circle
    ! Area of circle inside rectangle, coordinates and variables of various calculations.
    real(dp) :: area, x0, y0, x1, y1, x2, y2, cross_prod, dot_prod, fraction
    ! Sampled radius and position angles of objects.
    real(dp) :: r(25000), t(25000)
    ! Distance between circle and a particular rectangle edge; Haversine distance; amount of circle outside a
    ! particular rectangle edge; and point-inside-hull parameters.
    real(dp) :: h, d, chord_area_overlap, sum_of_angles, theta

!$OMP PARALLEL DO DEFAULT(NONE) PRIVATE(i, j, k, area, h, d, chord_area_overlap, sum_of_angles, theta, x0, y0, x1, y1, x2, y2, &
!$OMP& circle_too_near_edge, hull_point_inside_circle, cross_prod, dot_prod, fraction, r, t) &
!$OMP& SHARED(density_radius, cat_ax1, cat_ax2, circ_overlap_area, hull_ax1, hull_ax2, seed)
    do j = 1, size(cat_ax1)
        call random_seed(put=seed(:, j))
        ! First, check if the circle overlaps any edges. If not, then we know what the area is trivially.
        circle_too_near_edge = .false.
        x0 = cat_ax1(j)
        y0 = cat_ax2(j)
        do k = 1, size(hull_ax1)-1
            x1 = hull_ax1(k)
            y1 = hull_ax2(k)
            x2 = hull_ax1(k+1)
            y2 = hull_ax2(k+1)
            ! https://mathworld.wolfram.com/Point-LineDistance2-Dimensional.html
            d = abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if (d <= density_radius) then
                circle_too_near_edge = .true.
                exit
            end if
        end do

        if (circle_too_near_edge) then
            ! Inside the "circle overlaps edges" part, we can see if it's a straight-line overlap or not.
            hull_point_inside_circle = .false.
            do k = 1, size(hull_ax1)-1
                call haversine(hull_ax1(k), x0, hull_ax2(k), y0, d)
                if (d <= density_radius) then
                    hull_point_inside_circle = .true.
                    exit
                end if
            end do

            if (hull_point_inside_circle) then
                ! Draw samples to determine fraction of circle inside polygon of odd shape.
                call random_number(t(:))
                call random_number(r(:))
                t(:) = t(:) * 2.0_dp * pi
                r(:) = sqrt(r(:)) * density_radius
                fraction = 0.0_dp
                do k = 1, size(t)
                    x0 = cat_ax1(j) + r(k) * cos(t(k))
                    y0 = cat_ax2(j) + r(k) * sin(t(k))
                    sum_of_angles = 0.0_dp
                    do i = 1, size(hull_ax1)-1
                        x1 = hull_ax1(i) - x0
                        y1 = hull_ax2(i) - y0
                        x2 = hull_ax1(i+1) - x0
                        y2 = hull_ax2(i+1) - y0

                        dot_prod = x1*x2 + y1*y2
                        cross_prod = x1*y2 - x2*y1
                        theta = atan2(cross_prod, dot_prod)
                        sum_of_angles = sum_of_angles + theta
                    end do
                    if (abs(sum_of_angles) > pi) then
                        fraction = fraction + 1.0_dp
                    end if
                end do
                fraction = fraction / real(size(t), dp)
                area = fraction * pi * density_radius**2
            else
                ! If no point directly inside the circle, then the area of missing circle is defined by
                ! a straight line intersecting the circle, and hence a chord integral. Taking height to
                ! be sqrt(r**2 - x**2) - h, h being the distance from the circle center to the straight
                ! line, with x the straight-line-parallel distance from the circle center, we can
                ! integrate across the chord from x = -sqrt(r**2 - h**2) to +sqrt(r**2 - h**2), giving
                x0 = cat_ax1(j)
                y0 = cat_ax2(j)
                do k = 1, size(hull_ax1)-1
                    x1 = hull_ax1(k)
                    y1 = hull_ax2(k)
                    x2 = hull_ax1(k+1)
                    y2 = hull_ax2(k+1)
                    h = abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    if (h <= density_radius) then
                        exit
                    end if
                end do
                chord_area_overlap = density_radius**2 * atan(sqrt(density_radius**2 - h**2) / h) - &
                                     h * sqrt(density_radius**2 - h**2)
                area = pi * density_radius**2 - chord_area_overlap
            end if
        else
            area = pi * density_radius**2
        end if
        circ_overlap_area(j) = area
    end do
!$OMP END PARALLEL DO

end subroutine get_circle_area_overlap

subroutine perturb_aufs(Narray, magarray, r, dr, rbins, j0s, mag_D, dmag_D, Ds, N_norm, num_int, dmcut, psfr, &
    psfsig, lentrials, seed, dd_params, l_cut, algorithm_type, Fracgrid, Fluxav, fouriergrid, rgrid, intrgrid)
    ! Fortran wrapper for the perturbation AUF component calculation for a set of density-magnitude
    ! combinations, creating the various parameters needed to use the distribution of perturbations.
    integer, parameter :: dp = kind(0.0d0)  ! double precision
    ! Number of simulated PSFs to generate.
    integer, intent(in) :: lentrials
    ! Number of bins to draw simulated perturbers from, below the brightness of the central source.
    integer, intent(in) :: num_int(:)
    ! RNG seed.
    integer, intent(in) :: seed(:, :)
    ! Arrays of local densities and central source brightnesses to generate simulated PSFs for.
    real(dp), intent(in) :: Narray(:), magarray(:)
    ! Real space coordinates: middle of bins, bin widths, and bin edges (hence size(r)+1 == size(rbins)).
    real(dp), intent(in) :: r(:), dr(:), rbins(:)
    ! Bessel Function of First Kind of Zeroth Order, evaluated at various r-rho combinations.
    real(dp), intent(in) :: j0s(:, :)
    ! Magnitudes, magnitude bin widths, and logarithmic source number densities, from which to draw
    ! Poissonian average sources per PSF circle.
    real(dp), intent(in) :: mag_D(:), dmag_D(:), Ds(:)
    ! Normalising density of simulated sources.
    real(dp), intent(in) :: N_norm
    ! Relative fluxes, in magnitude offset, above which to record whether a central object suffers
    ! a contaminating source or not.
    real(dp), intent(in) :: dmcut(:)
    ! Radius and Gaussian-sigma of PSF for given filter, used to define the PSF circle inside which to draw contaminants.
    real(dp), intent(in) :: psfr, psfsig
    ! Parameters describing the skew-normal distribution, and relative flux regimes in
    ! which certain algorithm approximations apply, for the background dominated PSF regime algorithm.
    real(dp), intent(in) :: dd_params(:, :, :), l_cut(:)
    ! Algorithm type, either "fw" or "psf", for flux-weighted or PSF photometry methods.
    character(len=3), intent(in) :: algorithm_type
    ! Fraction of sources with contaminant above dmcut, and average contamination of, density-magnitude
    ! combinations to consider for this filter-sightline pair.
    real(dp), intent(out) :: Fracgrid(size(dmcut), size(Narray)), Fluxav(size(Narray))
    ! Fourier, real, and cumulative integral of real, representations of the distribution of perturbations
    ! simulated for the lentrials number of PSFs.
    real(dp), intent(out) :: fouriergrid(size(j0s, 2), size(Narray)), rgrid(size(r), size(Narray)), intrgrid(size(r), size(Narray))
    ! Loop counters.
    integer :: j, k
    ! Variables to define the sizes and positions of various arrays: defines allocatable length of dm;
    ! position in mag_D; and maximum number of simulated perturbers in a single PSF.
    integer :: lendm, mag_Dindex, maxk
    ! Temporary storage of various parameters: individual perturbations of a given density-brightness
    ! combination, fractions of PSF realisations for one N-m pair that have contaminants above dmcut
    ! relative fluxes, and the average flux within each PSF realisation for the given PSF setup.
    real(dp) :: offsets(lentrials), fraccontam(size(dmcut)), fluxcontam(lentrials)
    ! Individual PSF setup distribution functions: histogram of perturbations, cumulative integral of
    ! perturbations, and Fourier-space representation of perturbation distribution.
    real(dp) :: hist(size(r)), cumulathist(size(r)), fourierhist(size(j0s, 2))
    ! Central source magnitude and local normalising density at which to simulate PSF contaminations.
    real(dp) :: mag, N_b, midr(size(r))
    ! Define the number of sources per PSF circle in each magnitude bin range (from central source
    ! brightness to num_int bins fainter), the magnitude offsets (relative fluxes) of those bins,
    ! and the widths of those magnitude offset bins.
    real(dp), allocatable :: dNs(:), dms(:), ddms(:)

!$OMP PARALLEL DO DEFAULT(NONE) PRIVATE(j, k, N_b, mag, mag_Dindex, lendm, dms, dNs, ddms, offsets, fraccontam, maxk, &
!$OMP& fluxcontam, hist, cumulathist, fourierhist, midr) &
!$OMP& SHARED(Narray, magarray, mag_D, psfr, dmcut, lentrials, num_int, Ds, dmag_D, N_norm, rbins, r, dr, j0s, rgrid, &
!$OMP& seed, fouriergrid, intrgrid, Fracgrid, Fluxav, psfsig, dd_params, l_cut, algorithm_type) SCHEDULE(guided)
    do j = 1, size(Narray)
        N_b = Narray(j)
        mag = magarray(j)
        mag_Dindex = minloc(mag_D, mask=(mag_D >= mag), dim=1)
        lendm = min(num_int(j), size(mag_D)-mag_Dindex+1)
        allocate(dms(lendm))
        allocate(ddms(lendm))
        allocate(dNs(lendm))
        do k = 1, lendm
            dNs(k) = 10**Ds(mag_Dindex+k-1) * dmag_D(mag_Dindex+k-1) * pi * (psfr/3600.0_dp)**2 * N_b / N_norm
            dms(k) = mag_D(mag_Dindex+k-1) - mag
            ddms(k) = dmag_D(mag_Dindex+k-1)
        end do
        maxk = max(5, int(10*maxval(dNs)))
        call scatter_perturbers(dNs, dms, psfr, maxk, dmcut, psfsig, offsets, fraccontam, fluxcontam, dd_params, l_cut, ddms, &
                                algorithm_type, lentrials, seed(:, j))
        call histogram1d_dp(offsets, rbins(1), rbins(size(rbins)), size(r), midr, hist)

        ! r is middle of bins, which are represented by rbins; there's a shift of dr/2 between the two (minus rbins(lenr+1))
        hist = hist / (pi * ((r + dr/2.0_dp)**2 - (r - dr/2.0_dp)**2) * sum(hist))
        cumulathist(1) = hist(1) * pi * ((r(1) + dr(1)/2.0_dp)**2 - (r(1) - dr(1)/2.0_dp)**2)
        do k = 2, size(r)
            cumulathist(k) = cumulathist(k-1) + hist(k) * pi * ((r(k) + dr(k)/2.0_dp)**2 - (r(k) - dr(k)/2.0_dp)**2)  
        end do
        call fourier_transform(hist, r, dr, j0s, fourierhist)
        fouriergrid(:, j) = fourierhist
        rgrid(:, j) = hist

        intrgrid(:, j) = cumulathist
        Fracgrid(:, j) = fraccontam
        Fluxav(j) = sum(fluxcontam) / real(lentrials, dp)

        deallocate(dms)
        deallocate(ddms)
        deallocate(dNs)
    end do
!$OMP END PARALLEL DO

end subroutine perturb_aufs

subroutine scatter_perturbers(dNs, dms, psfr, maxk, dmcut, psfsig, offsets, fraccontam, fluxcontam, dd_params, l_cut, ddms, &
    algorithm_type, lentrials, seed)
    ! Given a set of average numbers of sources per PSF circle for a series of relative fluxes, populate
    ! a bright, central source's PSF with randomly placed sources, and calculate the flux brightening
    ! and expected PSF centroid shift.
    integer, parameter :: dp = kind(0.0d0)  ! double precision
    ! Number of simulated PSFs to generate.
    integer, intent(in) :: lentrials
    ! Maximum allowed number of sources of a given magnitude offset in a PSF circle.
    integer, intent(in) :: maxk
    ! RNG seed.
    integer, intent(in) :: seed(:)
    ! Average numbers of sources per PSF circle for each magnitude offset; magnitude offsets (or
    ! relative fluxes) to be populated within the PSF, and the bin width of the magnitude offsets.
    real(dp), intent(in) :: dNs(:), dms(:), ddms(:)
    ! PSF radius and Gaussian-sigma, defined by the Rayleigh criterion based on the full-width at half-maximum and
    ! the relation to FWHM respectively.
    real(dp), intent(in) :: psfr, psfsig
    ! Magnitude offsets above which to consider if a PSF has been contaminated by a source of
    ! this relative flux.
    real(dp), intent(in) :: dmcut(:)
    ! Parameters describing the skew-normal distribution, and relative flux regimes in
    ! which certain algorithm approximations apply, for the background dominated PSF regime algorithm.
    real(dp), intent(in) :: dd_params(:, :, :), l_cut(:)
    ! Algorithm type, either "fw" or "psf", for flux-weighted or PSF photometry methods.
    character(len=3), intent(in) :: algorithm_type

    real(dp), intent(out) :: offsets(lentrials), fraccontam(size(dmcut)), fluxcontam(lentrials)
    ! Loop counters, and number of sources to be populated within a given PSF for a small dm slice.
    integer :: i, j, k, loopk
    ! Flag to indicate whether each PSF realisation has been contaminated by a source brighter
    ! than each dmcut relative flux.
    integer :: ncontams(size(dmcut), lentrials)
    ! Variables related to position within the PSF circle for each object in a dm slice.
    real(dp) :: x(maxk), y(maxk), xav, yav, tmp_xav, tmp_yav, t(maxk), r(maxk)
    ! Poissonian-related values, defining the randomly drawn number of small magnitude range objects
    ! in a given PSF.
    real(dp) :: factorial(maxk+1), powercounter, numchance, expdns(size(dNs)), cumulativepoisson(maxk+1, size(dNs))
    ! Variables related to the relative flux of each simulated perturbing source, and the total flux
    ! within a PSF.
    real(dp) :: fluxes(size(dNs)), dfluxes(2, size(dNs)), df(maxk), f0, normf
    ! Variables related to the parameterisation of the background-dominated PSF algorithm case.
    real(dp) :: recorded_x(maxk, size(dNs)), recorded_y(maxk, size(dNs)), recorded_f(maxk, size(dNs))

    factorial(1) = 1.0_dp
    do i = 1, maxk
        factorial(i+1) = factorial(i) * real(i, dp)
    end do

    call random_seed(put=seed)

    fluxes = 10**(-dms/2.5_dp)
    ! Asymmetric bins in flux space (for symmetric mag bins) needs an upper and lower bin width
    dfluxes(1, :) = 10**(-(dms-ddms/2.0_dp)/2.5_dp) - 10**(-dms/2.5_dp)
    dfluxes(2, :) = 10**(-dms/2.5_dp) - 10**(-(dms+ddms/2.0_dp)/2.5_dp)
    expdns = exp(-dNs)

    ncontams = 0
    offsets = 0.0_dp
    fluxcontam = 0.0_dp
    ! Cumulative poisson = exp(-l) sum_i=0^floor(k) l^i / i!; l^0 / 0! = 1
    cumulativepoisson(1, :) = 1.0_dp
    do j = 1, size(dNs)
        powercounter = dNs(j)
        do k = 1, maxk
            cumulativepoisson(k+1, j) = cumulativepoisson(k, j) + powercounter / factorial(k+1) 
            powercounter = powercounter * dNs(j)
        end do
    end do
    do i = 1, lentrials
        xav = 0.0_dp
        yav = 0.0_dp
        normf = 1.0_dp
        if (algorithm_type == 'psf') then
            recorded_x(:, :) = 0.0d0
            recorded_y(:, :) = 0.0d0
            recorded_f(:, :) = -1.0d0 ! initialise fluxes as negative to skip in the fitting process
        end if
        do j = 1, size(dNs)
            call random_number(numchance)
            numchance = numchance / expdns(j)
            if (cumulativepoisson(1, j) > numchance) then
                loopk = 0
            else
                if (cumulativepoisson(maxk, j) < numchance) then
                    loopk = maxk
                else
                    loopk = maxk
                    do k = 1, maxk
                        if (cumulativepoisson(k+1, j) > numchance) then
                            loopk = k
                            exit
                        end if
                    end do
                end if
                do k = 1, size(dmcut)
                    if (ncontams(k, i) == 0 .and. dms(j) < dmcut(k)) then
                        ncontams(k, i) = 1
                    end if
                end do
                call random_number(t(:loopk))
                call random_number(r(:loopk))
                call random_number(df(:loopk))
                t(:loopk) = t(:loopk) * 2.0_dp * pi
                r(:loopk) = sqrt(r(:loopk)) * psfr

                ! fluxes is middle of bin
                df(:loopk) = df(:loopk) * (dfluxes(1, j) + dfluxes(2, j))
                f0 = fluxes(j) - dfluxes(2, j)

                x(:loopk) = r(:loopk) * sin(t(:loopk))
                y(:loopk) = r(:loopk) * cos(t(:loopk))

                if (algorithm_type == 'psf') then
                    recorded_x(:loopk, j) = x(:loopk)
                    recorded_y(:loopk, j) = y(:loopk)
                    recorded_f(:loopk, j) = f0 + df(:loopk)
                end if

                if (algorithm_type == 'psf') then
                    call psf_perturb(x(:loopk), y(:loopk), r(:loopk), f0+df(:loopk), dd_params, fluxes(j), psfsig, &
                                     l_cut, tmp_xav, tmp_yav)
                    ! Cumulatively keep track of xav and yav.
                    xav = xav + tmp_xav
                    yav = yav + tmp_yav
                else
                    xav = xav + sum(x(:loopk) * (f0+df(:loopk)))
                    yav = yav + sum(y(:loopk) * (f0+df(:loopk)))
                    normf = normf + sum(f0+df(:loopk))
                end if
            end if
        end do
        if (algorithm_type == 'fw') then
            xav = xav / normf
            yav = yav / normf
        end if
        offsets(i) = sqrt(xav**2 + yav**2)

        if (algorithm_type == 'fw') then
            fluxcontam(i) = normf - 1.0_dp
        else
            fluxcontam(i) = exp(-0.25_dp * offsets(i)**2 / psfsig**2) - 1.0_dp
            do k = 1, size(dNs)
                do j = 1, maxk
                    if (recorded_f(j, k) >= 0.0_dp) then
                        fluxcontam(i) = fluxcontam(i) + recorded_f(j, k) * exp(-0.25_dp * ((recorded_x(j, k) - xav)**2 + &
                                                                                           (recorded_y(j, k) - yav)**2)/psfsig**2)
                    end if
                end do
            end do
            if (fluxcontam(i) < 0.0_dp) then
                fluxcontam(i) = 0.0_dp
            end if
        end if
    end do
    ! TODO: update with mean/median/model/percentiles
    do k = 1, size(dmcut)
        fraccontam(k) = real(sum(ncontams(k, :)), dp) / real(lentrials, dp)
    end do

end subroutine scatter_perturbers

subroutine psf_perturb(x, y, r, fs, dd_params, flux, psfsig, l_cut, xav, yav)
    ! For a set of cartesian coordinates and relative fluxes for perturbers, depending on the flux
    ! of the objects, calculate different perturbation offsets based on Gaussian PSF model fits with
    ! constant background noise.
    integer, parameter :: dp = kind(0.0d0)  ! double precision
    ! Positions and radii offsets for perturbers.
    real(dp), intent(in) :: x(:), y(:), r(:)
    ! Central and blurred relative fluxes of perturbers.
    real(dp), intent(in) :: fs(:), flux
    ! Parameters required to simulat perturbation based on simulations.
    real(dp), intent(in) :: dd_params(:, :, :), psfsig, l_cut(:)
    ! x and y coordinates of average perturbation for these objects.
    real(dp), intent(out) :: xav, yav
    ! Loop counter
    integer :: k
    ! Temporary parameters for deriving polynomial fits based on dd_params.
    integer :: which_dd_poly
    real(dp) :: ddparams(size(dd_params, 1)), dddparamsdf(size(dd_params, 1)), param_temp, dparam_temp
    ! Placeholder for fit_skew calculation
    real(dp) :: g1

    xav = 0.0_dp
    yav = 0.0_dp
    if (flux >= l_cut(3)) then
        ! If equal-brightness, or close enough as defined by l_cut, "binary" star(s), then the offsets are just
        ! flux weighted between the central sources and perturbers, on an individual basis.
        xav = xav + sum(x*fs / (1.0_dp + fs))
        yav = yav + sum(y*fs / (1.0_dp + fs))
    else if (flux >= l_cut(1)) then  ! otherwise, if relative flux too large to approximate, fit distribution
        ! Select which side of the relative flux polynomial we want to model:
        if (flux >= l_cut(2)) then
            which_dd_poly = 2
        else
            which_dd_poly = 1
        end if
        ddparams(:) = dd_params(:, 1, which_dd_poly)
        dddparamsdf(:) = 0.0_dp
        param_temp = flux
        dparam_temp = 1.0_dp
        ! To avoid computing these polynomials for every source, we compute the bin middle per flux bin and
        ! compute the linear gradient of each parameter wrt flux, and assume bins are small enough that the
        ! polynomial parameters at arbitrary flux within a bin can be well modelled as a linear slope.
        do k = 2, size(dd_params, 2)
            ! m = sum_i=0^N p_i,m f**i
            ddparams(:) = ddparams(:) + dd_params(:, k, which_dd_poly) * param_temp
            param_temp = param_temp * flux
            ! dm/df = sum_i=1^N p_i,m f**(i-1) * i
            dddparamsdf(:) = dddparamsdf(:) + dd_params(:, k, which_dd_poly) * dparam_temp * real(k-1)
            dparam_temp = dparam_temp * flux
        end do
        do k = 1, size(x)
            ! When calling the skew normal remember we fit the paramerisation in psf sigma normalised space, so
            ! first divide the offset by psfsig, then multiply back out again at the other end.
            if (abs(x(k)) < (ddparams(5) + dddparamsdf(5) * (fs(k) - flux))*psfsig) then
                ! if offset is small, just use flux-weighted individual offset
                xav = xav + sign(1.0_dp, x(k)) * abs(x(k)) * fs(k) / (1 + fs(k))
            ! Otherwise we fit a skew-normal distribution for the individual vector perturbation.
            else
                call fit_skew(ddparams(:4) + dddparamsdf(:4) * (fs(k) - flux), abs(x(k))/psfsig, fs(k), g1)
                xav = xav + sign(1.0_dp, x(k)) * g1 * psfsig
            end if
            if (abs(y(k)) < (ddparams(5) + dddparamsdf(5) * (fs(k) - flux))*psfsig) then
                yav = yav + sign(1.0_dp, y(k)) * abs(y(k)) * fs(k) / (1 + fs(k))
            else
                call fit_skew(ddparams(:4) + dddparamsdf(:4) * (fs(k) - flux), abs(y(k))/psfsig, fs(k), g1)
                yav = yav + sign(1.0_dp, y(k)) * g1 * psfsig
            end if
        end do
    else
        xav = xav + sum(fs * x * exp(-0.25_dp * r**2 / psfsig**2))
        yav = yav + sum(fs * y * exp(-0.25_dp * r**2 / psfsig**2))
    end if

end subroutine psf_perturb

subroutine fit_skew(params, x, L, y)
    ! Calculate the skew-normal perturber effect, evaluated at some perturber separation x.
    integer, parameter :: dp = kind(0.0d0)  ! double precision
    ! The skew-normal parameters array, the value at which to evaluate the function, and the relative
    ! flux of the perturber.
    real(dp), intent(in) :: params(:), x, L
    ! Perturbation effect experienced by perturber of relative flux L at x.
    real(dp), intent(out) :: y
    ! Sigma scale, mean mu, shape alpha, and amplitude T of the scaled skew-normal distribution.
    real(dp) :: c, u, a, t
    ! Temporary variables holding the shifted and scaled separation, and skew-normal PDF and CDF.
    real(dp) :: x2, psipdf, psicdf

    c = params(1)
    u = params(2)
    a = params(3)
    t = params(4)
    x2 = (x - u) / c

    psipdf = 1.0_dp/sqrt(2.0_dp * pi) * exp(-0.5_dp * x2**2)
    psicdf = 0.5_dp * (1.0_dp + erf(a * x2 / sqrt(2.0_dp)))
    y = 2 * t / c * L * psipdf * psicdf

end subroutine fit_skew

! ------------------------------------------------------------------------------
! Copyright (c) 2009-13, Thomas P. Robitaille
!
! All rights reserved.
!
! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions are met:
!
!  * Redistributions of source code must retain the above copyright notice, this
!    list of conditions and the following disclaimer.
!
!  * Redistributions in binary form must reproduce the above copyright notice,
!    this list of conditions and the following disclaimer in the documentation
!    and/or other materials provided with the distribution.
!
! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
! AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
! IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
! DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
! FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
! DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
! SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
! CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
! OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
! OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
! ------------------------------------------------------------------------------
! (Applies to xval_dp, ipos_dp, and histogram1d_dp)

subroutine xval_dp(xmin,xmax,i,nbin, xval)
  ! Find central value of a bin for a regular histogram

  integer, parameter :: dp = kind(0.0d0)  ! double precision

  real(dp),intent(in) :: xmin,xmax
  ! range of values

  integer,intent(in) :: i
  ! the bin number

  integer,intent(in) :: nbin
  ! number of bins

  real(dp), intent(out) :: xval

  real(dp) :: frac

  frac=(real(i-1)+0.5)/real(nbin)

  xval=frac*(xmax-xmin)+xmin

end subroutine xval_dp

subroutine ipos_dp(xmin,xmax,x,nbin, ipos)
! Find bin a value falls in for a regular histogram

    integer, parameter :: dp = kind(0.0d0)  ! double precision

    real(dp),intent(in) :: xmin,xmax
    ! range of values

    real(dp),intent(in) :: x
    ! the value to bin

    integer,intent(in) :: nbin
    ! number of bins

    integer, intent(out) :: ipos

    real(dp) :: frac

    if(xmax > xmin) then

       if(x < xmin) then
          ipos = 0
       else if(x > xmax) then
          ipos = nbin+1
       else if(x < xmax) then
          frac=(x-xmin)/(xmax-xmin)
          ipos=int(frac*real(nbin, dp))+1
       else  ! x == xmax
          ipos = nbin
       end if

    else

       if(x < xmax) then
          ipos = 0
       else if(x > xmin) then
          ipos = nbin+1
       else if(x < xmin) then
          frac=(x-xmin)/(xmax-xmin)
          ipos=int(frac*real(nbin, dp))+1
       else  ! x == xmin
          ipos = nbin
       end if

    end if

end subroutine ipos_dp

subroutine histogram1d_dp(array,xmin,xmax,nbin,hist_x,hist_y,mask,weights)
  ! Bin 1D array of values into 1D regular histogram

  integer, parameter :: dp = kind(0.0d0)  ! double precision

  real(dp),dimension(:),intent(in) :: array
  real(dp),dimension(:),intent(in),optional :: weights
  ! the array of values to bin

  real(dp),intent(in) :: xmin,xmax
  ! the range of the histogram

  integer,intent(in) :: nbin
  ! number of bins

  real(dp),dimension(nbin),intent(out) :: hist_x,hist_y
  ! the histogram

  integer :: i,ibin
  ! binning variables

  logical,optional,intent(in) :: mask(:)
  logical,allocatable:: keep(:)

  allocate(keep(size(array)))

  if(present(mask)) then
     keep = mask
  else
     keep = .true.
  end if

  hist_x=0._dp ; hist_y=0._dp

  do i=1,size(array)
     if(keep(i)) then
        call ipos_dp(xmin,xmax,array(i),nbin, ibin)
        if(ibin.ge.1.and.ibin.le.nbin) then
           if(present(weights)) then
              hist_y(ibin)=hist_y(ibin)+weights(i)
           else
              hist_y(ibin)=hist_y(ibin)+1._dp
           end if
        end if
     end if
  end do

  do ibin=1,nbin
     call xval_dp(xmin,xmax,ibin,nbin, hist_x(ibin))
  end do

  deallocate(keep)

end subroutine histogram1d_dp

subroutine fourier_transform(pr, r, dr, j0s, G)
    ! Calculates the Fourier-Bessel transform, or Hankel transform of zeroth order, of a function.
    ! This is equivalent to a two-dimensional Fourier transform in the limiting case of circular
    ! symmetry.
    integer, parameter :: dp = kind(0.0d0)  ! double precision
    ! Function to be fourier transformed, in units of per area.
    real(dp), intent(in) :: pr(:)
    ! Real space coordinates and bin widths.
    real(dp), intent(in) :: r(:), dr(:)
    ! Bessel function of first kind of zeroth order, evaluated at 2 * pi * r * rho, where rho
    ! is the fourier space coordinates of interest to capture the fourier transformation.
    real(dp), intent(in) :: j0s(:, :)
    ! The Fourier-Bessel transform of pr, evaluated at the appropriate fourier space coordinates
    ! rho, as desired.
    real(dp), intent(out) :: G(size(j0s, 2))
    ! Loop counters.
    integer :: i, j
    ! Temporary variables
    real(dp) :: r_pr_dr(size(r))
    r_pr_dr = r * pr * dr
    G(:) = 0.0_dp
    ! Following notation of J. W. Goodman, Introduction to Fourier Optics (1996), equation 2-31,
    ! G0 is the Hankel transform of gR, and the inverse is the opposite.
    ! For a Hankel transform (of zero order; or Fourier-Bessel transform) the "r" variable is r, with
    ! rho along the other axis of j0s, but for an inverse transformation "r" becomes rho with r
    ! represented along the second axis of j0s, because the two transformations are symmetric.
    
    do i = 1, size(j0s, 2)
        do j = 1, size(j0s, 1)
            G(i) = G(i) + r_pr_dr(j) * j0s(j, i)
        end do
        G(i) = G(i) * 2.0_dp * pi
    end do

end subroutine fourier_transform

subroutine get_random_seed_size(size)
    ! Number of initial seeds expected by random_seed, to be initialised for a specified RNG setup.
    integer, intent(out) :: size

    call random_seed(size=size)

end subroutine get_random_seed_size

end module perturbation_auf_fortran