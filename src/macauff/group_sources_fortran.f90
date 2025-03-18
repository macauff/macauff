! Licensed under a 3-clause BSD style license - see LICENSE

module group_sources_fortran
! This module provides the Fortran code for the handling of the grouping of sources between
! the two catalogues being cross-matched, identifying astrometric overlaps.

implicit none

integer, parameter :: dp = kind(0.0d0)  ! double precision
real(dp), parameter :: pi = 4.0_dp*atan(1.0_dp)

contains

subroutine calc_j1s(rho, r, j1s)
    ! Calculate the Bessel Function of First Kind of First Order for a set of fourier-space coordinates.
    integer, parameter :: dp = kind(0.0d0)  ! double precision
    ! Fourier- and real-space coordinates at which to evaluate J1.
    real(dp), intent(in) :: rho(:), r(:)
    ! Bessel Function of First kind of First order
    real(dp), intent(out) :: j1s(size(rho), size(r))
    integer :: j, k

!$OMP PARALLEL DO DEFAULT(NONE) PRIVATE(j, k) SHARED(rho, r, j1s)
    do k = 1, size(rho)
        do j = 1, size(r)
            call jy01a_j1(2.0_dp*pi*rho(k)*r(j), j1s(k, j))
        end do
    end do
!$OMP END PARALLEL DO

end subroutine calc_j1s

subroutine cumulative_fourier_probability(pr, drho, dist, j1s, cumulative_prob)
    ! Calculates the cumulative 2-D spacial integral of the convolved AUFs of two sources,
    ! based on an inverse 2-D Fourier (Hankel) transform of the fourier-space representation
    ! of the individual AUF components, returning the probability under a specific radial distance.
    integer, parameter :: dp = kind(0.0d0)  ! double precision
    ! Fourier-space representation of PDF to be fourier transformed.
    real(dp), intent(in) :: pr(:)
    ! Real--space coordinate widths for fourier transformation.
    real(dp), intent(in) :: drho(:)
    ! Real-space distance out to which to perform cumulative convolution integral.
    real(dp), intent(in) :: dist
    ! J1, the Bessel Function of First Kind of First Order, evaluated at all dist-rho combinations.
    real(dp), intent(in) :: j1s(:)
    ! Loop counter.
    integer :: j
    ! Total integral of the convolved PDFs, evaluated via fourier transformation.
    real(dp), intent(out) :: cumulative_prob

    cumulative_prob = 0.0_dp
    do j = 1, size(drho)
        cumulative_prob = cumulative_prob + pr(j) * j1s(j) * drho(j)
    end do
    cumulative_prob = dist * cumulative_prob * 2.0_dp * pi

end subroutine cumulative_fourier_probability

subroutine get_overlap_indices(a_ax_1, a_ax_2, b_ax_1, b_ax_2, a_inds, a_size, b_inds, b_size, amax, bmax, a_axerr, b_axerr, r, &
    rho, drho, j1s, afouriergrid, bfouriergrid, amodrefind, bmodrefind, max_frac, aindices, bindices, anumoverlap, bnumoverlap, &
    a_auf_cdf, b_auf_cdf)
    ! Once total potential overlap is found in getmaxn, we can keep track of each individual
    ! source overlap between the two catalogues, storing their respective indices to keep
    ! links of potential counterparts between the two catalogues.
    integer, parameter :: dp = kind(0.0d0)  ! double precision
    ! 4-D index maps from each source into the fourier-space perturbation AUF component grids.
    integer, intent(in) :: amodrefind(:, :), bmodrefind(:, :)
    ! Largest size of overlaps of each catalogue, from getmaxn.
    integer, intent(in) :: amax, bmax
    ! Orthogonal sky coordinate arrays and circular coordinate uncertainties for catalogue a and b.
    real(dp), intent(in) :: a_ax_1(:), a_ax_2(:), b_ax_1(:), b_ax_2(:), a_axerr(:), b_axerr(:)
    ! Number of overlaps, and indices of the overlaps, of all sources in catalogue "b" for each "a" object (and vice versa).
    integer, intent(in) :: a_size(:), a_inds(:, :), b_size(:), b_inds(:, :)
    ! Real- and fourier-space values for empirical AUF convolutions and integrals. Note that r
    ! evalutaes at a bin middle (i.e., r'+dr'/2) while rho is left-hand bin edges and drho
    ! bin widths.
    real(dp), intent(in) :: r(:), rho(:), drho(:)
    ! J1, the Bessel Function of First Kind of First Order, evaluated at all r-rho bin middle combinations.
    real(dp), intent(in) :: j1s(:, :)
    ! Grid of fourier-space representations of the perturbation components of each AUF.
    real(dp), intent(in) :: afouriergrid(:, :, :, :), bfouriergrid(:, :, :, :)
    ! Largest considered AUF integral to be considered as potential counterparts.
    real(dp), intent(in) :: max_frac
    ! Indices of overlaps from catalogue a into b (and b into a).
    integer, intent(out) :: aindices(amax, size(a_ax_1)), bindices(bmax, size(b_ax_1))
    ! Number of overlaps of each source into the opposing catalogue.
    integer, intent(out) :: anumoverlap(size(a_ax_1)), bnumoverlap(size(b_ax_1))
    ! Cumulative Distribution Functions evaluates for each potential pairing, within max_frac, matching [ab]indices.
    real(dp), intent(out) :: a_auf_cdf(amax, size(a_ax_1)), b_auf_cdf(bmax, size(b_ax_1))
    ! Loop counters.
    integer :: i, j, k, l
    ! Sky offsets and uncertainties
    real(dp) :: oa, ob
    ! Respective fourier-space variables.
    real(dp) :: afourier(size(afouriergrid, 1)), bfourier(size(bfouriergrid, 1)), four(size(afouriergrid, 1))
    ! Cumulative real-space probability.
    real(dp) :: cumulative
    ! The offset between opposing catalogue sources, as given by the Haversine formula.
    real(dp) :: dist
    ! Temporary flag arrays.
    integer :: tempcounter, changeflag
    ! Allocatable temporary index array.
    integer, allocatable :: tempind(:)
    ! Allocatable temporary CDF array.
    real(dp), allocatable :: tempcdf(:)

    a_auf_cdf = 2
    b_auf_cdf = 2
    aindices = -1
    bindices = -1
    anumoverlap = 0
    bnumoverlap = 0

    allocate(tempind(amax))
    allocate(tempcdf(amax))
!$OMP PARALLEL DO DEFAULT(NONE) PRIVATE(i, j, k, l, dist, afourier, bfourier, four, cumulative, tempind, tempcdf, tempcounter, &
!$OMP& changeflag, oa, ob) SHARED(a_ax_1, a_ax_2, b_ax_1, b_ax_2, aindices, anumoverlap, afouriergrid, bfouriergrid, &
!$OMP& amodrefind, bmodrefind, r, rho, drho, max_frac, a_axerr, b_axerr, j1s, a_auf_cdf, a_inds, a_size)
    do j = 1, size(a_ax_1)
        tempind = -1
        tempcounter = 1
        tempcdf = 2
        changeflag = 0
        oa = a_axerr(j)
        do k = 1, a_size(j)
            i = a_inds(k, j) + 1
            ob = b_axerr(i)
            afourier = afouriergrid(:, amodrefind(1, j)+1, amodrefind(2, j)+1, amodrefind(3, j)+1)
            bfourier = bfouriergrid(:, bmodrefind(1, i)+1, bmodrefind(2, i)+1, bmodrefind(3, i)+1)
            four = afourier*bfourier*exp(-2.0_dp * pi**2 * (rho+drho/2.0_dp)**2 * (oa**2 + ob**2))
            call haversine(a_ax_1(j), b_ax_1(i), a_ax_2(j), b_ax_2(i), dist)
            l = minloc(abs(r - dist*3600.0_dp), dim=1)
            call cumulative_fourier_probability(four, drho, dist*3600.0_dp, j1s(:, l), cumulative)
            if (cumulative < max_frac) then
                ! Python's zero indexing!
                tempind(tempcounter) = i - 1
                tempcdf(tempcounter) = cumulative
                tempcounter = tempcounter + 1
                changeflag = 1
            end if
        end do
        if (changeflag > 0) then
            aindices(:, j) = tempind
            a_auf_cdf(:, j) = tempcdf
            anumoverlap(j) = tempcounter - 1
        end if
    end do
!$OMP END PARALLEL DO

    deallocate(tempind)
    allocate(tempind(bmax))
    deallocate(tempcdf)
    allocate(tempcdf(bmax))

!$OMP PARALLEL DO DEFAULT(NONE) PRIVATE(i, j, k, l, dist, afourier, bfourier, four, cumulative, tempind, tempcdf, tempcounter, &
!$OMP& changeflag, oa, ob) SHARED(a_ax_1, a_ax_2, b_ax_1, b_ax_2, bindices, bnumoverlap, afouriergrid, bfouriergrid, &
!$OMP& amodrefind, bmodrefind, r, rho, drho, max_frac, a_axerr, b_axerr, j1s, b_auf_cdf, b_inds, b_size)
    do i = 1, size(b_ax_1)
        tempind = -1
        tempcounter = 1
        tempcdf = 2
        changeflag = 0
        ob = b_axerr(i)
        do k = 1, b_size(i)
            j = b_inds(k, i) + 1
            oa = a_axerr(j)
            afourier = afouriergrid(:, amodrefind(1, j)+1, amodrefind(2, j)+1, amodrefind(3, j)+1)
            bfourier = bfouriergrid(:, bmodrefind(1, i)+1, bmodrefind(2, i)+1, bmodrefind(3, i)+1)
            four = afourier*bfourier*exp(-2.0_dp * pi**2 * (rho+drho/2.0_dp)**2 * (oa**2 + ob**2))
            call haversine(a_ax_1(j), b_ax_1(i), a_ax_2(j), b_ax_2(i), dist)
            l = minloc(abs(r - dist*3600.0_dp), dim=1)
            call cumulative_fourier_probability(four, drho, dist*3600.0_dp, j1s(:, l), cumulative)
            if (cumulative < max_frac) then
                tempind(tempcounter) = j - 1
                tempcdf(tempcounter) = cumulative
                tempcounter = tempcounter + 1
                changeflag = 1
            end if
        end do
        if (changeflag > 0) then
            bindices(:, i) = tempind
            b_auf_cdf(:, i) = tempcdf
            bnumoverlap(i) = tempcounter - 1
        end if
    end do
!$OMP END PARALLEL DO

end subroutine get_overlap_indices

subroutine get_integral_length(a_err, b_err, r, rho, drho, j1s, a_fouriergrid, b_fouriergrid, a_modrefind, b_modrefind, &
    a_inds, a_size, frac_array, int_areas)
    ! Calculate the "error circle" areas for sources, based on various cumulative integral fractions
    ! of the convolution of all overlapping opposing catalogue source AUFs with the given source's AUF.
    integer, parameter :: dp = kind(0.0d0)  ! double precision
    ! Indices for each source into the grid of perturbation component AUFs.
    integer, intent(in) :: a_modrefind(:, :), b_modrefind(:, :)
    ! Number of overlaps, and indices of the overlaps, of all sources in catalogue "b" for each "a" object.
    integer, intent(in) :: a_size(:), a_inds(:, :)
    ! Astrometric uncertainties for the sources in the two catalogues.
    real(dp), intent(in) :: a_err(:), b_err(:)
    ! Real- and fourier-space arrays, and the Bessel Function of First Kind of First Order for those
    ! r-rho combinations.
    real(dp), intent(in) :: r(:), rho(:), drho(:), j1s(:, :)
    ! Grids of fourier representations of the perturbation component of the AUF for the catalogue sources.
    real(dp), intent(in) :: a_fouriergrid(:, :, :, :), b_fouriergrid(:, :, :, :)
    ! Fractions of integral to consider when integrating convolutions of AUFs; likely to be the "bright"
    ! and "field" source identifications necessary to derive photometric likelihoods and priors.
    real(dp), intent(in) :: frac_array(:)
    ! Output source error circle areas (probably "bright" and "field") for all catalogue "a" sources.
    real(dp), intent(out) :: int_areas(size(a_err), size(frac_array))
    ! Loop counters, and counter for keeping average error circle area.
    integer :: i, j, k, l, int_areas_n(size(a_err), size(frac_array))
    ! Real-space lengthscales: singular astrometric uncertainties, and cumulative convolution
    ! integral distance.
    real(dp) :: ao, bo, cumulative_dists(size(frac_array))
    ! Respective fourier-space variables.
    real(dp) :: afourier(size(a_fouriergrid, 1)), bfourier(size(b_fouriergrid, 1)), four(size(a_fouriergrid, 1))

    int_areas = 0.0_dp

    int_areas_n = 0

!$OMP PARALLEL DO DEFAULT(NONE) PRIVATE(i, j, k, l, ao, bo, afourier, bfourier, four, cumulative_dists) SHARED(a_err, &
!$OMP& a_fouriergrid, a_modrefind, r, rho, drho, j1s, a_size, a_inds, b_err, b_fouriergrid, b_modrefind, frac_array, &
!$OMP& int_areas, int_areas_n)
    do i = 1, size(a_err)
        ao = a_err(i)
        afourier = a_fouriergrid(:, a_modrefind(1, i)+1, a_modrefind(2, i)+1, a_modrefind(3, i)+1)
        do j = 1, a_size(i)
            k = a_inds(j, i) + 1
            bo = b_err(k)
            bfourier = b_fouriergrid(:, b_modrefind(1, k)+1, b_modrefind(2, k)+1, b_modrefind(3, k)+1)
            four = afourier*bfourier*exp(-2.0_dp * pi**2 * (rho+drho/2.0_dp)**2 * (ao**2 + bo**2))
            call cumulative_fourier_distance(four, r, drho, frac_array, j1s, cumulative_dists)
            do l = 1, size(frac_array)
                int_areas(i, l) = int_areas(i, l) + cumulative_dists(l)
                int_areas_n(i, l) = int_areas_n(i, l) + 1
            end do
        end do
    end do
!$OMP END PARALLEL DO

    do i = 1, size(a_err)
        do l = 1, size(frac_array)
            if (int_areas_n(i, l) > 0) then
                int_areas(i, l) = pi * (int_areas(i, l) / int_areas_n(i, l))**2
            end if
        end do
    end do

end subroutine get_integral_length

subroutine cumulative_fourier_distance(f, r, drho, probs, j1s, cumulative_dists)
    ! Calculates the cumulative 2-D spacial integral of the convolved AUFs of two sources,
    ! based on an inverse 2-D Fourier (Hankel) transform of the fourier-space representation
    ! of the individual AUF components, up to a specified set of probabilities, returning the
    ! radial distance to reach those integral fractions.
    integer, parameter :: dp = kind(0.0d0)  ! double precision
    ! Fourier-space representation of PDF to be fourier transformed.
    real(dp), intent(in) :: f(:)
    ! Real- and fourier-space coordinates for fourier transformation.
    real(dp), intent(in) :: r(:), drho(:)
    ! Integral fractions out to which to perform cumulative convolution integral.
    real(dp), intent(in) :: probs(:)
    ! J0, the Bessel Function of First Kind of First Order, evaluated at all r-rho combinations.
    real(dp), intent(in) :: j1s(:, :)
    ! Loop counters.
    integer :: j, k, loop
    ! Flags, one per probability in probs, to indicate we have reached the distance out to this
    ! probability integral and to stop considering it any further.
    integer :: flags(size(probs))
    ! Individual probability, calculated by the integral of the Hankel transformation of f.
    real(dp) :: cumulative_prob
    ! Temporary distance, to keep for the distance search.
    real(dp) :: dist_low, dist_high, dist
    ! Distances of integrals of the convolved PDFs, evaluated via fourier transformation.
    real(dp), intent(out) :: cumulative_dists(size(probs))

    do k = 1, size(flags)
        flags(k) = 0
        dist_low = 0
        dist_high = r(size(r))
        do loop = 1, 15
            dist = 0.5_dp * (dist_high + dist_low)
            j = minloc(r, mask=(r >= dist), dim=1)
            call cumulative_fourier_probability(f, drho, dist, j1s(:, j), cumulative_prob)
            if (cumulative_prob >= probs(k)) then
                dist_high = dist
            else
                dist_low = dist
            end if
        end do
        cumulative_dists(k) = dist/3600.0_dp
    end do

end subroutine cumulative_fourier_distance

end module group_sources_fortran