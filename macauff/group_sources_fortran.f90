! Licensed under a 3-clause BSD style license - see LICENSE

module group_sources_fortran
! This module provides the Fortran code for the handling of the grouping of sources between
! the two catalogues being cross-matched, identifying astrometric overlaps.
use perturbation_auf_fortran, only: haversine
implicit none
integer, parameter :: dp = kind(0.0d0)  ! double precision
real(dp), parameter :: pi = 4.0_dp*atan(1.0_dp)

contains

subroutine get_max_overlap(a_ax_1, a_ax_2, b_ax_1, b_ax_2, max_sep, a_axerr, b_axerr, r, dr, rho, drho, j0s, afouriergrid, &
    bfouriergrid, amodrefind, bmodrefind, max_frac, anumoverlap, bnumoverlap)
    ! Loop through all pairs of sources, recording all of those within maximum separation,
    ! keeping a running total of how many overlaps there are for each source in the two
    ! catalogues.

    integer, parameter :: dp = kind(0.0d0)  ! double precision

    ! 4-D index maps from each source into the cumulative offset perturbation AUF component grids.
    integer, intent(in) :: amodrefind(:, :), bmodrefind(:, :)
    ! Orthogonal sky coordinate arrays and circular coordinate uncertainties for catalogue a and b.
    real(dp), intent(in) :: a_ax_1(:), a_ax_2(:), b_ax_1(:), b_ax_2(:), a_axerr(:), b_axerr(:)
    ! Largest allowed separation between objects across catalogue a and b to be considered
    ! potential counterparts to one another.
    real(dp), intent(in) :: max_sep
    ! Physical sky coordinates and fourier space values for empirical AUF convolutions and integrals.
    real(dp), intent(in) :: r(:), dr(:), rho(:), drho(:)
    ! J0, the Bessel Function of First kind of Zeroth Order, evaluated at all r-rho combinations.
    real(dp), intent(in) :: j0s(:, :)
    ! Grid of fourier-space representations of the perturbation components of each AUF.
    real(dp), intent(in) :: afouriergrid(:, :, :, :, :), bfouriergrid(:, :, :, :, :)
    ! Largest considered AUF integral to be considered as potential counterparts.
    real(dp), intent(in) :: max_frac
    ! Number of sources overlapping each source in a given catalogue from the opposing catalogue.
    integer, intent(out) :: anumoverlap(size(a_ax_1)), bnumoverlap(size(b_ax_1))
    ! Loop counters.
    integer :: i, j, l
    ! Sky offsets and uncertainties
    real(dp) :: dax2, oa, ob
    ! Respective fourier-space variables.
    real(dp) :: afourier(size(afouriergrid, 1)), bfourier(size(bfouriergrid, 1)), four(size(afouriergrid, 1))
    ! Cumulative real-space probability.
    real(dp) :: cumulative
    ! Sky separations -- square of maximum allowed separation, max_sep, and the offset between
    ! opposing catalogue sources, as given by the Haversine formula.
    real(dp) :: max_sep2, dist

    max_sep2 = max_sep**2

    anumoverlap = 0
!$OMP PARALLEL DO DEFAULT(NONE) PRIVATE(i, j, l, dax2, afourier, bfourier, cumulative, oa, ob, dist, four) &
!$OMP& SHARED(a_ax_1, a_ax_2, b_ax_1, b_ax_2, max_sep, max_sep2, anumoverlap, afouriergrid, bfouriergrid, amodrefind, &
!$OMP& bmodrefind, a_axerr, b_axerr, max_frac, r, dr, rho, drho, j0s) REDUCTION(+:bnumoverlap)
    do j = 1, size(a_ax_1)
        oa = a_axerr(j)
        do i = 1, size(b_ax_1)
            ob = b_axerr(i)
            dax2 = a_ax_2(j) - b_ax_2(i)
            if (abs(dax2) <= max_sep) then
                call haversine(a_ax_1(j), b_ax_1(i), a_ax_2(j), b_ax_2(i), dist)
                dist = dist * 3600.0_dp  ! haversine returns distance in degrees
                if (max_sep2 >= dist**2) then
                    ! Calculate the cumulative integral probability by inverse Fourier transforming
                    ! the multiplication of the various Fourier-space PDFs of each convolution
                    ! component.
                    afourier = afouriergrid(:, amodrefind(1, j)+1, amodrefind(2, j)+1, &
                                               amodrefind(3, j)+1, amodrefind(4, j)+1)
                    bfourier = bfouriergrid(:, bmodrefind(1, i)+1, bmodrefind(2, i)+1, &
                                                   bmodrefind(3, i)+1, bmodrefind(4, i)+1)
                    four = afourier*bfourier*exp(-2.0_dp * pi**2 * rho**2 * (oa**2 + ob**2))
                    call cumulative_fourier_transform(four, r, dr, rho, drho, dist, j0s, cumulative)
                    if (cumulative < max_frac) then
                        anumoverlap(j) = anumoverlap(j) + 1
                        bnumoverlap(i) = bnumoverlap(i) + 1
                    end if
                end if
            end if
        end do
    end do
!$OMP END PARALLEL DO

end subroutine get_max_overlap

subroutine cumulative_fourier_transform(f, r, dr, rho, drho, dist, j0s, cumulative_prob)
    integer, parameter :: dp = kind(0.0d0)  ! double precision
    real(dp), intent(in) :: f(:), r(:), dr(:), rho(:), drho(:), dist, j0s(:, :)
    integer :: i, j
    real(dp) :: pr

    real(dp), intent(out) :: cumulative_prob

    cumulative_prob = 0.0_dp
    do j = 1, size(r)
        if (r(j) >= dist) then
            exit
        end if
        pr = 0.0_dp
        do i = 1, size(rho)
            pr = pr + rho(i) * f(i) * j0s(i, j) * drho(i)
        end do
        ! pr, the fourier transform of f, ends up being in probability density of per
        ! unit area, so we have to account for that when integrating. The final term
        ! is, essentially, 2 pi r dr
        cumulative_prob = cumulative_prob + 2.0_dp * pi * pr * (pi * ((r(j)+dr(j)/2.0_dp)**2 - (r(j)-dr(j)/2.0_dp)**2))
    end do

end subroutine cumulative_fourier_transform

end module group_sources_fortran