! Licensed under a 3-clause BSD style license - see LICENSE

module counterpart_pairing_fortran
! This module provides the Fortran code for the final act of cross-matching, in which
! sources are paired within their respective islands, and counterparts are established.

implicit none

integer, parameter :: dp = kind(0.0d0)  ! double precision
real(dp), parameter :: pi = 4.0_dp*atan(1.0_dp)

contains

subroutine contam_match_prob(Fcc, Fcn, Fnc, Fnn, rho, drho, sep, Gcc, Gcn, Gnc, Gnn)
    ! Calculate the probability density of two sources being two detections of a single sky object
    ! given the separation between them, including the combinations of hypotheses that the sources
    ! independently suffer, or do not suffer, from a statistical description of blended source
    ! perturbation.
    integer, parameter :: dp = kind(0.0d0)  ! double precision
    ! Combinations of fourier-space representations of convolutions of AUFs. "c" and "n" represent
    ! "contaminated" and "non-contaminated", for catalogue "a" then "b" for F[a][b]. Thus the four
    ! arrays are the four combinations of AUFs that suffer perturbation or not for the two sources.
    real(dp), intent(in) :: Fcc(:), Fcn(:), Fnc(:), Fnn(:)
    ! Fourier-space representation of the F** arrays.
    real(dp), intent(in) :: rho(:), drho(:)
    ! Sky separation, in arcseconds, between the two objects in question.
    real(dp), intent(in) :: sep
    ! Output probability densities -- again, per combination of (non-)contamination -- that the
    ! two objects are a single source detected twice given their on-sky separation.
    real(dp), intent(out) :: Gcc, Gcn, Gnc, Gnn
    ! Loop counter.
    integer :: j
    ! Hankel transform parameters.
    real(dp) :: j0, z
    Gcc = 0.0_dp
    Gcn = 0.0_dp
    Gnc = 0.0_dp
    Gnn = 0.0_dp

    do j = 1, size(rho)
        z = rho(j)*sep*2.0_dp*pi
        call jy01a_j0(z, j0)
        Gcc = Gcc + rho(j) * Fcc(j) * j0 * drho(j)
        Gcn = Gcn + rho(j) * Fcn(j) * j0 * drho(j)
        Gnc = Gnc + rho(j) * Fnc(j) * j0 * drho(j)
        Gnn = Gnn + rho(j) * Fnn(j) * j0 * drho(j)
    end do

    Gcc = Gcc * 2.0_dp * pi
    Gcn = Gcn * 2.0_dp * pi
    Gnc = Gnc * 2.0_dp * pi
    Gnn = Gnn * 2.0_dp * pi

end subroutine contam_match_prob

end module counterpart_pairing_fortran