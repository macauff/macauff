! Licensed under a 3-clause BSD style license - see LICENSE

module counterpart_pairing_fortran
! This module provides the Fortran code for the final act of cross-matching, in which
! sources are paired within their respective islands, and counterparts are established.

implicit none

integer, parameter :: dp = kind(0.0d0)  ! double precision
real(dp), parameter :: pi = 4.0_dp*atan(1.0_dp)

contains

subroutine contam_match_prob(Fcc, Fcn, Fnc, Fnn, rho, drho, sep, Gcc, Gcn, Gnc, Gnn)
    implicit none
    double precision, intent(in) :: Fcc(:), Fcn(:), Fnc(:), Fnn(:), rho(:), drho(:), sep
    double precision, intent(out) :: Gcc, Gcn, Gnc, Gnn
    integer :: j
    double precision :: j0, z
    Gcc = 0.0_dp
    Gcn = 0.0_dp
    Gnc = 0.0_dp
    Gnn = 0.0_dp

    do j = 1, size(rho)
        z = rho(j)*sep*2.0_dp*pi
        call jy01a(z, j0)
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