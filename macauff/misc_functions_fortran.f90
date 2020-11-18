! Licensed under a 3-clause BSD style license - see LICENSE

module misc_functions_fortran
! This module provides the Fortran code for the handling of various miscellaneous functions,
! not tied to one specific sub-section of the codebase.
implicit none
integer, parameter :: dp = kind(0.0d0)  ! double precision
real(dp), parameter :: pi = 4.0_dp*atan(1.0_dp)

contains

subroutine calc_j0(r, r0, j0s)
    ! Wrapper for jy01a, to calculate the Bessel Function of First Kind of
    ! Zeroth Order for arrays of rs and rhos.
    integer, parameter :: dp = kind(0.0d0)  ! double precision
    ! Input r/rho arrays.
    real(dp), intent(in) :: r(:), r0(:)
    ! Output J0(x) values, where x = 2 pi r rho.
    real(dp), intent(out) :: j0s(size(r), size(r0))
    ! Loop counter integers.
    integer :: i, j
    ! Temporary holding variable, where z = 2 pi r rho.
    real(dp) :: z

!$OMP PARALLEL DO DEFAULT(NONE) PRIVATE(i, j, z) SHARED(r, r0, j0s) COLLAPSE(2)
    do i = 1, size(r0)
        do j = 1, size(r)
            z = r(j)*r0(i)*2.0_dp*pi
            call jy01a(z, j0s(j, i))
        end do
    end do
!$OMP END PARALLEL DO

end subroutine calc_j0

end module misc_functions_fortran