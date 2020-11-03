! Licensed under a 3-clause BSD style license - see LICENSE

module misc_functions_fortran
! This module provides the Fortran code for the handling of various miscenalleous functions,
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

subroutine jy01a (x, bj0)

!*****************************************************************************80
!
!! JY01A computes Bessel functions J0(x).
!
!  Licensing:
!
!    This routine is copyrighted by Shanjie Zhang and Jianming Jin.  However, 
!    they give permission to incorporate this routine into a user program 
!    provided that the copyright is acknowledged.
!
!  Modified:
!
!    01 August 2012
!
!  Author:
!
!    Shanjie Zhang, Jianming Jin
!
!  Reference:
!
!    Shanjie Zhang, Jianming Jin,
!    Computation of Special Functions,
!    Wiley, 1996,
!    ISBN: 0-471-11963-6,
!    LC: QA351.C45.
!
!  Parameters:
!
!    Input, real ( kind = 8 ) X, the argument.
!
!    Output, real ( kind = 8 ) BJ0,
!    the value of J0(x).
!
!  Modified slightly by TJW, 2020.
!
  implicit none

  real ( kind = 8 ), save, dimension(12) :: a = (/ &
    -0.7031250000000000D-01, 0.1121520996093750D+00, &
    -0.5725014209747314D+00, 0.6074042001273483D+01, &
    -0.1100171402692467D+03, 0.3038090510922384D+04, &
    -0.1188384262567832D+06, 0.6252951493434797D+07, &
    -0.4259392165047669D+09, 0.3646840080706556D+11, &
    -0.3833534661393944D+13, 0.4854014686852901D+15 /)
  real ( kind = 8 ), save, dimension(12) :: b = (/ &
    0.7324218750000000D-01, -0.2271080017089844D+00, &
    0.1727727502584457D+01, -0.2438052969955606D+02, &
    0.5513358961220206D+03, -0.1825775547429318D+05, &
    0.8328593040162893D+06, -0.5006958953198893D+08, &
    0.3836255180230433D+10, -0.3649010818849833D+12, &
    0.4218971570284096D+14, -0.5827244631566907D+16 /)
  real ( kind = 8 ) bj0
  real ( kind = 8 ) cu

  integer ( kind = 4 ) k
  integer ( kind = 4 ) k0
  real ( kind = 8 ) p0
  real ( kind = 8 ) q0
  real ( kind = 8 ) r
  real ( kind = 8 ) rp2
  real ( kind = 8 ) t1
  real ( kind = 8 ) x
  real ( kind = 8 ) x2

  rp2 = 0.63661977236758D+00
  x2 = x * x

  if ( abs( x ) <= 1.0D-05 ) then
    bj0 = 1.0D+00
    return
  end if

  if ( x <= 12.0D+00 ) then

    bj0 = 1.0D+00
    r = 1.0D+00
    do k = 1,30
      r = -0.25D+00 * r * x2 / ( k * k )
      bj0 = bj0 + r
      if ( abs ( r ) < abs ( bj0 ) * 1.0D-15 ) then
        exit
      end if
    end do

  else

    if ( x < 35.0D+00 ) then
      k0 = 12
    else if ( x < 50.0D+00 ) then
      k0 = 10
    else
      k0 = 8
    end if

    t1 = x - 0.25D+00 * pi
    p0 = 1.0D+00
    q0 = -0.125D+00 / x
    do k = 1, k0
      p0 = p0 + a(k) * x ** ( - 2 * k )
      q0 = q0 + b(k) * x ** ( - 2 * k - 1 )
    end do
    cu = sqrt ( rp2 / x )
    bj0 = cu * ( p0 * cos ( t1 ) - q0 * sin ( t1 ) )

  end if
  return
end

end module misc_functions_fortran