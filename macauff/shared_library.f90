! Licensed under a 3-clause BSD style license - see LICENSE

! This file provides the Fortran code for shared functions used by other fortran
! files, allowing them to share the linked subroutines.

subroutine haversine(lon1, lon2, lat1, lat2, hav_dist)
! Function to calculate the haversine formula.

implicit none

integer, parameter :: dp = kind(0.0d0)  ! double precision
real(dp), parameter :: pi = 4.0_dp*atan(1.0_dp)

! The longitudes and latitudes (orthogonal celestial sphere coordinates, in a consistent frame)
! to compute the great-circle distance of, in degrees.
real(dp), intent(in) :: lon1, lon2, lat1, lat2
! Longitude and latitude separations, and individual latitudes, but in radians.
real(dp) :: dlon_rad, dlat_rad, lat1_rad, lat2_rad
! Haversine distance, in degrees.
real(dp), intent(out) :: hav_dist

dlon_rad = (lon1 - lon2) / 180.0_dp * pi
dlat_rad = (lat1 - lat2) / 180.0_dp * pi
lat1_rad = lat1 / 180.0_dp * pi
lat2_rad = lat2 / 180.0_dp * pi

hav_dist = 2.0_dp * asin(min(1.0_dp, sqrt((sin(dlat_rad/2.0_dp))**2 + cos(lat1_rad)*cos(lat2_rad)*(sin(dlon_rad/2.0_dp)**2))))
hav_dist = hav_dist * 180.0_dp / pi  ! convert radians to degrees

end subroutine haversine

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
  real ( kind = 8 ) pi
  real ( kind = 8 ) q0
  real ( kind = 8 ) r
  real ( kind = 8 ) rp2
  real ( kind = 8 ) t1
  real ( kind = 8 ) x
  real ( kind = 8 ) x2

  pi = 3.141592653589793D+00
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
