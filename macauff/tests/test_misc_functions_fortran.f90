! Licensed under a 3-clause BSD style license - see LICENSE

subroutine haversine_wrapper(lon1, lon2, lat1, lat2, hav_dist)
! Wrapper for the haversine formula.

integer, parameter :: dp = kind(0.0d0)  ! double precision

real(dp), intent(in) :: lon1, lon2, lat1, lat2
real(dp), intent(out) :: hav_dist

call haversine(lon1, lon2, lat1, lat2, hav_dist)

end subroutine haversine_wrapper