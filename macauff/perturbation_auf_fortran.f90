module perturbation_auf_fortran
implicit none
private
public haversine
integer, parameter :: dp = kind(0.0d0)  ! double precision
real(dp), parameter :: pi = 4.0_dp*atan(1.0_dp)

contains

subroutine haversine(lon1, lon2, lat1, lat2, hav_dist)
! Function to calculate the haversine formula.

integer, parameter :: dp = kind(0.0d0)  ! double precision

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

hav_dist = 2.0_dp * asin(sqrt((sin(dlat_rad/2.0_dp))**2 + cos(lat1_rad)*cos(lat2_rad)*(sin(dlon_rad)/2.0_dp)**2))
hav_dist = hav_dist * 180.0_dp / pi  ! convert radians to degrees

end subroutine haversine

! subroutine find_nearest_auf_point()

! end subroutine find_nearest_auf_point

end module perturbation_auf_fortran