! Licensed under a 3-clause BSD style license - see LICENSE

module perturbation_auf_fortran
! This module provides the Fortran code for the handling of the creation of the perturbation
! component of the AUF.

implicit none

integer, parameter :: dp = kind(0.0d0)  ! double precision
real(dp), parameter :: pi = 4.0_dp*atan(1.0_dp)

contains

subroutine find_nearest_auf_point(source_lon, source_lat, auf_point_lon, auf_point_lat, auf_point_ind)
! Find the nearest on-sky distance between a series of coordinates and another list of positions.

integer, parameter :: dp = kind(0.0d0)  ! double precision

! 1-D arrays of longitudes and latitudes (or equivalent) to search the nearest point of.
real(dp), intent(in) :: source_lon(:), source_lat(:)
! 1-D arrays of lon & lat (consistent with source_lon & source_lat) points that AUFs are sampled
! from in create_perturb_auf.
real(dp), intent(in) :: auf_point_lon(:), auf_point_lat(:)

! 1-D array of indices, for each source position, of the closest AUF position.
real(dp), intent(out) :: auf_point_ind(size(source_lon))

! Loop counters
integer :: i, j
! Great-circle distance
real(dp) :: hav_dist, min_hav_dist

!$OMP PARALLEL DO DEFAULT(NONE) PRIVATE(i, min_hav_dist, hav_dist) SHARED(source_lon, source_lat, auf_point_lon, &
!$OMP& auf_point_lat, auf_point_ind)
do i = 1, size(source_lon)
    min_hav_dist = 9999.9_dp
    do j = 1, size(auf_point_lon)
        call haversine(source_lon(i), auf_point_lon(j), source_lat(i), auf_point_lat(j), hav_dist)
        if (hav_dist < min_hav_dist) then
            auf_point_ind(i) = j - 1  ! pre-convert one-index fortran to zero-index python indices
            min_hav_dist = hav_dist
        end if
    end do
end do
!$OMP END PARALLEL DO

end subroutine find_nearest_auf_point

end module perturbation_auf_fortran