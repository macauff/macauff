! Licensed under a 3-clause BSD style license - see LICENSE

module misc_functions_fortran
! This module provides the Fortran code for the handling of various miscellaneous functions,
! not tied to one specific sub-section of the codebase.
implicit none
integer, parameter :: dp = kind(0.0d0)  ! double precision
real(dp), parameter :: pi = 4.0_dp*atan(1.0_dp)

contains

subroutine find_nearest_point(source_lon, source_lat, point_lon, point_lat, point_ind)
! Find the nearest on-sky distance between a series of coordinates and another list of positions.

implicit none

integer, parameter :: dp = kind(0.0d0)  ! double precision

! 1-D arrays of longitudes and latitudes (or equivalent) to search the nearest point of.
real(dp), intent(in) :: source_lon(:), source_lat(:)
! 1-D arrays of lon & lat (consistent with source_lon & source_lat) points that positons are
! sampled from.
real(dp), intent(in) :: point_lon(:), point_lat(:)

! 1-D array of indices, for each source position, of the closest position.
integer, intent(out) :: point_ind(size(source_lon))

! Loop counters
integer :: i, j
! Great-circle distance
real(dp) :: hav_dist, min_hav_dist

!$OMP PARALLEL DO DEFAULT(NONE) PRIVATE(i, min_hav_dist, hav_dist) SHARED(source_lon, source_lat, point_lon, &
!$OMP& point_lat, point_ind)
do i = 1, size(source_lon)
    min_hav_dist = 9999.9_dp
    do j = 1, size(point_lon)
        call haversine(source_lon(i), point_lon(j), source_lat(i), point_lat(j), hav_dist)
        if (hav_dist < min_hav_dist) then
            point_ind(i) = j - 1  ! pre-convert one-index fortran to zero-index python indices
            min_hav_dist = hav_dist
        end if
    end do
end do
!$OMP END PARALLEL DO

end subroutine find_nearest_point

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