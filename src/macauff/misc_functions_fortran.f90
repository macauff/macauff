! Licensed under a 3-clause BSD style license - see LICENSE

module misc_functions_fortran
! This module provides the Fortran code for the handling of various miscellaneous functions,
! not tied to one specific sub-section of the codebase.
implicit none
integer, parameter :: dp = kind(0.0d0)  ! double precision
real(dp), parameter :: pi = 4.0_dp*atan(1.0_dp)

contains

subroutine haversine_wrapper(lon1, lon2, lat1, lat2, hav_dist)
    ! Wrapper for the haversine formula in shared_library.f90.
    integer, parameter :: dp = kind(0.0d0)  ! double precision
    ! On sky longitudes and latitudes of the two sources for which we wish to find a separation.
    real(dp), intent(in) :: lon1, lon2, lat1, lat2
    ! On-sky, Haversine separation of the two sources in question.
    real(dp), intent(out) :: hav_dist

    call haversine(lon1, lon2, lat1, lat2, hav_dist)

end subroutine haversine_wrapper

subroutine find_nearest_point(source_lon, source_lat, point_lon, point_lat, point_ind)
    ! Find the nearest on-sky distance between a series of coordinates and another list of positions.
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
    ! Wrapper for jy01a_j0, to calculate the Bessel Function of First Kind of
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
            call jy01a_j0(z, j0s(j, i))
        end do
    end do
!$OMP END PARALLEL DO

end subroutine calc_j0

subroutine get_circle_area_overlap(cat_ax1, cat_ax2, density_radius, hull_ax1, hull_ax2, seed, circ_overlap_area)
    ! Calculates the amount of circle overlap with a rectangle of particular coordinates. Adapts
    ! code provided by B. Retter, from Retter, Hatchell & Naylor (2019, MNRAS, 487, 887).
    integer, parameter :: dp = kind(0.0d0)  ! double precision
    ! Coordinates in orthogonal sky axes, and radius of, circles to calculate the overlap with sky position of.
    real(dp), intent(in) :: cat_ax1(:), cat_ax2(:), density_radius
    ! Sky coordinates of the points defining the sky cutout region. Note that this should be cyclical, such
    ! that hull_ax1(1) == hull_ax1(size(hull_ax1)), hull_ax2(1) == hull_ax2(size(hull_ax2)).
    real(dp), intent(in) :: hull_ax1(:), hull_ax2(:)
    ! RNG seed.
    integer, intent(in) :: seed(:, :)
    ! Relative amount of circle inside rectangle for each unique point.
    real(dp), intent(out) :: circ_overlap_area(size(cat_ax1))

    ! Loop counters.
    integer :: i, j, k
    ! Flags for forks in the logic of calculating circle area.
    logical :: circle_too_near_edge, hull_point_inside_circle
    ! Area of circle inside rectangle, coordinates and variables of various calculations.
    real(dp) :: area, x0, y0, x1, y1, x2, y2, cross_prod, dot_prod, fraction
    ! Sampled radius and position angles of objects.
    real(dp) :: r(25000), t(25000)
    ! Distance between circle and a particular rectangle edge; Haversine distance; amount of circle outside a
    ! particular rectangle edge; and point-inside-hull parameters.
    real(dp) :: h, d, chord_area_overlap, sum_of_angles, theta

!$OMP PARALLEL DO DEFAULT(NONE) PRIVATE(i, j, k, area, h, d, chord_area_overlap, sum_of_angles, theta, x0, y0, x1, y1, x2, y2, &
!$OMP& circle_too_near_edge, hull_point_inside_circle, cross_prod, dot_prod, fraction, r, t) &
!$OMP& SHARED(density_radius, cat_ax1, cat_ax2, circ_overlap_area, hull_ax1, hull_ax2, seed)
    do j = 1, size(cat_ax1)
        call random_seed(put=seed(:, j))
        ! First, check if the circle overlaps any edges. If not, then we know what the area is trivially.
        circle_too_near_edge = .false.
        x0 = cat_ax1(j)
        y0 = cat_ax2(j)
        do k = 1, size(hull_ax1)-1
            x1 = hull_ax1(k)
            y1 = hull_ax2(k)
            x2 = hull_ax1(k+1)
            y2 = hull_ax2(k+1)
            ! https://mathworld.wolfram.com/Point-LineDistance2-Dimensional.html
            d = abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if (d <= density_radius) then
                circle_too_near_edge = .true.
                exit
            end if
        end do

        if (circle_too_near_edge) then
            ! Inside the "circle overlaps edges" part, we can see if it's a straight-line overlap or not.
            hull_point_inside_circle = .false.
            do k = 1, size(hull_ax1)-1
                call haversine(hull_ax1(k), x0, hull_ax2(k), y0, d)
                if (d <= density_radius) then
                    hull_point_inside_circle = .true.
                    exit
                end if
            end do

            if (hull_point_inside_circle) then
                ! Draw samples to determine fraction of circle inside polygon of odd shape.
                call random_number(t(:))
                call random_number(r(:))
                t(:) = t(:) * 2.0_dp * pi
                r(:) = sqrt(r(:)) * density_radius
                fraction = 0.0_dp
                do k = 1, size(t)
                    x0 = cat_ax1(j) + r(k) * cos(t(k))
                    y0 = cat_ax2(j) + r(k) * sin(t(k))
                    sum_of_angles = 0.0_dp
                    do i = 1, size(hull_ax1)-1
                        x1 = hull_ax1(i) - x0
                        y1 = hull_ax2(i) - y0
                        x2 = hull_ax1(i+1) - x0
                        y2 = hull_ax2(i+1) - y0

                        dot_prod = x1*x2 + y1*y2
                        cross_prod = x1*y2 - x2*y1
                        theta = atan2(cross_prod, dot_prod)
                        sum_of_angles = sum_of_angles + theta
                    end do
                    if (abs(sum_of_angles) > pi) then
                        fraction = fraction + 1.0_dp
                    end if
                end do
                fraction = fraction / real(size(t), dp)
                area = fraction * pi * density_radius**2
            else
                ! If no point directly inside the circle, then the area of missing circle is defined by
                ! a straight line intersecting the circle, and hence a chord integral. Taking height to
                ! be sqrt(r**2 - x**2) - h, h being the distance from the circle center to the straight
                ! line, with x the straight-line-parallel distance from the circle center, we can
                ! integrate across the chord from x = -sqrt(r**2 - h**2) to +sqrt(r**2 - h**2), giving
                x0 = cat_ax1(j)
                y0 = cat_ax2(j)
                do k = 1, size(hull_ax1)-1
                    x1 = hull_ax1(k)
                    y1 = hull_ax2(k)
                    x2 = hull_ax1(k+1)
                    y2 = hull_ax2(k+1)
                    h = abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    if (h <= density_radius) then
                        exit
                    end if
                end do
                chord_area_overlap = density_radius**2 * atan(sqrt(density_radius**2 - h**2) / h) - &
                                     h * sqrt(density_radius**2 - h**2)
                area = pi * density_radius**2 - chord_area_overlap
            end if
        else
            area = pi * density_radius**2
        end if
        circ_overlap_area(j) = area
    end do
!$OMP END PARALLEL DO

end subroutine get_circle_area_overlap

subroutine get_random_seed_size(size)
    ! Number of initial seeds expected by random_seed, to be initialised for a specified RNG setup.
    integer, intent(out) :: size

    call random_seed(size=size)

end subroutine get_random_seed_size

end module misc_functions_fortran