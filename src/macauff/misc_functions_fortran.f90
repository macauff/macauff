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
    logical :: circle_too_near_edge
    ! Area of circle inside rectangle, coordinates and variables of various calculations.
    real(dp) :: area, x0, y0, x0s(100), y0s(100), x1, y1, x2, y2, cross_prod, dot_prod, fraction
    ! Sampled radius and position angles of objects.
    real(dp) :: r(100), t(100)
    ! Distance between circle and a particular rectangle edge; Haversine distance; minimum-vector coordinates;
    ! amount of circle outside a particular rectangle edge; and point-inside-hull parameters.
    real(dp) :: h, d, z, s, xn, yn, chord_area_overlap, sum_of_angles, theta

!$OMP PARALLEL DO DEFAULT(NONE) PRIVATE(i, j, k, area, h, d, z, s, xn, yn, chord_area_overlap, sum_of_angles, theta, x0, y0, x0s, &
!$OMP& y0s, x1, y1, x2, y2, circle_too_near_edge, cross_prod, dot_prod, fraction, r, t) &
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
            if (abs(x2 - x1) <= 1e-8_dp) then
                ! If we have flat lines in either direction then the intersection point should be trivial.
                xn = x1
                yn = y0
            else if (abs(y2 - y1) <= 1e-8_dp) then
                xn = x0
                yn = y1
            else
                ! Solving s * (x2-x1, y2-y1) + (x1, y1) = t * (y2-y1, -(x2-x1)) + (x0, y0) for s,
                ! re-arranging both x- and y-versions for t and setting equal, yields
                z = (y2 - y1) / (x2 - x1)
                s = -1 / (1/z + z) * ((x1 - x0) / (y2 - y1) + (y1 - y0) / (x2 - x1))
                xn = s * (x2 - x1) + x1
                yn = s * (y2 - y1) + y1
            end if
            ! Then, despite using euclidean geometry above, we can use the more-accurate
            ! great-circle distance here for whether objects fall outside of the larger
            ! area or not.
            call haversine(xn, x0, yn, y0, d)
            if (d <= density_radius) then
                circle_too_near_edge = .true.
                exit
            end if
        end do

        if (circle_too_near_edge) then
            ! Draw samples to determine fraction of circle inside polygon of odd shape.
            ! Use the "sunflower seed arrangement" pattern to get a uniform distribution.
            do k = 1, size(t)
                t(k) = 2.0_dp * pi * real(k, dp) / ((sqrt(5.0_dp)+1.0_dp)/2.0_dp)**2  ! golden ratio phi
            end do
            do k = 1, size(r)
                ! Note that these are angular offsets and hence convert from degrees to radians!
                r(k) = sqrt(real(k, dp) - 0.5_dp)/sqrt(real(size(r), dp) - 0.5_dp) * density_radius / 180.0_dp * pi
            end do

            call distribute_objects_in_circle(cat_ax1(j), cat_ax2(j), r, t, x0s, y0s)

            fraction = 0.0_dp
            do k = 1, size(t)
                x0 = x0s(k)
                y0 = y0s(k)
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
            area = pi * density_radius**2
        end if
        circ_overlap_area(j) = area
    end do
!$OMP END PARALLEL DO

end subroutine get_circle_area_overlap

subroutine distribute_objects_in_circle(x0, y0, r, t, x0s, y0s)
    ! Distribute objects around x0, y0, such that they are a great-circle distance r from
    ! the central coordinate by an angle t.
    integer, parameter :: dp = kind(0.0d0)  ! double precision
    ! Input central coordinates, and great-circle separation and relative angle to place
    ! simulated objects.
    real(dp), intent(in) :: x0, y0, r(:), t(:)
    ! Output coordinates of simulated objects.
    real(dp), intent(out) :: x0s(size(r)), y0s(size(r))

    ! Loop counter.
    integer :: k
    ! Coordinates of input object
    real(dp) :: a, d, ne(3), x, y, z
    ! Components of the angles t and r.
    real(dp) :: cost, sint, cosr, sinr
    ! Composite in-tangent-plane vector for moving simulated object in the direction of
    ! some angle t; the vector orthogonal to this vector and its components; and the
    ! rotation matrix around vector nr.
    real(dp) :: na(3), nr(3), ux, uy, uz, r_xyz(3, 3)
    ! Cartesian and spherical coordinates of simulated source.
    real(dp) :: new_xyz(3), new_a, new_d

    a = x0 / 180.0_dp * pi
    d = y0 / 180.0_dp * pi
    ne = (/ cos(a) * cos(d), sin(a) * cos(d), sin(d) /)
    x = ne(1)
    y = ne(2)
    z = ne(3)
    do k = 1, size(t)
        cost = cos(t(k))
        sint = sin(t(k))
        na = (/ -1.0_dp * sint * sin(a) - cost * cos(a) * sin(d), &
                sint * cos(a) - cost * sin(a) * sin(d), cost * cos(d)/)
        na = na / sqrt(na(1)**2 + na(2)**2 + na(3)**2)
        ! nr = na x ne
        nr = (/ na(2)*ne(3) - na(3)*ne(2), na(3)*ne(1) - na(1)*ne(3), na(1)*ne(2) - na(2)*ne(1) /)

        ux = nr(1)
        uy = nr(2)
        uz = nr(3)
        cosr = cos(r(k))
        sinr = sin(r(k))
        ! Force C-ordered array indexing just so that the matrix multiplication below looks "neater".
        r_xyz = reshape((/ cosr + ux**2 * (1 - cosr), ux * uy * (1 - cosr) - uz * sinr, ux * uz * (1 - cosr) + uy * sinr, &
                           ux * uy * (1 - cosr) + uz * sinr, cosr + uy**2 * (1 - cosr), uy * uz * (1 - cosr) - ux * sinr, &
                           ux * uz * (1 - cosr) - uy * sinr, uy * uz * (1 - cosr) + ux * sinr, cosr + uz**2 * (1 - cosr)/), &
                        shape(r_xyz), order=(/ 2, 1 /))

        new_xyz = (/ r_xyz(1, 1) * x + r_xyz(1, 2) * y + r_xyz(1, 3) * z, &
                     r_xyz(2, 1) * x + r_xyz(2, 2) * y + r_xyz(2, 3) * z, &
                     r_xyz(3, 1) * x + r_xyz(3, 2) * y + r_xyz(3, 3) * z /)

        new_a = atan2(new_xyz(2), new_xyz(1))
        if (new_a < 0) then
            new_a = new_a + 2 * pi
        end if
        new_d = new_xyz(3) / sqrt(new_xyz(1)**2 + new_xyz(2)**2 + new_xyz(3)**2)
        new_d = asin(max(-1.0_dp, min(1.0_dp, new_d)))
        x0s(k) = new_a * 180.0_dp / pi
        y0s(k) = new_d * 180.0_dp / pi
    end do

end subroutine distribute_objects_in_circle

subroutine get_random_seed_size(size)
    ! Number of initial seeds expected by random_seed, to be initialised for a specified RNG setup.
    integer, intent(out) :: size

    call random_seed(size=size)

end subroutine get_random_seed_size

end module misc_functions_fortran