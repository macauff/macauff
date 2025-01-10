! Licensed under a 3-clause BSD style license - see LICENSE

module photometric_likelihood_fortran
! This module provides the Fortran code for the handling of the creation of photometric
! likelihood and priors based on a bandpass-bandpass pairing between catalogues.

implicit none

integer, parameter :: dp = kind(0.0d0)  ! double precision
real(dp), parameter :: pi = 4.0_dp*atan(1.0_dp)

contains

subroutine find_mag_bin_inds(mags, flags, bins, cuts)
    ! Find the indices of the bins to place all magnitudes in for a given magnitude bin array.
    integer, parameter :: dp = kind(0.0d0)  ! double precision
    ! Magnitudes to place in the appropriate magnitude bins.
    real(dp), intent(in) :: mags(:), bins(:)
    ! Flags for whether to use this source for populating magnitude bins.
    logical, intent(in) :: flags(:)
    ! Slices for each magnitude bin, indicating which sources fall into that particular bin.
    integer, intent(out) :: cuts(size(bins)-1, size(mags))

    integer :: i, j

    cuts(:, :) = 0

!$OMP PARALLEL DO DEFAULT(NONE) PRIVATE(i, j) SHARED(mags, flags, bins, cuts)
    do i = 1, size(mags)
        if (flags(i)) then
            do j = 1, size(bins)-1
                if (mags(i) >= bins(j) .and. mags(i) < bins(j+1)) then
                    cuts(j, i) = 1
                    exit
                end if
            end do
        end if
    end do
!$OMP END PARALLEL DO

end subroutine find_mag_bin_inds

subroutine get_field_dists(auf_cdf_a, a_indices, a_overlap, f_frac, a_flags, b_flags, b_mag, low_mag, upp_mag, a_mask_ind)
    ! Derive the distribution of "field" sources, those with no counterpart in the opposing
    ! catalogue, by removing any source within the "field" integral fraction radius of any object.
    integer, parameter :: dp = kind(0.0d0)  ! double precision
    ! Indices into opposing catalogue from the "primary" catalogue, and the number of
    ! overlaps for a source.
    integer, intent(in) :: a_indices(:, :), a_overlap(:)
    ! Boolean flags indicating whether a given object in a catalogue is detected in its bandpass.
    logical, intent(in) :: a_flags(:), b_flags(:)
    ! Evaluation of the AUF, integrated to the separation between potential counterparts, and fraction of
    ! counterpart objects to remove (and hence also remove "field" objects).
    real(dp), intent(in) :: auf_cdf_a(:, :), f_frac
    ! Second catalogue magnitude for this bandpass, and limits between which to consider removing
    ! sources from the field source distribution.
    real(dp), intent(in) :: b_mag(:), low_mag, upp_mag
    ! Integer representation of boolean flag determining whether a source remains as a potential
    ! field source, or if it is too close to a primary catalogue object and thus not used to
    ! construct the "unmatched" distribution.
    integer, intent(out) :: a_mask_ind(size(a_overlap))
    ! Loop counters.
    integer :: i, j, k

    a_mask_ind = 1

!$OMP PARALLEL DO DEFAULT(NONE) PRIVATE(i, j, k) SHARED(auf_cdf_a, a_indices, a_overlap, a_flags, b_flags, b_mag, &
!$OMP& low_mag, upp_mag, f_frac, a_mask_ind)
    do j = 1, size(a_overlap)
        if (a_flags(j)) then
            do k = 1, a_overlap(j)
                ! Index arrays have come from python and are zero-indexed, so correct offset here.
                i = a_indices(k, j) + 1
                if (i > 0) then
                    if (auf_cdf_a(k, j) < f_frac .and. b_flags(i) .and. b_mag(i) >= low_mag .and. b_mag(i) <= upp_mag) then
                        a_mask_ind(j) = 0
                        exit
                    end if
                end if
            end do
        else
            a_mask_ind(j) = 0
        end if
    end do
!$OMP END PARALLEL DO

end subroutine get_field_dists

subroutine brightest_mag(auf_cdf_a, a_mag, b_mag, a_indices, a_overlap, b_frac, a_err_circ, a_flags, b_flags, a_bin, &
    mag_mask, av_area)
    ! Derive the distribution of brightest sources of one catalogue inside "error circles" of
    ! another catalogue.
    integer, parameter :: dp = kind(0.0d0)  ! double precision
    ! Indices into opposing catalogue from the "primary" catalogue, and the number of
    ! overlaps for a source.
    integer, intent(in) :: a_indices(:, :), a_overlap(:)
    ! Boolean flags indicating whether a given object in a catalogue is detected in its bandpass.
    logical, intent(in) :: a_flags(:), b_flags(:)
    ! "Primary" catalogue error circle radius, evaluation of the AUF, integrated to the separation
    ! between potential counterparts, and fraction of overlapping objects to keep when considering
    ! brightest object in a given error circle.
    real(dp), intent(in) :: a_err_circ(:), auf_cdf_a(:, :), b_frac
    ! Catalogue magnitude for these bandpasses, and bins into which to divide the "a" catalogue
    ! magnitudes.
    real(dp), intent(in) :: a_mag(:), b_mag(:), a_bin(:)
    ! Integer representation of boolean array, indicating the brightest sources in catalogue "b"
    ! for all sources within bin sizes set by a_bin.
    integer, intent(out) :: mag_mask(size(b_mag), size(a_bin)-1)
    ! The average error circle size of sources in each a_bin magnitude range.
    real(dp), intent(out) :: av_area(size(a_bin)-1)
    ! Loop counters.
    integer :: i, j, k
    ! Indices keeping track of the various bins sources are assigned in this subroutine.
    integer :: brightindex, m, r
    ! Counter for the number of sources in each a_bin magnitude range.
    integer :: n(size(a_bin)-1)
    ! Brightest magnitude inside an error circle.
    real(dp) :: brightmag

    mag_mask = 0
    n = 0
    av_area = 0

!$OMP PARALLEL DO DEFAULT(NONE) PRIVATE(i, m, r) SHARED(a_err_circ, a_bin, a_mag, a_flags) REDUCTION(+: n, av_area)
    do i = 1, size(a_overlap)
        if (a_err_circ(i) > 0 .and. a_flags(i)) then
            r = 1
            do m = 2, size(a_bin)
                if ((a_bin(m) >= a_mag(i)) .and. (a_bin(m-1) < a_mag(i))) then
                    r = m-1
                    exit
                end if
            end do
            av_area(r) = av_area(r) + a_err_circ(i)**2
            n(r) = n(r) + 1
        end if
    end do
!$OMP END PARALLEL DO
    do i = 1, size(a_bin)-1
        if (n(i) > 0) then
            av_area(i) = pi*av_area(i)/n(i)
        end if
    end do

!$OMP PARALLEL DO DEFAULT(NONE) PRIVATE(i, j, k, brightindex, brightmag, m, r) SHARED(auf_cdf_a, b_mag, mag_mask, a_indices, &
!$OMP& a_overlap, a_err_circ, a_flags, b_flags, a_bin, a_mag, b_frac)
    do i = 1, size(a_overlap)
        if (a_flags(i)) then
            r = 1
            do m = 2, size(a_bin)
                if ((a_bin(m) >= a_mag(i)) .and. (a_bin(m-1) < a_mag(i))) then
                    r = m-1
                    exit
                end if
            end do
            brightindex = -1
            brightmag = 999
            do k = 1, a_overlap(i)
                j = a_indices(k, i) + 1
                if (j > 0) then
                    if (b_flags(j)) then
                        if (auf_cdf_a(k, i) < b_frac .and. b_mag(j) < brightmag) then
                            brightindex = j
                            brightmag = b_mag(j)
                        end if
                    end if
                end if
            end do
            if (brightindex /= -1) then
                mag_mask(brightindex, r) = 1
            end if
        end if
    end do
!$OMP END PARALLEL DO

end subroutine brightest_mag

end module photometric_likelihood_fortran