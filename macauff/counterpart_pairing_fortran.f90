! Licensed under a 3-clause BSD style license - see LICENSE

module counterpart_pairing_fortran
! This module provides the Fortran code for the final act of cross-matching, in which
! sources are paired within their respective islands, and counterparts are established.

use, intrinsic :: ieee_arithmetic, only: IEEE_VALUE, IEEE_QUIET_NAN

implicit none

integer, parameter :: dp = kind(0.0d0)  ! double precision
real(dp), parameter :: pi = 4.0_dp*atan(1.0_dp)

contains

subroutine find_island_probabilities(a_astro, a_photo, b_astro, b_photo, alist, alist_, blist, blist_, agrplen, bgrplen, c_array, &
    fa_array, fb_array, c_priors, fa_priors, fb_priors, amagref, bmagref, amodrefind, bmodrefind, abinsarray, abinlengths, &
    bbinsarray, bbinlengths, afrac_grids, aflux_grids, bfrac_grids, bflux_grids, afourier_grids, bfourier_grids, a_sky_inds, &
    b_sky_inds, rho, drho, n_fracs, large_len, mini_crpts_len, acountinds, bcountinds, afieldinds, bfieldinds, acontamprob, &
    bcontamprob, etaarray, xiarray, acontamflux, bcontamflux, probcarray, crptseps, probfaarray, afieldfluxs, afieldseps, &
    afieldetas, afieldxis, probfbarray, bfieldfluxs, bfieldseps, bfieldetas, bfieldxis)
    ! Calculate the final cross-matches. For each island, unique permutations of match and non-match are
    ! compared, with the highest probability combination stored. The matches and non-matches then have
    ! various secondary parameters, such as log-likelihood-ratio of match or their contamination probability
    ! derived and returned.
    integer, parameter :: dp = kind(0.0d0)  ! double precision
    ! Small and full-index versions of the island group numbers, and their lengths in the two catalogues.
    integer, intent(in) :: alist(:, :), alist_(:, :), blist(:, :), blist_(:, :), agrplen(:), bgrplen(:)
    ! Magnitude bin lengths for filters in the two catalogues.
    integer, intent(in) :: abinlengths(:, :), bbinlengths(:, :)
    ! Indices into filter, specific perturbation AUF model, and sky slice, for the respective catalogues.
    integer, intent(in) :: amagref(:), bmagref(:), amodrefind(:, :), bmodrefind(:, :), a_sky_inds(:), b_sky_inds(:)
    ! Number of contaminant fractions simulated in the perturbation AUF, and total lengths of small and
    ! large catalogues.
    integer, intent(in) :: n_fracs, large_len, mini_crpts_len
    ! TODO: *binsarray, *binslength, *_priors and *_array are all memmapped. Does this matter in practice?
    ! Should be able to take unique-sky-slice cutouts, but this'll need convenience functions for the slicing.
    ! Astrometry and photometry for the two respective catalogues.
    real(dp), intent(in) :: a_astro(:, :), a_photo(:, :), b_astro(:, :), b_photo(:, :)
    ! Photometric "counterpart" and "field" likelihood and prior arrays.
    real(dp), intent(in) :: c_array(:, :, :, :, :), fa_array(:, :, :, :), fb_array(:, :, :, :), c_priors(:, :, :), &
    fa_priors(:, :, :), fb_priors(:, :, :)
    ! Magnitude bins for both catalogues.
    real(dp), intent(in) :: abinsarray(:, :, :), bbinsarray(:, :, :)
    ! Perturbation AUF derivates: fraction of contaminated objects, average flux contamination, and AUF
    ! component fourier transforms for both catalogues.
    real(dp), intent(in) :: afrac_grids(:, :, :, :), aflux_grids(:, :, :), afourier_grids(:, :, :, :), bfrac_grids(:, :, :, :), &
    bflux_grids(:, :, :), bfourier_grids(:, :, :, :)
    ! Fourier-space grids for perturbation AUF.
    real(dp), intent(in) :: rho(:), drho(:)
    ! Indices of matches and non-matches in the two catalogues.
    integer, intent(out) :: acountinds(mini_crpts_len), bcountinds(mini_crpts_len), &
    afieldinds(size(a_astro, 1)), bfieldinds(size(b_astro, 1))
    ! Contamination probabilities of each source in the pairs of matches.
    real(dp), intent(out) :: acontamprob(n_fracs, mini_crpts_len), bcontamprob(n_fracs, mini_crpts_len)
    ! Eta and Xi -- log-likelihood-ratios in astrometry/photometry -- of each match, based on a one-to-one match.
    ! In the non-match case, nearest opposing island object's position and brightness are used.
    real(dp), intent(out) :: etaarray(mini_crpts_len), xiarray(mini_crpts_len), afieldetas(size(a_astro, 1)), &
    afieldxis(size(a_astro, 1)), bfieldetas(size(b_astro, 1)), bfieldxis(size(b_astro, 1))
    ! Average simulated contaminating flux for each match or non-match.
    real(dp), intent(out) :: acontamflux(mini_crpts_len), bcontamflux(mini_crpts_len), afieldfluxs(size(a_astro, 1)), &
    bfieldfluxs(size(b_astro, 1))
    ! Probability of (non-)match.
    real(dp), intent(out) :: probcarray(mini_crpts_len), probfaarray(size(a_astro, 1)), probfbarray(size(b_astro, 1))
    ! Separations between pair of matches, or of nearest opposing island object if a non-match.
    real(dp), intent(out) :: crptseps(mini_crpts_len), afieldseps(size(a_astro, 1)), bfieldseps(size(b_astro, 1))
    ! Indices into islands, or to map each island to a unique part of the larger array to avoid index clashes.
    integer :: i, j, match_cumulat_ind(size(alist, 2)), afield_cumulat_ind(size(alist, 2)), bfield_cumulat_ind(size(alist, 2))
    ! NaN, and temporary non-normalised posterior, and evidence, holders.
    real(dp) :: nan, prob, integral
    ! Smaller arrays holding one islands' worth of each output described above.
    real(dp), allocatable :: acrptscontp(:, :), bcrptscontp(:, :), etacrpts(:), xicrpts(:), acrptflux(:), &
    bcrptflux(:), crptsep(:), afieldflux(:), bfieldflux(:), afieldsep(:), afieldeta(:), afieldxi(:), &
    bfieldsep(:), bfieldeta(:), bfieldxi(:)
    integer, allocatable :: acrpts(:), bcrpts(:), afield(:), bfield(:)

    nan = IEEE_VALUE(nan, IEEE_QUIET_NAN)

    ! Start by calculating the cumulative maximum index of each island, assuming either 0%
    ! or 100% match rate.
    match_cumulat_ind(1) = 1
    afield_cumulat_ind(1) = 1
    bfield_cumulat_ind(1) = 1
    do i = 2, size(alist, 2)
        match_cumulat_ind(i) = match_cumulat_ind(i-1) + min(agrplen(i-1), bgrplen(i-1))
        afield_cumulat_ind(i) = afield_cumulat_ind(i-1) + agrplen(i-1)
        bfield_cumulat_ind(i) = bfield_cumulat_ind(i-1) + bgrplen(i-1)
    end do

    acountinds(:) = large_len + 1
    bcountinds(:) = large_len + 1
    etaarray(:) = -100.0_dp
    xiarray(:) = -100.0_dp
    acontamprob(:, :) = -100.0_dp
    bcontamprob(:, :) = -100.0_dp
    acontamflux(:) = -100.0_dp
    bcontamflux(:) = -100.0_dp
    probcarray(:) = -100.0_dp
    crptseps(:) = -100.0_dp

    afieldinds(:) = large_len + 1
    afieldfluxs(:) = -100.0_dp
    afieldseps(:) = -100.0_dp
    afieldetas(:) = -100.0_dp
    afieldxis(:) = -100.0_dp
    probfaarray(:) = -100.0_dp

    bfieldinds(:) = large_len + 1
    bfieldfluxs(:) = -100.0_dp
    bfieldseps(:) = -100.0_dp
    bfieldetas(:) = -100.0_dp
    bfieldxis(:) = -100.0_dp
    probfbarray(:) = -100.0_dp

!$OMP PARALLEL DO DEFAULT(NONE) PRIVATE(i, j, acrpts, bcrpts, acrptscontp, bcrptscontp, etacrpts, xicrpts, acrptflux, bcrptflux, &
!$OMP& crptsep, afield, bfield, afieldflux, bfieldflux, afieldsep, afieldeta, afieldxi, bfieldsep, bfieldeta, bfieldxi, prob, &
!$OMP& integral) SHARED(a_astro, a_photo, b_astro, b_photo, c_array, fa_array, fb_array, c_priors, fa_priors, fb_priors, &
!$OMP& abinsarray, bbinsarray, abinlengths, bbinlengths, afrac_grids, aflux_grids, afourier_grids, bfrac_grids, bflux_grids, &
!$OMP& bfourier_grids, rho, drho, n_fracs, large_len, alist, blist, alist_, blist_, amagref, a_sky_inds, bmagref, b_sky_inds, &
!$OMP& amodrefind, bmodrefind, acountinds, bcountinds, afieldinds, bfieldinds, acontamprob, bcontamprob, etaarray, xiarray, &
!$OMP& acontamflux, bcontamflux, probcarray, crptseps, probfaarray, afieldfluxs, afieldseps, afieldetas, afieldxis, probfbarray, &
!$OMP& bfieldfluxs, bfieldseps, bfieldetas, bfieldxis, agrplen, bgrplen, match_cumulat_ind, afield_cumulat_ind, &
!$OMP& bfield_cumulat_ind, nan)
    do i = 1, size(alist, 2)
        if (agrplen(i) == 0) then
            bfieldinds(bfield_cumulat_ind(i):bfield_cumulat_ind(i)+bgrplen(i)-1) = blist_(:bgrplen(i), i)
            probfbarray(bfield_cumulat_ind(i):bfield_cumulat_ind(i)+bgrplen(i)-1) = 1
            do j = 1, bgrplen(i)
                bfieldfluxs(bfield_cumulat_ind(i)+j-1) = bflux_grids( &
                    bmodrefind(1, blist(j, i)+1)+1, bmodrefind(2, blist(j, i)+1)+1, bmodrefind(3, blist(j, i)+1)+1)
            end do
            bfieldseps(bfield_cumulat_ind(i):bfield_cumulat_ind(i)+bgrplen(i)-1) = nan
            bfieldetas(bfield_cumulat_ind(i):bfield_cumulat_ind(i)+bgrplen(i)-1) = nan
            bfieldxis(bfield_cumulat_ind(i):bfield_cumulat_ind(i)+bgrplen(i)-1) = nan
        else if (bgrplen(i) == 0) then
            afieldinds(afield_cumulat_ind(i):afield_cumulat_ind(i)+agrplen(i)-1) = alist_(:agrplen(i), i)
            probfaarray(afield_cumulat_ind(i):afield_cumulat_ind(i)+agrplen(i)-1) = 1
            do j = 1, agrplen(i)
                afieldfluxs(afield_cumulat_ind(i)+j-1) = aflux_grids( &
                    amodrefind(1, alist(j, i)+1)+1, amodrefind(2, alist(j, i)+1)+1, amodrefind(3, alist(j, i)+1)+1)
            end do
            afieldseps(afield_cumulat_ind(i):afield_cumulat_ind(i)+agrplen(i)-1) = nan
            afieldetas(afield_cumulat_ind(i):afield_cumulat_ind(i)+agrplen(i)-1) = nan
            afieldxis(afield_cumulat_ind(i):afield_cumulat_ind(i)+agrplen(i)-1) = nan
        else
            ! Allocate all of the outputs from find_single_island_prob...
            allocate( &
                acrpts(min(agrplen(i), bgrplen(i))), bcrpts(min(agrplen(i), bgrplen(i))), &
                acrptscontp(n_fracs, min(agrplen(i), bgrplen(i))), bcrptscontp(n_fracs, min(agrplen(i), bgrplen(i))), &
                etacrpts(min(agrplen(i), bgrplen(i))), xicrpts(min(agrplen(i), bgrplen(i))), &
                acrptflux(min(agrplen(i), bgrplen(i))), bcrptflux(min(agrplen(i), bgrplen(i))), &
                crptsep(min(agrplen(i), bgrplen(i))), afield(agrplen(i)), bfield(bgrplen(i)), afieldflux(agrplen(i)), &
                bfieldflux(bgrplen(i)), afieldsep(agrplen(i)), afieldeta(agrplen(i)), afieldxi(agrplen(i)), &
                bfieldsep(bgrplen(i)), bfieldeta(bgrplen(i)), bfieldxi(bgrplen(i)))
            ! Lots of +1 indexing here as the pointers were all generated in Python. Note that *list_ does NOT
            ! get +1 indexed because it's never used and only ends up documenting the full-scale indexes of
            ! (non-)matches, and might as well stay in Python-indexing throughout.
            call find_single_island_prob(a_astro, a_photo, b_astro, b_photo, c_array, fa_array, fb_array, c_priors, fa_priors, &
                fb_priors, abinsarray, bbinsarray, abinlengths, bbinlengths, afrac_grids, aflux_grids, afourier_grids, &
                bfrac_grids, bflux_grids, bfourier_grids, rho, drho, n_fracs, large_len, alist(:agrplen(i), i)+1, &
                blist(:bgrplen(i), i)+1, alist_(:agrplen(i), i), blist_(:bgrplen(i), i), amagref(alist(:agrplen(i), i)+1)+1, &
                a_sky_inds(alist(:agrplen(i), i)+1)+1, bmagref(blist(:bgrplen(i), i)+1)+1, b_sky_inds(blist(:bgrplen(i), i)+1)+1, &
                amodrefind(:, alist(:agrplen(i), i)+1)+1, bmodrefind(:, blist(:bgrplen(i), i)+1)+1, acrpts, bcrpts, acrptscontp, &
                bcrptscontp, etacrpts, xicrpts, acrptflux, bcrptflux, crptsep, afield, bfield, &
                afieldflux, bfieldflux, afieldsep, afieldeta, afieldxi, bfieldsep, bfieldeta, bfieldxi, prob, integral)
            ! Similar to in the main counterpart_pairing loop, we can assume either 100% or 0%
            ! match rates, and handle the missing values later. This assignment is therefore just
            ! a version of "know how large each island is, and jump the N/M a/b sources for each
            ! island".
            afieldinds(afield_cumulat_ind(i):afield_cumulat_ind(i)+agrplen(i)-1) = afield
            probfaarray(afield_cumulat_ind(i):afield_cumulat_ind(i)+agrplen(i)-1) = prob / integral
            afieldfluxs(afield_cumulat_ind(i):afield_cumulat_ind(i)+agrplen(i)-1) = afieldflux
            afieldseps(afield_cumulat_ind(i):afield_cumulat_ind(i)+agrplen(i)-1) = afieldsep
            afieldetas(afield_cumulat_ind(i):afield_cumulat_ind(i)+agrplen(i)-1) = afieldeta
            afieldxis(afield_cumulat_ind(i):afield_cumulat_ind(i)+agrplen(i)-1) = afieldxi

            bfieldinds(bfield_cumulat_ind(i):bfield_cumulat_ind(i)+bgrplen(i)-1) = bfield
            probfbarray(bfield_cumulat_ind(i):bfield_cumulat_ind(i)+bgrplen(i)-1) = prob / integral
            bfieldfluxs(bfield_cumulat_ind(i):bfield_cumulat_ind(i)+bgrplen(i)-1) = bfieldflux
            bfieldseps(bfield_cumulat_ind(i):bfield_cumulat_ind(i)+bgrplen(i)-1) = bfieldsep
            bfieldetas(bfield_cumulat_ind(i):bfield_cumulat_ind(i)+bgrplen(i)-1) = bfieldeta
            bfieldxis(bfield_cumulat_ind(i):bfield_cumulat_ind(i)+bgrplen(i)-1) = bfieldxi

            acountinds(match_cumulat_ind(i):match_cumulat_ind(i)+size(acrpts)-1) = acrpts
            bcountinds(match_cumulat_ind(i):match_cumulat_ind(i)+size(acrpts)-1) = bcrpts
            etaarray(match_cumulat_ind(i):match_cumulat_ind(i)+size(acrpts)-1) = etacrpts
            xiarray(match_cumulat_ind(i):match_cumulat_ind(i)+size(acrpts)-1) = xicrpts
            acontamprob(:, match_cumulat_ind(i):match_cumulat_ind(i)+size(acrpts)-1) = acrptscontp
            bcontamprob(:, match_cumulat_ind(i):match_cumulat_ind(i)+size(acrpts)-1) = bcrptscontp
            acontamflux(match_cumulat_ind(i):match_cumulat_ind(i)+size(acrpts)-1) = acrptflux
            bcontamflux(match_cumulat_ind(i):match_cumulat_ind(i)+size(acrpts)-1) = bcrptflux
            probcarray(match_cumulat_ind(i):match_cumulat_ind(i)+size(acrpts)-1) = prob / integral
            crptseps(match_cumulat_ind(i):match_cumulat_ind(i)+size(acrpts)-1) = crptsep

            deallocate(acrpts, bcrpts, acrptscontp, bcrptscontp, etacrpts, xicrpts, acrptflux, bcrptflux, crptsep, afield, bfield, &
            afieldflux, bfieldflux, afieldsep, afieldeta, afieldxi, bfieldsep, bfieldeta, bfieldxi)
        end if
    end do
!$OMP END PARALLEL DO

end subroutine find_island_probabilities

subroutine find_single_island_prob(a_astro, a_photo, b_astro, b_photo, c_array, fa_array, fb_array, c_priors, fa_priors, &
    fb_priors, abinsarray, bbinsarray, abinlengths, bbinlengths, afrac_grids, aflux_grids, afourier_grids, bfrac_grids, &
    bflux_grids, bfourier_grids, rho, drho, n_fracs, large_len, aperm, bperm, aperm_, bperm_, aused, qa, bused, qb, arefinds, &
    brefinds, acrpts, bcrpts, acrptscontp, bcrptscontp, etacrpts, xicrpts, acrptflux, bcrptflux, crptseps, afield, bfield, &
    afieldflux, bfieldflux, afieldseps, afieldeta, afieldxi, bfieldseps, bfieldeta, bfieldxi, prob, integral)
    ! Calculate a single island match and non-match permutation. For each combination of N catalogue "a"
    ! and M catalogue "b" sources, find the pairings and non-pairings that maximise the likelihood. Also
    ! calculates secondary parameters such as likelihood ratios for astrometry and photometry and
    ! contamination probabilities for matches.
    integer, parameter :: dp = kind(0.0d0)  ! double precision
    integer, parameter :: int64 = selected_int_kind(15)  ! 64-bit integer
    ! Small- and large-index versions of the indices of the objects in each catalogue in this island.
    integer, intent(in) :: aperm(:), bperm(:), aperm_(:), bperm_(:)
    ! Magnitude and sky filter indices for catalogues 'a' and 'b'.
    integer, intent(in) :: aused(:), qa(:), bused(:), qb(:)
    ! Perturbation AUF grid indices for each source in the two catalogues.
    integer, intent(in) :: arefinds(:, :), brefinds(:, :)
    ! Lengths of the magnitude bin arrays for each catalogue.
    integer, intent(in) :: abinlengths(:, :), bbinlengths(:, :)
    ! Number of contamination fractions recorded in each simulated perturbation AUF, and largest index value.
    integer, intent(in) :: n_fracs, large_len
    ! Astrometry and photometry for the two respective catalogues.
    real(dp), intent(in) :: a_astro(:, :), a_photo(:, :), b_astro(:, :), b_photo(:, :)
    ! Photometric "counterpart" and "field" likelihood and prior arrays.
    real(dp), intent(in) :: c_array(:, :, :, :, :), fa_array(:, :, :, :), fb_array(:, :, :, :), c_priors(:, :, :), &
    fa_priors(:, :, :), fb_priors(:, :, :)
    ! Magnitude bins for both catalogues.
    real(dp), intent(in) :: abinsarray(:, :, :), bbinsarray(:, :, :)
    ! Perturbation AUF derivates: fraction of contaminated objects, average flux contamination, and AUF
    ! component fourier transforms for both catalogues.
    real(dp), intent(in) :: afrac_grids(:, :, :, :), aflux_grids(:, :, :), afourier_grids(:, :, :, :), bfrac_grids(:, :, :, :), &
    bflux_grids(:, :, :), bfourier_grids(:, :, :, :)
    ! Fourier-space grids for perturbation AUF.
    real(dp), intent(in) :: rho(:), drho(:)
    ! Contamination probabilities of matches and non-matches from the island.
    real(dp), intent(out) :: acrptscontp(n_fracs, min(size(aperm), size(bperm))), &
    bcrptscontp(n_fracs, min(size(aperm), size(bperm)))
    ! Log-likelihood-ratios for astrometry and photometry for matches and non-matches. For non-matches,
    ! the nearest neighbour is chosen.
    real(dp), intent(out) :: etacrpts(min(size(aperm), size(bperm))), xicrpts(min(size(aperm), size(bperm))), &
    afieldeta(size(aperm)), afieldxi(size(aperm)), bfieldeta(size(bperm)), bfieldxi(size(bperm))
    ! Average contaminating simulated flux in the perturbation AUF generated for these (non-)matches.
    real(dp), intent(out) :: acrptflux(min(size(aperm), size(bperm))), bcrptflux(min(size(aperm), size(bperm))), &
    afieldflux(size(aperm)), bfieldflux(size(bperm))
    ! Separations between matches and non-matches. For non-matches, the nearest neighbour is picked.
    real(dp), intent(out) :: crptseps(min(size(aperm), size(bperm))), afieldseps(size(aperm)), bfieldseps(size(bperm))
    ! Non-normalised posterior and accompanying normalising evidence for the maximum-likelihood match.
    real(dp), intent(out) :: prob, integral
    ! Pairs of matched indices, and individual non-match indices, for sources in this island.
    integer, intent(out) :: acrpts(min(size(aperm), size(bperm))), bcrpts(min(size(aperm), size(bperm))), &
    afield(size(aperm)), bfield(size(bperm))
    ! Loop counter and indices.
    integer :: i, j, k, l, ff, N, t_ind
    ! Number of permutations or combinations for a given number of chosen items from an array.
    integer :: a_n_combs, b_n_combs, b_n_perms_per_comb
    ! Indices into magnitude bins for each source in the islands.
    integer :: bina(size(aperm)), binb(size(bperm))
    ! Counterpart and "field" probability for all combinations of match between opposing catalogue objects.
    real(dp) :: counterpartgrid(size(aperm), size(bperm)), afieldarray(size(aperm)), bfieldarray(size(bperm))
    ! Log-likelihood-ratio storage arrays for astrometry and photometry.
    real(dp) :: etagrid(size(aperm), size(bperm)), xigrid(size(aperm), size(bperm))
    ! Average simulated contaminating flux for each source in the island.
    real(dp) :: acontamfluxgrid(size(aperm)), bcontamfluxgrid(size(bperm))
    ! "Field" source prior and likelihood for each object in the island.
    real(dp) :: Nfa(size(aperm)), Nfb(size(bperm)), fa(size(aperm)), fb(size(bperm))
    ! Storage for average flux contamination and probability of flux contamination of potential matches.
    real(dp) :: aF(n_fracs), bF(n_fracs), acontamprobgrid(n_fracs, size(aperm), size(bperm)), &
    bcontamprobgrid(n_fracs, size(aperm), size(bperm))
    ! Fourier-space representations of centroiding component of AUF, and its corresponding centroid
    ! uncertainty, and temporary storage of a particular simulated perturbation AUF component.
    real(dp) :: afourgauss(size(drho)), bfourgausses(size(rho)-1, size(bperm)), aoffs(size(drho)), asig, boffs(size(drho)), bsig
    ! Real-space probabilities of match in various contamination combinations, and fourier-space
    ! representations of AUFs.
    real(dp) :: G, G0cc(size(drho)), G0cn(size(drho)), G0nc(size(drho)), G0nn(size(drho)), Gcc, Gcn, Gnc, Gnn
    ! Temporary storage: counterpart prior and likelihood, match evidence, posterior probability,
    ! and sky separation; NaN storage.
    real(dp) :: Nc, cdmdm, pr, tempprob, sep, tempsep, seps(size(aperm), size(bperm)), nan
    ! Arrays to keep track of which island indices are matches and which non-matches for a given permutation.
    integer, allocatable :: ya(:), yb(:), ta(:), tb(:)
    ! Flag for removing the ya or yb indices from aperm or bperm to get ta or tb.
    logical :: flag
    ! Combinations of catalogue a indices, and permutations of catalogue b indices, to provide the overall
    ! set of all a-b matches and non-matches.
    integer, allocatable :: a_combinations(:, :), b_permutations(:, :)
    ! Temporary storage of 64-bit integer factorial numbers.
    integer(int64) :: fac1, fac2

    nan = IEEE_VALUE(nan, IEEE_QUIET_NAN)

    do j = 1, size(aperm)
        bina(j) = maxloc(a_photo(aperm(j), aused(j)) - abinsarray(:abinlengths(aused(j), qa(j)), aused(j), qa(j)), &
                         mask=(a_photo(aperm(j), aused(j)) - abinsarray(:abinlengths(aused(j), qa(j)), aused(j), qa(j)) >= 0), &
                         dim=1)
        ! For the field sources we don't know which other filter to use, so we
        ! just default to using the first filter in the opposing catalogue,
        ! but it shouldn't matter since it ought to be independent.
        Nfa(j) = fa_priors(1, aused(j), qa(j))
        fa(j) = fa_array(bina(j), 1, aused(j), qa(j))
        acontamfluxgrid(j) = aflux_grids(arefinds(1, j), arefinds(2, j), arefinds(3, j))
    end do

    do j = 1, size(bperm)
        binb(j) = maxloc(b_photo(bperm(j), bused(j)) - bbinsarray(:bbinlengths(bused(j), qb(j)), bused(j), qb(j)), &
                         mask=(b_photo(bperm(j), bused(j)) - bbinsarray(:bbinlengths(bused(j), qb(j)), bused(j), qb(j)) >= 0), &
                         dim=1)
        Nfb(j) = fb_priors(bused(j), 1, qb(j))
        fb(j) = fb_array(binb(j), bused(j), 1, qb(j))
        bcontamfluxgrid(j) = bflux_grids(brefinds(1, j), brefinds(2, j), brefinds(3, j))
    end do

    do j = 1, size(bperm)
        bsig = b_astro(bperm(j), 3)
        bfourgausses(:, j) = exp(-2.0_dp * pi**2 * (rho(:size(rho)-1)+drho/2.0_dp)**2 * bsig**2)
    end do

    afieldarray = Nfa*fa
    bfieldarray = Nfb*fb

    do j = 1, size(aperm)
        aF = afrac_grids(:, arefinds(1, j), arefinds(2, j), arefinds(3, j))
        aoffs = afourier_grids(:, arefinds(1, j), arefinds(2, j), arefinds(3, j))
        asig = a_astro(aperm(j), 3)
        afourgauss = exp(-2.0_dp * pi**2 * (rho(:size(rho)-1)+drho/2.0_dp)**2 * asig**2)
        do k = 1, size(bperm)
            call haversine(a_astro(aperm(j), 1), b_astro(bperm(k), 1), a_astro(aperm(j), 2), b_astro(bperm(k), 2), sep)
            ! sep comes out of haversine in degrees, but contam_match_prob
            ! assumes everything is in arcseconds, so convert sep here.
            sep = sep * 3600.0_dp
            bF = bfrac_grids(:, brefinds(1, k), brefinds(2, k), brefinds(3, k))
            boffs = bfourier_grids(:, brefinds(1, k), brefinds(2, k), brefinds(3, k))

            ! Calculate the probability densities of all four combinations
            ! of perturbation and non-perturbation AUFs.
            G0nn = afourgauss*bfourgausses(:, k)
            G0cc = aoffs*boffs*G0nn
            G0cn = aoffs*G0nn
            G0nc = boffs*G0nn
            call contam_match_prob(G0cc, G0cn, G0nc, G0nn, rho(:size(rho)-1)+drho/2.0_dp, drho, sep, Gcc, Gcn, Gnc, Gnn)
            ! G would be in units of per square arcseconds, but we need it
            ! in units of per square degree to compare to Nf.
            G = Gcc * 3600.0_dp**2

            do ff = 1, n_fracs
                pr = aF(ff)*bF(ff)*Gcc + aF(ff)*(1-bF(ff))*Gcn + (1-aF(ff))*bF(ff)*Gnc + (1-aF(ff))*(1-bF(ff))*Gnn
                ! Marginalise over the opposite source contamination probability
                ! to calculate specific source's contamination chance.
                acontamprobgrid(ff, j, k) = min(1.0_dp, max(0.0_dp, (aF(ff)*bF(ff)*Gcc + aF(ff)*(1-bF(ff))*Gcn)/pr))
                bcontamprobgrid(ff, j, k) = min(1.0_dp, max(0.0_dp, (aF(ff)*bF(ff)*Gcc + (1-aF(ff))*bF(ff)*Gnc)/pr))
            end do
            Nc = c_priors(bused(k), aused(j), qb(k))
            cdmdm = c_array(binb(k), bina(j), bused(k), aused(j), qb(k))
            counterpartgrid(j, k) = Nc * G * cdmdm

            if (abs(fa(j) * fb(k)) < 1e-7_dp) then
                etagrid(j, k) = 10.0_dp
            else if (abs(cdmdm) < 1e-7_dp) then
                etagrid(j, k) = -10.0_dp
            else
                etagrid(j, k) = log10(cdmdm / (fa(j) * fb(k)))
            end if

            if (abs(Nfa(j) * Nfb(k)) < 1e-7_dp) then
                xigrid(j, k) = 10.0_dp
            else if (abs(Nc * G) < 1e-7_dp) then
                xigrid(j, k) = -10.0_dp
            else
                xigrid(j, k) = log10(Nc * G / (Nfa(j) * Nfb(k)))
            end if

            seps(j, k) = sep
        end do
    end do

    ! Start with the case of no matches between any island objects. Assume
    ! by default that no sources match in the island. Add a small normalising constant just in case.
    tempprob = product(afieldarray) * product(bfieldarray) + 1e-10_dp
    integral = tempprob
    prob = tempprob

    ! Default values for arrays need assigning to counterpart grids, and the field arrays fully
    ! filling in to initialise everything.
    acrpts(:) = large_len + 1
    bcrpts(:) = large_len + 1
    acrptscontp(:, :) = -100.0_dp
    bcrptscontp(:, :) = -100.0_dp
    etacrpts(:) = -100.0_dp
    xicrpts(:) = -100.0_dp
    acrptflux(:) = -100.0_dp
    bcrptflux(:) = -100.0_dp
    crptseps(:) = -100.0_dp

    afield(:) = aperm_(:)
    bfield(:) = bperm_(:)
    afieldflux(:) = acontamfluxgrid(:)
    bfieldflux(:) = bcontamfluxgrid(:)

    do j = 1, size(aperm)
        tempsep = 1e10_dp
        do k = 1, size(bperm)
            if (seps(j, k) < tempsep) then
                afieldseps(j) = seps(j, k)
                afieldeta(j) = etagrid(j, k)
                afieldxi(j) = xigrid(j, k)
                tempsep = seps(j, k)
            end if
        end do
    end do

    do k = 1, size(bperm)
        tempsep = 1e10_dp
        do j = 1, size(aperm)
            if (seps(j, k) < tempsep) then
                bfieldseps(k) = seps(j, k)
                bfieldeta(k) = etagrid(j, k)
                bfieldxi(k) = xigrid(j, k)
                tempsep = seps(j, k)
            end if
        end do
    end do

    do N = 1, min(size(aperm), size(bperm))
        ! Combinations of m-pick-n are m! / n! / (m - n)!
        call factorial(int(size(aperm), int64), int(N, int64), fac1)
        ! fac1 = m! / (m - n)!, fac2 = n! / (n - n)! = n!
        call factorial(int(N, int64), int(N, int64), fac2)
        a_n_combs = int(fac1 / fac2)
        call factorial(int(size(bperm), int64), int(N, int64), fac1)
        call factorial(int(N, int64), int(N-1, int64), fac2)
        b_n_combs = int(fac1 / fac2)
        ! Permutations of an n-sized array (n-pick-n) are n!
        call factorial(int(N, int64), int(N-1, int64), fac1)
        b_n_perms_per_comb = int(fac1)
        allocate(a_combinations(N, a_n_combs), b_permutations(N, b_n_combs*b_n_perms_per_comb))
        call calc_combs(size(aperm), a_n_combs, N, a_combinations)
        call calc_permcombs(size(bperm), N, b_n_perms_per_comb, b_n_combs, b_permutations)
        allocate(ya(size(a_combinations, 1)), yb(size(b_permutations, 1)))
        allocate(ta(size(aperm, 1) - size(a_combinations, 1)), tb(size(bperm, 1) - size(b_permutations, 1)))
        do i = 1, size(a_combinations, 2)
            do j = 1, size(b_permutations, 2)
                ! For paired sources, order matters, so we have to find the
                ! index of the array holding the source's overall catalogue
                ! index that matches that index in the permutation list.
                ya = a_combinations(:, i)
                yb = b_permutations(:, j)
                ! This should be the same as np.delete(np.arange(0, len(aperm)), ya), in which we look
                ! at each aperm index (np.arange(0, len(aperm))), ask if it's in ya at all, and if it
                ! isn't then we keep it in ta.
                t_ind = 1
                do k = 1, size(aperm)
                    flag = .True.
                    do l = 1, size(ya)
                        if (k == ya(l)) then
                            flag = .False.
                        end if
                    end do
                    if (flag) then
                        ta(t_ind) = k
                        t_ind = t_ind + 1
                    end if
                end do
                t_ind = 1
                do k = 1, size(bperm)
                    flag = .True.
                    do l = 1, size(yb)
                        if (k == yb(l)) then
                            flag = .False.
                        end if
                    end do
                    if (flag) then
                        tb(t_ind) = k
                        t_ind = t_ind + 1
                    end if
                end do
                tempprob = 1
                do k = 1, size(ya)
                    tempprob = tempprob * counterpartgrid(ya(k), yb(k))
                end do
                do k = 1, size(ta)
                    tempprob = tempprob * afieldarray(ta(k))
                end do
                do k = 1, size(tb)
                    tempprob = tempprob * bfieldarray(tb(k))
                end do
                integral = integral + tempprob
                if (tempprob > prob) then
                    prob = tempprob
                    ! Reset the grids to include "default" values:
                    acrpts(:) = large_len + 1
                    bcrpts(:) = large_len + 1
                    acrptscontp(:, :) = -100.0_dp
                    bcrptscontp(:, :) = -100.0_dp
                    etacrpts(:) = -100.0_dp
                    xicrpts(:) = -100.0_dp
                    acrptflux(:) = -100.0_dp
                    bcrptflux(:) = -100.0_dp
                    crptseps(:) = -100.0_dp

                    afield(:) = large_len + 1
                    bfield(:) = large_len + 1
                    afieldflux(:) = -100.0_dp
                    bfieldflux(:) = -100.0_dp
                    do k = 1, size(ya)
                        acrpts(k) = aperm_(ya(k))
                        acrptflux(k) = acontamfluxgrid(ya(k))
                        bcrpts(k) = bperm_(yb(k))
                        bcrptflux(k) = bcontamfluxgrid(yb(k))
                        acrptscontp(:, k) = acontamprobgrid(:, ya(k), yb(k))
                        bcrptscontp(:, k) = bcontamprobgrid(:, ya(k), yb(k))
                        etacrpts(k) = etagrid(ya(k), yb(k))
                        xicrpts(k) = xigrid(ya(k), yb(k))
                        crptseps(k) = seps(ya(k), yb(k))
                    end do
                    do k = 1, size(ta)
                        afield(k) = aperm_(ta(k))
                        afieldflux(k) = acontamfluxgrid(ta(k))
                    end do
                    do k = 1, size(tb)
                        bfield(k) = bperm_(tb(k))
                        bfieldflux(k) = bcontamfluxgrid(tb(k))
                    end do
                    afieldseps(:) = nan
                    afieldeta(:) = nan
                    afieldxi(:) = nan
                    ! TODO: Do we want these to be nearest of all sources, or nearest non-paired object?
                    do k = 1, size(ta)
                        tempsep = 1e10_dp
                        do l = 1, size(tb)
                            if (seps(ta(k), tb(l)) < tempsep) then
                                afieldseps(k) = seps(ta(k), tb(l))
                                afieldeta(k) = etagrid(ta(k), tb(l))
                                afieldxi(k) = xigrid(ta(k), tb(l))
                                tempsep = seps(ta(k), tb(l))
                            end if
                        end do
                    end do
                    bfieldseps(:) = nan
                    bfieldeta(:) = nan
                    bfieldxi(:) = nan
                    do l = 1, size(tb)
                        tempsep = 1e10_dp
                        do k = 1, size(ta)
                            if (seps(ta(k), tb(l)) < tempsep) then
                                bfieldseps(l) = seps(ta(k), tb(l))
                                bfieldeta(l) = etagrid(ta(k), tb(l))
                                bfieldxi(l) = xigrid(ta(k), tb(l))
                                tempsep = seps(ta(k), tb(l))
                            end if
                        end do
                    end do
                end if
            end do
        end do
        deallocate(a_combinations, b_permutations)
        deallocate(ya, yb, ta, tb)
    end do

end subroutine find_single_island_prob

subroutine factorial(N, M, g)
    ! Calculates N * (N-1) * (N-2) * ... * (N-M+1), equivalent to N! / (N-M)!.
    ! If M = N or M = N - 1 this is the same as N! and if M = 1 it is equal to N.
    integer, parameter :: int64 = selected_int_kind(15)  ! 64-bit integer
    ! The number to calculate the factorial of.
    integer(int64), intent(in) :: N, M
    ! The returned value g = N! / (N-M)!.
    integer(int64), intent(out) :: g
    ! Loop counter.
    integer(int64) :: k

    g = 1
    do k = N-M+1, N
        g = g * k
    end do

end subroutine factorial

subroutine contam_match_prob(Fcc, Fcn, Fnc, Fnn, rho, drho, sep, Gcc, Gcn, Gnc, Gnn)
    ! Calculate the probability density of two sources being two detections of a single sky object
    ! given the separation between them, including the combinations of hypotheses that the sources
    ! independently suffer, or do not suffer, from a statistical description of blended source
    ! perturbation.
    integer, parameter :: dp = kind(0.0d0)  ! double precision
    ! Combinations of fourier-space representations of convolutions of AUFs. "c" and "n" represent
    ! "contaminated" and "non-contaminated", for catalogue "a" then "b" for F[a][b]. Thus the four
    ! arrays are the four combinations of AUFs that suffer perturbation or not for the two sources.
    real(dp), intent(in) :: Fcc(:), Fcn(:), Fnc(:), Fnn(:)
    ! Fourier-space representation of the F** arrays.
    real(dp), intent(in) :: rho(:), drho(:)
    ! Sky separation, in arcseconds, between the two objects in question.
    real(dp), intent(in) :: sep
    ! Output probability densities -- again, per combination of (non-)contamination -- that the
    ! two objects are a single source detected twice given their on-sky separation.
    real(dp), intent(out) :: Gcc, Gcn, Gnc, Gnn
    ! Loop counter.
    integer :: j
    ! Hankel transform parameters.
    real(dp) :: j0, z
    Gcc = 0.0_dp
    Gcn = 0.0_dp
    Gnc = 0.0_dp
    Gnn = 0.0_dp

    do j = 1, size(rho)
        z = rho(j)*sep*2.0_dp*pi
        call jy01a_j0(z, j0)
        Gcc = Gcc + rho(j) * Fcc(j) * j0 * drho(j)
        Gcn = Gcn + rho(j) * Fcn(j) * j0 * drho(j)
        Gnc = Gnc + rho(j) * Fnc(j) * j0 * drho(j)
        Gnn = Gnn + rho(j) * Fnn(j) * j0 * drho(j)
    end do

    Gcc = Gcc * 2.0_dp * pi
    Gcn = Gcn * 2.0_dp * pi
    Gnc = Gnc * 2.0_dp * pi
    Gnn = Gnn * 2.0_dp * pi

end subroutine contam_match_prob

recursive subroutine perm(i, a, perm_grid, n, ind)
    ! Iterate over a to find all permutations of the array. Adapted from
    ! http://rosettacode.org/wiki/Permutations#Fortran.
    ! Iteration counter, and size of array a.
    integer, intent(in) :: i, n
    ! Temporary storage array, array into which to save all permutations, and recursion counter.
    integer, intent(inout) :: a(:), perm_grid(:, :), ind
    ! Loop counter and temporary storage variable.
    integer :: j, t
    if (i == n) then
        perm_grid(:, ind) = a
        ind = ind + 1
    else
        do j = i, n
            t = a(i)
            a(i) = a(j)
            a(j) = t
            call perm(i + 1, a, perm_grid, n, ind)
            t = a(i)
            a(i) = a(j)
            a(j) = t
        end do
    end if

end subroutine

subroutine calc_combs(n, n_combs, m_pick, comb_grid)
    ! Calculate all combinations of length m_pick from an array of length n.
    ! Size of array, total number of combinations, and size of sub-set array to select combinations of.
    integer, intent(in) :: n, n_combs, m_pick
    ! Output grid containing all n-pick-m combinations.
    integer, intent(out) :: comb_grid(m_pick, n_combs)
    ! Temporary array to store m_pick-combinations to shuffle, and a recursive index counter.
    integer :: a(m_pick), ind
    ind = 1
    call comb(1, a, comb_grid, n, ind, m_pick)

end subroutine calc_combs

recursive subroutine comb(m, a, comb_grid, n_max, ind, m_pick)
    ! Iterate over all m_pick-combinations of a (of length n_max), storing them in comb_grid.
    ! Adapted from https://rosettacode.org/wiki/Combinations#Fortran.
    ! Recursive counter, size of combinations to select, and size of array to pick items from.
    integer, intent (in) :: m, m_pick, n_max
    ! Temporary storage for swapping items, array to save combinations to, and recursion counter.
    integer, intent(inout) :: a(:), comb_grid(:, :), ind
    ! Loop counter.
    integer :: n

    if (m > m_pick) then
      comb_grid(:, ind) = a
      ind = ind + 1
    else
      do n = 1, n_max
        if (m == 1) then
          a(m) = n
          call comb(m + 1, a, comb_grid, n_max, ind, m_pick)
        else if (n > a(m - 1)) then
          a(m) = n
          call comb(m + 1, a, comb_grid, n_max, ind, m_pick)
        end if
      end do
    end if

end subroutine comb

subroutine calc_permcombs(n, m_pick, n_perms, n_combs, combperm_grid)
    ! Calculate all permutations of selecting M items from an array of length N.
    ! Length of input array, length of output permutations array, and the number of combinations from
    ! n-pick-m and the number of permutations of each unique combination for the array.
    integer, intent(in) :: n, m_pick, n_perms, n_combs
    ! Grid iterating over every permutation of m_pick items from an n-length array.
    integer, intent(out) :: combperm_grid(m_pick, n_perms*n_combs)
    ! Loop counters.
    integer :: i, ind
    ! Temporary storage to shuffle combinations, and storage for all unique combinations of n-pick-m.
    integer :: a(m_pick), comb_grid(m_pick, n_combs)
    ind = 1
    call comb(1, a, comb_grid, n, ind, m_pick)
    ind = 1
    do i = 1, n_combs
        call perm(1, comb_grid(:, i), combperm_grid, m_pick, ind)
    end do

end subroutine calc_permcombs

end module counterpart_pairing_fortran