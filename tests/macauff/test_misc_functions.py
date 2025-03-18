# Licensed under a 3-clause BSD style license - see LICENSE
'''
Tests for the "misc_functions" module.
'''

import numpy as np
import pytest
import scipy.special
from numpy.testing import assert_allclose
from scipy.stats import binned_statistic

# pylint: disable=import-error,no-name-in-module
from macauff.get_trilegal_wrapper import get_av_infinity
from macauff.misc_functions import (
    _load_rectangular_slice,
    convex_hull_area,
    coord_inside_convex_hull,
    create_auf_params_grid,
    generate_avs_inside_hull,
    hav_dist_constant_lat,
    load_small_ref_auf_grid,
    min_max_lon,
)
from macauff.misc_functions_fortran import misc_functions_fortran as mff

# pylint: enable=import-error,no-name-in-module


def test_closest_auf_point():
    test_points = np.array([[1, 1], [50, 50]])
    source_points = np.array([[3, 2], [80, 50], [40, 30]])
    inds = mff.find_nearest_point(source_points[:, 0], source_points[:, 1],
                                  test_points[:, 0], test_points[:, 1])
    assert np.all(inds == np.array([0, 1, 1]))


def test_calc_j0():
    r = np.linspace(0, 5, 5000)
    rho = np.linspace(0, 100, 5000)
    j0 = mff.calc_j0(r, rho)
    # pylint: disable-next=no-member
    assert_allclose(j0, scipy.special.j0(2 * np.pi * r.reshape(-1, 1) * rho.reshape(1, -1)),
                    rtol=1e-5)


def test_create_fourier_offsets_grid():
    a_len = np.array([[5, 10, 5], [15, 4, 8]], order='F')
    auf_pointings = np.array([[10, 20], [50, 50], [100, -40]])
    filt_names = ['W1', 'W2']
    r = np.linspace(0, 5, 10)
    p_a_o = {}
    for j, auf_pointing in enumerate(auf_pointings):
        ax1, ax2 = auf_pointing
        for i, filt in enumerate(filt_names):
            perturb_auf_combo = f'{ax1}-{ax2}-{filt}'
            s_p_a_o = {}
            s_p_a_o['fourier'] = (i + len(filt_names)*j)*np.ones((len(r[:-1]), a_len[i, j]), float)
            p_a_o[perturb_auf_combo] = s_p_a_o

    a = create_auf_params_grid(p_a_o, auf_pointings, filt_names, 'fourier', a_len,
                               len_first_axis=len(r)-1)
    assert np.all(a.shape == (9, 15, 2, 3))
    a_manual = -1*np.ones((9, 15, 2, 3), float, order='F')
    for j in range(0, len(auf_pointings)):
        for i in range(0, len(filt_names)):
            a_manual[:, :a_len[i, j], i, j] = i + len(filt_names)*j
    assert np.all(a == a_manual)


def test_load_small_ref_ind_fourier_grid():
    a_len = np.array([[6, 10, 7], [15, 9, 8], [7, 10, 12], [8, 8, 11]], order='F')
    auf_pointings = np.array([[10, 20], [50, 50], [100, -40]])
    filt_names = ['W1', 'W2', 'W3', 'W4']
    a = np.empty(dtype=float, shape=(9, 15, 4, 3), order='F')
    for j in range(0, len(auf_pointings)):
        for i in range(0, len(filt_names)):
            a[:, :a_len[i, j], i, j] = (i*a_len[i, j] + a_len[i, j]*len(filt_names)*j +
                                        np.arange(a_len[i, j]).reshape(1, -1))
    p_a_o = {}
    p_a_o['fourier_grid'] = a
    # Unique indices: 0, 1, 2, 5; 0, 3; 0, 1, 2
    # These map to 0, 1, 2, 3; 0, 1; 0, 1, 2
    modrefind = np.array([[0, 2, 0, 2, 1, 5], [0, 3, 3, 3, 3, 0], [0, 1, 2, 1, 2, 1]])
    [a], b = load_small_ref_auf_grid(modrefind, p_a_o, ['fourier'])

    new_small_modrefind = np.array([[0, 2, 0, 2, 1, 3], [0, 1, 1, 1, 1, 0], [0, 1, 2, 1, 2, 1]])
    new_small_fouriergrid = np.empty((9, 4, 2, 3), float, order='F')
    for j, j_old in enumerate([0, 1, 2]):
        for i, i_old in enumerate([0, 3]):
            for k, k_old in enumerate([0, 1, 2, 5]):
                new_small_fouriergrid[:, k, i, j] = (
                    k_old + i_old*a_len[i_old, j_old] + a_len[i_old, j_old]*len(filt_names)*j_old)

    assert np.all(b.shape == (3, 6))
    assert np.all(a.shape == (9, 4, 2, 3))
    assert np.all(b == new_small_modrefind)
    assert np.all(a == new_small_fouriergrid)


def test_hav_dist_constant_lat():
    lon1s = [0, 124.1, 65.34, 180, 324, 96.34]
    lon2s = [359.1, 150.23, 165.3, 210, 10.3, 60.34]

    for lat in [-86.4, -40.3, -10.1, 0, 15.5, 45.1, 73.14, 88.54]:
        for lon1, lon2 in zip(lon1s, lon2s):
            a = mff.haversine_wrapper(lon1, lon2, lat, lat)
            b = hav_dist_constant_lat(lon1, lat, lon2)
            assert_allclose(a, b)


def test_load_rectangular_slice():
    rng = np.random.default_rng(4324324432)
    for x, y, padding in zip([2, -1, -1, 45], [3, 1, 1, 46], [0.05, 0, 0.02, 0.1]):
        a = rng.uniform(x, y, size=(5000, 2))
        if x < 0:
            a[a[:, 0] < 0, 0] = a[a[:, 0] < 0, 0] + 360
        lon1, lon2, lat1, lat2 = x+0.2, x+0.4, x+0.1, x+0.3
        sky_cut = _load_rectangular_slice(a, lon1, lon2, lat1, lat2, padding)
        for i in range(len(a)):
            within_range = np.empty(4, bool)
            if x > 0:
                within_range[0] = (a[i, 0] >= lon1) | (
                    np.degrees(2 * np.arcsin(np.abs(np.cos(np.radians(a[i, 1])) *
                                             np.sin(np.radians(a[i, 0] - lon1)/2)))) <= padding)
                within_range[1] = (a[i, 0] <= lon2) | (
                    np.degrees(2 * np.arcsin(np.abs(np.cos(np.radians(a[i, 1])) *
                                             np.sin(np.radians(a[i, 0] - lon2)/2)))) <= padding)
            else:
                if a[i, 0] < 180:
                    within_range[0] = (a[i, 0] >= lon1) | (
                        np.degrees(2 * np.arcsin(np.abs(np.cos(np.radians(a[i, 1])) *
                                                 np.sin(np.radians(a[i, 0] - lon1)/2)))) <= padding)
                else:
                    within_range[0] = (a[i, 0] - 360 >= lon1) | (
                        np.degrees(2 * np.arcsin(np.abs(np.cos(np.radians(a[i, 1])) *
                                                 np.sin(np.radians(a[i, 0] - lon1)/2)))) <= padding)
                if a[i, 0] < 180:
                    within_range[1] = (a[i, 0] <= lon2) | (
                        np.degrees(2 * np.arcsin(np.abs(np.cos(np.radians(a[i, 1])) *
                                                 np.sin(np.radians(a[i, 0] - lon2)/2)))) <= padding)
                else:
                    within_range[1] = (a[i, 0] - 360 <= lon2) | (
                        np.degrees(2 * np.arcsin(np.abs(np.cos(np.radians(a[i, 1])) *
                                                 np.sin(np.radians(a[i, 0] - lon2)/2)))) <= padding)
            within_range[2] = a[i, 1] >= lat1-padding
            within_range[3] = a[i, 1] <= lat2+padding
            if sky_cut[i]:
                assert np.all(within_range)
            else:
                assert not np.all(within_range)


def test_min_max_lon():
    rng = np.random.default_rng(seed=435834534)
    for min_lon, max_lon in zip([0, 10, 90, 340, 355], [360, 20, 350, 20, 5]):
        if min_lon < max_lon:
            a = rng.uniform(min_lon, max_lon, size=50000)
            min_n, max_n = min_lon, max_lon
        else:
            a = rng.uniform(min_lon-360, max_lon, size=50000)
            a[a < 0] = a[a < 0] + 360
            min_n, max_n = min_lon - 360, max_lon
        new_min_lon, new_max_lon = min_max_lon(a)
        assert_allclose([new_min_lon, new_max_lon], [min_n, max_n], rtol=0.01)


@pytest.mark.parametrize("shape", ["circle", "rectangle"])
@pytest.mark.parametrize("overlay_origin", [False, "-1", "359"])
@pytest.mark.parametrize("high_lat", [True, False])
class TestConvexHull:
    def generate_points(self, shape, overlay_origin, high_lat):
        self.rng = np.random.default_rng(seed=6723486457)
        self.x = self.rng.uniform(3, 13, size=300000)
        self.y = self.rng.uniform(-2, 1, size=300000)
        if high_lat:
            self.y += 70
        if overlay_origin in ("-1", "359"):
            self.x -= 4
        if overlay_origin == "359":
            self.x[self.x < 0] = self.x[self.x < 0] + 360
        if high_lat:
            self.y_mid = -0.5 + 70
        else:
            self.y_mid = -0.5
        if overlay_origin in ("-1", "359"):
            self.x_mid = 4
        else:
            self.x_mid = 8

        self.r = 1.5

        if shape == "circle":
            q = np.array([mff.haversine_wrapper(a, self.x_mid, b, self.y_mid) for a, b in
                          zip(self.x, self.y)]) <= self.r
            self.x, self.y = self.x[q], self.y[q]

    def test_convex_hull_area(self, shape, overlay_origin, high_lat):
        self.generate_points(shape, overlay_origin, high_lat)
        hull_area = convex_hull_area(self.x, self.y)

        ax1_min, ax1_max = min_max_lon(self.x)
        ax2_min = np.amin(self.y)
        ax2_max = np.amax(self.y)
        if shape == "rectangle":
            fake_area = (ax1_max - ax1_min) * (
                np.sin(np.radians(ax2_max)) - np.sin(np.radians(ax2_min))) * 180/np.pi
        else:
            y_bins = np.linspace(-2, 1, 100)
            dys = np.diff(y_bins)
            if high_lat:
                y_bins += 70
            if overlay_origin == "359":
                self.x[self.x > 180] = self.x[self.x > 180] - 360
            min_xs, _, _ = binned_statistic(self.y, self.x, statistic='min', bins=y_bins)
            max_xs, _, _ = binned_statistic(self.y, self.x, statistic='max', bins=y_bins)
            fake_area = 0
            for y, dy, min_x, max_x in zip(0.5*(y_bins[1:]+y_bins[:-1]), dys, min_xs, max_xs):
                fake_area += (max_x - min_x) * np.cos(np.radians(y)) * dy

        assert_allclose(hull_area, fake_area, rtol=0.01)

    def test_coord_in_convex_hull(self, shape, overlay_origin, high_lat):
        self.generate_points(shape, overlay_origin, high_lat)
        _, hull_points, x_shift = convex_hull_area(self.x, self.y, return_hull=True)

        ax1_min, ax1_max = min_max_lon(self.x)
        ax2_min = np.amin(self.y)
        ax2_max = np.amax(self.y)
        points_x = self.rng.uniform(ax1_min-0.5, ax1_max+0.5, size=1000)
        points_y = self.rng.uniform(ax2_min-0.5, ax2_max+0.5, size=1000)
        for point_x, point_y in zip(points_x, points_y):
            if shape == "circle":
                point_dist = mff.haversine_wrapper(point_x, self.x_mid, point_y, self.y_mid)
                if self.r*0.998 <= point_dist <= self.r*1.002:
                    # Polygon precision means we can't verify points that lie
                    # close to the circle edge, as the true distance isn't
                    # an accurate measure of the polygon inside/outside.
                    continue
            inside = coord_inside_convex_hull([point_x + x_shift, point_y], hull_points)
            if shape == "circle":
                true_inside = point_dist <= self.r
            else:
                true_inside = ((point_x <= ax1_max) & (point_x >= ax1_min) &
                               (point_y <= ax2_max) & (point_y >= ax2_min))

            assert inside == true_inside

    def test_generate_avs_inside_hull(self, shape, overlay_origin, high_lat):
        self.generate_points(shape, overlay_origin, high_lat)
        _, hull_points, x_shift = convex_hull_area(self.x, self.y, return_hull=True)

        ax1_min, ax1_max = min_max_lon(self.x)
        ax2_min = np.amin(self.y)
        ax2_max = np.amax(self.y)

        avs = generate_avs_inside_hull(ax1_min - 2, ax1_max + 2, ax2_min - 2, ax2_max + 2, hull_points,
                                       x_shift, 'galactic')

        assert len(avs) >= 30

        n_dim = 7
        while True:
            ax1s = np.linspace(ax1_min - 2, ax1_max + 2, n_dim)
            ax2s = np.linspace(ax2_min - 2, ax2_max + 2, n_dim)
            ax1s, ax2s = np.meshgrid(ax1s, ax2s, indexing='xy')
            ax1s, ax2s = ax1s.flatten(), ax2s.flatten()
            # Basically just reproduce the code in generate_avs_inside_hull,
            # except this bit where we use the known "inside" flags.
            if shape == "circle":
                check = np.array([mff.haversine_wrapper(a, self.x_mid, b, self.y_mid) <= self.r for
                                  a, b in zip(ax1s, ax2s)])
            else:
                check = np.array([(a >= ax1_min) & (a <= ax1_max) & (b >= ax2_min) & (b <= ax2_max)
                                  for a, b in zip(ax1s, ax2s)])
            if np.sum(check) >= 30:
                break
            n_dim += 1
        ax1s, ax2s = ax1s[check], ax2s[check]
        other_avs = np.array([get_av_infinity(ax1, ax2, frame='galactic')[0] for ax1, ax2 in zip(ax1s, ax2s)])
        assert_allclose(avs, other_avs, rtol=0.01)


@pytest.mark.parametrize("coord", ["central", "0-360", "pole"])
@pytest.mark.parametrize("position", ["inside", "corner", "edge", "random"])
# pylint: disable-next=too-many-branches,too-many-statements
def test_circle_area(position, coord):
    rng = np.random.default_rng(7891246734)
    r = 0.1

    if coord == "central":
        x_edges = np.array([50, 51])
        y_edges = np.array([0, 1])
    if coord == "0-360":
        x_edges = np.array([0, 360])
        y_edges = np.array([0, 1])
    if coord == "pole":
        x_edges = np.array([150, 152])
        y_edges = np.array([70, 71])

    hull_x = x_edges[[0, 0, 1, 1, 0]]  # pylint: disable=possibly-used-before-assignment
    hull_y = y_edges[[0, 1, 1, 0, 0]]  # pylint: disable=possibly-used-before-assignment

    if position == "inside":
        # If circle is inside rectangle, get full area:
        done = 0
        while done < 100:
            seed = rng.choice(100000, size=(mff.get_random_seed_size(), 1))
            x = rng.uniform(x_edges[0], x_edges[1])
            y = rng.uniform(y_edges[0], y_edges[1])
            if (x - r/np.cos(np.radians(y_edges[1])) >= x_edges[0] and
                    x + r/np.cos(np.radians(y_edges[1])) <= x_edges[1] and
                    y - r >= y_edges[0] and y + r <= y_edges[1]):
                calc_area = mff.get_circle_area_overlap([x], [y], r, hull_x, hull_y, seed)
                assert_allclose(calc_area, np.pi * r**2)
                done += 1

    if position == "corner":
        # Now, if the circle is exactly on the corners of the rectangle
        # we should have a quarter the area:
        x0s = [x_edges[0], x_edges[0], x_edges[1], x_edges[1]]
        y0s = [y_edges[0], y_edges[1], y_edges[0], y_edges[1]]
        for x, y in zip(x0s, y0s):
            seed = rng.choice(100000, size=(mff.get_random_seed_size(), 1))
            calc_area = mff.get_circle_area_overlap([x], [y], r, hull_x, hull_y, seed)
            # We have a random process in this calculation, will result in small
            # variations in the area.
            if coord != "0-360":
                assert_allclose(calc_area, np.pi * r**2 / 4, rtol=0.02, atol=5e-5)
            else:
                assert_allclose(calc_area, np.pi * r**2 / 2, rtol=0.02, atol=5e-5)

    # pylint: disable-next=too-many-nested-blocks
    if position == "edge":
        # In the middle of an edge we should have half the circle area:
        for _ in range(100):
            x0s = [x_edges[0], 0.5*np.sum(x_edges), x_edges[1], 0.5*np.sum(x_edges)]
            y0s = [0.5*np.sum(y_edges), y_edges[0], 0.5*np.sum(y_edges), y_edges[1]]
            for x0, y0 in zip(x0s, y0s):
                if x0 == 0.5*np.sum(x_edges):
                    x = rng.uniform(x_edges[0] + r/np.cos(np.radians(y_edges[1])) + 1e-4,
                                    x_edges[1] - r/np.cos(np.radians(y_edges[1])) - 1e-4, size=1)
                else:
                    x = x0
                if y0 == 0.5*np.sum(y_edges):
                    y = rng.uniform(y_edges[0] + r + 1e-4, y_edges[1] - r - 1e-4, size=1)
                else:
                    y = y0
                seed = rng.choice(100000, size=(mff.get_random_seed_size(), 1))
                calc_area = mff.get_circle_area_overlap([x], [y], r, hull_x, hull_y, seed)
                if coord != "0-360" or x0 == 0.5*np.sum(x_edges):
                    assert_allclose(calc_area, np.pi * r**2 / 2, rtol=0.02, atol=5e-5)
                else:
                    assert_allclose(calc_area, np.pi * r**2, rtol=0.02, atol=5e-5)

        # Otherwise, we have a more random amount of missing circle:
        for _ in range(100):
            x0s = [x_edges[0], 0.5*np.sum(x_edges), x_edges[1], 0.5*np.sum(x_edges)]
            y0s = [0.5*np.sum(y_edges), y_edges[0], 0.5*np.sum(y_edges), y_edges[1]]
            for x0, y0 in zip(x0s, y0s):
                if x0 == 0.5*np.sum(x_edges):
                    x = rng.uniform(x_edges[0] + r/np.cos(np.radians(y_edges[1])) + 1e-4,
                                    x_edges[1] - r/np.cos(np.radians(y_edges[1])) - 1e-4, size=1)
                    if y0 == y_edges[0]:
                        y = rng.uniform(y0 + 1e-4, y0 + r - 1e-4, size=1)
                    else:
                        y = rng.uniform(y0 - r + 1e-4, y0 - 1e-4, size=1)
                if y0 == 0.5*np.sum(y_edges):
                    y = rng.uniform(y_edges[0] + r + 1e-4, y_edges[1] - r - 1e-4, size=1)
                    if x0 == x_edges[0]:
                        x = rng.uniform(x0 + 1e-4, x0 + r/np.cos(np.radians(y_edges[1])) - 1e-4, size=1)
                    else:
                        x = rng.uniform(x0 - r/np.cos(np.radians(y_edges[1])) + 1e-4, x0 - 1e-4, size=1)
                seed = rng.choice(100000, size=(mff.get_random_seed_size(), 1))
                calc_area = mff.get_circle_area_overlap([x], [y], r, hull_x, hull_y, seed)
                if coord != "0-360" or x0 == 0.5*np.sum(x_edges):
                    if x0 == 0.5*np.sum(x_edges):
                        if y0 == y_edges[0]:
                            h = r - (y - y0)
                        else:
                            h = r - (y0 - y)
                    if y0 == 0.5*np.sum(y_edges):
                        if x0 == x_edges[0]:
                            h = r - (x - x0)*np.cos(np.radians(y_edges[1]))
                        else:
                            h = r - (x0 - x)*np.cos(np.radians(y_edges[1]))
                    # pylint: disable-next=possibly-used-before-assignment
                    chord_area = r**2 * np.arccos(1 - h / r) - (r - h) * np.sqrt(r**2 - (r - h)**2)
                    remaining_area = np.pi * r**2 - chord_area
                    assert_allclose(calc_area, remaining_area, rtol=0.02, atol=5e-5)
                else:
                    assert_allclose(calc_area, np.pi * r**2, rtol=0.02, atol=5e-5)

    # pylint: disable-next=too-many-nested-blocks
    if position == "random":
        # Verify a few randomly placed circles too:
        done = 0
        while done < 100:
            seed = rng.choice(100000, size=(mff.get_random_seed_size(), 1))
            x = rng.uniform(x_edges[0], x_edges[1])
            y = rng.uniform(y_edges[0], y_edges[1])
            if np.any([x - r/np.cos(np.radians(y_edges[1])) < x_edges[0],
                       x + r/np.cos(np.radians(y_edges[1])) > x_edges[1],
                       y - r < y_edges[0], y + r > y_edges[1]]):
                calc_area = mff.get_circle_area_overlap([x], [y], r, hull_x, hull_y, seed)
                if "0-360" not in coord:
                    xp = np.linspace(max(x_edges[0], x - 1.1*r/np.cos(np.radians(y_edges[1]))),
                                     min(x_edges[1], x + 1.1*r/np.cos(np.radians(y_edges[1]))), 300)
                else:
                    xp = np.linspace(x - 1.1*r/np.cos(np.radians(y_edges[1])),
                                     x + 1.1*r/np.cos(np.radians(y_edges[1])), 300)
                yp = np.linspace(max(y_edges[0], y - 1.1*r), min(y_edges[1], y + 1.1*r), 300)
                dx, dy = xp[1] - xp[0], yp[1] - yp[0]
                manual_area = 0
                for x_p in xp:
                    for y_p in yp:
                        if mff.haversine_wrapper(x_p, x, y_p, y) <= r:
                            manual_area += dx*dy * np.cos(np.radians(y_edges[1]))
                assert_allclose(calc_area, manual_area, rtol=0.05, atol=5e-5)
            done += 1
