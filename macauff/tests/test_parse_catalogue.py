# Licensed under a 3-clause BSD style license - see LICENSE
'''
Tests for the "parse_catalogue" module.
'''

import numpy as np
from numpy.testing import assert_allclose

from ..parse_catalogue import csv_to_npy


class TestParseCatalogue:
    def setup_class(self):
        rng = np.random.default_rng(seed=45555)

        self.N = 1000000
        data = rng.standard_normal(size=(self.N, 7))
        data[:, 6] = np.round(data[:, 6]).astype(int)
        nan_cols = [rng.choice(self.N, size=(100,), replace=False),
                    rng.choice(self.N, size=(100,), replace=False)]
        data[nan_cols[0], 4] = np.nan
        data[nan_cols[1], 5] = np.nan
        self.data = data

    def test_csv_to_npy(self):
        # Convert data to string to get expected Pandas-esque .csv formatting where
        # NaN values are empty strings.
        data1 = self.data.astype(str)
        data1[data1 == 'nan'] = ''

        for header_text, header in zip(['', '# a, b, c, d, e, f, g'], [False, True]):
            np.savetxt('test_data.csv', data1, delimiter=',', fmt='%s', header=header_text)

            csv_to_npy('.', 'test_data', '.', [0, 1, 2], [4, 5], 6, header=header)

            astro = np.load('con_cat_astro.npy')
            photo = np.load('con_cat_photo.npy')
            best_index = np.load('magref.npy')

            assert np.all(astro.shape == (self.N, 3))
            assert np.all(photo.shape == (self.N, 2))
            assert np.all(best_index.shape == (self.N,))
            assert_allclose(astro, self.data[:, [0, 1, 2]])
            assert_allclose(photo, self.data[:, [4, 5]])
            assert_allclose(best_index, self.data[:, 6])
