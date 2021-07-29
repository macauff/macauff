# Licensed under a 3-clause BSD style license - see LICENSE
'''
Tests for the "parse_catalogue" module.
'''

import os
import pandas as pd
import numpy as np
from numpy.testing import assert_allclose

from ..parse_catalogue import csv_to_npy, npy_to_csv


class TestParseCatalogue:
    def setup_class(self):
        rng = np.random.default_rng(seed=45555)

        self.N = 100000
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

    def test_npy_to_csv(self):
        # Convert data to string to get expected Pandas-esque .csv formatting where
        # NaN values are empty strings.
        data1 = self.data.astype(str)
        data1[data1 == 'nan'] = ''
        data1[:, 0] = ['Gaia {}'.format(i) for i in data1[:, 0]]
        np.savetxt('test_a_data.csv', data1, delimiter=',', fmt='%s', header='')

        rng = np.random.default_rng(seed=43587232)

        self.Nb = 70000
        data = rng.standard_normal(size=(self.Nb, 8))
        data[:, 7] = np.round(data[:, 7]).astype(int)
        nan_cols = [rng.choice(self.Nb, size=(200,), replace=False),
                    rng.choice(self.Nb, size=(200,), replace=False)]
        data[nan_cols[0], 4] = np.nan
        data[nan_cols[1], 5] = np.nan
        data2 = data.astype(str)
        data2[data2 == 'nan'] = ''
        data2[:, 0] = ['J{}'.format(i) for i in data2[:, 0]]
        np.savetxt('test_b_data.csv', data2, delimiter=',', fmt='%s', header='')

        # Fake 3x match probability, eta/xi/2x contamination/match+non-match
        # index arrays.
        os.system('rm -r test_folder')
        os.makedirs('test_folder/pairing', exist_ok=True)
        N_match = int(0.6*self.N)
        ac = rng.choice(self.N, size=N_match, replace=False)
        np.save('test_folder/pairing/ac.npy', ac)
        bc = rng.choice(self.Nb, size=N_match, replace=False)
        np.save('test_folder/pairing/bc.npy', bc)
        af = np.delete(np.arange(0, self.N), ac)
        np.save('test_folder/pairing/af.npy', af)
        bf = np.delete(np.arange(0, self.Nb), bc)
        np.save('test_folder/pairing/bf.npy', bf)

        pc = rng.uniform(0.5, 1, size=N_match)
        np.save('test_folder/pairing/pc.npy', pc)
        pfa = rng.uniform(0.5, 1, size=len(af))
        np.save('test_folder/pairing/pfa.npy', pfa)
        pfb = rng.uniform(0.5, 1, size=len(bf))
        np.save('test_folder/pairing/pfb.npy', pfb)

        eta = rng.uniform(-10, 10, size=N_match)
        np.save('test_folder/pairing/eta.npy', eta)
        xi = rng.uniform(-10, 10, size=N_match)
        np.save('test_folder/pairing/xi.npy', xi)

        pac = rng.uniform(0, 1, size=(N_match, 2))
        np.save('test_folder/pairing/pacontam.npy', pac)
        pbc = rng.uniform(0, 1, size=(N_match, 2))
        np.save('test_folder/pairing/pbcontam.npy', pbc)

        acf = rng.uniform(0, 0.2, size=N_match)
        np.save('test_folder/pairing/acontamflux.npy', acf)
        bcf = rng.uniform(0, 3, size=N_match)
        np.save('test_folder/pairing/bcontamflux.npy', bcf)

        a_cols = ['A_Designation', 'A_RA', 'A_Dec', 'G_BP', 'G', 'G_RP']
        b_cols = ['B_Designation', 'B_RA', 'B_Dec', 'W1', 'W2', 'W3', 'W4']
        extra_cols = ['MATCH_P', 'ETA', 'XI', 'A_AVG_CONT', 'B_AVG_CONT', 'A_CONT_F1',
                      'A_CONT_F10', 'B_CONT_F1', 'B_CONT_F10']

        npy_to_csv(['.', '.'], 'test_folder', '.', ['test_a_data', 'test_b_data'],
                   ['match_csv', 'a_nonmatch_csv', 'b_nonmatch_csv'], [a_cols, b_cols],
                   [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5, 6]], ['A', 'B'], 20,
                   headers=[False, False])

        assert os.path.isfile('match_csv.csv')
        assert os.path.isfile('a_nonmatch_csv.csv')

        names = np.append(np.append(a_cols, b_cols), extra_cols)

        df = pd.read_csv('match_csv.csv', header=None, names=names)
        for i, col in zip([1, 2, 3, 4, 5], a_cols[1:]):
            assert_allclose(df[col], self.data[ac, i])
        # data1 and data2 are the string representations of catalogues a/b.
        assert np.all([df[a_cols[0]].iloc[i] == data1[ac[i], 0] for i in range(len(ac))])
        # self.data kept as catalogue "a", and regular variable data cat "b".
        for i, col in zip([1, 2, 3, 4, 5, 6], b_cols[1:]):
            assert_allclose(df[col], data[bc, i])
        assert np.all([df[b_cols[0]].iloc[i] == data2[bc[i], 0] for i in range(len(bc))])

        for f, col in zip([pc, eta, xi, acf, bcf, pac[:, 0], pac[:, 1], pbc[:, 0], pbc[:, 1]],
                          extra_cols):
            assert_allclose(df[col], f)

        names = np.append(a_cols, ['MATCH_P'])
        df = pd.read_csv('a_nonmatch_csv.csv', header=None, names=names)
        for i, col in zip([1, 2, 3, 4, 5], a_cols[1:]):
            assert_allclose(df[col], self.data[af, i])
        assert np.all([df[a_cols[0]].iloc[i] == data1[af[i], 0] for i in range(len(af))])
        assert_allclose(df['MATCH_P'], pfa)
        names = np.append(b_cols, ['MATCH_P'])
        df = pd.read_csv('b_nonmatch_csv.csv', header=None, names=names)
        for i, col in zip([1, 2, 3, 4, 5, 6], b_cols[1:]):
            assert_allclose(df[col], data[bf, i])
        assert np.all([df[b_cols[0]].iloc[i] == data2[bf[i], 0] for i in range(len(bf))])
        assert_allclose(df['MATCH_P'], pfb)
