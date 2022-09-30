# Licensed under a 3-clause BSD style license - see LICENSE
'''
Tests for the "parse_catalogue" module.
'''

import os
import pandas as pd
import numpy as np
import pytest
from numpy.testing import assert_allclose

from ..parse_catalogue import csv_to_npy, npy_to_csv, rect_slice_npy, rect_slice_csv


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

    def test_csv_to_npy_process_uncert(self):
        # Convert data to string to get expected Pandas-esque .csv formatting where
        # NaN values are empty strings.
        data1 = self.data.astype(str)
        data1[data1 == 'nan'] = ''

        header_text, header = '# a, b, c, d, e, f, g', True
        np.savetxt('test_data.csv', data1, delimiter=',', fmt='%s', header=header_text)

        with pytest.raises(ValueError, match='process_uncerts must either be True or'):
            csv_to_npy('.', 'test_data', '.', [0, 1, 2], [4, 5], 6, header=header,
                       process_uncerts=None)
        with pytest.raises(ValueError, match='astro_sig_fits_filepath must given if process'):
            csv_to_npy('.', 'test_data', '.', [0, 1, 2], [4, 5], 6, header=header,
                       process_uncerts=True)
        with pytest.raises(ValueError, match='cat_in_radec must given if process_uncerts is '):
            csv_to_npy('.', 'test_data', '.', [0, 1, 2], [4, 5], 6, header=header,
                       process_uncerts=True, astro_sig_fits_filepath='test_sig_folder')
        with pytest.raises(ValueError, match='If process_uncerts is True, cat_in_radec must '):
            csv_to_npy('.', 'test_data', '.', [0, 1, 2], [4, 5], 6, header=header,
                       process_uncerts=True, astro_sig_fits_filepath='test_sig_folder',
                       cat_in_radec='something else')

        if os.path.exists('test_sig_folder'):
            os.system('rm -rf ./test_sig_folder')

        with pytest.raises(ValueError, match='astro_sig_fits_filepath does not exist.'):
            csv_to_npy('.', 'test_data', '.', [0, 1, 2], [4, 5], 6, header=header,
                       process_uncerts=True, astro_sig_fits_filepath='test_sig_folder',
                       cat_in_radec=False)

        os.makedirs('test_sig_folder')
        np.save('test_sig_folder/m_sigs_array.npy', np.array([2]))
        np.save('test_sig_folder/n_sigs_array.npy', np.array([0.01]))
        np.save('test_sig_folder/lmids.npy', np.array([10.0]))
        np.save('test_sig_folder/bmids.npy', np.array([0.0]))

        csv_to_npy('.', 'test_data', '.', [0, 1, 2], [4, 5], 6, header=header,
                   process_uncerts=True, astro_sig_fits_filepath='test_sig_folder',
                   cat_in_radec=False)

        astro = np.load('con_cat_astro.npy')
        photo = np.load('con_cat_photo.npy')
        best_index = np.load('magref.npy')

        assert np.all(astro.shape == (self.N, 3))
        assert np.all(photo.shape == (self.N, 2))
        assert np.all(best_index.shape == (self.N,))
        assert_allclose(astro[:, [0, 1]], self.data[:, [0, 1]])
        assert_allclose(astro[:, 2], np.sqrt((2*self.data[:, 2])**2 + 0.01**2))
        assert_allclose(photo, self.data[:, [4, 5]])
        assert_allclose(best_index, self.data[:, 6])

    def test_rect_slice_npy(self):
        np.save('con_cat_astro.npy', self.data[:, [1, 2, 3]])
        np.save('con_cat_photo.npy', self.data[:, [4, 5]])
        np.save('magref.npy', self.data[:, 6])

        os.makedirs('dummy_folder', exist_ok=True)

        rc = [-0.3, 0.3, -0.1, 0.2]
        for pad in [0.03, 0]:
            if os.path.isfile('_temporary_sky_slice_1.npy'):
                for n in ['1', '2', '3', '4', 'combined']:
                    os.system('rm _temporary_sky_slice_{}.npy'.format(n))
                for f in ['con_cat_astro', 'con_cat_photo', 'magref']:
                    os.system('rm dummy_folder/{}.npy'.format(f))

            rect_slice_npy('.', 'dummy_folder', rc, pad, 10)

            astro = np.load('dummy_folder/con_cat_astro.npy')
            photo = np.load('dummy_folder/con_cat_photo.npy')
            best_index = np.load('dummy_folder/magref.npy')

            cosd = np.cos(np.radians(self.data[:, 2]))
            qa = (self.data[:, 1] >= rc[0]-pad/cosd) & (self.data[:, 1] <= rc[1]+pad/cosd)
            qd = (self.data[:, 2] >= rc[2]-pad) & (self.data[:, 2] <= rc[3]+pad)
            q = qa & qd

            assert np.sum(q) == astro.shape[0]
            assert np.sum(q) == photo.shape[0]
            assert np.sum(q) == len(best_index)

            assert_allclose(self.data[q][:, [1, 2, 3]], astro)
            assert_allclose(self.data[q][:, [4, 5]], photo)
            assert_allclose(self.data[q][:, 6], best_index)

    def test_rect_slice_csv(self):
        # Convert data to string to get expected Pandas-esque .csv formatting where
        # NaN values are empty strings.
        data1 = self.data.astype(str)
        data1[data1 == 'nan'] = ''
        data1[:, 0] = ['Gaia {}'.format(i) for i in data1[:, 0]]
        rc = [-0.3, 0.3, -0.1, 0.2]
        col_names = ['A_Designation', 'A_RA', 'A_Dec', 'A_Err', 'G', 'G_RP', 'Best_Index']
        for header_text, header in zip(['', '# a, b, c, d, e, f, g'], [False, True]):
            np.savetxt('test_data.csv', data1, delimiter=',', fmt='%s', header=header_text)

            for pad in [0.03, 0]:
                rect_slice_csv('.', '.', 'test_data', 'test_data_small', rc, pad, [1, 2], 20,
                               header=header)
                df = pd.read_csv('test_data_small.csv', header=None, names=col_names)
                cosd = np.cos(np.radians(self.data[:, 2]))
                qa = (self.data[:, 1] >= rc[0]-pad/cosd) & (self.data[:, 1] <= rc[1]+pad/cosd)
                qd = (self.data[:, 2] >= rc[2]-pad) & (self.data[:, 2] <= rc[3]+pad)
                q = qa & qd

                assert np.sum(q) == len(df)

                for i, col in zip([1, 2, 3, 4, 5, 6], col_names[1:]):
                    assert_allclose(df[col], self.data[q, i])
                assert np.all([df[col_names[0]].iloc[i] == data1[q, 0][i] for i in
                               range(np.sum(q))])


class TestParseCatalogueNpyToCsv:
    def setup_class(self):
        os.system('rm *.csv')
        rng = np.random.default_rng(seed=45555)

        self.N = 100000
        data = rng.standard_normal(size=(self.N, 7))
        data[:, 6] = np.round(data[:, 6]).astype(int)
        nan_cols = [rng.choice(self.N, size=(100,), replace=False),
                    rng.choice(self.N, size=(100,), replace=False)]
        data[nan_cols[0], 4] = np.nan
        data[nan_cols[1], 5] = np.nan
        self.data = data

        # Convert data to string to get expected Pandas-esque .csv formatting where
        # NaN values are empty strings.
        self.data1 = self.data.astype(str)
        self.data1[self.data1 == 'nan'] = ''
        self.data1[:, 0] = ['Gaia {}'.format(i) for i in self.data1[:, 0]]
        np.savetxt('test_a_data.csv', self.data1, delimiter=',', fmt='%s', header='')

        rng = np.random.default_rng(seed=43587232)

        self.Nb = 70000
        self.datab = rng.standard_normal(size=(self.Nb, 8))
        self.datab[:, 7] = np.round(self.datab[:, 7]).astype(int)
        nan_cols = [rng.choice(self.Nb, size=(200,), replace=False),
                    rng.choice(self.Nb, size=(200,), replace=False)]
        self.datab[nan_cols[0], 4] = np.nan
        self.datab[nan_cols[1], 5] = np.nan
        self.data2 = self.datab.astype(str)
        self.data2[self.data2 == 'nan'] = ''
        self.data2[:, 0] = ['J{}'.format(i) for i in self.data2[:, 0]]
        np.savetxt('test_b_data.csv', self.data2, delimiter=',', fmt='%s', header='')

        # Fake 3x match probability, eta/xi/2x contamination/match+non-match
        # index arrays.
        os.system('rm -r test_folder')
        os.makedirs('test_folder/pairing', exist_ok=True)
        self.N_match = int(0.6*self.N)
        self.ac = rng.choice(self.N, size=self.N_match, replace=False)
        np.save('test_folder/pairing/ac.npy', self.ac)
        self.bc = rng.choice(self.Nb, size=self.N_match, replace=False)
        np.save('test_folder/pairing/bc.npy', self.bc)
        self.af = np.delete(np.arange(0, self.N), self.ac)
        np.save('test_folder/pairing/af.npy', self.af)
        self.bf = np.delete(np.arange(0, self.Nb), self.bc)
        np.save('test_folder/pairing/bf.npy', self.bf)

        self.pc = rng.uniform(0.5, 1, size=self.N_match)
        np.save('test_folder/pairing/pc.npy', self.pc)
        self.pfa = rng.uniform(0.5, 1, size=len(self.af))
        np.save('test_folder/pairing/pfa.npy', self.pfa)
        self.pfb = rng.uniform(0.5, 1, size=len(self.bf))
        np.save('test_folder/pairing/pfb.npy', self.pfb)

        self.eta = rng.uniform(-10, 10, size=self.N_match)
        np.save('test_folder/pairing/eta.npy', self.eta)
        self.xi = rng.uniform(-10, 10, size=self.N_match)
        np.save('test_folder/pairing/xi.npy', self.xi)

        self.pac = rng.uniform(0, 1, size=(self.N_match, 2))
        np.save('test_folder/pairing/pacontam.npy', self.pac)
        self.pbc = rng.uniform(0, 1, size=(self.N_match, 2))
        np.save('test_folder/pairing/pbcontam.npy', self.pbc)

        self.acf = rng.uniform(0, 0.2, size=self.N_match)
        np.save('test_folder/pairing/acontamflux.npy', self.acf)
        self.bcf = rng.uniform(0, 3, size=self.N_match)
        np.save('test_folder/pairing/bcontamflux.npy', self.bcf)

        self.aff = rng.uniform(0, 0.2, size=len(self.af))
        np.save('test_folder/pairing/afieldflux.npy', self.aff)
        self.bff = rng.uniform(0, 3, size=len(self.bf))
        np.save('test_folder/pairing/bfieldflux.npy', self.bff)

        self.csep = rng.uniform(0, 0.5, size=self.N_match)
        np.save('test_folder/pairing/crptseps.npy', self.csep)

        self.afs = rng.uniform(0, 0.5, size=len(self.af))
        np.save('test_folder/pairing/afieldseps.npy', self.afs)
        self.afeta = rng.uniform(-3, 0, size=len(self.af))
        np.save('test_folder/pairing/afieldeta.npy', self.afeta)
        self.afxi = rng.uniform(-3, 0, size=len(self.af))
        np.save('test_folder/pairing/afieldxi.npy', self.afxi)

        self.bfs = rng.uniform(0, 0.5, size=len(self.bf))
        np.save('test_folder/pairing/bfieldseps.npy', self.bfs)
        self.bfeta = rng.uniform(0, 0.5, size=len(self.bf))
        np.save('test_folder/pairing/bfieldeta.npy', self.bfeta)
        self.bfxi = rng.uniform(0, 0.5, size=len(self.bf))
        np.save('test_folder/pairing/bfieldxi.npy', self.bfxi)

    def test_npy_to_csv(self):
        a_cols = ['A_Designation', 'A_RA', 'A_Dec', 'G', 'G_RP']
        b_cols = ['B_Designation', 'B_RA', 'B_Dec', 'W1', 'W2', 'W3']
        extra_cols = ['MATCH_P', 'SEPARATION', 'ETA', 'XI', 'A_AVG_CONT', 'B_AVG_CONT',
                      'A_CONT_F1', 'A_CONT_F10', 'B_CONT_F1', 'B_CONT_F10']

        npy_to_csv(['.', '.'], 'test_folder', '.', ['test_a_data', 'test_b_data'],
                   ['match_csv', 'a_nonmatch_csv', 'b_nonmatch_csv'], [a_cols, b_cols],
                   [[0, 1, 2, 4, 5], [0, 1, 2, 4, 5, 6]], ['A', 'B'], 20,
                   headers=[False, False], input_npy_folders=[None, None])

        assert os.path.isfile('match_csv.csv')
        assert os.path.isfile('a_nonmatch_csv.csv')

        names = np.append(np.append(a_cols, b_cols), extra_cols)

        df = pd.read_csv('match_csv.csv', header=None, names=names)
        for i, col in zip([1, 2, 4, 5], a_cols[1:]):
            assert_allclose(df[col], self.data[self.ac, i])
        # data1 and data2 are the string representations of catalogues a/b.
        assert np.all([df[a_cols[0]].iloc[i] == self.data1[self.ac[i], 0] for i in
                       range(len(self.ac))])
        # self.data kept as catalogue "a", and datab for cat "b".
        for i, col in zip([1, 2, 4, 5, 6], b_cols[1:]):
            assert_allclose(df[col], self.datab[self.bc, i])
        assert np.all([df[b_cols[0]].iloc[i] == self.data2[self.bc[i], 0] for i in
                       range(len(self.bc))])

        for f, col in zip([self.pc, self.csep, self.eta, self.xi, self.acf, self.bcf,
                           self.pac[:, 0], self.pac[:, 1], self.pbc[:, 0], self.pbc[:, 1]],
                          extra_cols):
            assert_allclose(df[col], f)

        names = np.append(a_cols, ['MATCH_P', 'NNM_SEPARATION', 'NNM_ETA', 'NNM_XI', 'A_AVG_CONT'])
        df = pd.read_csv('a_nonmatch_csv.csv', header=None, names=names)
        for i, col in zip([1, 2, 4, 5], a_cols[1:]):
            assert_allclose(df[col], self.data[self.af, i])
        assert np.all([df[a_cols[0]].iloc[i] == self.data1[self.af[i], 0] for i in
                       range(len(self.af))])
        assert_allclose(df['MATCH_P'], self.pfa)
        assert_allclose(df['A_AVG_CONT'], self.aff)
        assert_allclose(df['NNM_SEPARATION'], self.afs)
        assert_allclose(df['NNM_ETA'], self.afeta)
        assert_allclose(df['NNM_XI'], self.afxi)
        names = np.append(b_cols, ['MATCH_P', 'NNM_SEPARATION', 'NNM_ETA', 'NNM_XI', 'B_AVG_CONT'])
        df = pd.read_csv('b_nonmatch_csv.csv', header=None, names=names)
        for i, col in zip([1, 2, 4, 5, 6], b_cols[1:]):
            assert_allclose(df[col], self.datab[self.bf, i])
        assert np.all([df[b_cols[0]].iloc[i] == self.data2[self.bf[i], 0] for i in
                       range(len(self.bf))])
        assert_allclose(df['MATCH_P'], self.pfb)
        assert_allclose(df['B_AVG_CONT'], self.bff)
        assert_allclose(df['NNM_SEPARATION'], self.bfs)
        assert_allclose(df['NNM_ETA'], self.bfeta)
        assert_allclose(df['NNM_XI'], self.bfxi)

    def test_npy_to_csv_process_uncerts(self):
        a_cols = ['A_Designation', 'A_RA', 'A_Dec', 'G', 'G_RP']
        b_cols = ['B_Designation', 'B_RA', 'B_Dec', 'W1', 'W2', 'W3']
        extra_cols = ['MATCH_P', 'SEPARATION', 'ETA', 'XI', 'A_AVG_CONT', 'B_AVG_CONT',
                      'A_CONT_F1', 'A_CONT_F10', 'B_CONT_F1', 'B_CONT_F10']

        # Save original .npy files to test loading extra step
        os.makedirs('test_a_out', exist_ok=True)
        np.save('test_a_out/con_cat_astro.npy', self.data[:, [1, 2, 3]])
        os.makedirs('test_b_out', exist_ok=True)
        np.save('test_b_out/con_cat_astro.npy', self.datab[:, [1, 2, 3]])

        npy_to_csv(['.', '.'], 'test_folder', '.', ['test_a_data', 'test_b_data'],
                   ['match_csv', 'a_nonmatch_csv', 'b_nonmatch_csv'], [a_cols, b_cols],
                   [[0, 1, 2, 4, 5], [0, 1, 2, 4, 5, 6]], ['A', 'B'], 20,
                   headers=[False, False], input_npy_folders=['test_a_out', 'test_b_out'])

        assert os.path.isfile('match_csv.csv')
        assert os.path.isfile('a_nonmatch_csv.csv')

        names = np.append(np.append(np.append(a_cols, b_cols), extra_cols),
                          ['A_FIT_SIG', 'B_FIT_SIG'])

        df = pd.read_csv('match_csv.csv', header=None, names=names)
        for i, col in zip([1, 2, 4, 5], a_cols[1:]):
            assert_allclose(df[col], self.data[self.ac, i])
        # data1 and data2 are the string representations of catalogues a/b.
        assert np.all([df[a_cols[0]].iloc[i] == self.data1[self.ac[i], 0] for i in
                       range(len(self.ac))])
        # self.data kept as catalogue "a", and datab for cat "b".
        for i, col in zip([1, 2, 4, 5, 6], b_cols[1:]):
            assert_allclose(df[col], self.datab[self.bc, i])
        assert np.all([df[b_cols[0]].iloc[i] == self.data2[self.bc[i], 0] for i in
                       range(len(self.bc))])
        assert_allclose(df['A_FIT_SIG'], self.data[self.ac, 3])
        assert_allclose(df['B_FIT_SIG'], self.datab[self.bc, 3])

        names = np.append(a_cols, ['MATCH_P', 'NNM_SEPARATION', 'NNM_ETA', 'NNM_XI',
                                   'A_AVG_CONT', 'A_FIT_SIG'])
        df = pd.read_csv('a_nonmatch_csv.csv', header=None, names=names)
        for i, col in zip([1, 2, 4, 5], a_cols[1:]):
            assert_allclose(df[col], self.data[self.af, i])
        assert np.all([df[a_cols[0]].iloc[i] == self.data1[self.af[i], 0] for i in
                       range(len(self.af))])
        assert_allclose(df['MATCH_P'], self.pfa)
        assert_allclose(df['A_FIT_SIG'], self.data[self.af, 3])

        names = np.append(b_cols, ['MATCH_P', 'NNM_SEPARATION', 'NNM_ETA', 'NNM_XI',
                                   'B_AVG_CONT', 'B_FIT_SIG'])
        df = pd.read_csv('b_nonmatch_csv.csv', header=None, names=names)
        for i, col in zip([1, 2, 4, 5, 6], b_cols[1:]):
            assert_allclose(df[col], self.datab[self.bf, i])
        assert np.all([df[b_cols[0]].iloc[i] == self.data2[self.bf[i], 0] for i in
                       range(len(self.bf))])
        assert_allclose(df['NNM_XI'], self.bfxi)
        assert_allclose(df['B_FIT_SIG'], self.datab[self.bf, 3])

    def test_npy_to_csv_cols_out_of_order(self):
        a_cols = ['A_RA', 'A_Dec', 'A_Designation', 'G', 'G_RP']
        b_cols = ['W1', 'W2', 'W3', 'B_Designation', 'B_RA', 'B_Dec']
        extra_cols = ['MATCH_P', 'SEPARATION', 'ETA', 'XI', 'A_AVG_CONT', 'B_AVG_CONT',
                      'A_CONT_F1', 'A_CONT_F10', 'B_CONT_F1', 'B_CONT_F10']

        npy_to_csv(['.', '.'], 'test_folder', '.', ['test_a_data', 'test_b_data'],
                   ['match_csv', 'a_nonmatch_csv', 'b_nonmatch_csv'], [a_cols, b_cols],
                   [[1, 2, 0, 4, 5], [4, 5, 6, 0, 1, 2]], ['A', 'B'], 20,
                   headers=[False, False], input_npy_folders=[None, None])

        assert os.path.isfile('match_csv.csv')
        assert os.path.isfile('a_nonmatch_csv.csv')

        names = np.append(np.append(a_cols, b_cols), extra_cols)

        df = pd.read_csv('match_csv.csv', header=None, names=names)
        for i, col in zip([1, 2, 4, 5], np.array(a_cols)[[0, 1, 3, 4]]):
            assert_allclose(df[col], self.data[self.ac, i])
        # data1 and data2 are the string representations of catalogues a/b.
        assert np.all([df[a_cols[2]].iloc[i] == self.data1[self.ac[i], 0] for i in
                       range(len(self.ac))])
        # self.data kept as catalogue "a", and datab for cat "b".
        for i, col in zip([4, 5, 6, 1, 2], np.array(b_cols)[[0, 1, 2, 4, 5]]):
            assert_allclose(df[col], self.datab[self.bc, i])
        assert np.all([df[b_cols[3]].iloc[i] == self.data2[self.bc[i], 0] for i in
                       range(len(self.bc))])

        for f, col in zip([self.pc, self.csep, self.eta, self.xi, self.acf, self.bcf,
                           self.pac[:, 0], self.pac[:, 1], self.pbc[:, 0], self.pbc[:, 1]],
                          extra_cols):
            assert_allclose(df[col], f)

        names = np.append(a_cols, ['MATCH_P', 'NNM_SEPARATION', 'NNM_ETA', 'NNM_XI', 'A_AVG_CONT'])
        df = pd.read_csv('a_nonmatch_csv.csv', header=None, names=names)
        for i, col in zip([1, 2, 4, 5], np.array(a_cols)[[0, 1, 3, 4]]):
            assert_allclose(df[col], self.data[self.af, i])
        assert np.all([df[a_cols[2]].iloc[i] == self.data1[self.af[i], 0] for i in
                       range(len(self.af))])
        assert_allclose(df['MATCH_P'], self.pfa)
        assert_allclose(df['A_AVG_CONT'], self.aff)
        assert_allclose(df['NNM_SEPARATION'], self.afs)
        assert_allclose(df['NNM_ETA'], self.afeta)
        assert_allclose(df['NNM_XI'], self.afxi)
        names = np.append(b_cols, ['MATCH_P', 'NNM_SEPARATION', 'NNM_ETA', 'NNM_XI', 'B_AVG_CONT'])
        df = pd.read_csv('b_nonmatch_csv.csv', header=None, names=names)
        for i, col in zip([4, 5, 6, 1, 2], np.array(b_cols)[[0, 1, 2, 4, 5]]):
            assert_allclose(df[col], self.datab[self.bf, i])
        assert np.all([df[b_cols[3]].iloc[i] == self.data2[self.bf[i], 0] for i in
                       range(len(self.bf))])
        assert_allclose(df['MATCH_P'], self.pfb)
        assert_allclose(df['B_AVG_CONT'], self.bff)
        assert_allclose(df['NNM_SEPARATION'], self.bfs)
        assert_allclose(df['NNM_ETA'], self.bfeta)
        assert_allclose(df['NNM_XI'], self.bfxi)

    def test_npy_to_csv_incorrect_extra_cols(self):
        a_cols = ['A_Designation', 'A_RA', 'A_Dec', 'G', 'G_RP']
        b_cols = ['B_Designation', 'B_RA', 'B_Dec', 'W1', 'W2', 'W3']

        with pytest.raises(UserWarning, match="either both need to be None, or both"):
            npy_to_csv(['.', '.'], 'test_folder', '.', ['test_a_data', 'test_b_data'],
                       ['match_csv', 'a_nonmatch_csv', 'b_nonmatch_csv'], [a_cols, b_cols],
                       [[0, 1, 2, 4, 5], [0, 1, 2, 4, 5, 6]], ['A', 'B'], 20,
                       headers=[False, False], extra_col_name_lists=[[1], [2]],
                       input_npy_folders=[None, None])

    def test_npy_to_csv_both_cat_extra_col(self):
        a_cols = ['A_Designation', 'A_RA', 'A_Dec', 'G', 'G_RP']
        b_cols = ['B_Designation', 'B_RA', 'B_Dec', 'W1', 'W2', 'W3']
        extra_cols = ['MATCH_P', 'SEPARATION', 'ETA', 'XI', 'A_AVG_CONT', 'B_AVG_CONT',
                      'A_CONT_F1', 'A_CONT_F10', 'B_CONT_F1', 'B_CONT_F10']

        add_a_cols = ['A_Err']
        add_b_cols = ['B_Err']
        add_a_nums = [3]
        add_b_nums = [3]

        npy_to_csv(['.', '.'], 'test_folder', '.', ['test_a_data', 'test_b_data'],
                               ['match_csv', 'a_nonmatch_csv', 'b_nonmatch_csv'], [a_cols, b_cols],
                               [[0, 1, 2, 4, 5], [0, 1, 2, 4, 5, 6]], ['A', 'B'], 20,
                               headers=[False, False],
                               extra_col_name_lists=[add_a_cols, add_b_cols],
                               extra_col_num_lists=[add_a_nums, add_b_nums],
                               input_npy_folders=[None, None])

        assert os.path.isfile('match_csv.csv')
        assert os.path.isfile('a_nonmatch_csv.csv')

        add_cols = np.append(add_a_cols, add_b_cols)
        names = np.append(np.append(np.append(a_cols, b_cols), extra_cols), add_cols)

        df = pd.read_csv('match_csv.csv', header=None, names=names)
        for i, col in zip([1, 2, 4, 5], a_cols[1:]):
            assert_allclose(df[col], self.data[self.ac, i])
        for i, col in zip(add_a_nums, add_a_cols):
            assert_allclose(df[col], self.data[self.ac, i])
        # data1 and data2 are the string representations of catalogues a/b.
        assert np.all([df[a_cols[0]].iloc[i] == self.data1[self.ac[i], 0] for i in
                       range(len(self.ac))])
        # self.data kept as catalogue "a", and datab for cat "b".
        for i, col in zip([1, 2, 4, 5, 6], b_cols[1:]):
            assert_allclose(df[col], self.datab[self.bc, i])
        for i, col in zip(add_b_nums, add_b_cols):
            assert_allclose(df[col], self.datab[self.bc, i])
        assert np.all([df[b_cols[0]].iloc[i] == self.data2[self.bc[i], 0] for i in
                       range(len(self.bc))])

        for f, col in zip([self.pc, self.csep, self.eta, self.xi, self.acf, self.bcf,
                           self.pac[:, 0], self.pac[:, 1], self.pbc[:, 0], self.pbc[:, 1]],
                          extra_cols):
            assert_allclose(df[col], f)

        names = np.append(np.append(a_cols, ['MATCH_P', 'NNM_SEPARATION', 'NNM_ETA', 'NNM_XI',
                                             'A_AVG_CONT']), add_a_cols)
        df = pd.read_csv('a_nonmatch_csv.csv', header=None, names=names)
        for i, col in zip([1, 2, 4, 5], a_cols[1:]):
            assert_allclose(df[col], self.data[self.af, i])
        for i, col in zip(add_a_nums, add_a_cols):
            assert_allclose(df[col], self.data[self.af, i])
        assert np.all([df[a_cols[0]].iloc[i] == self.data1[self.af[i], 0] for i in
                       range(len(self.af))])
        assert_allclose(df['MATCH_P'], self.pfa)
        assert_allclose(df['A_AVG_CONT'], self.aff)
        assert_allclose(df['NNM_SEPARATION'], self.afs)
        assert_allclose(df['NNM_ETA'], self.afeta)
        assert_allclose(df['NNM_XI'], self.afxi)
        names = np.append(np.append(b_cols, ['MATCH_P', 'NNM_SEPARATION', 'NNM_ETA', 'NNM_XI',
                                             'B_AVG_CONT']), add_b_cols)
        df = pd.read_csv('b_nonmatch_csv.csv', header=None, names=names)
        for i, col in zip([1, 2, 4, 5, 6], b_cols[1:]):
            assert_allclose(df[col], self.datab[self.bf, i])
        for i, col in zip(add_b_nums, add_b_cols):
            assert_allclose(df[col], self.datab[self.bf, i])
        assert np.all([df[b_cols[0]].iloc[i] == self.data2[self.bf[i], 0] for i in
                       range(len(self.bf))])
        assert_allclose(df['MATCH_P'], self.pfb)
        assert_allclose(df['B_AVG_CONT'], self.bff)
        assert_allclose(df['NNM_SEPARATION'], self.bfs)
        assert_allclose(df['NNM_ETA'], self.bfeta)
        assert_allclose(df['NNM_XI'], self.bfxi)

    def test_npy_to_csv_one_cat_extra_col(self):
        a_cols = ['A_Designation', 'A_RA', 'A_Dec', 'G', 'G_RP']
        b_cols = ['B_Designation', 'B_RA', 'B_Dec', 'W1', 'W2', 'W3']
        extra_cols = ['MATCH_P', 'SEPARATION', 'ETA', 'XI', 'A_AVG_CONT', 'B_AVG_CONT',
                      'A_CONT_F1', 'A_CONT_F10', 'B_CONT_F1', 'B_CONT_F10']

        add_a_cols = ['A_Err']
        add_b_cols = []
        add_a_nums = [3]
        add_b_nums = []

        npy_to_csv(['.', '.'], 'test_folder', '.', ['test_a_data', 'test_b_data'],
                               ['match_csv', 'a_nonmatch_csv', 'b_nonmatch_csv'], [a_cols, b_cols],
                               [[0, 1, 2, 4, 5], [0, 1, 2, 4, 5, 6]], ['A', 'B'], 20,
                               headers=[False, False],
                               extra_col_name_lists=[add_a_cols, add_b_cols],
                               extra_col_num_lists=[add_a_nums, add_b_nums],
                               input_npy_folders=[None, None])

        assert os.path.isfile('match_csv.csv')
        assert os.path.isfile('a_nonmatch_csv.csv')

        add_cols = np.append(add_a_cols, add_b_cols)
        names = np.append(np.append(np.append(a_cols, b_cols), extra_cols), add_cols)

        df = pd.read_csv('match_csv.csv', header=None, names=names)
        for i, col in zip([1, 2, 4, 5], a_cols[1:]):
            assert_allclose(df[col], self.data[self.ac, i])
        for i, col in zip(add_a_nums, add_a_cols):
            assert_allclose(df[col], self.data[self.ac, i])
        # data1 and data2 are the string representations of catalogues a/b.
        assert np.all([df[a_cols[0]].iloc[i] == self.data1[self.ac[i], 0] for i in
                       range(len(self.ac))])
        # self.data kept as catalogue "a", and datab for cat "b".
        for i, col in zip([1, 2, 4, 5, 6], b_cols[1:]):
            assert_allclose(df[col], self.datab[self.bc, i])
        # Only extra column in test_b_data.csv is the "B_Err" column from above,
        # self.data2[:, 3]. Test that these data do not match any output data:
        for n in names[np.delete(np.arange(len(names)), [0, 5])]:
            with pytest.raises(AssertionError, match='Not equal to tolerance'):
                assert_allclose(df[n], self.datab[self.bc, 3])
        assert np.all([df[b_cols[0]].iloc[i] == self.data2[self.bc[i], 0] for i in
                       range(len(self.bc))])

        for f, col in zip([self.pc, self.csep, self.eta, self.xi, self.acf, self.bcf,
                           self.pac[:, 0], self.pac[:, 1], self.pbc[:, 0], self.pbc[:, 1]],
                          extra_cols):
            assert_allclose(df[col], f)

        names = np.append(np.append(a_cols, ['MATCH_P', 'NNM_SEPARATION', 'NNM_ETA', 'NNM_XI',
                                             'A_AVG_CONT']), add_a_cols)
        df = pd.read_csv('a_nonmatch_csv.csv', header=None, names=names)
        for i, col in zip([1, 2, 4, 5], a_cols[1:]):
            assert_allclose(df[col], self.data[self.af, i])
        for i, col in zip(add_a_nums, add_a_cols):
            assert_allclose(df[col], self.data[self.af, i])
        assert np.all([df[a_cols[0]].iloc[i] == self.data1[self.af[i], 0] for i in
                       range(len(self.af))])
        assert_allclose(df['MATCH_P'], self.pfa)
        assert_allclose(df['A_AVG_CONT'], self.aff)
        assert_allclose(df['NNM_SEPARATION'], self.afs)
        assert_allclose(df['NNM_ETA'], self.afeta)
        assert_allclose(df['NNM_XI'], self.afxi)
        names = np.append(np.append(b_cols, ['MATCH_P', 'NNM_SEPARATION', 'NNM_ETA', 'NNM_XI',
                                             'B_AVG_CONT']), add_b_cols)
        df = pd.read_csv('b_nonmatch_csv.csv', header=None, names=names)
        for i, col in zip([1, 2, 4, 5, 6], b_cols[1:]):
            assert_allclose(df[col], self.datab[self.bf, i])
        # Only extra column in test_b_data.csv is the "B_Err" column from above,
        # self.data2[:, 3]. Test that these data do not match any output data:
        for n in names[1:]:
            with pytest.raises(AssertionError, match='Not equal to tolerance'):
                assert_allclose(df[n], self.datab[self.bf, 3])
        assert np.all([df[b_cols[0]].iloc[i] == self.data2[self.bf[i], 0] for i in
                       range(len(self.bf))])
        assert_allclose(df['MATCH_P'], self.pfb)
        assert_allclose(df['B_AVG_CONT'], self.bff)
        assert_allclose(df['NNM_SEPARATION'], self.bfs)
        assert_allclose(df['NNM_ETA'], self.bfeta)
        assert_allclose(df['NNM_XI'], self.bfxi)
