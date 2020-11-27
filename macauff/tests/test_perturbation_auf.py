# Licensed under a 3-clause BSD style license - see LICENSE
'''
Tests for the "perturbation_auf" module.
'''

import pytest
import os
import numpy as np

from ..matching import CrossMatch


class TestCreatePerturbAUF:
    def setup_class(self):
        self.cm = CrossMatch(os.path.join(os.path.dirname(__file__), 'data/crossmatch_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_a_params.txt'),
                             os.path.join(os.path.dirname(__file__), 'data/cat_b_params.txt'))
        self.cm.a_auf_region_points = np.array([[0, 0], [50, 50]], dtype=float)
        self.cm.b_auf_region_points = np.array([[0, 0], [50, 50]], dtype=float)
        self.cm.mem_chunk_num = 4
        self.files_per_auf_sim = 7

    @pytest.mark.filterwarnings("ignore:Incorrect number of files in")
    def test_not_implemented_error(self):
        # Reset any saved files from these tests
        os.system("rm -rf {}/*".format(self.cm.a_auf_folder_path))
        os.system("rm -rf {}/*".format(self.cm.b_auf_folder_path))
        self.cm.include_perturb_auf = True
        self.cm.a_psf_fwhms = np.array([0.1, 0.1, 0.1])
        self.cm.a_download_tri = False
        with pytest.raises(NotImplementedError, match="Perturbation AUF components are not "
                           "currently included"):
            self.cm.create_perturb_auf(self.files_per_auf_sim)

    def test_no_perturb_outputs(self):
        # Randomly generate two catalogues (x3 files) between coordinates
        # 0, 0 and 50, 50.
        rng = np.random.default_rng()
        for path, Nf, size in zip([self.cm.a_cat_folder_path, self.cm.b_cat_folder_path], [3, 4],
                                  [25, 54]):
            cat = np.zeros((size, 3), float)
            rand_inds = rng.permutation(cat.shape[0])[:size // 2 - 1]
            cat[rand_inds, 0] = 50
            cat[rand_inds, 1] = 50
            cat += rng.uniform(-0.1, 0.1, cat.shape)
            np.save('{}/con_cat_astro.npy'.format(path), cat)

            cat = rng.uniform(10, 20, (size, Nf))
            np.save('{}/con_cat_photo.npy'.format(path), cat)

            cat = rng.choice(Nf, size=(size,))
            np.save('{}/magref.npy'.format(path), cat)

        self.cm.include_perturb_auf = False
        self.cm.run_auf = True
        self.cm.create_perturb_auf(self.files_per_auf_sim)
        lenr = len(self.cm.r)
        lenrho = len(self.cm.rho)
        for coord in ['0.0', '50.0']:
            for filt in ['W1', 'W2', 'W3', 'W4']:
                path = '{}/{}/{}/{}'.format(self.cm.b_auf_folder_path, coord, coord, filt)
                for filename, shape in zip(['frac', 'flux', 'offset', 'cumulative', 'fourier',
                                            'N', 'mag'],
                                           [(2, 1), (1,), (lenr-1, 1), (lenr-1, 1), (lenrho-1, 1),
                                            (1, 1), (1, 1)]):
                    assert os.path.isfile('{}/{}.npy'.format(path, filename))
                    file = np.load('{}/{}.npy'.format(path, filename))
                    assert np.all(file.shape == shape)
                assert np.all(np.load('{}/frac.npy'.format(path)) == 0)
                assert np.all(np.load('{}/cumulative.npy'.format(path)) == 1)
                assert np.all(np.load('{}/fourier.npy'.format(path)) == 1)
                assert np.all(np.load('{}/mag.npy'.format(path)) == 1)
                file = np.load('{}/offset.npy'.format(path))
                assert np.all(file[1:] == 0)
                assert file[0] == 1/(2 * np.pi * (self.cm.r[0] + self.cm.dr[0]/2) * self.cm.dr[0])

        file = np.load('{}/modelrefinds.npy'.format(self.cm.a_auf_folder_path))
        assert np.all(file[0, :] == 0)
        assert np.all(file[1, :] == np.load('{}/magref.npy'.format(self.cm.a_cat_folder_path)))

        # Select AUF pointing index based on a 0 vs 50 cut in longitude.
        cat = np.load('{}/con_cat_astro.npy'.format(self.cm.a_cat_folder_path))
        inds = np.ones(file.shape[1], int)
        inds[np.where(cat[:, 0] < 1)[0]] = 0
        assert np.all(file[2, :] == inds)

    def test_run_auf_file_number(self):
        # Reset any saved files from the above tests
        os.system("rm -rf {}/*".format(self.cm.a_auf_folder_path))
        os.system("rm -rf {}/*".format(self.cm.b_auf_folder_path))
        self.cm.run_auf = False
        with pytest.warns(UserWarning, match='Incorrect number of files in catalogue "a"'):
            self.cm.create_perturb_auf(self.files_per_auf_sim)

        # Now create fake files to simulate catalogue "a" having the right files.
        # For 2 AUF pointings this comes to 5 + 2*N_filt*files_per_auf_sim files.
        os.system("rm -rf {}/*".format(self.cm.a_auf_folder_path))
        for i in range(5 + 2 * 3 * self.files_per_auf_sim):
            np.save('{}/random_file_{}.npy'.format(self.cm.a_auf_folder_path, i), np.zeros(1))

        # This should still return the same warning, just for catalogue "b" now.
        with pytest.warns(UserWarning) as record:
            self.cm.create_perturb_auf(self.files_per_auf_sim)
        assert len(record) == 1
        assert 'Incorrect number of files in catalogue "b"' in record[0].message.args[0]

    @pytest.mark.filterwarnings("ignore:Incorrect number of files in")
    def test_load_auf_print(self, capsys):
        # Reset any saved files from the above tests
        os.system("rm -rf {}/*".format(self.cm.a_auf_folder_path))
        os.system("rm -rf {}/*".format(self.cm.b_auf_folder_path))

        # Generate new dummy data for catalogue "b"'s AUF folder.
        for i in range(5 + 2 * 4 * self.files_per_auf_sim):
            np.save('{}/random_file_{}.npy'.format(self.cm.b_auf_folder_path, i), np.zeros(1))
        capsys.readouterr()
        # This test will create catalogue "a" files because of the wrong
        # number of files (zero) in the folder.
        self.cm.create_perturb_auf(self.files_per_auf_sim)
        output = capsys.readouterr().out
        assert 'Loading empirical crowding AUFs for catalogue "a"' not in output
        assert 'Loading empirical crowding AUFs for catalogue "b"' in output

        os.system("rm -rf {}/*".format(self.cm.a_auf_folder_path))
        os.system("rm -rf {}/*".format(self.cm.b_auf_folder_path))
        # Generate new dummy data for each catalogue's AUF folder.
        for path, fn in zip([self.cm.a_auf_folder_path, self.cm.b_auf_folder_path], [3, 4]):
            for i in range(5 + 2 * fn * self.files_per_auf_sim):
                np.save('{}/random_file_{}.npy'.format(path, i), np.zeros(1))
        capsys.readouterr()
        self.cm.create_perturb_auf(self.files_per_auf_sim)
        output = capsys.readouterr().out
        assert 'Loading empirical crowding AUFs for catalogue "a"' in output
        assert 'Loading empirical crowding AUFs for catalogue "b"' in output
