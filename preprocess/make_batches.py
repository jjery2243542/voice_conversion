import sys
sys.path.append('..')
from utils import Sampler
import h5py
import numpy as np

max_step=5
batch_size=16
n_batches=20000

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('usage: python3 make_batches.py [in_h5py_path] [speaker_sex_path] [out_h5py_path]')
        exit(0)
    sampler = Sampler(sys.argv[1], sys.argv[2], max_step)
    with h5py.File(sys.argv[3], 'w') as f_h5:
        for i in range(n_batches):
            X_i_t, X_i_tk, X_i_tk_prime, X_j = [], [], [], []
            for j in range(batch_size):
                x_i_t, x_i_tk, x_i_tk_prime, x_j = sampler.sample()
                X_i_t.append(x_i_t)
                X_i_tk.append(x_i_tk)
                X_i_tk_prime.append(x_i_tk_prime)
                X_j.append(x_j)
            X_i_t = np.array(X_i_t, dtype=np.float32)
            X_i_tk = np.array(X_i_tk, dtype=np.float32)
            X_i_tk_prime = np.array(X_i_tk_prime, dtype=np.float32)
            X_j = np.array(X_j, dtype=np.float32)
            f_h5.create_dataset('{}/X_i_t'.format(i), data=X_i_t, dtype=np.float32)
            f_h5.create_dataset('{}/X_i_tk'.format(i), data=X_i_tk, dtype=np.float32)
            f_h5.create_dataset('{}/X_i_tk_prime'.format(i), data=X_i_tk_prime, dtype=np.float32)
            f_h5.create_dataset('{}/X_j'.format(i), data=X_j, dtype=np.float32)
            if i % 10 == 0:
                print('process [{}/{}]'.format(i, n_batches))

